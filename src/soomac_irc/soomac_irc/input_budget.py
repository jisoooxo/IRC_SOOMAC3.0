import json

# 2026-11-01: 모델 입력이 서빙 최대 길이를 넘지 않게 맞춘다.
# vLLM 서버는 "입력 + 출력"이 max_model_len을 넘는 요청을 거부한다. 넘기 전에 가장 오래된 대화부터 뺀다.
# 학습(r5)은 대화를 자르지 않는다(초과 행은 오류). 학습 행의 최대 입력이 이 한도보다 짧아 학습에서 본 입력은 여기서도 그대로다.
# HF 경로에도 똑같이 적용해 백엔드가 바뀌어도 모델이 보는 입력이 같게 한다.
SERVING_MAX_MODEL_LEN = 16384  # tools/vllm_server/start_vllm_server.sh의 VLLM_MAX_LEN과 같아야 한다

# 대화 기록을 모두 뺀 뒤 남는 부분(주문 상태·이번 발화 등)이 커질 여유. 시작 시 검사에 쓴다.
DYNAMIC_INPUT_RESERVE_TOKENS = 1024


class InputBudgetError(ValueError):
    pass


def input_token_limit(max_new_tokens: int) -> int:
    # 출력 자리를 미리 비워 두어 대답이 중간에 잘리지 않게 한다.
    return SERVING_MAX_MODEL_LEN - max_new_tokens


def render_inputs(processor, system_prompt: str, model_input: dict):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(model_input, ensure_ascii=False, separators=(",", ":"))},
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    return messages, inputs


def fit_recent_history(processor, system_prompt: str, model_input: dict, limit: int, agent: str):
    """입력이 limit 이하가 될 때까지 recent_history에서 가장 오래된 메시지부터 뺀다.

    한 번에 가장 오래된 메시지 하나와 그 뒤의 assistant 메시지를 함께 빼서 남은 기록이 항상 user로 시작한다.
    (시작 안내가 assistant 단독으로 들어가 기록 길이가 홀수일 때 user 없는 답이 맨 앞에 남지 않게 한다.)

    평소처럼 limit 이하면 아무것도 바꾸지 않는다. 반환: (messages, inputs, 뺀 메시지 수).
    """
    dropped = 0
    while True:
        messages, inputs = render_inputs(processor, system_prompt, model_input)
        tokens = inputs["input_ids"].shape[1]
        if tokens <= limit:
            return messages, inputs, dropped
        history = model_input.get("recent_history") or []
        if not history:
            raise InputBudgetError(
                f"{agent} 입력 {tokens}토큰이 한도 {limit}를 넘는데 뺄 대화 기록이 없음(프롬프트·상태가 너무 김)"
            )
        removed = 1
        while removed < len(history) and history[removed].get("role") != "user":
            removed += 1
        model_input["recent_history"] = history[removed:]
        dropped += removed


def check_static_budget(processor, system_prompt: str, limit: int, agent: str) -> None:
    # 노드 시작 시 검사: 대화 기록 없이도 system prompt가 한도 안에 여유 있게 들어가야 한다.
    # 데이터 쪽에서 프롬프트를 늘려 한도를 넘게 되면 대화 도중이 아니라 시작 순간에 알린다.
    _, inputs = render_inputs(processor, system_prompt, {})
    tokens = inputs["input_ids"].shape[1]
    if tokens + DYNAMIC_INPUT_RESERVE_TOKENS > limit:
        raise InputBudgetError(
            f"{agent} system prompt {tokens}토큰 + 여유 {DYNAMIC_INPUT_RESERVE_TOKENS} > 입력 한도 {limit}. "
            "프롬프트를 줄이거나 SERVING_MAX_MODEL_LEN·VLLM_MAX_LEN을 함께 올려야 함"
        )
