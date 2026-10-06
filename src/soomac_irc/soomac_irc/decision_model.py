import copy
import json
import time

import torch
import xgrammar as xgr
from xgrammar.contrib.hf import LogitsProcessor as XGrammarLogitsProcessor

from soomac_irc.agent_contract import (
    DECISION_SCHEMA,
    DuplicateDecisionKeyError,
    NormalizedDecision,
    SessionState,
    normalize_decision,
)
from soomac_irc.agent_prompts import DECISION_SYSTEM

DECISION_HISTORY_TURNS = 10
DECISION_INPUT_MAX_TOKENS = 8192
DECISION_MAX_TOKENS = 1024
DECISION_KEYS_ANY_ORDER = True


#################### Decision 모델 입력 만들기 ####################

def build_decision_model_input(session: SessionState, user_text: str, robot_state: dict, repair: dict | None = None) -> dict:
    # 모델은 최근 자연어 대화와 현재 확정 state를 함께 본다.
    # 과거 의미를 Python이 다시 구조화하지 않고 history 원문을 그대로 전달한다.
    model_input = {
        "recent_history": copy.deepcopy(session["history"][-(DECISION_HISTORY_TURNS * 2):]),
        "order": copy.deepcopy(session["order"]),
        "preferences": copy.deepcopy(session["preferences"]),
        "pending": copy.deepcopy(session.get("pending")),
        "robot_state": copy.deepcopy(robot_state),
        "message": user_text.strip(),
    }
    if repair is not None:
        model_input["repair"] = copy.deepcopy(repair)
    return model_input


def build_decision_inputs(session: SessionState, user_text: str, robot_state: dict, processor, repair: dict | None = None):
    # processor가 만든 실제 token 수를 기준으로 입력 길이를 맞춘다.
    # 길면 가장 오래된 user/assistant 한 쌍부터 제거하고 현재 state와 message는 끝까지 보존한다.
    model_input = build_decision_model_input(session, user_text, robot_state, repair)

    while True:
        messages = [
            {"role": "system", "content": DECISION_SYSTEM},
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
        prompt_tokens = inputs["input_ids"].shape[1]
        if prompt_tokens <= DECISION_INPUT_MAX_TOKENS:
            return messages, inputs, model_input

        history = model_input["recent_history"]
        if history:
            model_input["recent_history"] = history[2:] if len(history) >= 2 else []
            continue
        raise ValueError(f"Decision 입력 token 초과: {prompt_tokens}/{DECISION_INPUT_MAX_TOKENS}")


#################### 모델 JSON을 내부 Decision으로 변환 ####################

def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict:
    # json.loads 기본 동작은 중복 key의 마지막 값만 남긴다.
    # 실행 관련 key가 조용히 바뀌지 않도록 중복이 하나라도 있으면 해당 턴을 거절한다.
    parsed = {}
    for key, value in pairs:
        if key in parsed:
            raise DuplicateDecisionKeyError(f"Decision JSON duplicate key: {key}")
        parsed[key] = value
    return parsed


def _parse_sparse_decision(raw: str) -> dict:
    # repair 입력에는 모델이 실제로 출력한 외부 계약 JSON을 그대로 보존해야 한다.
    # 내부 기본값을 채우기 전 단계에서도 duplicate key 거절 정책은 동일하게 적용한다.
    return json.loads(raw, object_pairs_hook=_reject_duplicate_json_keys)


#################### 실제 Decision 추론 함수 만들기 ####################

def make_call_decision(model, processor, logger=None):
    # 모델과 tokenizer마다 stop token이 다를 수 있어 유효한 id만 중복 없이 사용한다.
    tokenizer = processor.tokenizer
    stop_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<turn|>")]
    stop_ids = list(dict.fromkeys(i for i in stop_ids if isinstance(i, int) and i >= 0))
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=len(tokenizer), stop_token_ids=stop_ids)
    # JSON key 순서는 의미가 아니므로 top-level과 중첩 객체 모두 자유롭게 출력하게 한다.
    # 필드 이름·자료형·허용값 검사는 그대로 유지하며 여러 의미를 한 JSON에 함께 담을 수 있다.
    compiled_grammar = xgr.GrammarCompiler(tokenizer_info).compile_json_schema(
        DECISION_SCHEMA,
        any_order=DECISION_KEYS_ANY_ORDER,
    )

    @torch.inference_mode()
    def call_decision(
        session: SessionState,
        user_text: str,
        robot_state: dict,
        repair: dict | None = None,
    ) -> NormalizedDecision:
        # 한 호출의 입력·원문 출력·정규화 결과·지연 시간을 trace 하나에 모은다.
        # 호출마다 초기화해 이전 턴의 sparse 출력이 실패한 호출에 섞이지 않게 한다.
        call_decision.last_external_output = None
        started = time.perf_counter()
        trace = {
            "stage": "decision",
            "repair": copy.deepcopy(repair),
            "user_text": user_text.strip(),
            "robot_state": copy.deepcopy(robot_state),
        }
        try:
            messages, inputs, model_input = build_decision_inputs(session, user_text, robot_state, processor, repair)
            inputs = inputs.to(model.device)
            prompt_tokens = inputs["input_ids"].shape[1]

            # sampling 없이 같은 입력은 같은 Decision이 나오게 하고, XGrammar로 JSON 모양을 보장한다.
            output = model.generate(
                **inputs,
                max_new_tokens=DECISION_MAX_TOKENS,
                do_sample=False,
                use_cache=True,
                eos_token_id=stop_ids,
                logits_processor=[XGrammarLogitsProcessor(compiled_grammar)],
            )
            output_tokens = output[0].shape[0] - prompt_tokens
            raw = processor.decode(output[0][prompt_tokens:], skip_special_tokens=True).strip()
            sparse = _parse_sparse_decision(raw)
            parsed = normalize_decision(sparse)
            call_decision.last_external_output = copy.deepcopy(sparse)
            trace.update({
                "model_input": model_input,
                "messages": copy.deepcopy(messages),
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "raw": raw,
                "external": copy.deepcopy(sparse),
                "parsed": copy.deepcopy(parsed),
            })
            if logger is not None:
                logger.info(f"Decision raw: {raw}")
            return parsed
        except Exception as error:
            trace["error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            trace["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
            call_decision.trace_events.append(copy.deepcopy(trace))

    call_decision.trace_events = []
    call_decision.last_external_output = None
    return call_decision
