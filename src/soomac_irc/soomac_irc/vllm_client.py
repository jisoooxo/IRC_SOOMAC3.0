import base64
import contextlib
import io
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

# 2026-11-01: LLM 노드가 모델을 직접 올리지 않고 vLLM 서버(tools/vllm_server/start_vllm_server.sh)에 요청하는 경로.
# 표준 라이브러리만 쓴다(ROS Python 3.10 환경에 새 패키지를 늘리지 않는다).
VLLM_BASE_URL = "http://127.0.0.1:8000"
VLLM_BASE_MODEL = "gemma4-base"  # 서버의 --served-model-name. Adapter 없이 base를 부를 때 쓴다(VLM).
VLLM_TIMEOUT_S = 120.0


class VllmError(RuntimeError):
    pass


def _post(path: str, payload: dict, timeout: float = VLLM_TIMEOUT_S) -> dict:
    request = urllib.request.Request(
        VLLM_BASE_URL + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise VllmError(f"vLLM {path} HTTP {error.code}: {error.read().decode('utf-8', 'replace')[:500]}") from error
    except urllib.error.URLError as error:
        raise VllmError(f"vLLM 서버 연결 실패({VLLM_BASE_URL}): {error.reason}") from error


def wait_until_ready(timeout_s: float = 600.0, interval_s: float = 2.0) -> None:
    # 서버가 모델·Adapter를 다 올릴 때까지 /health를 기다린다.
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            with urllib.request.urlopen(VLLM_BASE_URL + "/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        if time.monotonic() > deadline:
            raise VllmError(f"vLLM 서버가 {timeout_s:.0f}초 안에 준비되지 않음: {VLLM_BASE_URL}")
        time.sleep(interval_s)


def complete_tokens(
    model: str,
    prompt_token_ids: list[int],
    max_tokens: int,
    stop_token_ids: list[int],
    grammar: str | None = None,
) -> dict:
    # HF processor가 만든 토큰 ID를 그대로 보낸다(채팅 템플릿 차이가 끼어들지 않게).
    # model에는 Adapter 이름(decision/recommendation/response)을 넣는다. greedy·비thinking은 호출자 입력과 같다.
    payload = {
        "model": model,
        "prompt": prompt_token_ids,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stop_token_ids": stop_token_ids,
        "skip_special_tokens": True,
        "return_token_ids": True,
    }
    if grammar is not None:
        payload["structured_outputs"] = {"grammar": grammar}
    body = _post("/v1/completions", payload)
    choice = body["choices"][0]
    usage = body.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return {
        "text": choice["text"],
        "token_ids": choice.get("token_ids"),
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cached_tokens": details.get("cached_tokens"),
    }


def chat_with_images(
    system_prompt: str,
    user_text: str,
    pil_images: list,
    max_tokens: int,
    model: str = VLLM_BASE_MODEL,
) -> dict:
    # VLM 판정용. 이미지는 PNG base64 data URL로 보낸다. Adapter 없이 base 모델을 부른다.
    content = []
    for image in pil_images:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}})
    content.append({"type": "text", "text": user_text})
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    body = _post("/v1/chat/completions", payload)
    choice = body["choices"][0]
    usage = body.get("usage") or {}
    return {
        "text": choice["message"]["content"] or "",
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


#################### LLM 노드용 백엔드: HF 모델 자리에 그대로 넣는 대리 객체 ####################



class VllmModel:
    """decision/recommendation/response의 `model.generate(...)` 호출부를 바꾸지 않고 vLLM 서버로 보낸다.

    - set_adapter / disable_adapter / active_adapter: PEFT 모델과 같은 이름으로 Adapter를 고른다(None = base).
    - generate: input_ids를 그대로 서버에 보내고, XGrammar 처리기에서 문법 문자열을 꺼내 서버 문법으로 넘긴다.
      반환은 HF와 같은 [1, 입력+출력] 텐서라 호출부의 decode·토큰 수 계산이 그대로 동작한다.
    - greedy만 지원한다(현재 Agent 호출은 전부 do_sample=False).
    """

    device = "cpu"

    def __init__(self, max_model_len: int | None = None):
        self.active_adapter = None
        self.last_usage = None
        # 서버 최대 길이(입력+출력). 출력 한도를 남은 길이로 줄여 서버 거부를 막는다(실제 출력은 수십 토큰).
        self.max_model_len = max_model_len

    def set_adapter(self, adapter_name: str) -> None:
        self.active_adapter = adapter_name

    @contextlib.contextmanager
    def disable_adapter(self):
        previous, self.active_adapter = self.active_adapter, None
        try:
            yield
        finally:
            self.active_adapter = previous

    def generate(self, input_ids, max_new_tokens, eos_token_id, logits_processor=None, do_sample=False, **_unused):
        import torch

        if do_sample:
            raise VllmError("vLLM 백엔드는 greedy(do_sample=False)만 지원함")
        if input_ids.shape[0] != 1:
            raise VllmError("vLLM 백엔드는 batch 1만 지원함")
        grammar = None
        for processor in logits_processor or []:
            compiled = getattr(processor, "compiled_grammars", None)
            if compiled:
                grammar = str(compiled[0].grammar)
        stop_ids = list(eos_token_id) if isinstance(eos_token_id, (list, tuple)) else [int(eos_token_id)]
        prompt = input_ids[0].tolist()
        max_tokens = int(max_new_tokens)
        if self.max_model_len is not None:
            room = self.max_model_len - len(prompt)
            if room <= 0:
                raise VllmError(f"입력 {len(prompt)}토큰이 vLLM 서버 최대 길이 {self.max_model_len} 이상. 서버를 VLLM_MAX_LEN을 올려 재시작")
            max_tokens = min(max_tokens, room)
        result = complete_tokens(self.active_adapter or VLLM_BASE_MODEL, prompt, max_tokens, stop_ids, grammar)
        if result.get("token_ids") is None:
            # 빈 출력으로 넘기면 호출부가 JSON 파싱 오류로 오인한다. 전송 계약 위반으로 바로 알린다.
            raise VllmError("vLLM 응답에 token_ids가 없음(return_token_ids 미지원 서버?)")
        self.last_usage = {key: result[key] for key in ("prompt_tokens", "completion_tokens", "cached_tokens", "finish_reason")}
        return torch.tensor([prompt + list(result["token_ids"])], dtype=torch.long)


def served_models() -> list[dict]:
    # 서버 /v1/models 목록(base + Adapter). Adapter의 root는 서버가 읽은 경로다.
    try:
        with urllib.request.urlopen(VLLM_BASE_URL + "/v1/models", timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))["data"]
    except (urllib.error.URLError, ConnectionError, TimeoutError) as error:
        raise VllmError(f"vLLM 모델 목록 조회 실패: {error}") from error


def served_adapter_roots() -> dict:
    return {model["id"]: model.get("root") for model in served_models() if model["id"] != VLLM_BASE_MODEL}


def check_served_adapters(expected: dict, served_roots: dict) -> None:
    # 서버가 올린 변환본이 지금 model_runtime이 가리키는 학습 결과에서 만들어졌는지 확인한다.
    # 새 Adapter로 바꿨는데 변환·서버 재시작을 잊으면 예전 Adapter로 조용히 도는 것을 막는다.
    # 한계: 같은 경로에 다시 변환만 하고 서버를 재시작하지 않은 경우는 잡지 못한다(변환 후 반드시 재시작).
    for name, source in expected.items():
        if source is None:
            continue
        root = served_roots.get(name)
        if root is None:
            raise VllmError(f"vLLM 서버에 {name} Adapter가 등록되지 않음. 서버 실행 인자 확인")
        record = Path(root) / "CONVERSION.json"
        if not record.is_file():
            raise VllmError(f"{name} vLLM용 변환 기록 없음: {record}. convert_adapters_for_vllm.py로 만든 경로인지 확인")
        converted_from = Path(json.loads(record.read_text(encoding="utf-8"))["source"]).resolve()
        if converted_from != Path(source).resolve():
            raise VllmError(f"{name} 서버 Adapter가 현재 Adapter와 다름: 변환 원본 {converted_from} / 현재 {source}. 다시 변환 후 서버 재시작")


def load_vllm_backend(model_path: str, adapter_paths: dict):
    # 노드는 모델을 GPU에 올리지 않는다. 토큰화·채팅 템플릿용 processor만 CPU로 읽는다.
    from transformers import AutoProcessor

    wait_until_ready()
    models = served_models()
    check_served_adapters(adapter_paths, {m["id"]: m.get("root") for m in models if m["id"] != VLLM_BASE_MODEL})
    max_model_len = next((m.get("max_model_len") for m in models if m["id"] == VLLM_BASE_MODEL), None)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    return VllmModel(max_model_len=max_model_len), processor


def make_call_vlm_remote(max_tokens: int, logger=None):
    # model_runtime.make_call_vlm과 같은 입출력. 이미지는 서버가 전처리하고 Adapter 없이 base로 판정한다.
    def call_vlm(pil_images: list, system_prompt: str, user_text: str) -> str:
        if not isinstance(pil_images, list) or not pil_images:
            raise ValueError("VLM 입력 이미지가 없음")
        result = chat_with_images(system_prompt, user_text, pil_images, max_tokens)
        reply = result["text"].strip()
        # 비thinking이어도 "thought" 글자를 흉내 내는 경우가 있어 HF 경로와 같이 잘라 낸다.
        if reply.startswith("thought"):
            reply = reply[len("thought"):].lstrip()
        if logger is not None:
            logger.info(f"VLM prompt tokens: {result['prompt_tokens']}, images: {len(pil_images)} (vLLM)")
            logger.info(f"VLM output: {reply}")
        return reply

    return call_vlm
