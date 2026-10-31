import copy
import hashlib

import torch

# 2026-11-01: Agent별 고정 system prompt(+few-shot)를 매 호출 다시 계산하지 않기 위한 접두 KV 캐시.
# 실측 기준 고정 앞부분은 Decision 6040 / Recommendation 3338 / Response(task) 5280 토큰이고
# 호출마다 바뀌는 history·state·사용자 말은 그 뒤에 붙는 수백 토큰뿐이다.
# False로 바꾸면 종전처럼 매 호출 전체 입력을 처음부터 계산한다.
PREFIX_CACHE_ENABLED = True

# system 뒤에 붙는 user 내용 자리. 실제 입력과 토큰이 갈라지는 지점을 찾는 용도로만 쓴다.
_PLACEHOLDER_USER = "\u0000"


def _is_hf_model(model) -> bool:
    # 테스트용 가짜 모델은 캐시 경로를 타지 않고 종전 generate를 그대로 호출한다.
    return isinstance(model, torch.nn.Module) and hasattr(model, "config")


class PrefixKVCache:
    """(Adapter, system prompt)마다 앞부분 KV를 한 번만 계산해 두고 호출마다 복사해서 이어 쓴다.

    - KV는 Adapter 가중치에 따라 달라지므로 Adapter 이름을 key에 넣어 D/Q/P끼리 절대 섞지 않는다.
    - 매 호출 실제 input_ids 앞부분이 저장된 토큰과 완전히 같을 때만 사용한다(아니면 종전 경로).
    - 이전 답을 재사용하는 것이 아니다. 앞부분 계산 결과만 재사용하고 뒷부분·출력은 매번 새로 계산한다.
    - Adapter 가중치 버전은 key에 없다. 프로세스 안에서 Adapter를 다시 로드하면 반드시 clear()를 부른다.
    - 잠금이 없다. 현재처럼 워커 스레드 하나에서 순차 호출할 때만 안전하다.
    - 캐시 유무에 따라 bf16/NF4 계산 순서가 달라져 거의 동점인 토큰에서 greedy 출력이 바뀔 수 있다(버그 아님).
      품질 판정은 토큰 일치가 아니라 의미 시나리오 회귀로 한다.
    """

    def __init__(self, processor, logger=None, enabled: bool | None = None):
        self.processor = processor
        self.logger = logger
        # None이면 생성 시점의 PREFIX_CACHE_ENABLED를 따른다(시험에서 모듈 값만 바꿔 OFF/ON 비교).
        self.enabled = PREFIX_CACHE_ENABLED if enabled is None else enabled
        self.entries = {}  # (adapter key, system prompt sha256) -> (prefix_ids, cache)
        self.last_event = None  # 마지막 호출의 hit/build/miss/off 기록. trace에 그대로 남긴다.

    def clear(self):
        self.entries.clear()

    def _build(self, model, key, system_prompt: str, input_ids: torch.Tensor):
        # system + 자리표시 user를 렌더링해 실제 입력과 처음 달라지는 토큰 직전까지를 접두로 잡는다.
        # 생성에는 최소 1개 새 토큰이 필요하므로 전체 길이 - 1을 넘지 않게 자른다.
        rendered = self.processor.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": _PLACEHOLDER_USER}],
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        candidate = rendered["input_ids"][0].to(input_ids.device)
        limit = min(candidate.shape[0], input_ids.shape[1] - 1)
        mismatch = (candidate[:limit] != input_ids[0, :limit]).nonzero()
        length = int(mismatch[0]) if mismatch.numel() else limit
        if length <= 0:
            return None

        prefix_ids = input_ids[:, :length].clone()
        # 접두 6천 토큰 전체 logits(수 GB)를 만들지 않도록 마지막 1개만 계산한다.
        output = model(
            input_ids=prefix_ids,
            attention_mask=torch.ones_like(prefix_ids),
            use_cache=True,
            logits_to_keep=1,
        )
        cache = output.past_key_values
        # transformers 5.14+ 슬라이딩 층은 최근 1023개만 남기지만 원본 텐서의 view로 저장한다.
        # view가 살아 있으면 접두 전체 길이의 원본 메모리가 해제되지 않으므로 실제 크기로 복사해 둔다.
        # (Decision 접두 기준 1.97 GiB → 0.44 GiB 실측. 값·길이 정보는 그대로다.)
        # inference_mode에서는 view의 _base가 기록되지 않으므로 "저장소가 내용보다 큰가"로 판단한다.
        for layer in getattr(cache, "layers", []):
            for name in ("keys", "values"):
                tensor = getattr(layer, name, None)
                if (isinstance(tensor, torch.Tensor)
                        and tensor.untyped_storage().nbytes() > tensor.numel() * tensor.element_size()):
                    setattr(layer, name, tensor.clone(memory_format=torch.contiguous_format))
        entry = (prefix_ids, cache)
        self.entries[key] = entry
        if self.logger is not None:
            self.logger.info(f"Prefix KV cache 생성: adapter={key[0]}, tokens={length}")
        return entry

    def generate(self, model, adapter_key: str | None, system_prompt: str, generate_args: dict):
        # 호출자가 Adapter 활성/비활성을 이미 맞춘 상태에서 부른다. 캐시도 그 상태로 만들어진다.
        input_ids = generate_args.get("input_ids")
        if not (self.enabled and _is_hf_model(model)) or input_ids is None or input_ids.shape[0] != 1:
            self.last_event = {"status": "off"}
            return model.generate(**generate_args)

        key = (adapter_key or "__base__", hashlib.sha256(system_prompt.encode("utf-8")).hexdigest())
        entry = self.entries.get(key)
        status = "hit"
        if entry is None:
            entry = self._build(model, key, system_prompt, input_ids)
            status = "build"
        if entry is not None:
            prefix_ids, cache = entry
            length = prefix_ids.shape[1]
            if input_ids.shape[1] > length and torch.equal(input_ids[0, :length], prefix_ids[0]):
                self.last_event = {"status": status, "cached_tokens": length}
                # 저장본은 다음 호출에도 써야 하므로 복사본에 이어 쓴다.
                # input_ids/attention_mask는 전체를 넘겨야 position이 맞게 잡힌다(generate가 캐시 길이만큼 잘라 계산).
                return model.generate(**generate_args, past_key_values=copy.deepcopy(cache))

        self.last_event = {"status": "miss"}
        if self.logger is not None:
            self.logger.warning(f"Prefix KV cache 미사용: adapter={key[0]} 입력 앞부분이 저장된 접두와 다름")
        return model.generate(**generate_args)
