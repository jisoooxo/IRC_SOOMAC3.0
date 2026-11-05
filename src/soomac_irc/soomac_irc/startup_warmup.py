"""Prime shared D/Q/P prompt prefixes before accepting any live input."""
import time

from soomac_irc import vllm_client
from soomac_irc.vllm_client import VllmError, VllmModel


def _prefix_tokens(processor, system_prompt):
    # Also include empty user content: no shared synthetic user text may enter
    # the prefix even if the two nonempty probes tokenize with a common start.
    variants = []
    for user_text in ("", "A", "B"):
        inputs = processor.apply_chat_template(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": user_text}],
            add_generation_prompt=True, tokenize=True, return_dict=True,
            return_tensors="pt", enable_thinking=False,
        )
        variants.append(inputs["input_ids"][0].tolist())
    common = 0
    for tokens in zip(*variants):
        if len(set(tokens)) != 1:
            break
        common += 1
    if common == 0 or common >= min(map(len, variants)):
        raise VllmError("워밍업 공통 프롬프트 경계를 확인할 수 없음")
    return variants[0][:common]


def _warmup_specs(processor):
    import xgrammar as xgr
    from soomac_irc.agent_contract import DECISION_SCHEMA, RECOMMENDATION_SCHEMA, RESPONSE_SCHEMA
    from soomac_irc.agent_prompts import DECISION_SYSTEM, RECOMMENDATION_SYSTEM
    from soomac_irc.decision_model import DECISION_KEYS_ANY_ORDER
    from soomac_irc.response_model import RESPONSE_SYSTEM_BY_ROUTE

    tokenizer = processor.tokenizer
    stop_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<turn|>")]
    stop_ids = list(dict.fromkeys(i for i in stop_ids if isinstance(i, int) and i >= 0))
    info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=len(tokenizer), stop_token_ids=stop_ids)
    compiler = xgr.GrammarCompiler(info)
    for role, system, schema, options in (
        ("decision", DECISION_SYSTEM, DECISION_SCHEMA, {"any_order": DECISION_KEYS_ANY_ORDER}),
        ("recommendation", RECOMMENDATION_SYSTEM, RECOMMENDATION_SCHEMA, {}),
        ("response", RESPONSE_SYSTEM_BY_ROUTE["task"], RESPONSE_SCHEMA, {}),
    ):
        grammar = str(compiler.compile_json_schema(schema, **options).grammar)
        yield role, _prefix_tokens(processor, system), stop_ids, grammar


def warmup_vllm_agents(model, processor, adapter_names, logger=None):
    """Prefill shared prompts + one disposable token per adapter; return timings.

    No session, graph, publisher or live history is used. Partial output (usually
    finish_reason=length) is intentional and never parsed as a model answer.
    P/task is warmed; general/mixed and evicted cache entries may still be cold.
    HTTP requests use the existing 120-second timeout, without retries.
    """
    if not isinstance(model, VllmModel):
        raise TypeError("Startup warmup requires the vLLM client backend")
    if set(adapter_names) != {"decision", "recommendation", "response"} or not all(adapter_names.values()):
        raise ValueError("Startup warmup requires all three explicit adapter names")
    previous_adapter, previous_usage = model.active_adapter, model.last_usage
    started = time.perf_counter()
    timings = {}
    role = "initialization"
    try:
        for role, prefix, stop_ids, grammar in _warmup_specs(processor):
            if model.max_model_len is None or len(prefix) + 1 > model.max_model_len:
                raise VllmError(f"{role} 워밍업 입력이 서버 길이 한도를 초과함")
            if logger is not None:
                logger.info(f"LLM 시작 워밍업: {role} 공통 접두 {len(prefix)}토큰")
            role_started = time.perf_counter()
            result = vllm_client.complete_tokens(adapter_names[role], prefix, 1, stop_ids, grammar)
            if result.get("finish_reason") not in ("stop", "length") or not result.get("token_ids"):
                raise VllmError(f"{role} 워밍업 응답 미완료")
            timings[role] = {"seconds": time.perf_counter() - role_started,
                             "prefix_tokens": len(prefix),
                             **{k: result.get(k) for k in ("prompt_tokens", "completion_tokens", "cached_tokens", "finish_reason")}}
    except Exception as error:
        raise VllmError(f"LLM 시작 워밍업 실패 ({role}): {error}") from error
    finally:
        model.active_adapter, model.last_usage = previous_adapter, previous_usage
    result = {"seconds": time.perf_counter() - started, "roles": timings}
    if logger is not None:
        logger.info(f"D/Q/P 워밍업 완료 ({result['seconds']:.2f}초); 임시 출력 폐기")
    return result
