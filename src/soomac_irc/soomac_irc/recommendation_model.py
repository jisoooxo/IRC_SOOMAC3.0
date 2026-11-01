import copy
import json
import time

import torch
import xgrammar as xgr
from xgrammar.contrib.hf import LogitsProcessor as XGrammarLogitsProcessor

from soomac_irc.adapter_runtime import activate_adapter
from soomac_irc.agent_contract import (
    RECOMMENDATION_SCHEMA,
    NormalizedDecision,
    SessionState,
)
from soomac_irc.agent_prompts import RECOMMENDATION_SYSTEM
from soomac_irc.domain import AMOUNTS, NOODLE_TYPES, SAUCES, TOPPINGS
from soomac_irc.input_budget import check_static_budget, fit_recent_history, input_token_limit
from soomac_irc.llm_policy import build_recommendation_scope
from soomac_irc.prefix_cache import PrefixKVCache
from soomac_irc.response_contract import build_recommendation_model_input

RECOMMENDATION_MAX_TOKENS = 1024
RECOMMENDATION_HISTORY_TURNS = 10
# 2026-11-01: 서빙 한도(10240) - 출력 1024 = 9216. 넘으면 오래된 대화부터 뺀다(현재 실측 최대 약 3.9k).
RECOMMENDATION_INPUT_MAX_TOKENS = input_token_limit(RECOMMENDATION_MAX_TOKENS)


#################### 추천 모델 호출 함수 만들기 ####################

def make_call_recommendation(model, processor, logger=None, adapter_name: str | None = None):
    # 추천 결과는 항상 완성된 proposal 모양으로 나오게 XGrammar를 한 번만 준비한다.
    tokenizer = processor.tokenizer
    stop_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<turn|>")]
    stop_ids = list(dict.fromkeys(i for i in stop_ids if isinstance(i, int) and i >= 0))
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=len(tokenizer), stop_token_ids=stop_ids)
    compiled_grammar = xgr.GrammarCompiler(tokenizer_info).compile_json_schema(RECOMMENDATION_SCHEMA)
    check_static_budget(processor, RECOMMENDATION_SYSTEM, RECOMMENDATION_INPUT_MAX_TOKENS, "Recommendation")
    prefix_cache = PrefixKVCache(processor, logger)

    @torch.inference_mode()
    def call_recommendation(
        session: SessionState,
        decision: NormalizedDecision,
        robot_state: dict,
        user_text: str | None,
    ) -> dict:
        # Decision이 추천 조건을 별도 field로 번역하지 않는다.
        # 추천 모델이 사용자 원문·최근 대화·현재 주문을 직접 읽고 후보를 만든다.
        started = time.perf_counter()
        model_input = build_recommendation_model_input(session, decision, robot_state, user_text)
        messages = [
            {"role": "system", "content": RECOMMENDATION_SYSTEM},
            {"role": "user", "content": json.dumps(model_input, ensure_ascii=False, separators=(",", ":"))},
        ]
        trace = {"stage": "recommendation", "model_input": copy.deepcopy(model_input), "messages": copy.deepcopy(messages)}
        try:
            # 서빙 한도를 넘으면 가장 오래된 대화부터 뺀다. 평소(한도 이하)에는 입력이 그대로다.
            messages, inputs, dropped = fit_recent_history(
                processor, RECOMMENDATION_SYSTEM, model_input, RECOMMENDATION_INPUT_MAX_TOKENS, "Recommendation",
            )
            if dropped:
                trace.update({"model_input": copy.deepcopy(model_input), "messages": copy.deepcopy(messages),
                              "history_trimmed_messages": dropped})
            inputs = inputs.to(model.device)
            prompt_tokens = inputs["input_ids"].shape[1]
            generate_args = {
                **inputs,
                "max_new_tokens": RECOMMENDATION_MAX_TOKENS,
                "do_sample": False,
                "use_cache": True,
                "eos_token_id": stop_ids,
                "logits_processor": [XGrammarLogitsProcessor(compiled_grammar)],
            }

            # A/B를 통과한 Recommendation Adapter가 있으면 이를 사용한다.
            # adapter_name이 없으면 종전처럼 Base Model을 사용한다.
            # 고정 system prompt 앞부분은 해당 Adapter(또는 Base) 전용 접두 KV를 재사용한다.
            if activate_adapter(model, adapter_name):
                output = prefix_cache.generate(model, adapter_name, RECOMMENDATION_SYSTEM, generate_args)
            elif hasattr(model, "disable_adapter"):
                with model.disable_adapter():
                    output = prefix_cache.generate(model, None, RECOMMENDATION_SYSTEM, generate_args)
            else:
                output = prefix_cache.generate(model, None, RECOMMENDATION_SYSTEM, generate_args)
            trace["prefix_cache"] = copy.deepcopy(prefix_cache.last_event)
            raw = processor.decode(output[0][prompt_tokens:], skip_special_tokens=True).strip()
            parsed = json.loads(raw)
            trace.update({
                "prompt_tokens": prompt_tokens,
                "output_tokens": output[0].shape[0] - prompt_tokens,
                "raw": raw,
                "parsed": copy.deepcopy(parsed),
            })
            if logger is not None:
                logger.info(f"Recommendation raw: {raw}")
            return parsed
        except Exception as error:
            trace["error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            trace["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
            call_recommendation.trace_events.append(copy.deepcopy(trace))

    call_recommendation.trace_events = []
    call_recommendation.prefix_cache = prefix_cache
    return call_recommendation
