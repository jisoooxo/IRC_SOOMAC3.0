import copy
import json
import time

import torch
import xgrammar as xgr
from xgrammar.contrib.hf import LogitsProcessor as XGrammarLogitsProcessor

from soomac_irc.adapter_runtime import activate_adapter
from soomac_irc.agent_contract import RESPONSE_SCHEMA, SessionState
from soomac_irc.agent_prompts import GENERAL_RESPONSE_SYSTEM, MIXED_RESPONSE_SYSTEM, TASK_RESPONSE_SYSTEM
from soomac_irc.input_budget import check_static_budget, fit_recent_history, input_token_limit
from soomac_irc.prefix_cache import PrefixKVCache
from soomac_irc.response_contract import (
    RESPONSE_HISTORY_TURNS, build_response_model_input, validate_response_output,
)

RESPONSE_MAX_TOKENS = 512
# 서빙 한도16384 - 출력512 = 입력15872. 한도 내에서는 기존 history를 유지한다.
RESPONSE_INPUT_MAX_TOKENS = input_token_limit(RESPONSE_MAX_TOKENS)
RESPONSE_SYSTEM_BY_ROUTE = {
    "task": TASK_RESPONSE_SYSTEM,
    "general": GENERAL_RESPONSE_SYSTEM,
    "mixed": MIXED_RESPONSE_SYSTEM,
}


#################### 최종 사용자 응답 모델 호출 함수 만들기 ####################

def make_call_response(model, processor, logger=None, adapter_name: str | None = None):
    # Response는 답변 문자열 하나만 내도록 JSON 문법 객체를 미리 만든다.
    tokenizer = processor.tokenizer
    stop_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<turn|>")]
    stop_ids = list(dict.fromkeys(i for i in stop_ids if isinstance(i, int) and i >= 0))
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=len(tokenizer), stop_token_ids=stop_ids)
    compiled_grammar = xgr.GrammarCompiler(tokenizer_info).compile_json_schema(RESPONSE_SCHEMA)
    for route_name, route_system in RESPONSE_SYSTEM_BY_ROUTE.items():
        check_static_budget(processor, route_system, RESPONSE_INPUT_MAX_TOKENS, f"Response/{route_name}")
    prefix_cache = PrefixKVCache(processor, logger)

    @torch.inference_mode()
    def call_response(
        user_text: str | None,
        session: SessionState,
        policy: dict,
        applied_changes: dict,
        future_changes: list[dict],
        recommendation_result: dict | None,
        next_prompt: dict | None,
        robot_state: dict,
        route: str,
    ) -> str:
        # route에 맞는 말투를 고르되, 주문 사실은 항상 Python이 확정한 값만 전달한다.
        started = time.perf_counter()
        route = route if route in RESPONSE_SYSTEM_BY_ROUTE else "task"
        model_input = build_response_model_input(
            user_text, session, policy, applied_changes, future_changes,
            recommendation_result, next_prompt, robot_state, route,
        )
        messages = [
            {"role": "system", "content": RESPONSE_SYSTEM_BY_ROUTE[route]},
            {"role": "user", "content": json.dumps(model_input, ensure_ascii=False, separators=(",", ":"))},
        ]
        trace = {"stage": "response", "route": route, "model_input": copy.deepcopy(model_input), "messages": copy.deepcopy(messages)}
        try:
            # 서빙 한도를 넘으면 가장 오래된 대화부터 뺀다. 평소(한도 이하)에는 입력이 그대로다.
            messages, inputs, dropped = fit_recent_history(
                processor, RESPONSE_SYSTEM_BY_ROUTE[route], model_input, RESPONSE_INPUT_MAX_TOKENS, f"Response/{route}",
            )
            if dropped:
                trace.update({"model_input": copy.deepcopy(model_input), "messages": copy.deepcopy(messages),
                              "history_trimmed_messages": dropped})
            inputs = inputs.to(model.device)
            prompt_tokens = inputs["input_ids"].shape[1]
            generate_args = {
                **inputs,
                "max_new_tokens": RESPONSE_MAX_TOKENS,
                "do_sample": False,
                "use_cache": True,
                "eos_token_id": stop_ids,
                "logits_processor": [XGrammarLogitsProcessor(compiled_grammar)],
            }

            # 학습된 Response Adapter가 있으면 해당 LoRA로 답변을 생성한다.
            # adapter_name을 생략한 기존 호출은 종전처럼 Base Model을 사용한다.
            # route마다 system이 다르므로 접두 KV도 (Adapter, route system)별로 따로 재사용한다.
            system_prompt = RESPONSE_SYSTEM_BY_ROUTE[route]
            if activate_adapter(model, adapter_name):
                output = prefix_cache.generate(model, adapter_name, system_prompt, generate_args)
            elif hasattr(model, "disable_adapter"):
                with model.disable_adapter():
                    output = prefix_cache.generate(model, None, system_prompt, generate_args)
            else:
                output = prefix_cache.generate(model, None, system_prompt, generate_args)
            trace["prefix_cache"] = copy.deepcopy(prefix_cache.last_event)
            raw = processor.decode(output[0][prompt_tokens:], skip_special_tokens=True).strip()
            parsed = json.loads(raw)
            trace.update({"raw": raw, "parsed": copy.deepcopy(parsed)})
            reply = validate_response_output(parsed)
            trace.update({
                "prompt_tokens": prompt_tokens,
                "output_tokens": output[0].shape[0] - prompt_tokens,
                "raw": raw,
                "parsed": copy.deepcopy(parsed),
                "reply": reply,
            })
            if not reply:
                raise ValueError("Response reply is empty")
            if logger is not None:
                logger.info(f"Response raw: {raw}")
            return reply
        except Exception as error:
            trace["error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            trace["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
            call_response.trace_events.append(copy.deepcopy(trace))

    call_response.trace_events = []
    call_response.prefix_cache = prefix_cache
    return call_response
