import copy
import json
import time

import torch
import xgrammar as xgr
from xgrammar.contrib.hf import LogitsProcessor as XGrammarLogitsProcessor

from soomac_irc.agent_contract import RESPONSE_SCHEMA, ResponseInput, SessionState
from soomac_irc.agent_prompts import GENERAL_RESPONSE_SYSTEM, MIXED_RESPONSE_SYSTEM, TASK_RESPONSE_SYSTEM
from soomac_irc.llm_policy import section_execution_items

RESPONSE_HISTORY_TURNS = 10
RESPONSE_MAX_TOKENS = 512
RESPONSE_SYSTEM_BY_ROUTE = {
    "task": TASK_RESPONSE_SYSTEM,
    "general": GENERAL_RESPONSE_SYSTEM,
    "mixed": MIXED_RESPONSE_SYSTEM,
}


#################### 최종 사용자 응답 모델 호출 함수 만들기 ####################

def make_call_response(model, processor, logger=None):
    # Response는 답변 문자열 하나만 내도록 JSON 문법 객체를 미리 만든다.
    tokenizer = processor.tokenizer
    stop_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<turn|>")]
    stop_ids = list(dict.fromkeys(i for i in stop_ids if isinstance(i, int) and i >= 0))
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=len(tokenizer), stop_token_ids=stop_ids)
    compiled_grammar = xgr.GrammarCompiler(tokenizer_info).compile_json_schema(RESPONSE_SCHEMA)

    @torch.inference_mode()
    def call_response(
        user_text: str,
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
        execution_authorized = bool(policy.get("execute"))
        model_input: ResponseInput = {
            "user_text": user_text,
            "recent_history": copy.deepcopy(session["history"][-(RESPONSE_HISTORY_TURNS * 2):]),
            "confirmed_order": copy.deepcopy(session["order"]),
            "preferences": copy.deepcopy(session["preferences"]),
            "pending": copy.deepcopy(session.get("pending")),
            "policy": copy.deepcopy(policy),
            "applied_this_turn": copy.deepcopy(applied_changes),
            "future_changes": copy.deepcopy(future_changes),
            "recommendation_result": copy.deepcopy(recommendation_result),
            "next_prompt": copy.deepcopy(next_prompt),
            "robot_state": copy.deepcopy(robot_state),
            "recent_action_history": copy.deepcopy(session.get("action_history", [])[-20:]),
            "execution_authorized": execution_authorized,
            "starting_now": (
                section_execution_items(session["order"], robot_state["section"])
                if execution_authorized
                else []
            ),
        }
        messages = [
            {"role": "system", "content": RESPONSE_SYSTEM_BY_ROUTE[route]},
            {"role": "user", "content": json.dumps(model_input, ensure_ascii=False, separators=(",", ":"))},
        ]
        trace = {"stage": "response", "route": route, "model_input": copy.deepcopy(model_input), "messages": copy.deepcopy(messages)}
        try:
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            ).to(model.device)
            prompt_tokens = inputs["input_ids"].shape[1]
            generate_args = {
                **inputs,
                "max_new_tokens": RESPONSE_MAX_TOKENS,
                "do_sample": False,
                "use_cache": True,
                "eos_token_id": stop_ids,
                "logits_processor": [XGrammarLogitsProcessor(compiled_grammar)],
            }

            # Decision Adapter가 응답 문체에 섞이지 않도록 base model로 답변만 생성한다.
            if hasattr(model, "disable_adapter"):
                with model.disable_adapter():
                    output = model.generate(**generate_args)
            else:
                output = model.generate(**generate_args)
            raw = processor.decode(output[0][prompt_tokens:], skip_special_tokens=True).strip()
            parsed = json.loads(raw)
            reply = parsed["reply"].strip()
            trace.update({
                "prompt_tokens": prompt_tokens,
                "output_tokens": output[0].shape[0] - prompt_tokens,
                "raw": raw,
                "parsed": copy.deepcopy(parsed),
                "reply": reply,
            })
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
    return call_response
