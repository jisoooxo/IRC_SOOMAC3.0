import copy
import json
import time

import torch
import xgrammar as xgr
from xgrammar.contrib.hf import LogitsProcessor as XGrammarLogitsProcessor

from soomac_irc.agent_contract import RESPONSE_SCHEMA
from soomac_irc.agent_prompts import RESPONSE_SYSTEM
from soomac_irc.llm_langgraph import SessionState


RESPONSE_HISTORY_TURNS = 8
RESPONSE_MAX_TOKENS = 512

def make_call_response(model, processor, logger=None):
    # state 변경이 끝난 뒤 Python이 확정한 사실만 읽는 Response Agent를 만든다.
    tokenizer = processor.tokenizer
    stop_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<turn|>")]
    stop_ids = list(dict.fromkeys(token_id for token_id in stop_ids if isinstance(token_id, int) and token_id >= 0))

    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer,
        vocab_size=len(tokenizer),
        stop_token_ids=stop_ids,
    )
    compiled_grammar = xgr.GrammarCompiler(tokenizer_info).compile_json_schema(RESPONSE_SCHEMA)

    @torch.inference_mode()
    def call_response(user_text: str, session: SessionState, policy: dict, applied_changes: dict, future_changes: list[dict], recommendation_result: dict | None, queries: list[dict], next_prompt: dict | None, robot_state: dict) -> str:
        started = time.perf_counter()
        history = copy.deepcopy(session["history"][-(RESPONSE_HISTORY_TURNS * 2):])
        compact_session = {
            "order": session["order"],
            "preferences": session["preferences"],
            "recommendation": {
                "phase": session["recommendation"]["phase"],
                "last_proposal": session["recommendation"]["last_proposal"],
            },
        }
        needs_failure_history = any(
            query.get("type") == "robot_failure"
            for query in queries
        )
        recent_action_history = (
            session["action_history"][-12:]
            if needs_failure_history
            else []
        )
        model_input = {
            "user_text": user_text,
            "session": compact_session,
            "policy": policy,
            "applied_changes": applied_changes,
            "future_changes": future_changes,
            "recommendation_result": recommendation_result,
            "queries": queries,
            "next_prompt": next_prompt,
            "robot_state": robot_state,
            "recent_action_history": recent_action_history,
        }
        messages = [
            {"role": "system", "content": RESPONSE_SYSTEM},
            *history,
            {"role": "user", "content": json.dumps(model_input, ensure_ascii=False, separators=(",", ":"))},
        ]
        trace = {
            "stage": "response",
            "model_input": copy.deepcopy(model_input),
            "messages": copy.deepcopy(messages),
        }

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

            if hasattr(model, "disable_adapter"):
                with model.disable_adapter():
                    output = model.generate(**generate_args)
            else:
                output = model.generate(**generate_args)

            output_tokens = output[0].shape[0] - prompt_tokens
            raw = processor.decode(output[0][prompt_tokens:], skip_special_tokens=True).strip()
            parsed = json.loads(raw)
            reply = parsed["reply"].strip()
            trace.update({
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "raw": raw,
                "parsed": copy.deepcopy(parsed),
                "reply": reply,
            })

            if logger is not None:
                logger.info(f"Response raw: {raw}")

            return reply
        except Exception as error:
            trace["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            raise
        finally:
            trace["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
            call_response.trace_events.append(copy.deepcopy(trace))

    call_response.trace_events = []
    return call_response
