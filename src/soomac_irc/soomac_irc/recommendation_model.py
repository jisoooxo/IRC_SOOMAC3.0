import copy
import json
import time

import torch
import xgrammar as xgr
from xgrammar.contrib.hf import LogitsProcessor as XGrammarLogitsProcessor

from soomac_irc.agent_contract import RECOMMENDATION_SCHEMA
from soomac_irc.agent_prompts import RECOMMENDATION_SYSTEM
from soomac_irc.llm_langgraph import Decision, SessionState


RECOMMENDATION_MAX_TOKENS = 768

def make_call_recommendation(model, processor, logger=None):
    # 추천 요청이 있는 턴에서만 호출할 Recommendation Agent를 만든다.
    tokenizer = processor.tokenizer
    stop_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<turn|>")]
    stop_ids = list(dict.fromkeys(token_id for token_id in stop_ids if isinstance(token_id, int) and token_id >= 0))

    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer,
        vocab_size=len(tokenizer),
        stop_token_ids=stop_ids,
    )
    compiled_grammar = xgr.GrammarCompiler(tokenizer_info).compile_json_schema(RECOMMENDATION_SCHEMA)

    @torch.inference_mode()
    def call_recommendation(session: SessionState, decision: Decision, robot_state: dict, allowed_fields: list[str]) -> dict:
        started = time.perf_counter()
        model_input = {
            "order": session["order"],
            "restrictions": session["order"]["restrictions"],
            "preferences": session["preferences"],
            "previous_recommendation": session["recommendation"]["last_proposal"],
            "recommendation_request": decision["recommendation"],
            "explicit_order_patch": decision["order_patch"],
            "allowed_fields": allowed_fields,
            "robot_state": robot_state,
        }
        messages = [
            {"role": "system", "content": RECOMMENDATION_SYSTEM},
            {"role": "user", "content": json.dumps(model_input, ensure_ascii=False, separators=(",", ":"))},
        ]
        trace = {
            "stage": "recommendation",
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
                "max_new_tokens": RECOMMENDATION_MAX_TOKENS,
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
            trace.update({
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "raw": raw,
                "parsed": copy.deepcopy(parsed),
            })

            if logger is not None:
                logger.info(f"Recommendation raw: {raw}")

            return parsed
        except Exception as error:
            trace["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            raise
        finally:
            trace["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
            call_recommendation.trace_events.append(copy.deepcopy(trace))

    call_recommendation.trace_events = []
    return call_recommendation
