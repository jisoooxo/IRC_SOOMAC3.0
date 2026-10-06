import copy
import json
import time

import torch
import xgrammar as xgr
from xgrammar.contrib.hf import LogitsProcessor as XGrammarLogitsProcessor

from soomac_irc.agent_contract import (
    RECOMMENDATION_SCHEMA,
    NormalizedDecision,
    SessionState,
)
from soomac_irc.agent_prompts import RECOMMENDATION_SYSTEM
from soomac_irc.domain import AMOUNTS, NOODLE_TYPES, SAUCES, TOPPINGS

RECOMMENDATION_MAX_TOKENS = 1024
RECOMMENDATION_HISTORY_TURNS = 10


#################### 추천 모델 호출 함수 만들기 ####################

def make_call_recommendation(model, processor, logger=None):
    # 추천 결과는 항상 완성된 proposal 모양으로 나오게 XGrammar를 한 번만 준비한다.
    tokenizer = processor.tokenizer
    stop_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<turn|>")]
    stop_ids = list(dict.fromkeys(i for i in stop_ids if isinstance(i, int) and i >= 0))
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=len(tokenizer), stop_token_ids=stop_ids)
    compiled_grammar = xgr.GrammarCompiler(tokenizer_info).compile_json_schema(RECOMMENDATION_SCHEMA)

    @torch.inference_mode()
    def call_recommendation(
        session: SessionState,
        decision: NormalizedDecision,
        robot_state: dict,
        user_text: str,
    ) -> dict:
        # Decision이 추천 조건을 별도 field로 번역하지 않는다.
        # 추천 모델이 사용자 원문·최근 대화·현재 주문을 직접 읽고 후보를 만든다.
        started = time.perf_counter()
        model_input = {
            "current_user_text": user_text.strip(),
            "recent_history": copy.deepcopy(session["history"][-(RECOMMENDATION_HISTORY_TURNS * 2):]),
            "order": copy.deepcopy(session["order"]),
            "restrictions": copy.deepcopy(session["order"].get("restrictions", [])),
            "preferences": copy.deepcopy(session["preferences"]),
            "pending": copy.deepcopy(session.get("pending")),
            "recommendation_action": decision["recommendation"]["action"],
            "robot_state": copy.deepcopy(robot_state),
            "supported_domain": {
                "sauces": list(SAUCES),
                "noodle_types": list(NOODLE_TYPES),
                "toppings": list(TOPPINGS),
                "amounts": list(AMOUNTS),
            },
        }
        messages = [
            {"role": "system", "content": RECOMMENDATION_SYSTEM},
            {"role": "user", "content": json.dumps(model_input, ensure_ascii=False, separators=(",", ":"))},
        ]
        trace = {"stage": "recommendation", "model_input": copy.deepcopy(model_input), "messages": copy.deepcopy(messages)}
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

            # Decision 전용 LoRA가 붙어 있으면 추천에는 base model을 사용한다.
            # 추천 후보는 뒤에서 Python validator를 다시 통과하므로 여기서는 state를 수정하지 않는다.
            if hasattr(model, "disable_adapter"):
                with model.disable_adapter():
                    output = model.generate(**generate_args)
            else:
                output = model.generate(**generate_args)
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
    return call_recommendation
