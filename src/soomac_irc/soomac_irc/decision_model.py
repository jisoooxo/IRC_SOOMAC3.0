import copy
import json
import time

import torch
import xgrammar as xgr
from xgrammar.contrib.hf import LogitsProcessor as XGrammarLogitsProcessor

from soomac_irc.agent_contract import DECISION_SCHEMA, normalize_decision
from soomac_irc.agent_prompts import DECISION_SYSTEM
from soomac_irc.llm_langgraph import Decision, DuplicateDecisionKeyError, SessionState
from soomac_irc.decision_overrides import post_decision_override, pre_decision_override
from soomac_irc.dialogue_focus import build_reference_context

# 설정값
DECISION_HISTORY_TURNS = 8       # Decision Agent에 전달할 최근 완료 대화 턴 수
DECISION_INPUT_MAX_TOKENS = 8192  # 모델 입력 상한. 출력 token 한도와 별개임.
DECISION_MAX_TOKENS = 1024        # sparse 출력이 잘리지 않도록 기존 여유를 유지.


def build_decision_inputs(session: SessionState, user_text: str, robot_state: dict, processor, repair: dict | None = None):
    # 전체 세션에서 token 한도에 맞는 최근 문맥과 현재 발화를 모델 입력으로 만든다.
    history = copy.deepcopy(session["history"][-(DECISION_HISTORY_TURNS * 2):])
    action_history = copy.deepcopy(session["action_history"])

    # 현재 user turn은 아직 history에 저장되기 전이므로 완료될 턴 번호를 여기서 계산한다.
    # LLM이 history에서 reference target을 새로 추측하기 전에 Python 결과를 입력으로 제공한다.
    current_history_turn = len(session["history"]) // 2 + 1
    reference_context = build_reference_context(user_text,session["dialogue_focus"], current_history_turn)

    while True:
        model_input = {
            "order": copy.deepcopy(session["order"]),
            "preferences": copy.deepcopy(session["preferences"]),
            "recommendation": copy.deepcopy(session["recommendation"]),
            "pending_confirmation": copy.deepcopy(session["pending_confirmation"]),
            "dialogue_focus": copy.deepcopy(session["dialogue_focus"]),
            "reference_context": copy.deepcopy(reference_context),
            "action_history": action_history,
            "robot_state": copy.deepcopy(robot_state),
            "message": user_text.strip(),
        }
        if repair is not None:
            model_input["repair"] = copy.deepcopy(repair)

        messages = [
            {"role": "system", "content": DECISION_SYSTEM},
            *history,
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
            return messages, inputs

        if history:
            # 가장 오래된 user/assistant 대화부터 제거한다.
            history = history[2:] if len(history) >= 2 else []
            continue

        if action_history:
            # 자연어 대화가 모두 빠진 뒤 실제 처리 기록을 제거한다.
            action_history = action_history[1:]
            continue

        raise ValueError(f"Decision 입력 token 초과: {prompt_tokens}/{DECISION_INPUT_MAX_TOKENS}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict:
    # json.loads의 object_pairs_hook은 dict로 합치기 전 key-value 순서를 그대로 넘긴다.
    # 일반 json.loads는 같은 key가 반복되면 앞의 값을 조용히 버리므로 여기서 즉시 막는다.
    parsed = {}

    for key, value in pairs:
        # 같은 JSON object 내부에서만 중복을 검사한다.
        # 서로 다른 queries 원소가 각각 target을 갖는 정상 구조는 문제없이 통과한다.
        if key in parsed:
            raise DuplicateDecisionKeyError(f"Decision JSON duplicate key: {key}")

        parsed[key] = value

    return parsed


def parse_decision(raw: str, session: SessionState | None = None) -> Decision:
    # sparse에서 scope를 생략한 경우에만 runtime 기본값을 채운 뒤 full Decision으로 정규화
    # 중복 key는 normalize 전에 거부해서 앞의 값이 조용히 사라지는 것을 막는다.
    sparse = json.loads(
        raw,
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    recommendation = sparse.get("recommendation")

    if recommendation is not None and "scope" not in recommendation:
        action = recommendation.get("action")

        if action == "request":
            recommendation["scope"] = "current"
        elif action == "revise" and session is not None:
            previous = session.get("recommendation", {}).get("last_proposal")
            recommendation["scope"] = previous.get("scope", "current") if previous else "current"

    return normalize_decision(sparse)


def make_call_decision(model, processor, logger=None):
    # XGrammar를 한 번 준비하고 Orchestrator 호출 함수를 만든다.
    tokenizer = processor.tokenizer
    stop_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<turn|>")]
    stop_ids = list(dict.fromkeys(token_id for token_id in stop_ids if isinstance(token_id, int) and token_id >= 0))

    tokenizer_info = xgr.TokenizerInfo.from_huggingface(
        tokenizer,
        vocab_size=len(tokenizer),
        stop_token_ids=stop_ids,
    )
    compiled_grammar = xgr.GrammarCompiler(tokenizer_info).compile_json_schema(DECISION_SCHEMA, any_order=True)

    @torch.inference_mode()
    def call_decision(session: SessionState, user_text: str, robot_state: dict, repair: dict | None = None) -> Decision:
        started = time.perf_counter()
        trace = {
            "stage": "decision",
            "repair": copy.deepcopy(repair),
            "user_text": user_text.strip(),
            "robot_state": copy.deepcopy(robot_state),
        }

        try:
            override = pre_decision_override(user_text, session, robot_state)

            if override is not None:
                call_decision.last_input = {
                    "override": "pre_decision",
                    "message": user_text.strip(),
                }
                call_decision.last_raw = None
                call_decision.last_prompt_tokens = 0
                trace.update({
                    "source": "pre_override",
                    "model_input": copy.deepcopy(call_decision.last_input),
                    "messages": None,
                    "raw": None,
                    "parsed_before_post": None,
                    "final": copy.deepcopy(override),
                    "prompt_tokens": 0,
                    "output_tokens": 0,
                })

                if logger is not None:
                    logger.info(f"Decision pre override: confirmation={override['confirmation']}")

                return override

            messages, inputs = build_decision_inputs(session, user_text, robot_state, processor, repair)
            inputs = inputs.to(model.device)
            prompt_tokens = inputs["input_ids"].shape[1]
            call_decision.last_input = json.loads(messages[-1]["content"])
            call_decision.last_prompt_tokens = prompt_tokens
            trace.update({
                "source": "model",
                "model_input": copy.deepcopy(call_decision.last_input),
                "messages": copy.deepcopy(messages),
                "prompt_tokens": prompt_tokens,
            })

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
            call_decision.last_raw = raw

            if logger is not None:
                logger.info(f"Decision tokens: {prompt_tokens} input, {output_tokens} output")
                logger.info(f"Decision raw: {raw}")

            parsed = parse_decision(raw, session)
            final = post_decision_override(user_text, parsed, session, robot_state)
            trace.update({
                "raw": raw,
                "parsed_before_post": copy.deepcopy(parsed),
                "final": copy.deepcopy(final),
                "post_override_applied": parsed != final,
                "output_tokens": output_tokens,
            })
            return final
        except Exception as error:
            trace["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            raise
        finally:
            trace["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
            call_decision.trace_events.append(copy.deepcopy(trace))

    call_decision.last_input = None
    call_decision.last_raw = None
    call_decision.last_prompt_tokens = None
    call_decision.trace_events = []
    return call_decision
