import copy
import json
import time

import torch
import xgrammar as xgr
from xgrammar.contrib.hf import LogitsProcessor as XGrammarLogitsProcessor

from soomac_irc.agent_contract import RESPONSE_SCHEMA
from soomac_irc.agent_prompts import (GENERAL_RESPONSE_SYSTEM,MIXED_RESPONSE_SYSTEM,TASK_RESPONSE_SYSTEM)
from soomac_irc.llm_langgraph import SessionState
from soomac_irc.dialogue_questions import active_question
from soomac_irc.llm_policy import compact_order_patch, section_execution_items


RESPONSE_HISTORY_TURNS = 8
RESPONSE_MAX_TOKENS = 512
# route는 Decision Agent가 이미 한 번의 inference로 확정했다.
# Response에서는 추가 inference 없이 해당 route의 system prompt만 Python mapping으로 선택한다.
RESPONSE_SYSTEM_BY_ROUTE = {
    "task": TASK_RESPONSE_SYSTEM,
    "general": GENERAL_RESPONSE_SYSTEM,
    "mixed": MIXED_RESPONSE_SYSTEM,
}

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
    def call_response(
        user_text: str,
        session: SessionState,
        policy: dict,
        applied_changes: dict,
        future_changes: list[dict],
        recommendation_result: dict | None,
        queries: list[dict],
        next_prompt: dict | None,
        robot_state: dict,
        route: str,
    ) -> str:
        started = time.perf_counter()
        # route는 schema enum을 통과한 값이며, 여기서는 Response prompt 선택에만 사용한다.
        # 새로운 Decision field나 별도 Router LLM 호출은 만들지 않는다.
        response_system = RESPONSE_SYSTEM_BY_ROUTE[route]
        response_system += """
        [처리 결과와 대화 표현의 구분]
        - dialogue_result는 이번 처리의 사실이다. user_text는 요청이며, 처리 결과가 아니다.
        - confirmed_order는 현재 저장된 주문이다.
        - applied_this_turn에 있는 변경만 이번에 반영됐다고 안내한다.
        - changed_this_turn이 false이면 이번 턴에 주문·제한·취향을 반영·해제·변경했다고 말하지 않는다.
        - 이전 assistant 발언의 변경 안내를 이번 턴 결과처럼 반복하지 않는다.
        - changed_this_turn이 false이면 user_text와 confirmed_order를 비교해, 요청한 값이 이미 주문에 있으면 이미 반영돼 있다고 답한다.
        - already_set의 order와 restrictions는 이번 턴 전부터 이미 있던 주문값과 제한(알러지 포함)이다. 새로 반영·추가했다고 말하지 말고 이미 들어 있다고 말한다.
        - unconfirmed_candidate는 아직 저장하지 않은 해석 후보이다.
        - 후보가 있으면 확정 주문처럼 안내하지 말고 question_to_ask에 따라 확인한다.
        - execution_authorized는 실행 허가이지 실제 작업 완료가 아니다.
        - execution_authorized가 false이면 담기·시작·진행한다고 절대 말하지 않는다. 실행 안내는 execution_authorized가 true일 때만 한다.
        - execution_authorized가 true이면 starting_now에 있는 재료만 지금 담기 시작한다고 말한다. history나 completed_tasks의 재료를 지금 담는다고 말하지 않는다.
        - 실제 진행과 완료는 active_task와 completed_tasks를 따른다.
        - 이전 assistant 발언보다 최신 dialogue_result를 우선한다.
        - question_to_ask가 있으면 그 대상과 목적에 맞게 자연스럽게 질문한다.
        - question_to_ask.type이 execution_offer이면 반영 내용을 말한 뒤 targets를 지금 바로 담기 시작할지 묻는다.
        - question_to_ask가 없으면 새로운 주문 확인이나 선택 질문을 임의로 만들지 않는다.
        - 일반 질문과 설명 요청은 계속 답한다.
        - 문구를 기계적으로 반복하지 말고 문맥에 맞게 자연스럽게 표현한다.
        - section_transition이 있으면 사용자 요청이 아닌 실제 단계 전환 사건에 답한다.
        - section_transition.outcome=skipped이면 해당 단계에서 담기 작업을 하지 않고 넘어간 것이다.
        - section_transition.outcome=completed일 때만 해당 단계의 담기 완료를 안내한다.
        - 전환 결과를 안내한 뒤 question_to_ask가 있으면 그 질문을 한다.
        """


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
        # model_input = {
        #     "user_text": user_text,
        #     "session": compact_session,
        #     "policy": policy,
        #     "applied_changes": applied_changes,
        #     "future_changes": future_changes,
        #     "recommendation_result": recommendation_result,
        #     "queries": queries,
        #     "next_prompt": next_prompt,
        #     "robot_state": robot_state,
        #     "recent_action_history": recent_action_history,
        # }

        question = active_question(session, robot_state["section"])
        candidate = None
        question_to_ask = copy.deepcopy(next_prompt)

        if (
            question is not None
            and question.get("type") == "menu_confirmation"
            and policy["reason"] == "menu_confirmation"
        ):
            candidate = compact_order_patch(question["proposal"])  # 비어 있는 필드는 질문 대상이 아니다
            question_to_ask = {
                "type": "confirm_menu_interpretation",
                "targets": copy.deepcopy(question["targets"]),
                "proposal": copy.deepcopy(candidate),
            }

        # 지난 질문과 이번 확인 후보를 구분한다.
        previous_question = copy.deepcopy(question)
        if (
            previous_question is not None
            and previous_question.get("type") == "menu_confirmation"
        ):
            previous_question = None

        dialogue_result = {
            "confirmed_order": copy.deepcopy(session["order"]),
            "applied_this_turn": copy.deepcopy(applied_changes),
            "unconfirmed_candidate": candidate,
            "processing_status": policy["status"],
            "processing_reason": policy["reason"],
            "execution_authorized": bool(policy["execute"]),
            # 실행 허가 시 이번 section에서 실제로 담을 재료. 응답 생성 뒤에 task가 만들어지므로 여기서 미리 계산한다.
            "starting_now": (
                section_execution_items(session["order"], robot_state["section"])
                if policy["execute"] else []
            ),
            "changed_this_turn": any(bool(value) for value in applied_changes.values()),
            "already_set": copy.deepcopy(policy.get("already_set") or {}),
            "active_task": copy.deepcopy(robot_state.get("active_task")),
            "completed_tasks": copy.deepcopy(robot_state["completed_tasks"]),
            "previous_question": previous_question,
            "question_to_ask": question_to_ask,
            "section_transition": copy.deepcopy(robot_state.get("section_transition")),
        }

        model_input = {
            "user_text": user_text,
            "session": compact_session,
            "policy": policy,
            "applied_changes": applied_changes,
            "future_changes": future_changes,
            "recommendation_result": recommendation_result,
            "queries": queries,
            "next_prompt": question_to_ask,
            "robot_state": robot_state,
            "recent_action_history": recent_action_history,
            "dialogue_result": dialogue_result,
        }
        messages = [
            {"role": "system", "content": response_system},
            *history,
            {"role": "user", "content": json.dumps(model_input, ensure_ascii=False, separators=(",", ":"))},
        ]
        trace = {
            "stage": "response",
            "route": route,
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
