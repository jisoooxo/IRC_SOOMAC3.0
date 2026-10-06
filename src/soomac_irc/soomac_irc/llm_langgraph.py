import copy
from typing import Callable, TypedDict

from soomac_irc.agent_contract import empty_order_patch, new_decision
from soomac_irc.llm_policy import (
    allowed_order_fields,
    apply_order_patch,
    apply_preference_options,
    apply_restriction_options,
    build_applied_changes,
    build_future_changes,
    build_turn_action_event,
    cancel_is_blocked,
    changed_order_keys_from_patch,
    current_section_changed,
    enforce_restrictions,
    missing_current_section,
    section_execution_items,
    validate_order_patch,
    validate_recommendation_proposal,
    validate_restriction_options,
)


class SessionState(TypedDict):
    # 확정된 주문 사실과 자연어 대화 기억만 session에 보관한다.
    # 아직 확정하지 않은 실행·추천은 pending 하나에서 종류로 구분한다.
    order: dict
    preferences: list[dict]
    pending: dict | None
    history: list[dict]
    action_history: list[dict]


class Decision(TypedDict):
    route: str
    understanding: str
    order_patch: dict
    restriction_options: list[dict]
    preference_options: list[dict]
    recommendation: dict
    commit: bool
    cancel: bool
    confirmation: str


class TurnState(TypedDict, total=False):
    session: SessionState
    previous_session: SessionState
    user_text: str
    robot_state: dict
    decision: Decision | None
    recommendation_result: dict | None
    policy: dict | None
    reply: str | None


class DuplicateDecisionKeyError(ValueError):
    pass


#################### 새 주문·세션·턴 state 만들기 ####################

def new_order_state() -> dict:
    # 실제 주문의 기준값이다. 추천 후보나 거절된 값은 이 dict에 넣지 않는다.
    return {
        "sauce": None,
        "noodle_type": None,
        "noodle_portion": None,
        "toppings": {},
        "restrictions": [],
    }


def new_session_state() -> SessionState:
    # 한 주문 대화가 시작될 때 만드는 최소 session 구조이다.
    return {
        "order": new_order_state(),
        "preferences": [],
        "pending": None,
        "history": [],
        "action_history": [],
    }


def new_turn_state(session: SessionState, user_text: str, robot_state: dict) -> TurnState:
    # Graph는 복사본에서만 동작한다. 중간 단계가 실패해도 실제 Node state가 오염되지 않는다.
    session_copy = copy.deepcopy(session)
    return {
        "session": session_copy,
        "previous_session": copy.deepcopy(session_copy),
        "user_text": user_text,
        "robot_state": copy.deepcopy(robot_state),
        "decision": None,
        "recommendation_result": None,
        "policy": None,
        "reply": None,
    }


#################### 주문 patch와 pending을 다루는 작은 도우미 ####################

def _patch_has_values(patch: dict) -> bool:
    # 주문 patch에 이번 턴의 실제 변경 후보가 하나라도 있는지 확인한다.
    return bool(
        patch.get("sauce") is not None
        or patch.get("noodle_type") is not None
        or patch.get("noodle_portion") is not None
        or patch.get("toppings")
    )


def _merge_patch(base: dict, overlay: dict) -> dict:
    # 추천 pending을 수락할 때 기존 후보 위에 같은 턴의 명시적 변경을 덮는다.
    merged = copy.deepcopy(base)
    for field in ("sauce", "noodle_type", "noodle_portion"):
        if overlay.get(field) is not None:
            merged[field] = copy.deepcopy(overlay[field])
    merged["toppings"].update(copy.deepcopy(overlay.get("toppings") or {}))
    return merged


def _remove_patch_from_order(order: dict, patch: dict) -> dict:
    # 미래 section의 preselected 실행을 거절하거나 교체할 때 그 후보만 주문에서 뺀다.
    updated = copy.deepcopy(order)
    if patch.get("sauce") is not None and updated.get("sauce") == patch.get("sauce"):
        updated["sauce"] = None
    if patch.get("noodle_type") is not None and updated.get("noodle_type") == patch.get("noodle_type"):
        updated["noodle_type"] = None
        updated["noodle_portion"] = None
    for item, amount in (patch.get("toppings") or {}).items():
        if updated.get("toppings", {}).get(item) == amount:
            updated["toppings"].pop(item, None)
    return updated


def build_preselected_section_confirmation(session: SessionState, section: str) -> dict | None:
    """미리 고른 미래 section 재료를 execution pending 하나로 묶는다."""
    if section not in ("veggie", "meat", "extra"):
        return None
    items = {}
    for key in allowed_order_fields(section):
        item = key.split(".", 1)[1]
        amount = session["order"]["toppings"].get(item)
        if amount is not None:
            items[item] = amount
    if not items:
        return None
    return {
        "type": "execution",
        "source": "preselected",
        "section": section,
        "targets": list(items),
        "items": items,
        "candidate": {"sauce": None, "noodle_type": None, "noodle_portion": None, "toppings": items},
    }


def build_preselected_confirmation_reply(pending: dict) -> str:
    # Graph 밖에서 section이 바뀔 때 사용할 안전한 기본 확인 문장이다.
    items = pending.get("items") or pending.get("candidate", {}).get("toppings", {})
    names = ", ".join(items) if items else "미리 선택한 재료"
    return f"{names}이(가) 미리 선택되어 있어요. 이대로 진행할까요?"


def _pending_prompt(pending: dict | None) -> dict | None:
    # Response Agent에는 내부 pending 전체가 아니라 사용자에게 물을 사실만 전달한다.
    if pending is None:
        return None
    ptype = pending.get("type")
    if ptype == "execution":
        return {
            "type": "execution",
            "source": pending.get("source", "current_update"),
            "section": pending.get("section"),
            "targets": copy.deepcopy(pending.get("targets", [])),
            "items": copy.deepcopy(pending.get("items", {})),
        }
    if ptype == "recommendation":
        return {
            "type": "recommendation",
            "candidate": copy.deepcopy(pending.get("candidate")),
            "reason_tags": copy.deepcopy(pending.get("reason_tags", [])),
        }
    return copy.deepcopy(pending)


def _call_recommendation_compat(call_recommendation, session, decision, robot_state, user_text):
    # 새 추천 함수에는 사용자 원문까지 넘긴다.
    # 이전 형식의 test double도 리팩터 전환 기간에는 그대로 호출할 수 있게 둔다.
    try:
        return call_recommendation(
            copy.deepcopy(session), copy.deepcopy(decision), copy.deepcopy(robot_state), None, user_text
        )
    except TypeError:
        return call_recommendation(
            copy.deepcopy(session), copy.deepcopy(decision), copy.deepcopy(robot_state), []
        )


def _has_explicit_new_semantics(decision: Decision) -> bool:
    # 사용자가 새 주문·제한·취향·추천·취소를 명시했는지 구조화 결과만 보고 판단한다.
    return bool(
        _patch_has_values(decision["order_patch"])
        or decision["restriction_options"]
        or decision["preference_options"]
        or decision["recommendation"]["action"] != "none"
        or decision["cancel"]
    )


def _resolve_pending(session: SessionState, decision: Decision) -> tuple[SessionState, Decision, dict | None]:
    """구조화된 pending의 수락·거절·교체만 처리한다.

    자연어 의미는 이미 Decision이 판단했으므로 여기서는 사용자 문장을 다시 해석하지 않는다.
    """
    working = copy.deepcopy(session)
    effective = copy.deepcopy(decision)
    pending = working.get("pending")
    result = None

    if pending is None:
        return working, effective, result

    # 새 주문이나 추천 요청이 들어오면 이전 대기 제안은 더 이상 현재 주제가 아니므로 닫는다.
    if effective["confirmation"] == "none" and _has_explicit_new_semantics(effective):
        if pending.get("type") == "execution" and pending.get("source") == "preselected":
            # 미래 미리 선택보다 사용자가 지금 직접 말한 새 선택을 우선한다.
            working["order"] = _remove_patch_from_order(
                working["order"], pending.get("candidate", empty_order_patch())
            )
        working["pending"] = None
        return working, effective, result

    # 수락·거절이 아니면 pending을 그대로 다음 턴까지 유지한다.
    if effective["confirmation"] not in ("accept", "reject"):
        return working, effective, result

    accepted = effective["confirmation"] == "accept"
    effective["confirmation"] = "none"
    working["pending"] = None
    ptype = pending.get("type")

    # 실행 확인 수락은 commit으로 바꾸고, 거절은 실행 대기만 닫는다.
    if ptype == "execution":
        if accepted:
            effective["commit"] = True
            result = {"type": "execution_accepted", "source": pending.get("source", "current_update")}
        else:
            if pending.get("source") == "preselected":
                working["order"] = _remove_patch_from_order(
                    working["order"], pending.get("candidate", empty_order_patch())
                )
            result = {"type": "confirmation_rejected", "pending_type": ptype}
        return working, effective, result

    # 추천 수락은 후보를 이번 턴의 order patch로 옮긴 뒤 일반 검증 경로를 다시 탄다.
    if ptype == "recommendation":
        if accepted:
            effective["order_patch"] = _merge_patch(pending["candidate"], effective["order_patch"])
            result = {
                "type": "recommendation_accepted",
                "proposal": copy.deepcopy(pending["candidate"]),
                "reason_tags": copy.deepcopy(pending.get("reason_tags", [])),
            }
        else:
            result = {"type": "confirmation_rejected", "pending_type": ptype}
        return working, effective, result

    return working, effective, result


#################### LangGraph stage 구성 ####################

def build_graph(
    call_decision: Callable,
    call_recommendation: Callable,
    call_response: Callable,
):
    from langgraph.graph import END, START, StateGraph

    def interpret_decision(state: TurnState) -> dict:
        # 모델 출력은 아직 후보일 뿐이다. duplicate key만 안전한 clarify로 바꾼다.
        try:
            decision = call_decision(
                copy.deepcopy(state["session"]),
                state["user_text"],
                copy.deepcopy(state["robot_state"]),
                None,
            )
        except DuplicateDecisionKeyError:
            decision = new_decision()
            decision["understanding"] = "clarify"
        return {"decision": decision}

    def route_by_decision(state: TurnState) -> str:
        # mixed는 주문 변경도 포함하므로 task 검증 경로를 사용한다.
        return "general" if state["decision"]["route"] == "general" else "task"

    def process_general(state: TurnState) -> dict:
        # 일반대화는 주문이나 pending을 바꾸지 않는다. 기존 pending은 다음 턴까지 유지한다.
        policy = {
            "status": "clarify" if state["decision"]["understanding"] == "clarify" else "pass",
            "reason": "understanding" if state["decision"]["understanding"] == "clarify" else "general",
            "execute": False,
            "conflicts": [],
        }
        return {
            "session": copy.deepcopy(state["session"]),
            "recommendation_result": None,
            "policy": policy,
        }

    def process_task(state: TurnState) -> dict:
        # 아래 분기는 모두 조기 반환한다. 위에서 아래로 읽으면 실제 처리 순서와 같다.
        previous = copy.deepcopy(state["session"])
        raw_decision = copy.deepcopy(state["decision"])

        # 1. 전체 주문 취소 의도는 모델이 내지만, 물리적으로 취소 가능한지는 코드가 결정한다.
        if raw_decision["cancel"]:
            if cancel_is_blocked(state["robot_state"]):
                return {
                    "session": previous,
                    "decision": raw_decision,
                    "recommendation_result": None,
                    "policy": {
                        "status": "blocked",
                        "reason": "cancel_after_start",
                        "execute": False,
                        "conflicts": [],
                    },
                }
            session = copy.deepcopy(previous)
            session["order"] = new_order_state()
            session["preferences"] = []
            session["pending"] = None
            return {
                "session": session,
                "decision": raw_decision,
                "recommendation_result": None,
                "policy": {
                    "status": "pass",
                    "reason": "order_cancelled",
                    "execute": False,
                    "conflicts": [],
                },
            }

        # 2. 이전 턴의 실행·추천 pending을 먼저 수락/거절/교체한다.
        session, decision, pending_result = _resolve_pending(previous, raw_decision)
        recommendation_result = None

        # 3. 모델도 의미를 확정하지 못한 턴은 state를 바꾸지 않고 다시 묻는다.
        if decision["understanding"] == "clarify":
            return {
                "session": session,
                "decision": decision,
                "recommendation_result": None,
                "policy": {"status": "clarify", "reason": "understanding", "execute": False, "conflicts": []},
            }

        # 4. 제한과 취향을 적용한 뒤, 주문 후보를 domain·물리 상태 기준으로 검증한다.
        restriction_validation = validate_restriction_options(
            session["order"], decision["restriction_options"]
        )
        session["order"] = apply_restriction_options(session["order"], restriction_validation["accepted"])
        session["preferences"] = apply_preference_options(session["preferences"], decision["preference_options"])

        order_validation = validate_order_patch(decision["order_patch"], state["robot_state"])
        accepted_patch = order_validation["accepted_patch"]
        temp_order = apply_order_patch(session["order"], accepted_patch)

        restriction_enforcement = enforce_restrictions(
            temp_order,
            newly_added=restriction_validation["newly_added"],
            changed_order_keys=changed_order_keys_from_patch(accepted_patch),
            robot_state=state["robot_state"],
        )
        session["order"] = restriction_enforcement["order"]

        # 5. 이미 담은 재료에 새 제한이 생겨도 실제로 담긴 주문 기록은 되돌리지 않는다.
        if restriction_enforcement["physical_conflicts"]:
            return {
                "session": session,
                "decision": decision,
                "recommendation_result": None,
                "policy": {
                    "status": "blocked",
                    "reason": "completed_restriction_conflict",
                    "execute": False,
                    "conflicts": copy.deepcopy(restriction_enforcement["physical_conflicts"]),
                    "restriction_blocked": copy.deepcopy(restriction_enforcement["blocked"]),
                },
            }

        # 6. 기존 안전 제한과 직접 충돌한 새 주문은 해당 값만 빼고 사용자에게 이유를 알린다.
        if restriction_enforcement["requested_conflicts"]:
            return {
                "session": session,
                "decision": decision,
                "recommendation_result": None,
                "policy": {
                    "status": "warning",
                    "reason": "restriction_conflict",
                    "execute": False,
                    "conflicts": copy.deepcopy(restriction_enforcement["requested_conflicts"]),
                    "restriction_blocked": copy.deepcopy(restriction_enforcement["blocked"]),
                },
            }

        # 7. 추천 요청은 원문과 history를 추천 모델이 직접 읽는다.
        # 추천 결과도 일반 주문과 같은 validator를 통과한 뒤 pending에만 저장한다.
        if decision["recommendation"]["action"] in ("request", "revise"):
            raw = _call_recommendation_compat(
                call_recommendation, session, decision, state["robot_state"], state["user_text"]
            )
            rec_validation = validate_recommendation_proposal(session, raw["proposal"], state["robot_state"])
            recommendation_result = {
                "proposal": copy.deepcopy(rec_validation["proposal"]),
                "reason_tags": copy.deepcopy(raw.get("reason_tags", [])),
                "accepted_fields": copy.deepcopy(rec_validation["accepted_fields"]),
                "rejected_fields": copy.deepcopy(rec_validation["rejected_fields"]),
            }
            if rec_validation["accepted_fields"]:
                session["pending"] = {
                    "type": "recommendation",
                    "candidate": copy.deepcopy(rec_validation["proposal"]),
                    "reason_tags": copy.deepcopy(raw.get("reason_tags", [])),
                }
                policy = {
                    "status": "pass",
                    "reason": "recommendation_proposed",
                    "execute": False,
                    "conflicts": [],
                }
            else:
                policy = {
                    "status": "clarify",
                    "reason": "recommendation_unavailable",
                    "execute": False,
                    "conflicts": [],
                    "rejected_fields": copy.deepcopy(rec_validation["rejected_fields"]),
                }
            return {
                "session": session,
                "decision": decision,
                "recommendation_result": recommendation_result,
                "policy": policy,
            }

        # 8. 일부 값이 거절돼도 나머지 정상 변경은 유지한다. 단, 같은 턴 자동 실행은 막는다.
        rejected_any = bool(
            order_validation["unsupported"]
            or order_validation["protected"]
            or order_validation["invalid"]
            or restriction_validation["unsupported"]
            or restriction_validation["invalid"]
        )
        if rejected_any:
            if order_validation["unsupported"] or restriction_validation["unsupported"]:
                reason = "unsupported"
            elif order_validation["protected"]:
                reason = "physical_state"
            else:
                reason = "invalid_candidate"
            return {
                "session": session,
                "decision": decision,
                "recommendation_result": recommendation_result,
                "policy": {
                    "status": "warning",
                    "reason": reason,
                    "execute": False,
                    "conflicts": [],
                    "unsupported": copy.deepcopy(order_validation["unsupported"] + restriction_validation["unsupported"]),
                    "protected": copy.deepcopy(order_validation["protected"]),
                    "invalid": copy.deepcopy(order_validation["invalid"] + restriction_validation["invalid"]),
                    "restriction_blocked": copy.deepcopy(restriction_enforcement["blocked"]),
                },
            }

        # 9. pending 거절은 state 변경 없이 정상적인 대화 결과로 끝낸다.
        if pending_result and pending_result.get("type") == "confirmation_rejected":
            return {
                "session": session,
                "decision": decision,
                "recommendation_result": recommendation_result,
                "policy": {"status": "pass", "reason": "confirmation_rejected", "execute": False, "conflicts": []},
            }

        # 10. commit 의미는 모델을 믿되 필수 주문과 robot busy는 코드가 마지막으로 확인한다.
        if decision["commit"]:
            missing = missing_current_section(session["order"], state["robot_state"]["section"])
            if missing:
                policy = {"status": "clarify", "reason": "missing_order", "execute": False, "missing": missing, "conflicts": []}
            elif state["robot_state"].get("active_task") is not None or state["robot_state"].get("task_queue"):
                policy = {"status": "blocked", "reason": "robot_busy", "execute": False, "conflicts": []}
            else:
                policy = {"status": "pass", "reason": "execution_allowed", "execute": True, "conflicts": []}
            return {
                "session": session,
                "decision": decision,
                "recommendation_result": recommendation_result,
                "policy": policy,
            }

        # 11. 현재 section 값만 바뀌었다면 바로 실행하지 않고 execution pending을 만든다.
        applied = build_applied_changes(previous, session)
        section = state["robot_state"]["section"]
        if (
            session.get("pending") is None
            and current_section_changed(applied, section)
            and section_execution_items(session["order"], section)
            and not missing_current_section(session["order"], section)
        ):
            items = section_execution_items(session["order"], section)
            session["pending"] = {
                "type": "execution",
                "source": "current_update",
                "section": section,
                "targets": [item["item"] for item in items],
            }

        return {
            "session": session,
            "decision": decision,
            "recommendation_result": recommendation_result,
            "policy": {
                "status": "pass",
                "reason": "state_update_only",
                "execute": False,
                "conflicts": [],
                "restriction_blocked": copy.deepcopy(restriction_enforcement["blocked"]),
                "restriction_unchanged": copy.deepcopy(restriction_validation["unchanged"]),
            },
        }

    def generate_response(state: TurnState) -> dict:
        # Response Agent에는 확정 state와 이번 턴 결과만 전달한다.
        # Response는 주문을 바꾸지 않고 사용자가 들을 문장만 만든다.
        applied = build_applied_changes(state["previous_session"], state["session"])
        future = build_future_changes(applied, state["robot_state"])
        next_prompt = _pending_prompt(state["session"].get("pending"))
        reason = state["policy"].get("reason")
        if reason == "missing_order":
            next_prompt = {"type": "missing_order", "fields": copy.deepcopy(state["policy"].get("missing", []))}
        elif reason == "unsupported":
            next_prompt = {"type": "unsupported", "items": copy.deepcopy(state["policy"].get("unsupported", []))}
        elif reason == "physical_state":
            next_prompt = {"type": "physical_state", "items": copy.deepcopy(state["policy"].get("protected", []))}
        elif reason == "restriction_conflict":
            next_prompt = {"type": "restriction_conflict", "conflicts": copy.deepcopy(state["policy"].get("conflicts", []))}
        elif reason == "completed_restriction_conflict":
            next_prompt = {"type": "completed_restriction_conflict", "conflicts": copy.deepcopy(state["policy"].get("conflicts", []))}

        reply = call_response(
            state["user_text"],
            copy.deepcopy(state["session"]),
            copy.deepcopy(state["policy"]),
            copy.deepcopy(applied),
            copy.deepcopy(future),
            copy.deepcopy(state.get("recommendation_result")),
            [],
            copy.deepcopy(next_prompt),
            copy.deepcopy(state["robot_state"]),
            state["decision"]["route"],
        )

        # 실제 변경 event와 자연어 대화를 session history에 마지막으로 기록한다.
        session = copy.deepcopy(state["session"])
        event = build_turn_action_event(applied, future)
        if event is not None:
            session["action_history"].append(event)
        if reason == "order_cancelled":
            session["action_history"].append({"type": "order_cancelled"})
        session["history"].append({"role": "user", "content": state["user_text"]})
        session["history"].append({"role": "assistant", "content": reply})
        return {"session": session, "policy": state["policy"], "reply": reply}

    # stage 선언 순서가 실제 한 턴의 흐름이다.
    # interpret → general/task → response 순서만 유지해 Graph를 평평하게 둔다.
    graph = StateGraph(TurnState)
    graph.add_node("interpret_decision", interpret_decision)
    graph.add_node("process_general", process_general)
    graph.add_node("process_task", process_task)
    graph.add_node("generate_response", generate_response)

    graph.add_edge(START, "interpret_decision")
    graph.add_conditional_edges(
        "interpret_decision",
        route_by_decision,
        {"general": "process_general", "task": "process_task"},
    )
    graph.add_edge("process_general", "generate_response")
    graph.add_edge("process_task", "generate_response")
    graph.add_edge("generate_response", END)
    return graph.compile()
