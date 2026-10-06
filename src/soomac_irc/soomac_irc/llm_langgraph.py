import copy
from typing import Callable

from soomac_irc.agent_contract import (
    DuplicateDecisionKeyError as _DuplicateDecisionKeyError,
    NormalizedDecision as _NormalizedDecision,
    SessionState as _SessionState,
    TurnState as _TurnState,
    empty_order_patch,
    new_decision,
)
from soomac_irc.llm_policy import (
    allowed_order_fields,
    apply_order_patch,
    apply_preference_options,
    apply_restriction_options,
    build_applied_changes,
    build_future_changes,
    build_turn_action_event,
    changed_order_keys_from_patch,
    current_section_changed,
    enforce_restrictions,
    missing_current_section,
    section_execution_items,
    validate_order_patch,
    validate_recommendation_proposal,
    validate_restriction_options,
)


#################### 주문 patch와 pending을 다루는 작은 도우미 ####################

def _patch_has_values(patch: dict) -> bool:
    # 주문 patch에 이번 턴의 실제 변경 후보가 하나라도 있는지 확인한다.
    return bool(
        patch.get("sauce") is not None
        or patch.get("noodle_type") is not None
        or patch.get("noodle_portion") is not None
        or patch.get("toppings")
    )


def _external_decision_view(decision: _NormalizedDecision) -> dict:
    """주입형 test double이 sparse 원문을 제공하지 않을 때 외부 계약 모양만 복원한다."""
    external = {"route": decision["route"]}
    patch = decision["order_patch"]
    order = {
        field: copy.deepcopy(patch[field])
        for field in ("sauce", "noodle_type", "noodle_portion")
        if patch.get(field) is not None
    }
    if patch.get("toppings"):
        order["toppings"] = copy.deepcopy(patch["toppings"])
    if order:
        external["order"] = order
    if decision["restriction_options"]:
        external["restrictions"] = copy.deepcopy(decision["restriction_options"])
    if decision["preference_options"]:
        external["preferences"] = copy.deepcopy(decision["preference_options"])
    if decision["recommendation"]["action"] != "none":
        external["recommendation"] = copy.deepcopy(decision["recommendation"])
    if decision["commit"]:
        external["commit"] = True
    if decision["confirmation"] != "none":
        external["confirmation"] = decision["confirmation"]
    if decision["understanding"] == "clarify":
        external["clarify"] = True
    return external


def find_structural_decision_errors(decision: _NormalizedDecision) -> list[str]:
    """Decision 내부 field끼리 동시에 성립할 수 없는 조합만 찾는다.

    사용자 문장·메뉴 domain·session·robot state는 보지 않는다. 지원 여부와 물리적
    실행 가능성은 이 검사 뒤의 기존 final validator가 그대로 담당한다.
    """
    has_resolved_task_semantics = bool(
        _patch_has_values(decision["order_patch"])
        or decision["restriction_options"]
        or decision["preference_options"]
        or decision["recommendation"]["action"] != "none"
        or decision["commit"]
        or decision["confirmation"] != "none"
    )
    errors = []

    if decision["route"] == "general" and has_resolved_task_semantics:
        errors.append("route=general cannot contain task mutation or action")

    if decision["understanding"] == "clarify" and has_resolved_task_semantics:
        errors.append("clarify decision cannot contain resolved mutation or action")

    # 면 종류를 지우면서 실제 면 양을 새로 지정하는 출력은 한 Decision 안에서 모순된다.
    # unsupported 값은 여기서 다루지 않고 기존 final validator로 그대로 보낸다.
    patch = decision["order_patch"]
    if (
        patch.get("noodle_type") == "none"
        and patch.get("noodle_portion") in ("low", "normal", "high")
    ):
        errors.append("noodle_type=none cannot contain an active noodle_portion")

    return errors


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


def build_preselected_section_confirmation(session: _SessionState, section: str) -> dict | None:
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


def _resolve_pending(
    session: _SessionState,
    decision: _NormalizedDecision,
) -> tuple[_SessionState, _NormalizedDecision, dict | None]:
    """구조화된 pending의 수락·거절·교체만 처리한다.

    자연어 의미는 이미 Decision이 판단했으므로 여기서는 사용자 문장을 다시 해석하지 않는다.
    """
    working = copy.deepcopy(session)
    effective = copy.deepcopy(decision)
    pending = working.get("pending")
    result = None

    if pending is None:
        return working, effective, result

    # 수락·거절이 아니면 새 구조화 의미가 함께 있어도 pending을 그대로 둔다.
    # 주문·제한 변경 뒤의 execution pending은 final order가 확정된 다음 다시 맞춘다.
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


def _refresh_execution_pending(session: _SessionState) -> None:
    """확정 주문과 execution pending의 대상만 다시 맞춘다.

    Decision이 만든 의미를 해석하지 않고 pending에 적힌 section과 canonical order만 본다.
    """
    pending = session.get("pending")
    if not isinstance(pending, dict) or pending.get("type") != "execution":
        return

    section = pending.get("section")
    items = section_execution_items(session["order"], section)
    if not items:
        session["pending"] = None
        return

    if pending.get("source") == "preselected":
        session["pending"] = build_preselected_section_confirmation(session, section)
        return

    refreshed = copy.deepcopy(pending)
    refreshed["targets"] = [item["item"] for item in items]
    session["pending"] = refreshed


#################### LangGraph stage 구성 ####################

def build_graph(
    call_decision: Callable,
    call_recommendation: Callable,
    call_response: Callable,
):
    from langgraph.graph import END, START, StateGraph

    def interpret_decision(state: _TurnState) -> dict:
        # 모델 출력은 아직 후보일 뿐이다. duplicate key는 기존 정책대로 바로 안전 종료한다.
        try:
            decision = call_decision(
                copy.deepcopy(state["session"]),
                state["user_text"],
                copy.deepcopy(state["robot_state"]),
                None,
            )
        except _DuplicateDecisionKeyError:
            decision = new_decision()
            decision["understanding"] = "clarify"
            return {"decision": decision}

        # 실제 모델 함수는 정규화 전 sparse JSON을 별도로 보존한다.
        # 주입형 함수가 해당 값을 제공하지 않을 때만 외부 key 이름으로 안전하게 복원한다.
        external_output = getattr(call_decision, "last_external_output", None)
        if not isinstance(external_output, dict):
            external_output = _external_decision_view(decision)

        errors = find_structural_decision_errors(decision)
        if not errors:
            return {"decision": decision}

        # 자연어를 Python이 다시 판단하지 않고, 모델이 만든 field 조합의 자기모순만 알려준다.
        repair = {
            "reason": "structural_contradiction",
            "errors": errors,
            "previous_output": copy.deepcopy(external_output),
            "instruction": (
                "현재 사용자 발화와 state를 다시 읽고, 의미를 임의로 추가하지 말고 "
                "서로 모순되는 Decision field만 최소한으로 수정하라."
            ),
        }
        try:
            repaired = call_decision(
                copy.deepcopy(state["session"]),
                state["user_text"],
                copy.deepcopy(state["robot_state"]),
                repair,
            )
        except _DuplicateDecisionKeyError:
            repaired = new_decision()
            repaired["understanding"] = "clarify"

        # 두 번째 결과도 모순이면 세 번째 호출 없이 아무 의미도 확정하지 않은 Decision으로 끝낸다.
        if find_structural_decision_errors(repaired):
            repaired = new_decision()
            repaired["understanding"] = "clarify"

        decision = repaired
        return {"decision": decision}

    def route_by_decision(state: _TurnState) -> str:
        # mixed는 주문 변경도 포함하므로 task 검증 경로를 사용한다.
        return "general" if state["decision"]["route"] == "general" else "task"

    def process_general(state: _TurnState) -> dict:
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

    def process_task(state: _TurnState) -> dict:
        # 아래 분기는 모두 조기 반환한다. 위에서 아래로 읽으면 실제 처리 순서와 같다.
        previous = copy.deepcopy(state["session"])
        raw_decision = copy.deepcopy(state["decision"])

        # 1. 이전 턴의 pending은 명시적 수락·거절만 먼저 처리한다.
        session, decision, pending_result = _resolve_pending(previous, raw_decision)
        recommendation_result = None

        # 2. 모델도 의미를 확정하지 못한 턴은 state를 바꾸지 않고 다시 묻는다.
        if decision["understanding"] == "clarify":
            return {
                "session": session,
                "decision": decision,
                "recommendation_result": None,
                "policy": {"status": "clarify", "reason": "understanding", "execute": False, "conflicts": []},
            }

        # 3. 제한과 취향을 적용한 뒤, 주문 후보를 domain·물리 상태 기준으로 검증한다.
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

        # 주문이나 제한이 바뀐 턴만 final order 기준으로 execution pending 대상을 갱신한다.
        # preference-only와 상태 질문은 기존 pending 객체를 그대로 보존한다.
        if _patch_has_values(decision["order_patch"]) or decision["restriction_options"]:
            _refresh_execution_pending(session)

        # 일반 state 변경 뒤에는 이전 주문 상태로 만든 추천 후보를 더 이상 확인 대상으로 두지 않는다.
        # request/revise 턴은 Recommendation Agent가 기존 후보까지 읽어야 하므로 호출 전까지 보존한다.
        applied = build_applied_changes(previous, session)
        if (
            isinstance(session.get("pending"), dict)
            and session["pending"].get("type") == "recommendation"
            and decision["recommendation"]["action"] == "none"
            and any(applied.values())
        ):
            session["pending"] = None

        # 4. 서로 다른 validator의 실패를 하나의 issues에 모아 Response가 모두 설명하게 한다.
        # 정상 field는 이미 반영했지만 issue가 하나라도 있으면 같은 턴의 commit은 실행하지 않는다.
        issues = {
            "unsupported": copy.deepcopy(
                order_validation["unsupported"] + restriction_validation["unsupported"]
            ),
            "invalid": copy.deepcopy(
                order_validation["invalid"] + restriction_validation["invalid"]
            ),
            "protected": copy.deepcopy(order_validation["protected"]),
            "restriction_conflicts": copy.deepcopy(
                restriction_enforcement["requested_conflicts"]
            ),
            "physical_conflicts": copy.deepcopy(
                restriction_enforcement["physical_conflicts"]
            ),
        }
        if any(issues.values()):
            if issues["physical_conflicts"]:
                status = "blocked"
                reason = "completed_restriction_conflict"
                conflicts = issues["physical_conflicts"]
            elif issues["restriction_conflicts"]:
                status = "warning"
                reason = "restriction_conflict"
                conflicts = issues["restriction_conflicts"]
            elif issues["unsupported"]:
                status = "warning"
                reason = "unsupported"
                conflicts = []
            elif issues["protected"]:
                status = "warning"
                reason = "physical_state"
                conflicts = []
            else:
                status = "warning"
                reason = "invalid_candidate"
                conflicts = []
            return {
                "session": session,
                "decision": decision,
                "recommendation_result": None,
                "policy": {
                    "status": status,
                    "reason": reason,
                    "execute": False,
                    "conflicts": copy.deepcopy(conflicts),
                    "issues": issues,
                    # 기존 Response/관측 코드가 읽던 top-level key도 유지한다.
                    "unsupported": copy.deepcopy(issues["unsupported"]),
                    "protected": copy.deepcopy(issues["protected"]),
                    "invalid": copy.deepcopy(issues["invalid"]),
                    "restriction_blocked": copy.deepcopy(restriction_enforcement["blocked"]),
                },
            }

        # 5. 추천 요청은 원문과 history를 추천 모델이 직접 읽는다.
        # 추천 결과도 일반 주문과 같은 validator를 통과한 뒤 pending에만 저장한다.
        if decision["recommendation"]["action"] in ("request", "revise"):
            raw = call_recommendation(
                copy.deepcopy(session),
                copy.deepcopy(decision),
                copy.deepcopy(state["robot_state"]),
                state["user_text"],
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
                # 새 추천 후보를 만들지 못했으면 이전 후보도 더 이상 확인 대상으로 남기지 않는다.
                session["pending"] = None
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

        # 6. pending 거절은 state 변경 없이 정상적인 대화 결과로 끝낸다.
        if pending_result and pending_result.get("type") == "confirmation_rejected":
            return {
                "session": session,
                "decision": decision,
                "recommendation_result": recommendation_result,
                "policy": {"status": "pass", "reason": "confirmation_rejected", "execute": False, "conflicts": []},
            }

        # 7. commit 의미는 모델을 믿되 필수 주문과 robot busy는 코드가 마지막으로 확인한다.
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

        # 8. 현재 section 값만 바뀌었다면 바로 실행하지 않고 execution pending을 만든다.
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

    def generate_response(state: _TurnState) -> dict:
        # Response Agent에는 확정 state와 이번 턴 결과만 전달한다.
        # Response는 주문을 바꾸지 않고 사용자가 들을 문장만 만든다.
        applied = build_applied_changes(state["previous_session"], state["session"])
        future = build_future_changes(applied, state["robot_state"])
        next_prompt = _pending_prompt(state["session"].get("pending"))
        reason = state["policy"].get("reason")
        if state["policy"].get("issues"):
            next_prompt = {
                "type": "validation_issues",
                "issues": copy.deepcopy(state["policy"]["issues"]),
            }
        elif reason == "missing_order":
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
            copy.deepcopy(next_prompt),
            copy.deepcopy(state["robot_state"]),
            state["decision"]["route"],
        )

        # 실제 변경 event와 자연어 대화를 session history에 마지막으로 기록한다.
        session = copy.deepcopy(state["session"])
        event = build_turn_action_event(applied, future)
        if event is not None:
            session["action_history"].append(event)
        session["history"].append({"role": "user", "content": state["user_text"]})
        session["history"].append({"role": "assistant", "content": reply})
        return {"session": session, "policy": state["policy"], "reply": reply}

    # stage 선언 순서가 실제 한 턴의 흐름이다.
    # interpret → general/task → response 순서만 유지해 Graph를 평평하게 둔다.
    graph = StateGraph(_TurnState)
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
