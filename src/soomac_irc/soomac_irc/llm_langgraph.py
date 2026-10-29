import copy
from typing import Callable
from soomac_irc.response_contract import RESPONSE_FAILURE_REPLY, validate_recommendation_result

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
    build_recommendation_scope,
    build_recommendation_unavailable_reply,
    build_turn_action_event,
    changed_order_keys_from_patch,
    current_section_changed,
    enforce_restrictions,
    find_restriction_conflicts,
    is_future_recommendation_pending,
    missing_current_section,
    pending_blocks_execution,
    physical_section_for_order_key,
    section_execution_items,
    validate_order_patch,
    validate_recommendation_proposal,
    validate_restriction_options,
    SAFETY_RESTRICTION_REASONS,
)


class _GraphTurnState(_TurnState, total=False):
    # 추천 전 계산한 임시 값만 잇는다. 공개 입출력·세션·모델 입력에는 포함하지 않는다.
    _task_context: dict | None


#################### 주문 patch와 pending을 다루는 작은 도우미 ####################

def _patch_has_values(patch: dict) -> bool:
    # 주문 patch에 이번 턴의 실제 변경 후보가 하나라도 있는지 확인한다.
    return bool(
        patch.get("sauce") is not None
        or patch.get("noodle_type") is not None
        or patch.get("noodle_portion") is not None
        or patch.get("toppings")
    )


def _canonicalize_structured_flavor_preference(
    decision: _NormalizedDecision,
) -> _NormalizedDecision:
    """구조화된 소프트 매운맛 기호만 기존 preference로 옮긴다.

    사용자 원문을 다시 해석하지 않고 Decision이 이미 구조화한 매운맛 회피 의미만
    canonical preference로 옮긴다. cannot_eat 등 강한 제한은 강도를 낮추지 않는다.
    """
    migrated = copy.deepcopy(decision)
    restrictions = []
    preferences = copy.deepcopy(migrated["preference_options"])
    preference_keys = {
        (entry.get("value"), entry.get("action"))
        for entry in preferences
        if isinstance(entry, dict)
    }
    for option in migrated["restriction_options"]:
        if (
            option.get("target") == "매운맛"
            and option.get("reason") == "dislike"
            and option.get("action") in ("add", "remove")
        ):
            canonical = {"value": "맵지 않게", "action": option["action"]}
            key = (canonical["value"], canonical["action"])
            if key not in preference_keys:
                preferences.append(canonical)
                preference_keys.add(key)
        else:
            restrictions.append(copy.deepcopy(option))
    migrated["restriction_options"] = restrictions
    migrated["preference_options"] = preferences
    return migrated


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
    if decision["skip_current"]:
        external["skip_current"] = True
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
        or decision["skip_current"]
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

    if decision["commit"] and decision["skip_current"]:
        errors.append("commit=true cannot be combined with skip_current=true")

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
    execution_items = section_execution_items(session["order"], section)
    items = {entry["item"]: entry.get("amount") for entry in execution_items}
    if not items:
        return None
    candidate = empty_order_patch()
    candidate["toppings"] = copy.deepcopy(items)
    return {
        "type": "execution",
        "source": "preselected",
        "section": section,
        "targets": list(items),
        "items": items,
        "candidate": candidate,
    }


def _with_object_particle(word: str) -> str:
    """한글 단어에 목적격 조사를 붙이고 그 외 문자열은 안전한 병기형을 쓴다."""
    if not word:
        return word
    code = ord(word[-1])
    if 0xAC00 <= code <= 0xD7A3:
        return f"{word}{'을' if (code - 0xAC00) % 28 else '를'}"
    return f"{word}을(를)"


def build_preselected_confirmation_reply(pending: dict) -> str:
    # Graph 밖에서 section이 바뀔 때 사용할 안전한 기본 확인 문장이다.
    items = pending.get("items") or pending.get("candidate", {}).get("toppings", {})
    amount_word = {"low": "적게", "normal": "보통", "high": "많이"}
    descriptions = []
    for name, amount in items.items():
        description = _with_object_particle(str(name))
        if amount in amount_word:
            description = f"{description} {amount_word[amount]}"
        descriptions.append(description)
    target = ", ".join(descriptions) if descriptions else "미리 고른 재료를"
    return f"미리 고른 {target} 담을까요?"


def build_execution_started_reply(session: _SessionState, section: str) -> str:
    """실제로 시작된 현재 section 작업만으로 결정적인 안내 문장을 만든다."""
    amount_word = {"low": "적게", "normal": "보통", "high": "많이"}
    items = section_execution_items(session["order"], section)
    if not items:
        raise ValueError(f"{section}에서 시작 안내를 만들 작업이 없음")

    descriptions = [
        f"{entry['item']} {amount_word[entry['amount']]}"
        if entry.get("amount") in amount_word
        else str(entry["item"])
        for entry in items
    ]
    return f"{', '.join(descriptions)} 담기를 시작해요."


def _pending_prompt(pending: dict | None) -> dict | None:
    # Response Agent에는 내부 pending 전체가 아니라 사용자에게 물을 사실만 전달한다.
    if pending is None:
        return None
    if pending.get("question_type") == "clarification":
        return {"type": "clarification"}
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


def _pending_confirmation_errors(session: _SessionState, decision: _NormalizedDecision) -> list[str]:
    # 질문 종류와 권한의 구조적 불일치만 검사한다. 사용자 문장은 읽지 않는다.
    pending = session.get("pending") or {}
    if (
        (pending.get("question_type") == "undelivered"
         or (pending.get("type") == "execution" and pending.get("question_type") == "clarification"))
        and decision["confirmation"] in ("accept", "reject")
        and not (pending.get("type") == "execution" and decision["commit"])
    ):
        return [
            "latest question is clarification or the pending question was not delivered; do not "
            "accept/reject an unasked question. Resolve explicit order meaning without confirmation, "
            "clarify if unresolved, or use commit only for explicit execution of a confirmed order"
        ]
    return []


def _record_question_delivery(session, next_prompt, *, delivered):
    # A successful generation satisfies the model-call contract, NOT proof that
    # the prose asks the intended question or that the user heard it. Decision
    # owns that semantic judgment from actual history; publisher failure restores
    # this marker. Never inspect sentence keywords here.
    pending = session.get("pending")
    if pending is None:
        return
    if not delivered:
        if pending.get("question_type") != "clarification":
            pending["question_type"] = "undelivered"
    elif next_prompt and next_prompt.get("type") == pending.get("type"):
        if pending.get("question_type") == "undelivered":
            pending.pop("question_type")


def generate_section_event_response(session, robot_state, call_recommendation, call_response, fallback_reply):
    """Read-only model orchestration for an actual conversational-stage event.

    Only validated pending candidates may change. No order, user history, Decision
    interpretation, execution authority or ROS control is created here.
    """
    working = copy.deepcopy(session)
    section = robot_state["section"]
    conflicts = [entry for entry in find_restriction_conflicts(working["order"])
                 if physical_section_for_order_key(entry["key"]) == section]
    policy = {"status": "pass", "reason": "section_transition", "execute": False, "conflicts": conflicts}
    result = None
    error = None
    recommendation_failed = False
    pending = working.get("pending")
    if (pending or {}).get("type") == "recommendation":
        result = validate_recommendation_result(working, {
            "proposal": pending.get("candidate"), "reason_tags": pending.get("reason_tags", []),
        }, robot_state)
        if "raw_proposal" in result:
            result["accepted_candidate"] = result.pop("raw_proposal")
            result["candidate_source"] = "revalidated_pending"
        working["pending"] = (
            {"type": "recommendation", "candidate": copy.deepcopy(result["proposal"]),
             "reason_tags": copy.deepcopy(result["reason_tags"])}
            if result["accepted_fields"] else None
        )
        recommendation_failed = not result["accepted_fields"] and bool(
            result["rejected_fields"] or result["reason_validation"]["schema_valid"] is False)
    if conflicts:
        policy.update(status="blocked", reason="restriction_conflict")
        if (working.get("pending") or {}).get("type") == "execution":
            working["pending"] = None
        next_prompt = {"type": "restriction_conflict", "conflicts": copy.deepcopy(conflicts)}
    else:
        if working.get("pending") is None:
            working["pending"] = build_preselected_section_confirmation(working, section)
            if working["pending"] is not None:
                # A valid confirmed order recovers an unusable old candidate.
                recommendation_failed = False
        if working.get("pending") is None:
            decision = new_decision()
            decision["recommendation"]["action"] = "request"
            try:
                raw = call_recommendation(copy.deepcopy(working), decision, copy.deepcopy(robot_state), None)
                result = validate_recommendation_result(working, raw, robot_state, current_event_only=True)
                recommendation_failed = not result["accepted_fields"] and bool(
                    result["rejected_fields"] or result["reason_validation"]["schema_valid"] is False)
                if result["accepted_fields"]:
                    working["pending"] = {
                        "type": "recommendation", "candidate": copy.deepcopy(result["proposal"]),
                        "reason_tags": copy.deepcopy(result["reason_tags"]),
                    }
            except Exception as exc:
                error = {"stage": "recommendation", "type": type(exc).__name__}
                recommendation_failed = True
        next_prompt = _pending_prompt(working.get("pending"))
        if next_prompt is None:
            next_prompt = {"type": "section_prompt", "section": section}
        if (working.get("pending") or {}).get("type") == "recommendation":
            policy["reason"] = "recommendation_proposed"
    if recommendation_failed:
        # Schema/domain failure is mechanical, not a new language judgment.
        # A valid empty optional proposal and safe partial proposals are not errors.
        error = error or {"stage": "recommendation", "type": "ValidationError"}
        _record_question_delivery(working, next_prompt, delivered=False)
        return {"session": working, "reply": RESPONSE_FAILURE_REPLY, "policy": policy,
                "recommendation_result": result, "next_prompt": next_prompt, "model_error": error}
    safe_fallback = fallback_reply
    if working["order"].get("restrictions") or (working.get("pending") or {}).get("type") == "recommendation":
        # A fault notice must not list prohibited options or ask execution
        # acceptance when pending actually points to a recommendation.
        safe_fallback = RESPONSE_FAILURE_REPLY
    try:
        # Rejected historical candidates remain audit evidence, not live proposals
        # when recovery now asks about the confirmed order or a restriction.
        response_recommendation = result if next_prompt.get("type") == "recommendation" else None
        reply = call_response(None, copy.deepcopy(working), copy.deepcopy(policy),
                              build_applied_changes(session, working), [],
                              copy.deepcopy(response_recommendation), copy.deepcopy(next_prompt), copy.deepcopy(robot_state), "task")
        if not isinstance(reply, str) or not reply.strip():
            raise ValueError("Response event reply is empty or not a string")
    except Exception as exc:
        error = {"stage": "response", "type": type(exc).__name__}
        reply = safe_fallback
        _record_question_delivery(working, next_prompt, delivered=(safe_fallback != RESPONSE_FAILURE_REPLY))
    else:
        _record_question_delivery(working, next_prompt, delivered=True)
    return {"session": working, "reply": reply, "policy": policy,
            "recommendation_result": result, "next_prompt": next_prompt, "model_error": error}


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

    ptype = pending.get("type")

    # execution pending 뒤의 명시적 commit은 confirmation field가 누락되어도 실행 수락과 같다.
    # 이렇게 해야 execute=true와 unresolved execution pending이 같은 Response 입력에 공존하지 않는다.
    if ptype == "execution" and effective["commit"] and effective["confirmation"] == "none":
        working["pending"] = None
        result = {
            "type": "execution_accepted",
            "source": pending.get("source", "current_update"),
            "section": pending.get("section"),
        }
        return working, effective, result

    # 수락·거절이 아니면 새 구조화 의미가 함께 있어도 pending을 그대로 둔다.
    # 주문·제한 변경 뒤의 execution pending은 final order가 확정된 다음 다시 맞춘다.
    if effective["confirmation"] not in ("accept", "reject"):
        return working, effective, result

    accepted = effective["confirmation"] == "accept"
    effective["confirmation"] = "none"
    working["pending"] = None

    # 강한 섭취 제한은 Decision이 삭제 의미를 확정해도 확인 수락 뒤에만 실제로 제거한다.
    if ptype == "restriction_removal":
        if accepted:
            candidate = pending.get("candidate") or {}
            removal = {
                "target": candidate.get("target"),
                "reason": candidate.get("reason"),
                "action": "remove",
            }
            working["order"] = apply_restriction_options(working["order"], [removal])
            result = {
                "type": "restriction_removal_accepted",
                "candidate": copy.deepcopy(candidate),
            }
        else:
            result = {
                "type": "restriction_removal_rejected",
                "pending_type": ptype,
                "candidate": copy.deepcopy(pending.get("candidate")),
            }
        # 한 번에 한 제한만 확인한다. 나머지 검증된 철회 요청은 pending 안에
        # 보존하여 다음 턴에 각각 묻고, 한 번의 동의로 전부 삭제하지 않는다.
        deferred = pending.get("deferred") or []
        if deferred:
            working["pending"] = {
                "type": "restriction_removal",
                "candidate": copy.deepcopy(deferred[0]),
            }
            if len(deferred) > 1:
                working["pending"]["deferred"] = copy.deepcopy(deferred[1:])
        return working, effective, result

    # 실행 확인 수락은 commit으로 바꾸고, 거절은 실행 대기만 닫는다.
    if ptype == "execution":
        if accepted:
            effective["commit"] = True
            result = {
                "type": "execution_accepted",
                "source": pending.get("source", "current_update"),
                "section": pending.get("section"),
            }
        else:
            if pending.get("source") == "preselected":
                working["order"] = _remove_patch_from_order(
                    working["order"], pending.get("candidate", empty_order_patch())
                )
            result = {"type": "execution_confirmation_rejected", "pending_type": ptype}
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
            result = {"type": "recommendation_rejected", "pending_type": ptype}
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
        refreshed = build_preselected_section_confirmation(session, section)
        if refreshed is not None and pending.get("question_type") in ("clarification", "undelivered"):
            refreshed["question_type"] = pending["question_type"]
        session["pending"] = refreshed
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

        errors = find_structural_decision_errors(decision) + _pending_confirmation_errors(state["session"], decision)
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
        if find_structural_decision_errors(repaired) or _pending_confirmation_errors(state["session"], repaired):
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
        session = copy.deepcopy(state["session"])
        if (policy["reason"] == "understanding" and (session.get("pending") or {}).get("type") == "execution"
                and session["pending"].get("question_type") != "undelivered"):
            session["pending"]["question_type"] = "clarification"
        return {
            "session": session,
            "recommendation_result": None,
            "policy": policy,
        }

    def prepare_task(state: _GraphTurnState) -> dict:
        # 아래 분기는 모두 조기 반환한다. 위에서 아래로 읽으면 실제 처리 순서와 같다.
        previous = copy.deepcopy(state["session"])
        raw_decision = _canonicalize_structured_flavor_preference(state["decision"])

        # 1. 이전 턴의 pending은 명시적 수락·거절만 먼저 처리한다.
        session, decision, pending_result = _resolve_pending(previous, raw_decision)
        recommendation_result = None

        # 2. 모델도 의미를 확정하지 못한 턴은 state를 바꾸지 않고 다시 묻는다.
        if decision["understanding"] == "clarify":
            if ((session.get("pending") or {}).get("type") == "execution"
                    and session["pending"].get("question_type") != "undelivered"):
                session["pending"]["question_type"] = "clarification"
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
        strong_removed_keys = {
            (entry["target"], entry["reason"])
            for entry in restriction_validation["newly_removed"]
            if entry["reason"] in SAFETY_RESTRICTION_REASONS
        }
        strong_removals = [
            option
            for option in restriction_validation["accepted"]
            if option["action"] == "remove"
            and (option["target"], option["reason"]) in strong_removed_keys
        ]
        immediately_applicable_restrictions = [
            option
            for option in restriction_validation["accepted"]
            if option not in strong_removals
        ]
        session["order"] = apply_restriction_options(
            session["order"], immediately_applicable_restrictions
        )
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

        # 추천과 무관한 현재 주문만 바뀌면 미래 후보는 보존한다. 제한/취향이나
        # 후보와 관련된 주문이 바뀌면 기존 무효화 정책을 그대로 적용한다.
        # request/revise 턴은 Recommendation Agent가 기존 후보까지 읽어야 하므로 호출 전까지 보존한다.
        applied = build_applied_changes(previous, session)
        pending = session.get("pending")
        independent_current_change = (
            isinstance(pending, dict)
            and pending.get("type") == "recommendation"
            and is_future_recommendation_pending(pending, state["robot_state"]["section"])
            and bool(applied["order_changes"])
            and all(
                physical_section_for_order_key(field) == state["robot_state"]["section"]
                for field in applied["order_changes"]
            )
            and not any(value for key, value in applied.items() if key != "order_changes")
        )
        if (
            isinstance(session.get("pending"), dict)
            and session["pending"].get("type") == "recommendation"
            and decision["recommendation"]["action"] == "none"
            and any(applied.values())
            and not independent_current_change
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

        # 강한 제한 삭제는 자연어 keyword가 아니라 검증된 구조적 remove 동작만 확인 대상으로 만든다.
        if strong_removals:
            if session.get("pending") is not None:
                policy = {
                    "status": "clarify",
                    "reason": "pending_confirmation_required",
                    "execute": False,
                    "conflicts": [],
                }
            else:
                candidate = {
                    "target": strong_removals[0]["target"],
                    "reason": strong_removals[0]["reason"],
                }
                session["pending"] = {
                    "type": "restriction_removal",
                    "candidate": candidate,
                }
                if len(strong_removals) > 1:
                    session["pending"]["deferred"] = [
                        {"target": option["target"], "reason": option["reason"]}
                        for option in strong_removals[1:]
                    ]
                policy = {
                    "status": "clarify",
                    "reason": "restriction_removal_confirmation_required",
                    "execute": False,
                    "conflicts": [],
                    "candidate": copy.deepcopy(candidate),
                    "deferred": copy.deepcopy(strong_removals[1:]),
                }
            return {
                "session": session,
                "decision": decision,
                "recommendation_result": recommendation_result,
                "policy": policy,
            }

        if (
            pending_result
            and pending_result["type"] in {"restriction_removal_accepted", "restriction_removal_rejected"}
            and (session.get("pending") or {}).get("type") == "restriction_removal"
        ):
            return {
                "session": session,
                "decision": decision,
                "recommendation_result": None,
                "policy": {
                    "status": "clarify", "reason": "restriction_removal_confirmation_required",
                    "execute": False, "conflicts": [],
                    "candidate": copy.deepcopy(session["pending"]["candidate"]),
                    "resolved_confirmation": copy.deepcopy(pending_result),
                },
            }

        # 추천 뒤에도 쓰는 계산 결과를 그대로 잇는다. 주문 변경을 다시 적용하지 않는다.
        return {
            "session": session,
            "decision": decision,
            "recommendation_result": None,
            "_task_context": {
                "pending_result": pending_result,
                "applied": applied,
                "accepted_patch": accepted_patch,
                "restriction_validation": restriction_validation,
                "restriction_enforcement": restriction_enforcement,
            },
        }

    def route_prepared_task(state: _GraphTurnState) -> str:
        if state.get("_task_context") is None:
            return "response"
        if state["decision"]["recommendation"]["action"] in ("request", "revise"):
            return "recommendation"
        return "finalize"

    def run_recommendation(state: _GraphTurnState) -> dict:
        # 이번 턴의 제한·취향·주문 변경을 먼저 반영한 상태로 한 번만 추천한다.
        session = state["session"]
        decision = state["decision"]
        try:
            raw = call_recommendation(
                copy.deepcopy(session), copy.deepcopy(decision),
                copy.deepcopy(state["robot_state"]), state["user_text"],
            )
        except Exception as error:
            raw = None
            recommendation_error = {"type": type(error).__name__}
        else:
            recommendation_error = None
        rec_validation = validate_recommendation_result(session, raw, state["robot_state"])
        recommendation_result = copy.deepcopy(rec_validation)
        if recommendation_error is not None:
            recommendation_result["model_error"] = recommendation_error

        # 추천은 검증된 후보를 pending에만 보관하며, 여기서 주문을 확정하지 않는다.
        if rec_validation["accepted_fields"]:
            session["pending"] = {
                "type": "recommendation",
                "candidate": copy.deepcopy(rec_validation["proposal"]),
                "reason_tags": copy.deepcopy(rec_validation["reason_tags"]),
            }
            policy = {
                "status": "pass", "reason": "recommendation_proposed",
                "execute": False, "conflicts": [],
            }
        else:
            # 실패한 새 추천으로 이전 후보를 확인 대상으로 남기지 않는다.
            session["pending"] = None
            policy = {
                "status": "clarify", "reason": "recommendation_unavailable",
                "execute": False, "conflicts": [],
                "rejected_fields": copy.deepcopy(rec_validation["rejected_fields"]),
            }

        # 미래 추천 때문에 독립된 현재 확정 주문의 명시적 실행 요청을 버리지 않는다.
        context = state["_task_context"]
        if not decision["commit"] or pending_blocks_execution(
            session.get("pending"), state["robot_state"]["section"]
        ):
            context = None
        return {
            "session": session,
            "decision": decision,
            "recommendation_result": recommendation_result,
            "policy": policy,
            "_task_context": context,
        }

    def route_recommended_task(state: _GraphTurnState) -> str:
        return "response" if state["_task_context"] is None else "finalize"

    def finalize_task(state: _GraphTurnState) -> dict:
        # 추천 전 계산 결과와 추천 후 pending을 함께 읽어 기존 실행 검사를 이어간다.
        session = state["session"]
        decision = state["decision"]
        recommendation_result = state["recommendation_result"]
        context = state["_task_context"]
        pending_result = context["pending_result"]
        applied = context["applied"]
        accepted_patch = context["accepted_patch"]
        restriction_validation = context["restriction_validation"]
        restriction_enforcement = context["restriction_enforcement"]

        # 6. 단순 거절은 여기서 끝내되, 현재 section을 새 값으로 교체한 복합 발화는
        # commit 검사 또는 새 execution pending 생성까지 계속 진행한다.
        section = state["robot_state"]["section"]
        direct_current_fields = (
            set(changed_order_keys_from_patch(accepted_patch))
            & set(allowed_order_fields(section))
        )
        if direct_current_fields and (session.get("pending") or {}).get("type") == "execution":
            # 현재 주문을 명확하게 지정한 유효 patch만 명확화 상태를 해소한다.
            # 실제 실행 여부는 아래 commit 검사에서 별도로 판단한다.
            session["pending"].pop("question_type", None)
        has_direct_current_replacement = (
            bool(direct_current_fields)
        )
        rejected_without_current_replacement = (
            pending_result
            and pending_result.get("type") in {
                "execution_confirmation_rejected",
                "recommendation_rejected",
                "restriction_removal_rejected",
            }
            and not has_direct_current_replacement
            and not decision["skip_current"]
        )
        if rejected_without_current_replacement:
            policy = {
                "status": "pass",
                "reason": pending_result["type"],
                "execute": False,
                "conflicts": [],
                "rejected_pending_type": pending_result["pending_type"],
            }
            if pending_result.get("candidate") is not None:
                policy["candidate"] = copy.deepcopy(pending_result["candidate"])
            return {
                "session": session,
                "decision": decision,
                "recommendation_result": recommendation_result,
                "policy": policy,
            }

        # 7. optional section 생략은 commit과 분리하고, 빈 선택 상태에서만 허용한다.
        if decision["skip_current"]:
            current_items = section_execution_items(session["order"], section)
            if section not in ("veggie", "meat", "extra"):
                policy = {
                    "status": "clarify", "reason": "invalid_skip_section",
                    "execute": False, "conflicts": [],
                }
            elif session.get("pending") is not None:
                policy = {
                    "status": "clarify", "reason": "pending_confirmation_required",
                    "execute": False, "conflicts": [],
                }
            elif current_items:
                policy = {
                    "status": "clarify", "reason": "skip_requires_empty_section",
                    "execute": False, "conflicts": [],
                    "selected": copy.deepcopy(current_items),
                }
            elif state["robot_state"].get("active_task") is not None or state["robot_state"].get("task_queue"):
                policy = {
                    "status": "blocked", "reason": "robot_busy",
                    "execute": False, "conflicts": [],
                }
            else:
                policy = {
                    "status": "pass", "reason": "section_skip_allowed",
                    "execute": True, "conflicts": [],
                }
            return {
                "session": session,
                "decision": decision,
                "recommendation_result": recommendation_result,
                "policy": policy,
            }

        # 8. commit 의미는 모델을 믿되 필수 주문과 robot busy는 코드가 마지막으로 확인한다.
        if decision["commit"]:
            missing = missing_current_section(session["order"], section)
            unresolved_pending = session.get("pending")
            explicit_current_change = current_section_changed(applied, section)
            current_items = section_execution_items(session["order"], section)
            accepted_section = (
                pending_result.get("section")
                if pending_result and pending_result.get("type") == "execution_accepted"
                else None
            )
            if accepted_section is not None and accepted_section != section:
                policy = {
                    "status": "clarify", "reason": "stale_execution_confirmation",
                    "execute": False, "conflicts": [],
                }
            elif pending_blocks_execution(unresolved_pending, section):
                policy = {
                    "status": "clarify", "reason": "pending_confirmation_required",
                    "execute": False, "conflicts": [],
                }
            elif section in ("veggie", "meat", "extra") and not current_items:
                policy = {
                    "status": "clarify", "reason": "empty_section_commit",
                    "execute": False, "conflicts": [],
                }
            elif missing:
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

        # 9. 현재 section 값만 바뀌었다면 바로 실행하지 않고 execution pending을 만든다.
        if (
            session.get("pending") is None
            and (current_section_changed(applied, section)
                 or (pending_result or {}).get("type") == "recommendation_accepted")
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

        final_reason = (
            "recommendation_accepted"
            if pending_result and pending_result.get("type") == "recommendation_accepted"
            else "state_update_only"
        )
        return {
            "session": session,
            "decision": decision,
            "recommendation_result": recommendation_result,
            "policy": {
                "status": "pass",
                "reason": final_reason,
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
        response_failed = False
        response_attempted = False
        future = build_future_changes(applied, state["robot_state"])
        next_prompt = _pending_prompt(state["session"].get("pending"))
        reason = state["policy"].get("reason")
        recommendation_result = copy.deepcopy(state.get("recommendation_result"))
        if reason == "recommendation_accepted" and recommendation_result is None:
            previous_pending = state["previous_session"].get("pending") or {}
            applied_proposal = empty_order_patch()
            for field, value in applied["order_changes"].items():
                if field.startswith("toppings."):
                    applied_proposal["toppings"][field.split(".", 1)[1]] = "none" if value is None else value
                else:
                    applied_proposal[field] = "none" if value is None else value
            recommendation_result = {
                "proposal": applied_proposal,
                "accepted_candidate": copy.deepcopy(previous_pending.get("candidate")),
                "reason_tags": copy.deepcopy(previous_pending.get("reason_tags", [])),
                "accepted_fields": list(applied["order_changes"]), "rejected_fields": [],
                "reason_validation": {"source": "pending_model", "schema_valid": None, "semantic_status": "unverified"},
            }
        if reason == "understanding":
            next_prompt = {"type": "clarification"}
        elif state["policy"].get("issues"):
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

        if state["policy"].get("execute"):
            # 실행 허용 상태는 Response 생성보다 실제 section 작업 목록을 단일 진실로 사용한다.
            # 빈 optional section의 명시적 제거+commit은 node가 section 전환 문장으로 교체한다.
            starting_items = section_execution_items(
                state["session"]["order"], state["robot_state"]["section"]
            )
            reply = (
                build_execution_started_reply(state["session"], state["robot_state"]["section"])
                if starting_items
                else ""
            )
        elif reason == "recommendation_unavailable":
            # 추천 후보가 전부 거절된 실패 턴은 Response 모델이 과거 답변을 반복하지 못하게 한다.
            # 사용자의 취향을 새로 해석하지 않고 현재부터 물리적으로 가능한 선택 범위만 안내한다.
            reply = build_recommendation_unavailable_reply(
                build_recommendation_scope(state["session"], state["robot_state"])
            )
        else:
            response_attempted = True
            try:
                reply = call_response(
                    state["user_text"], copy.deepcopy(state["session"]),
                    copy.deepcopy(state["policy"]), copy.deepcopy(applied), copy.deepcopy(future),
                    copy.deepcopy(recommendation_result), copy.deepcopy(next_prompt),
                    copy.deepcopy(state["robot_state"]), state["decision"]["route"],
                )
                if not isinstance(reply, str) or not reply.strip():
                    raise ValueError("Response reply is empty or not a string")
            except Exception as error:
                # Speech failure must not undo validated restrictions or a rejected execution.
                # This exceptional fault message never claims an order/start/completion.
                reply = RESPONSE_FAILURE_REPLY
                response_failed = True
                state["policy"] = dict(state["policy"], response_error={"type": type(error).__name__})

        # 실제 변경 event와 자연어 대화를 session history에 마지막으로 기록한다.
        session = copy.deepcopy(state["session"])
        if response_attempted:
            _record_question_delivery(session, next_prompt, delivered=not response_failed)
        event = build_turn_action_event(applied, future)
        if event is not None:
            session["action_history"].append(event)
        session["history"].append({"role": "user", "content": state["user_text"]})
        session["history"].append({"role": "assistant", "content": reply})
        return {"session": session, "policy": state["policy"], "reply": reply,
                "recommendation_result": recommendation_result, "_task_context": None}

    # 모델 역할을 노드로 드러내되 상태 반영·추천·실행 검사의 순서는 보존한다.
    # 임시 문맥은 내부 채널에만 두고, invoke 입출력은 기존 8개 공개 필드로 투영한다.
    graph = StateGraph(_GraphTurnState, input_schema=_TurnState, output_schema=_TurnState)
    graph.add_node("interpret_decision", interpret_decision)
    graph.add_node("process_general", process_general)
    graph.add_node("prepare_task", prepare_task)
    graph.add_node("run_recommendation", run_recommendation)
    graph.add_node("finalize_task", finalize_task)
    graph.add_node("generate_response", generate_response)

    graph.add_edge(START, "interpret_decision")
    graph.add_conditional_edges(
        "interpret_decision",
        route_by_decision,
        {"general": "process_general", "task": "prepare_task"},
    )
    graph.add_edge("process_general", "generate_response")
    graph.add_conditional_edges(
        "prepare_task", route_prepared_task,
        {"response": "generate_response", "recommendation": "run_recommendation", "finalize": "finalize_task"},
    )
    graph.add_conditional_edges(
        "run_recommendation", route_recommended_task,
        {"response": "generate_response", "finalize": "finalize_task"},
    )
    graph.add_edge("finalize_task", "generate_response")
    graph.add_edge("generate_response", END)
    return graph.compile()
