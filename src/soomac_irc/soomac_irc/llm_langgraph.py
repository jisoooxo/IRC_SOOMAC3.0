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
    build_recommendation_scope,
    build_recommendation_unavailable_reply,
    build_turn_action_event,
    changed_order_keys_from_patch,
    current_section_changed,
    enforce_restrictions,
    missing_current_section,
    physical_section_for_order_key,
    section_execution_items,
    validate_order_patch,
    validate_recommendation_proposal,
    validate_restriction_options,
    SAFETY_RESTRICTION_REASONS,
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


def _canonicalize_structured_flavor_preference(
    decision: _NormalizedDecision,
) -> _NormalizedDecision:
    """모델이 flavor preference를 unsupported restriction으로 낸 경우 의미를 보존한다.

    사용자 원문을 다시 해석하지 않고 Decision이 이미 구조화한 매운맛 회피 의미만
    canonical preference로 옮긴다.
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
            and option.get("reason") in ("cannot_eat", "dislike")
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


def _canonical_recommendation_reason_tags(
    session: _SessionState,
    robot_state: dict,
    proposal: dict,
) -> list[str]:
    """확정 state와 현재 section으로만 추천 근거를 만든다."""
    current_section = robot_state.get("section")
    labels = {
        "noodle": "면", "veggie": "야채", "meat": "육류",
        "extra": "추가 재료", "sauce": "소스",
    }
    proposal_fields = [
        field for field in ("sauce", "noodle_type", "noodle_portion")
        if proposal.get(field) is not None
    ]
    proposal_fields.extend(f"toppings.{item}" for item in (proposal.get("toppings") or {}))
    proposal_sections = {
        physical_section_for_order_key(field) for field in proposal_fields
        if physical_section_for_order_key(field) is not None
    }
    reason_section = next(iter(proposal_sections)) if len(proposal_sections) == 1 else current_section
    tags = [f"현재 {labels[reason_section]} 단계"] if reason_section in labels else []
    toppings = proposal.get("toppings") or {}
    preferences = {entry.get("value") for entry in session.get("preferences", [])}
    if reason_section == "extra" and "치즈" in toppings and "맵지 않게" in preferences:
        tags.insert(0, "매운맛 회피")
    if reason_section == "extra" and "페퍼론치노" in toppings and "매콤하게" in preferences:
        tags.insert(0, "매운맛 선호")
    if reason_section == "meat" and "소시지" in toppings:
        restricted_targets = {
            entry.get("target") for entry in session["order"].get("restrictions", [])
            if entry.get("reason") in SAFETY_RESTRICTION_REASONS
        }
        if restricted_targets & {"게살", "갑각류"}:
            tags.insert(0, "게살 섭취 제한 고려")
    return tags


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


def build_execution_rejected_reply() -> str:
    """실행 전 거절을 물리 중단이나 주문 취소로 과장하지 않는 고정 문장이다."""
    return "알겠어요. 이번 담기는 진행하지 않을게요."


def build_restriction_removal_rejected_reply(candidate: dict | None) -> str:
    """강한 제한 삭제 거절 뒤 유지되는 canonical safety state를 고정 문장으로 알린다."""
    target = (candidate or {}).get("target")
    if target:
        return f"알겠어요. {target} 섭취 제한은 그대로 유지할게요."
    return "알겠어요. 섭취 제한은 그대로 유지할게요."


def _describe_order_changes(order_changes: dict) -> str:
    amount_words = {"low": "적게", "normal": "보통으로", "high": "많이"}
    display_labels = {
        "sauce": "소스",
        "noodle_type": "면 종류",
        "noodle_portion": "면 양",
    }
    descriptions = []
    for field, value in order_changes.items():
        if value is None:
            continue
        item = field.split(".")[-1] if field.startswith("toppings.") else display_labels.get(field, field)
        if field in ("sauce", "noodle_type"):
            descriptions.append(_with_object_particle(str(value)))
        elif value in amount_words:
            descriptions.append(f"{_with_object_particle(item)} {amount_words[value]}")
        else:
            descriptions.append(f"{item} {_with_object_particle(str(value))}")
    return ", ".join(descriptions)


def build_recommendation_proposed_reply(recommendation_result: dict | None) -> str:
    """검증된 추천안과 canonical reason만 말하는 고정 문장이다."""
    result = recommendation_result or {}
    proposal = result.get("proposal") or {}
    changes = {
        field: proposal.get(field)
        for field in ("sauce", "noodle_type", "noodle_portion")
        if proposal.get(field) is not None
    }
    changes.update({f"toppings.{item}": amount for item, amount in (proposal.get("toppings") or {}).items()})
    description = _describe_order_changes(changes) or "이 구성으로"
    reason_tags = result.get("reason_tags") or []
    if "매운맛 회피" in reason_tags:
        prefix = "매운맛을 피하는 취향을 고려해 "
    elif "게살 섭취 제한 고려" in reason_tags:
        prefix = "게살 섭취 제한을 고려해 "
    else:
        prefix = ""
    return f"{prefix}{description} 추천해요. 적용할까요?"


def build_recommendation_accepted_reply(applied: dict, pending: dict | None) -> str:
    """추천 수락을 새 추천으로 되묻지 않고 확정 사실로 안내한다."""
    description = _describe_order_changes(applied.get("order_changes") or {}) or "추천한 구성으로"
    suffix = " 지금 담을까요?" if (pending or {}).get("type") == "execution" else ""
    return f"알겠어요. {description} 선택했어요.{suffix}"


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
        raw_decision = _canonicalize_structured_flavor_preference(state["decision"])

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
            canonical_reason_tags = _canonical_recommendation_reason_tags(
                session, state["robot_state"], rec_validation["proposal"]
            )
            recommendation_result = {
                "proposal": copy.deepcopy(rec_validation["proposal"]),
                "reason_tags": canonical_reason_tags,
                "accepted_fields": copy.deepcopy(rec_validation["accepted_fields"]),
                "rejected_fields": copy.deepcopy(rec_validation["rejected_fields"]),
            }
            if rec_validation["accepted_fields"]:
                session["pending"] = {
                    "type": "recommendation",
                    "candidate": copy.deepcopy(rec_validation["proposal"]),
                    "reason_tags": copy.deepcopy(canonical_reason_tags),
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

        # 6. 단순 거절은 여기서 끝내되, 현재 section을 새 값으로 교체한 복합 발화는
        # commit 검사 또는 새 execution pending 생성까지 계속 진행한다.
        section = state["robot_state"]["section"]
        direct_current_fields = (
            set(changed_order_keys_from_patch(accepted_patch))
            & set(allowed_order_fields(section))
        )
        has_direct_current_replacement = (
            bool(direct_current_fields)
            and current_section_changed(applied, section)
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
            elif unresolved_pending is not None:
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
        elif reason == "execution_confirmation_rejected":
            # pending은 이미 닫혔고 실행은 시작되지 않았다. 자유 생성이 과거 중단·남은 확인을
            # 만들지 못하도록 canonical event만으로 문장을 소유한다.
            reply = build_execution_rejected_reply()
        elif reason == "restriction_removal_rejected":
            # 강한 제한 유지도 모델의 추측 없이 Runtime이 확정 사실만 말한다.
            reply = build_restriction_removal_rejected_reply(state["policy"].get("candidate"))
        elif reason == "recommendation_proposed":
            reply = build_recommendation_proposed_reply(state.get("recommendation_result"))
        elif reason == "recommendation_accepted":
            reply = build_recommendation_accepted_reply(applied, state["session"].get("pending"))
        elif reason == "recommendation_unavailable":
            # 추천 후보가 전부 거절된 실패 턴은 Response 모델이 과거 답변을 반복하지 못하게 한다.
            # 사용자의 취향을 새로 해석하지 않고 현재부터 물리적으로 가능한 선택 범위만 안내한다.
            reply = build_recommendation_unavailable_reply(
                build_recommendation_scope(state["session"], state["robot_state"])
            )
        else:
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
