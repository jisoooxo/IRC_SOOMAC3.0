import copy

from soomac_irc.order_change import (
    apply_menu_changes,
    validate_menu_changes,
)
from soomac_irc.order import (
    EXTRAS,
    MEATS,
    NOODLE_TYPES,
    SAUCES,
    SECTION_ORDER,
    VEGGIES,
    missing_requirements,
    physical_section_for,
)
from soomac_irc.recommendation import revise_pending_recommendation, validate_recommended_order
from soomac_irc.restriction import (
    apply_restriction_changes,
    order_restriction_conflicts,
    strongest_restriction_for,
    validate_restriction_changes,
)


"""
Tool 요청을 실제로 허용할지 판단
로봇이 이미 처리한 재료 변경 차단
추천 확정·수정 처리
section 제외·진행·취소 판단
허용된 다음 주문 상태 생성
"""


def confirmation_block_reason(intent: bool | None) -> str:
    # 명시적 거절과 긍정 확인 부족을 transaction reason으로 바꾼다.
    # 입력: True, False, None
    # 상태 변경: 없음
    # 반환: confirm_validation에 기록할 reason
    if intent is False:
        return "explicit_rejection"

    return "explicit_confirmation_required"

"""
update() 함수는 딕셔너리(Dictionary)나 세트(Set) 같은 자료구조에서 값을 추가하거나 수정할 때 사용하는 메서드로,
맨 뒤에 들어간다고 보면 된다.
"""



def section_items_for(section: str) -> list[str] | None:
    # section 전체 제외 시 해당 section의 정본 재료를 반환한다.
    if section == "veggie":
        return list(VEGGIES)

    if section == "meat":
        return list(MEATS)

    if section == "extra":
        return list(EXTRAS)

    return None


def build_unavailable_section_skip(
    order: dict,
    execution: dict,
) -> dict | None:
    # 현재 section에 선택값이 없고 모든 재료가 제약으로 막혔을 때
    # 즉시 제외하거나 사용자 확인을 받을 proposal을 만든다.
    section = execution.get("section")
    items = section_items_for(section)

    if items is None:
        return None

    if any(item in order["toppings"] for item in items):
        return None

    restrictions = [
        strongest_restriction_for(order, item)
        for item in items
    ]

    if any(restriction is None for restriction in restrictions):
        return None

    reasons = [
        restriction["reason"]
        for restriction in restrictions
    ]

    # 취향 제약은 안전 제약과 달리 section 제외 전 사용자 확인을 받는다.
    needs_confirmation = "dislike" in reasons

    if needs_confirmation:
        reason = "dislike"
    else:
        reason = next(
            candidate
            for candidate in ("allergy", "cannot_eat", "dietary_rule")
            if candidate in reasons
        )

    return {
        "section": section,
        "reason": reason,
        "items": items,
        "restriction_changes": [],
        "source": "all_options_restricted",
        "needs_confirmation": needs_confirmation,
        "applied": False,
    }


def commit_section_skip(
    order: dict,
    execution: dict,
    recommendation: dict,
    proposal: dict,
) -> dict:
    # 확인을 받은 section 제외만 새 상태 복사본에 원자적으로 반영한다.
    # 입력 order·execution·recommendation은 직접 수정하지 않는다.
    order_after = copy.deepcopy(order)
    execution_after = copy.deepcopy(execution)
    recommendation_after = copy.deepcopy(recommendation)

    section = proposal.get("section")
    reason = proposal.get("reason")
    items = proposal.get("items")
    expected_items = section_items_for(section)

    if expected_items is None or items != expected_items:
        raise ValueError("확정할 section skip proposal이 잘못됨")

    restriction_changes = proposal.get("restriction_changes", [])

    clean_restrictions, invalid_restrictions = validate_restriction_changes(
        restriction_changes
    )

    if invalid_restrictions or len(clean_restrictions) != len(restriction_changes):
        raise ValueError("확정할 section restriction proposal이 잘못됨")

    restriction_applied, restriction_unchanged = apply_restriction_changes(
        order_after,
        clean_restrictions,
    )

    selected = execution_after.get("selected", [])

    if not isinstance(selected, list):
        raise ValueError("execution.selected는 list여야 함")

    active_task = execution_after.get("active_task")
    active_class = (
        active_task.get("class")
        if isinstance(active_task, dict)
        else None
    )

    completed_classes = {
        task.get("class")
        for task in execution_after.get("completed_tasks", [])
        if isinstance(task, dict)
    }

    conflicts = {
        conflict["item"]: conflict
        for conflict in order_restriction_conflicts(order_after)
    }

    blocked_constraint = []
    kept_physical_conflict = []

    for item in items:
        if item not in order_after["toppings"]:
            continue

        conflict = conflicts.get(item)

        # 이미 담기 시작했거나 완료한 재료는 주문에서도 삭제하지 않는다.
        if active_class == item or item in completed_classes:
            if conflict is not None:
                kept_physical_conflict.append(conflict)

            continue

        del order_after["toppings"][item]
        key = f"toppings.{item}"

        if key in selected:
            selected.remove(key)

        if conflict is not None:
            blocked_constraint.append(conflict)

    skipped_sections = execution_after.get("skipped_sections", {})

    if not isinstance(skipped_sections, dict):
        raise ValueError("execution.skipped_sections는 dict여야 함")

    skipped_sections[section] = reason
    execution_after["selected"] = selected
    execution_after["skipped_sections"] = skipped_sections

    confirmed_sections = recommendation_after.get(
        "confirmed_sections",
        [],
    )

    if not isinstance(confirmed_sections, list):
        raise ValueError("recommendation.confirmed_sections는 list여야 함")

    recommendation_after = {
        "phase": "idle",
        "confirmed_sections": [
            confirmed_section
            for confirmed_section in confirmed_sections
            if confirmed_section != section
        ],
    }

    return {
        "order": order_after,
        "execution": execution_after,
        "recommendation": recommendation_after,
        "restriction_applied": restriction_applied,
        "restriction_unchanged": restriction_unchanged,
        "blocked_constraint": blocked_constraint,
        "kept_physical_conflict": kept_physical_conflict,
    }
def new_turn_result(action: str, order_before: dict, order_after: dict) -> dict:
    # 한 턴에서 허용·차단된 내용을 기록할 빈 결과를 만든다.
    # 모델 출력의 changes와 다른 Python 내부 검증 결과이다.
    # 입력: Tool 이름, 변경 전 주문, 작업 중인 주문 복사본
    # 상태 변경: 없음
    # 반환: Reply와 commit이 함께 사용할 turn_result
    return { # 검수용 필드
        "action": action,
        "order_before": copy.deepcopy(order_before),
        "order_after": copy.deepcopy(order_after),
        "accepted": [],
        "ignored_same_value": [],
        "blocked_active_or_completed": [],
        "blocked_past_section": [],
        "blocked_constraint": [],          # 아직 담지 않았지만 restriction으로 제외
        "kept_physical_conflict": [],      # 이미 시작·완료·지나가서 유지
        "restriction_applied": [],         # 실제 추가·철회된 restriction
        "restriction_unchanged": [],       # 중복 추가 또는 없는 restriction 철회
        "invalid_value": [],
        "section_skip": None, # 명시적으로 건너뛴 section과 이유
        "confirm_validation": {
            "requested": False, # 현재 section 확정을 요청했는지
            "allowed": False,   # 로봇 작업을 시작해도 되는지
            "missing": {},
            "reason": None,
        },
        "cancel_validation": {
            "requested": False,         # 전체 취소 Tool이 들어왔는지
            "blocked": False,           # 이미 로봇 작업이 시작됐는지
            "needs_confirmation": False, # 실제 초기화 전 재확인이 필요한지
            "reason": None,
        },
        "recommendation_validation": {
            "clean": {},
            "dropped": [],
            "blocked": [],
            "proposal_created": False,
            "proposal_confirmed": False,
            "proposal_revised": False,
            "revision_reason": None,
            "needs_scope": False,
        },
    }

def physical_change_block_reason(execution: dict, field: str, item: str | None = None) -> str | None:
    # 주문값이 이미 로봇 작업에 들어갔거나 지나간 section인지 검사한다.
    # 반환: active_or_completed, past_section 또는 변경 가능한 경우 None
    section = execution.get("section")
    if section not in SECTION_ORDER:
        raise ValueError(f"섹션 이상함 : {section}")

    target_section = physical_section_for(field, item)
    if target_section not in SECTION_ORDER:
        raise ValueError(f"주문값의 물리 section을 찾을 수 없음 : {field}, {item}")

    active_task = execution.get("active_task")
    active_class = active_task.get("class") if isinstance(active_task, dict) else None
    completed_classes = {task.get("class") for task in execution.get("completed_tasks", []) if isinstance(task, dict)}

    # 면 종류와 면 양은 같은 면 투입 작업이 시작되면 함께 잠긴다.
    if field in ("noodle_type", "noodle_portion"):
        locked_classes = set(NOODLE_TYPES)
    elif field == "sauce":
        locked_classes = set(SAUCES)
    else:
        locked_classes = {item}

    if active_class in locked_classes or bool(completed_classes.intersection(locked_classes)):
        return "active_or_completed"
    if SECTION_ORDER.index(target_section) < SECTION_ORDER.index(section):
        return "past_section"
    return None


def remove_order_value(order: dict, selected: list[str], key: str, item: str) -> None:
    # restriction과 충돌한 아직 투입되지 않은 값을 주문 복사본에서 제거한다.
    # 입력 order와 selected는 order policy가 만든 복사본이어야 한다.
    if key == "sauce":
        order["sauce"] = None
    elif key == "noodle_type":
        order["noodle_type"] = None
    elif key.startswith("toppings."):
        order["toppings"].pop(item, None)
    else:
        raise ValueError(f"제약 충돌 주문 key가 잘못됨 : {key}")

    if key in selected:
        selected.remove(key)


def apply_restriction_conflict_policy(order_after: dict, execution_after: dict, turn_result: dict, newly_enabled: set[tuple[str, str]], confirmed_sections: list[str] | None = None) -> None:
    # 활성 restriction과 주문값의 충돌을 물리 진행 상태에 맞게 처리한다.
    # 이미 시작·완료·지나간 값은 유지하고 경고 대상으로 기록한다.
    # 아직 투입되지 않은 값은 주문 복사본에서 제거한다.
    selected = execution_after.get("selected", [])
    if not isinstance(selected, list):
        raise ValueError("execution.selected는 list여야 함")

    for conflict in order_restriction_conflicts(order_after):
        key = conflict["key"]
        item = conflict["item"]
        restriction = conflict["restriction"]
        restriction_key = (restriction["target"], restriction["reason"])

        # 이전 dislike는 사용자의 명시적 재선택을 막지 않는다.
        # 이번 턴에 새로 추가된 dislike만 즉시 주문에서 제외한다.
        if restriction["reason"] == "dislike" and restriction_key not in newly_enabled:
            continue

        if key == "noodle_type":
            field = "noodle_type"
            target_section = "noodle"
        elif key == "sauce":
            field = "sauce"
            target_section = "sauce"
        else:
            field = "toppings"
            target_section = physical_section_for("toppings", item)

        block_reason = physical_change_block_reason(execution_after, field, item)
        if block_reason is not None:
            turn_result["kept_physical_conflict"].append(conflict)
            continue

        remove_order_value(order_after, selected, key, item)
        if key in turn_result["accepted"]:
            turn_result["accepted"].remove(key)
        if confirmed_sections is not None and target_section in confirmed_sections:
            confirmed_sections.remove(target_section)

        turn_result["blocked_constraint"].append(conflict)

    execution_after["selected"] = selected
    turn_result["order_after"] = copy.deepcopy(order_after)

def build_policy_output(order_after: dict, execution_after: dict, recommendation_after: dict, turn_result: dict) -> dict:
    # Order Policy 결과를 Agent에 전달하는 공통 형식이다.
    # 이 함수는 실제 Node 상태를 commit하거나 ROS로 발행하지 않는다.
    turn_result["order_after"] = copy.deepcopy(order_after)
    return {
        "order": order_after,
        "execution": execution_after,
        "recommendation": recommendation_after,
        "turn_result": turn_result,
    }


def evaluate_confirm_request(order_after: dict, execution_after: dict, turn_result: dict, confirmation_intent: bool | None) -> None:
    # 일반 section 진행 요청을 검사한다.
    # 추천안 확인은 다음 추천 정책 함수에서 별도로 처리한다.
    section = execution_after.get("section")
    if section not in SECTION_ORDER:
        raise ValueError(f"섹션 이상함 : {section}")

    # 소스는 마지막에 투입하지만 주문 필수값은 noodle 단계에서 선택한다.
    check_sections = ["noodle"] if section == "sauce" else [section]
    missing = missing_requirements(order_after, check_sections)
    if confirmation_intent is not True:
        reason = confirmation_block_reason(confirmation_intent)
    elif missing:
        reason = "missing_required"
    else:
        reason = "ready"

    turn_result["confirm_validation"] = {
        "requested": True,
        "allowed": confirmation_intent is True and not missing,
        "missing": missing,
        "reason": reason,
    }


def evaluate_cancel_request(order_after: dict, execution_after: dict, turn_result: dict) -> None:
    # 전체 주문 취소는 지원하지 않고 현재 주문과 물리 상태를 유지한다.
    # 선택 여부와 로봇 시작 여부에 따라 사용자에게 설명할 reason만 만든다.
    active_task = execution_after.get("active_task")
    completed_tasks = execution_after.get("completed_tasks", [])
    robot_started = execution_after.get("robot_started", False)
    if not isinstance(completed_tasks, list):
        raise ValueError("execution.completed_tasks는 list여야 함")
    if type(robot_started) is not bool:
        raise ValueError("execution.robot_started는 bool이어야 함")

    has_order_value = bool(order_after.get("sauce") or order_after.get("noodle_type") or order_after.get("noodle_portion") or order_after.get("toppings") or order_after.get("restrictions"))
    physical_started = robot_started or active_task is not None or bool(completed_tasks)
    if physical_started:
        reason = "robot_already_started"
    elif has_order_value:
        reason = "order_already_selected"
    else:
        reason = "empty_order"

    turn_result["cancel_validation"] = {
        "requested": True,
        "blocked": True,
        "needs_confirmation": False,
        "reason": reason,
    }





def evaluate_order_policy(state: dict, confirmation_intent: bool | None = None) -> dict:
    # 원본 주문은 건드리지 않고 복사본에서만 검사·반영
    # 입력: 모델 Tool + llm_node의 주문·물리 진행 상태
    # 반환: 다음 order/execution, 필요한 분기의 recommendation, 이번 턴 검수표 turn_result
    # 여기서는 ROS 발행이나 사용자 문장 생성을 안 함. 물리 상태를 보고 가능한 변경만 확정함
    order_before = state["order"] # 주문 전 ㅇㅇ
    order_after = copy.deepcopy(order_before) # 복사한거
    execution_after = copy.deepcopy(state["execution"])

    recommendation_after = copy.deepcopy(state.get("recommendation", {"phase": "idle", "confirmed_sections": []}))
    if not isinstance(recommendation_after, dict):
        raise ValueError("recommendation은 dict여야 함")

    confirmed_sections = recommendation_after.get("confirmed_sections", [])
    if not isinstance(confirmed_sections, list):
        raise ValueError("recommendation.confirmed_sections는 list여야 함")

    tool_call = state.get("tool_call") # tool call dict 받아옴
    action = tool_call["name"] if tool_call is not None else "error"
    turn_result = new_turn_result(action, order_before, order_after)



    # 추천은 실제 order에 바로 반영하지 않고 확인용 proposal로만 만든다.
    if action == "recommend_order":
        changes = copy.deepcopy(tool_call["changes"])
        scope = changes["scope"]

        request = {
            # 사용자가 말한 추천 조건만 저장. adapter가 만든 메뉴값과 섞지 않음
            "scope": scope,
            "sections": copy.deepcopy(changes.get("sections", [])),
            "targets": copy.deepcopy(changes.get("targets", [])),
            "excluded": copy.deepcopy(changes.get("excluded", [])),
            "preferences": copy.deepcopy(changes.get("preferences", [])),
            "different": changes.get("different", False),
        }

        section = execution_after["section"]

        if section not in SECTION_ORDER:
            raise ValueError(f"섹션 이상함 : {section}")

        selected = execution_after.get("selected", [])

        if not isinstance(selected, list):
            raise ValueError("execution.selected는 list여야 함")

        active_task = execution_after.get("active_task")
        active_class = active_task.get("class") if isinstance(active_task, dict) else None

        completed_classes = {
            task.get("class")
            for task in execution_after.get("completed_tasks", [])
            if isinstance(task, dict)
        }

        # 추천과 restriction을 같이 말하면 restriction부터 주문 복사본에 적용
        restriction_changes = changes.get("restriction_changes", [])

        clean_restrictions, invalid_restrictions = validate_restriction_changes(restriction_changes)

        turn_result["invalid_value"].extend(invalid_restrictions)

        restriction_applied, restriction_unchanged = apply_restriction_changes(order_after, clean_restrictions)

        turn_result["restriction_applied"].extend(restriction_applied)
        turn_result["restriction_unchanged"].extend(restriction_unchanged)

        newly_enabled = {(change["target"], change["reason"]) for change in restriction_applied if change["enabled"]}

        # restriction과 충돌하는 기존 주문값은 물리 진행 상태에 따라 제거하거나 유지
        # 이전 dislike는 명시적 재선택을 막지 않지만 이번에 추가한 dislike는 적용
        # 이미 시작·완료됐거나 지나간 재료는 지우지 않고 안내 대상으로 남김
        execution_after["selected"] = selected
        apply_restriction_conflict_policy(order_after, execution_after, turn_result, newly_enabled, confirmed_sections)

        # 범위를 묻는 추천이어도 같은 발화의 restriction은 이미 안전하게 반영
        if scope == "ask":
            recommendation_after = {
                "phase": "await_scope",
                "request": request,
                "confirmed_sections": copy.deepcopy(confirmed_sections),
            }
            turn_result["recommendation_validation"]["needs_scope"] = True

            return {
                "order": order_after,
                "execution": execution_after,
                "recommendation": recommendation_after,
                "turn_result": turn_result,
            }

        food_sections = SECTION_ORDER[:4]

        if scope == "current":
            target_sections = [section] if section in food_sections else []

        elif scope == "remaining":
            target_sections = food_sections[food_sections.index(section):] if section in food_sections else []

        else:
            target_sections = changes["sections"].copy()

        recommended_order = copy.deepcopy(changes.get("recommended_order", {}))
        out_of_scope = []

        # 소스·면 종류·면 양은 주문상 noodle section에서 선택한다.
        for field in ("sauce", "noodle_type", "noodle_portion"):
            if field in recommended_order and "noodle" not in target_sections:
                out_of_scope.append({
                    "path": field,
                    "value": recommended_order[field],
                    "reason": "out_of_scope",
                })
                del recommended_order[field]

        for item, amount in list(recommended_order.get("toppings", {}).items()):
            item_section = physical_section_for("toppings", item)

            if item_section not in target_sections:
                out_of_scope.append({
                    "path": f"toppings.{item}",
                    "value": amount,
                    "reason": "out_of_scope",
                })
                del recommended_order["toppings"][item]

        if "toppings" in recommended_order and not recommended_order["toppings"]:
            del recommended_order["toppings"]

        recommendation_input = copy.deepcopy(changes)
        recommendation_input["recommended_order"] = recommended_order

        clean, dropped, blocked = validate_recommended_order(
            order_after, recommendation_input
        )
        dropped = out_of_scope + dropped

        proposal = copy.deepcopy(order_after)
        # proposal만 바꾸므로 사용자가 확인하기 전 실제 order는 그대로임
        applied, unchanged = apply_menu_changes(proposal, clean)

        turn_result["recommendation_validation"] = {
            "clean": copy.deepcopy(clean),
            "dropped": dropped,
            "blocked": blocked,
            "proposal_created": bool(applied),
            "proposal_confirmed": False, # 아직 추천을 보여주기만 했고 사용자가 확정하지 않음
            "needs_scope": False,
        }

        # 유효한 새 추천이 하나도 없으면 이전 proposal에 고착되지 않게 종료한다.
        if not applied:
            recommendation_after = {
                "phase": "idle",
                "confirmed_sections": copy.deepcopy(confirmed_sections),
            }

            return {
                "order": order_after,
                "execution": execution_after,
                "recommendation": recommendation_after,
                "turn_result": turn_result,
            }

        covered_sections = []

        for key in applied:
            if key in ("sauce", "noodle_type", "noodle_portion"):
                covered_section = "noodle"
            else:
                item = key.split(".", 1)[1]
                covered_section = physical_section_for("toppings", item)

            if covered_section not in covered_sections:
                covered_sections.append(covered_section)

        recommendation_after = {
            "phase": "confirming",
            "request": request,
            "proposal": proposal,
            "recommended_changes": copy.deepcopy(clean),
            "covered_sections": covered_sections,
            "confirmed_sections": copy.deepcopy(confirmed_sections),
        }

        return {
            "order": order_after,
            "execution": execution_after,
            "recommendation": recommendation_after,
            "turn_result": turn_result,
        }
    # 추천 확인 중의 메뉴 변경은 실제 주문이 아니라
    # 대기 중인 추천안에만 적용한다.
    if (
        action in ("set_order", "set_order_and_confirm")
        and recommendation_after.get("phase") == "confirming"
    ):
        changes = copy.deepcopy(tool_call["changes"])

        menu_changes = {
            field: changes[field]
            for field in (
                "sauce",
                "noodle_type",
                "noodle_portion",
                "toppings",
            )
            if field in changes
        }

        restriction_changes = changes.get(
            "restriction_changes",
            [],
        )

        # 알레르기·식이 조건만 말한 경우에는
        # 기존 restriction 안전 처리로 내려보낸다.
        if restriction_changes and not menu_changes:
            pass

        # 메뉴 수정과 안전 조건을 한 번에 요청하면
        # 어느 한쪽도 부분 반영하지 않는다.
        elif restriction_changes:
            turn_result["recommendation_validation"][
                "revision_reason"
            ] = "mixed_menu_and_restriction"

            return {
                "order": order_after,
                "execution": execution_after,
                "recommendation": recommendation_after,
                "turn_result": turn_result,
            }

        elif menu_changes:
            revision = revise_pending_recommendation(
                order_after,
                recommendation_after,
                menu_changes,
            )

            turn_result["recommendation_validation"].update({
                "clean": copy.deepcopy(
                    revision.get("clean", {})
                ),
                "dropped": copy.deepcopy(
                    revision.get("dropped", [])
                ),
                "blocked": copy.deepcopy(
                    revision.get("blocked", [])
                ),
                "proposal_revised": revision["applied"],
                "revision_reason": revision["reason"],
            })

            if revision["applied"]:
                recommendation_after = revision["recommendation"]

            # set_order_and_confirm으로 오분류되어도
            # 이 턴에는 실행하지 않는다.
            # 수정된 추천안을 보여주고 다음 턴에 다시 확인한다.
            return {
                "order": order_after,
                "execution": execution_after,
                "recommendation": recommendation_after,
                "turn_result": turn_result,
            }
    # 추천 확인 중 confirm_section이 들어와도 원문 동의가 있을 때만 반영한다.
    if action == "confirm_section" and recommendation_after.get("phase") == "confirming":

        if confirmation_intent is not True:
            turn_result["confirm_validation"] = {
                "requested": True,
                "allowed": False,
                "missing": {},
                "reason": confirmation_block_reason(confirmation_intent),
            }

            return {
                "order": order_after,
                "execution": execution_after,
                "recommendation": recommendation_after,
                "turn_result": turn_result,
            }

        # 입력은 현재 턴의 빈 changes가 아니라 직전 recommendation_state에 저장한 추천안임
        request = recommendation_after.get("request")
        recommended_changes = recommendation_after.get("recommended_changes")

        if not isinstance(request, dict):
            raise ValueError("recommendation.request는 dict여야 함")

        if not isinstance(recommended_changes, dict):
            raise ValueError("recommendation.recommended_changes는 dict여야 함")

        # 저장된 proposal을 그대로 믿지 않고 현재 주문·restriction으로 다시 검사
        recommendation_input = copy.deepcopy(request)
        recommendation_input["recommended_order"] = copy.deepcopy(recommended_changes)

        clean, dropped, blocked = validate_recommended_order(
            order_after, recommendation_input
        )

        confirmed_order = copy.deepcopy(order_after)
        applied, unchanged = apply_menu_changes(confirmed_order, clean)

        turn_result["accepted"].extend(applied)
        turn_result["ignored_same_value"].extend(unchanged)
        turn_result["recommendation_validation"] = {
            "clean": copy.deepcopy(clean),
            "dropped": dropped,
            "blocked": blocked,
            "proposal_created": False,
            "proposal_confirmed": bool(applied),
            "needs_scope": False,
        }

        selected = execution_after.get("selected", [])

        if not isinstance(selected, list):
            raise ValueError("execution.selected는 list여야 함")

        covered_sections = []

        for key in applied:
            if key not in selected:
                selected.append(key)

            if key in ("sauce", "noodle_type", "noodle_portion"):
                covered_section = "noodle"
            else:
                item = key.split(".", 1)[1]
                covered_section = physical_section_for("toppings", item)

            if covered_section not in covered_sections:
                covered_sections.append(covered_section)

        confirmed_sections_after = copy.deepcopy(confirmed_sections)

        for covered_section in covered_sections:
            if covered_section not in confirmed_sections_after:
                confirmed_sections_after.append(covered_section)

        order_after = confirmed_order
        execution_after["selected"] = selected
        turn_result["order_after"] = copy.deepcopy(order_after)

        # proposal은 소비했고, 확정된 section 목록만 다음 턴에 유지
        recommendation_after = {
            "phase": "idle",
            "confirmed_sections": confirmed_sections_after,
        }

        return {
            "order": order_after,
            "execution": execution_after,
            "recommendation": recommendation_after,
            "turn_result": turn_result,
        }

    # 현재 section의 재료 전체를 같은 이유로 거절
    if action == "refuse_section":
        changes = copy.deepcopy(tool_call["changes"])
        section = execution_after["section"]
        reason = changes.get("reason")
        items = changes.get("items")
        section_items = section_items_for(section)

        # 일부 재료만 거절했거나 다른 section 재료가 섞이면 전체 skip으로 인정하지 않음
        valid_items = (
            section_items is not None
            and isinstance(items, list)
            and all(isinstance(item, str) for item in items)
            and len(items) == len(section_items)
            and set(items) == set(section_items)
        )

        if not valid_items:
            turn_result["invalid_value"].append({
                "path": "items",
                "value": items,
                "reason": "invalid_section_refusal",
            })

            return {
                "order": order_after,
                "execution": execution_after,
                "recommendation": recommendation_after,
                "turn_result": turn_result,
            }

        restriction_changes = [
            {"target": item, "reason": reason, "enabled": True}
            for item in section_items
        ]
        clean_restrictions, invalid_restrictions = validate_restriction_changes(restriction_changes)

        turn_result["invalid_value"].extend(invalid_restrictions)

        # reason까지 정상인 경우에만 restriction과 skip을 적용
        if len(clean_restrictions) != len(section_items):
            return {
                "order": order_after,
                "execution": execution_after,
                "recommendation": recommendation_after,
                "turn_result": turn_result,
            }

        # dislike는 소프트 제약이므로 한 번 확인한 뒤 section을 건너뛴다.
        # 안전·식단 제약은 전체 재료가 제외되면 즉시 skip할 수 있다.
        needs_confirmation = reason == "dislike"
        section_skip = {
            "section": section,
            "reason": reason,
            "items": section_items,
            "restriction_changes": clean_restrictions,
            "source": "explicit_refusal",
            "needs_confirmation": needs_confirmation,
            "applied": False,
        }

        if needs_confirmation:
            # 사용자가 동의하기 전에 restriction·주문·추천 상태를 바꾸지 않는다.
            turn_result["section_skip"] = section_skip
        else:
            committed = commit_section_skip(
                order_after,
                execution_after,
                recommendation_after,
                section_skip,
            )
            order_after = committed["order"]
            execution_after = committed["execution"]
            recommendation_after = committed["recommendation"]
            turn_result["restriction_applied"].extend(
                committed["restriction_applied"]
            )
            turn_result["restriction_unchanged"].extend(
                committed["restriction_unchanged"]
            )
            turn_result["blocked_constraint"].extend(
                committed["blocked_constraint"]
            )
            turn_result["kept_physical_conflict"].extend(
                committed["kept_physical_conflict"]
            )
            section_skip["applied"] = True
            turn_result["section_skip"] = section_skip

        turn_result["order_after"] = copy.deepcopy(order_after)

        return {
            "order": order_after,
            "execution": execution_after,
            "recommendation": recommendation_after,
            "turn_result": turn_result,
        }


    # 추천 확인은 위에서 이미 처리됐고, 여기서는 일반 section 확정만 검사한다.
        # 추천 확인은 위에서 이미 처리됐고, 여기서는 일반 section 확정만 검사한다.
    if action == "confirm_section":
        evaluate_confirm_request(order_after, execution_after, turn_result, confirmation_intent)
        return build_policy_output(order_after, execution_after, recommendation_after, turn_result)

    if action == "cancel_order":
        # 전체 주문 취소는 지원하지 않고 현재 주문·물리 상태를 그대로 유지함
        evaluate_cancel_request(order_after, execution_after, turn_result)
        return build_policy_output(order_after, execution_after, recommendation_after, turn_result)

    # 나머지 Tool은 각 단계에서 별도로 처리한다.
    if action not in ("set_order", "set_order_and_confirm"):
        return build_policy_output(order_after, execution_after, recommendation_after, turn_result)


    changes = copy.deepcopy(tool_call["changes"])

    # restriction은 다음 단계에서 별도로 처리한다.
    menu_changes = {
        field: changes[field]
        for field in ("sauce", "noodle_type", "noodle_portion", "toppings")
        if field in changes
    }

    clean, invalid = validate_menu_changes(menu_changes)
    turn_result["invalid_value"].extend(invalid)

    section = execution_after["section"]

    if section not in SECTION_ORDER:
        raise ValueError(f"섹션 이상함 : {section}")

    selected = execution_after.get("selected", [])

    if not isinstance(selected, list):
        raise ValueError("execution.selected는 list여야 함")

    editable = {}
    # editable에 들어간 값만 마지막 apply 단계까지 살아남음

    # scalar와 topping을 같은 편집 가능성 검사에 넣는다.
    entries = [(field, None, clean[field]) for field in ("sauce", "noodle_type", "noodle_portion") if field in clean]
    entries.extend(("toppings", item, amount) for item, amount in clean.get("toppings", {}).items())

    for field, item, value in entries:
        key = f"toppings.{item}" if field == "toppings" else field

        # 면 양도 실제 면 작업과 함께 잠긴다.
        block_reason = physical_change_block_reason(execution_after, field, item)
        blocked = {"path": key, "value": value}

        if block_reason == "active_or_completed":
            turn_result["blocked_active_or_completed"].append(blocked)
            continue

        # 현재보다 앞선 물리 section은 이미 지나갔으므로 수정하지 않는다.
        if block_reason == "past_section":
            turn_result["blocked_past_section"].append(blocked)
            continue

        if field == "toppings":
            editable.setdefault("toppings", {})[item] = value
        else:
            editable[field] = value

    applied, unchanged = apply_menu_changes(order_after, editable)

    turn_result["accepted"].extend(applied)
    turn_result["ignored_same_value"].extend(unchanged)

    # selected에는 값이 아니라 검증 key만 저장한다.
    for key in applied:
        if (
            key.startswith("toppings.")
            and editable["toppings"][key.split(".", 1)[1]] == "none"
        ):
            if key in selected:
                selected.remove(key)

        elif key not in selected:
            selected.append(key)

    # restriction은 메뉴와 별도로 검증한 뒤 같은 주문 복사본에 반영한다.
    restriction_changes = changes.get("restriction_changes", [])

    clean_restrictions, invalid_restrictions = (
        validate_restriction_changes(restriction_changes)
    )

    turn_result["invalid_value"].extend(invalid_restrictions)

    restriction_applied, restriction_unchanged = (
        apply_restriction_changes(order_after, clean_restrictions)
    )

    turn_result["restriction_applied"].extend(restriction_applied)
    turn_result["restriction_unchanged"].extend(restriction_unchanged)

    newly_enabled = {
        (change["target"], change["reason"])
        for change in restriction_applied
        if change["enabled"]
    }

    # 새 restriction과 현재 주문의 충돌을 검사한다.
    # 기존 dislike는 소프트 제약이다.
    # 사용자가 해당 재료를 직접 다시 선택하면 허용한다.
    # 이미 물리적으로 시작됐거나 지나간 재료는 order에서도 지우지 않는다.
    # 적용 직후 restriction으로 빠졌다면 최종 accepted로 표시하지 않는다.
    # selected의 기존 반영 시점을 유지하도록 검사용 딕셔너리만 얕게 복사한다.
    execution_for_validation = execution_after.copy()
    execution_for_validation["selected"] = selected
    apply_restriction_conflict_policy(order_after, execution_for_validation, turn_result, newly_enabled)

    # 개별 restriction 반영 후 현재 section의 모든 선택지가 사라지면
    # 다음 턴의 일반 confirm으로 넘기지 않고 즉시 skip 또는 skip 재확인으로 연결한다.
    unavailable_skip = build_unavailable_section_skip(order_after, execution_after)

    if unavailable_skip is not None:
        if unavailable_skip["needs_confirmation"]:
            turn_result["section_skip"] = unavailable_skip
        else:
            committed = commit_section_skip(
                order_after,
                execution_after,
                recommendation_after,
                unavailable_skip,
            )
            order_after = committed["order"]
            execution_after = committed["execution"]
            recommendation_after = committed["recommendation"]
            unavailable_skip["applied"] = True
            turn_result["section_skip"] = unavailable_skip

        turn_result["order_after"] = copy.deepcopy(order_after)
        execution_after["selected"] = execution_after.get("selected", selected)

        return {
            "order": order_after,
            "execution": execution_after,
            "recommendation": recommendation_after,
            "turn_result": turn_result,
        }

    # 변경과 즉시 시작을 같이 요청했을 때는 전체 검증을 통과해야 확정한다.
    if action == "set_order_and_confirm":
        check_sections = ["noodle"] if section == "sauce" else [section]
        missing = missing_requirements(order_after, check_sections)
        has_problem = bool(
            turn_result["invalid_value"]
            or turn_result["blocked_active_or_completed"]
            or turn_result["blocked_past_section"]
            or turn_result["blocked_constraint"]
            or turn_result["kept_physical_conflict"]
        )

        if confirmation_intent is not True:
            confirm_reason = confirmation_block_reason(confirmation_intent)
        elif missing:
            confirm_reason = "missing_required"
        elif has_problem:
            confirm_reason = "partial_or_blocked_change"
        else:
            confirm_reason = "ready"

        turn_result["confirm_validation"] = {
            "requested": True,
            "allowed": (
                confirmation_intent is True
                and not missing
                and not has_problem
            ),
            "missing": missing,
            "reason": confirm_reason,
        }



    turn_result["order_after"] = copy.deepcopy(order_after)
    execution_after["selected"] = selected

    return {
        "order": order_after,
        "execution": execution_after,
        "turn_result": turn_result,
    }

