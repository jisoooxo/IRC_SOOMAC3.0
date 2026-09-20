import copy
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from soomac_irc.order_v6 import EXTRAS, MEATS, NOODLE_TYPES, SAUCES, SECTION_ORDER, VEGGIES, apply_menu_changes, apply_restriction_changes, missing_requirements, order_restriction_conflicts, physical_section_for, strongest_restriction_for, validate_menu_changes, validate_recommended_order, validate_restriction_changes

class ToolCall(TypedDict):
    # call_model_v6가 만든 한 턴의 최소 Tool 계약
    call_id: str
    name: str
    changes: dict

class AgentState(TypedDict, total=False): # total=False로 설정 -> 선택적으로 필드 설정 가능
    # llm_node가 한 턴마다 복사해서 전달하는 상태 ㅇㅇ
    user_text:str
    order: dict
    execution: dict # 실행중
    recommendation: dict
    history: list[dict]
    action_history: list[dict]

    # Tool 해석 결과
    tool_call: ToolCall | None
    parser_status: str # 전달 여부 ok / invalid
    transaction: dict  # 내 코드로 검증한 이번 턴 결과
    policy_reply: str | None # 안전·물리 상태 때문에 Python이 확정한 답변


def _section_items(section: str) -> list[str] | None:
    if section == "veggie":
        return list(VEGGIES)
    if section == "meat":
        return list(MEATS)
    if section == "extra":
        return list(EXTRAS)
    return None


def build_unavailable_section_skip(order: dict, execution: dict) -> dict | None:
    # 현재 section에 선택한 값이 없고 모든 재료가 restriction으로 막혔 때만 skip 후보를 만든다.
    section = execution.get("section")
    items = _section_items(section)

    if items is None or any(item in order["toppings"] for item in items):
        return None

    restrictions = [strongest_restriction_for(order, item) for item in items]

    if any(restriction is None for restriction in restrictions):
        return None

    reasons = [restriction["reason"] for restriction in restrictions]
    needs_confirmation = "dislike" in reasons
    reason = "dislike" if needs_confirmation else next(
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
    # 확인을 받은 section skip만 새 상태 복사본에 원자적으로 반영한다.
    order_after = copy.deepcopy(order)
    execution_after = copy.deepcopy(execution)
    recommendation_after = copy.deepcopy(recommendation)
    section = proposal.get("section")
    reason = proposal.get("reason")
    items = proposal.get("items")
    expected_items = _section_items(section)

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
    active_class = active_task.get("class") if isinstance(active_task, dict) else None
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
    confirmed_sections = recommendation_after.get("confirmed_sections", [])

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


def validate_transaction(state: AgentState) -> dict:
    # 원본 주문은 건드리지 않고 복사본에서만 검사·반영
    # 입력: 모델 Tool + llm_node의 주문·물리 진행 상태
    # 반환: 다음 order/execution, 필요한 분기의 recommendation, 이번 턴 검수표 transaction
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

    transaction = { # 검수용 필드
        "action": action,
        "order_before": copy.deepcopy(order_before),
        "order_after": order_after,
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
            "proposal_confirmed": False, # 추천안을 실제 주문에 반영했는지
            "needs_scope": False,
        },
    }


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

        transaction["invalid_value"].extend(invalid_restrictions)

        restriction_applied, restriction_unchanged = apply_restriction_changes(
            order_after, clean_restrictions
        )

        transaction["restriction_applied"].extend(restriction_applied)
        transaction["restriction_unchanged"].extend(restriction_unchanged)

        newly_enabled = {
            (change["target"], change["reason"])
            for change in restriction_applied
            if change["enabled"]
        }

        # restriction과 충돌하는 기존 주문값은 물리 진행 상태에 따라 제거하거나 유지
        for conflict in order_restriction_conflicts(order_after):
            key = conflict["key"]
            item = conflict["item"]
            restriction = conflict["restriction"]
            restriction_key = (restriction["target"], restriction["reason"])

            # 이전 dislike는 명시적 재선택을 막지 않지만 이번에 추가한 dislike는 적용
            if restriction["reason"] == "dislike" and restriction_key not in newly_enabled:
                continue

            if key == "noodle_type":
                target_section = "noodle"
                locked_classes = set(NOODLE_TYPES)
            elif key == "sauce":
                target_section = "sauce"
                locked_classes = set(SAUCES)
            else:
                target_section = physical_section_for("toppings", item)
                locked_classes = {item}

            locked = (
                active_class in locked_classes
                or bool(completed_classes.intersection(locked_classes))
            )
            past = SECTION_ORDER.index(target_section) < SECTION_ORDER.index(section)

            # 이미 시작·완료됐거나 지나간 재료는 지우지 않고 안내 대상으로 남김
            if locked or past:
                transaction["kept_physical_conflict"].append(conflict)
                continue

            if key == "sauce":
                order_after["sauce"] = None
            elif key == "noodle_type":
                order_after["noodle_type"] = None
            else:
                del order_after["toppings"][item]

            if key in selected:
                selected.remove(key)

            if target_section in confirmed_sections:
                confirmed_sections.remove(target_section)

            transaction["blocked_constraint"].append(conflict)

        execution_after["selected"] = selected
        transaction["order_after"] = copy.deepcopy(order_after)

        # 범위를 묻는 추천이어도 같은 발화의 restriction은 이미 안전하게 반영
        if scope == "ask":
            recommendation_after = {
                "phase": "await_scope",
                "request": request,
                "confirmed_sections": copy.deepcopy(confirmed_sections),
            }
            transaction["recommendation_validation"]["needs_scope"] = True

            return {
                "order": order_after,
                "execution": execution_after,
                "recommendation": recommendation_after,
                "transaction": transaction,
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

        transaction["recommendation_validation"] = {
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
                "transaction": transaction,
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
            "transaction": transaction,
        }

    # 추천 확인 중 confirm_section이 들어오면 추천안을 실제 주문에 반영
    if action == "confirm_section" and recommendation_after.get("phase") == "confirming":
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

        transaction["accepted"].extend(applied)
        transaction["ignored_same_value"].extend(unchanged)
        transaction["recommendation_validation"] = {
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
        transaction["order_after"] = copy.deepcopy(order_after)

        # proposal은 소비했고, 확정된 section 목록만 다음 턴에 유지
        recommendation_after = {
            "phase": "idle",
            "confirmed_sections": confirmed_sections_after,
        }

        return {
            "order": order_after,
            "execution": execution_after,
            "recommendation": recommendation_after,
            "transaction": transaction,
        }

    # 현재 section의 재료 전체를 같은 이유로 거절
    if action == "refuse_section":
        changes = copy.deepcopy(tool_call["changes"])
        section = execution_after["section"]
        reason = changes.get("reason")
        items = changes.get("items")
        section_items = _section_items(section)

        # 일부 재료만 거절했거나 다른 section 재료가 섞이면 전체 skip으로 인정하지 않음
        valid_items = (
            section_items is not None
            and isinstance(items, list)
            and all(isinstance(item, str) for item in items)
            and len(items) == len(section_items)
            and set(items) == set(section_items)
        )

        if not valid_items:
            transaction["invalid_value"].append({
                "path": "items",
                "value": items,
                "reason": "invalid_section_refusal",
            })

            return {
                "order": order_after,
                "execution": execution_after,
                "recommendation": recommendation_after,
                "transaction": transaction,
            }

        restriction_changes = [
            {"target": item, "reason": reason, "enabled": True}
            for item in section_items
        ]
        clean_restrictions, invalid_restrictions = validate_restriction_changes(restriction_changes)

        transaction["invalid_value"].extend(invalid_restrictions)

        # reason까지 정상인 경우에만 restriction과 skip을 적용
        if len(clean_restrictions) != len(section_items):
            return {
                "order": order_after,
                "execution": execution_after,
                "recommendation": recommendation_after,
                "transaction": transaction,
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
            transaction["section_skip"] = section_skip
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
            transaction["restriction_applied"].extend(
                committed["restriction_applied"]
            )
            transaction["restriction_unchanged"].extend(
                committed["restriction_unchanged"]
            )
            transaction["blocked_constraint"].extend(
                committed["blocked_constraint"]
            )
            transaction["kept_physical_conflict"].extend(
                committed["kept_physical_conflict"]
            )
            section_skip["applied"] = True
            transaction["section_skip"] = section_skip

        transaction["order_after"] = copy.deepcopy(order_after)

        return {
            "order": order_after,
            "execution": execution_after,
            "recommendation": recommendation_after,
            "transaction": transaction,
        }


    # 추천 확인은 위에서 이미 처리됐고, 여기서는 일반 section 확정만 검사한다.
    if action == "confirm_section":
        section = execution_after["section"]
        check_sections = ["noodle"] if section == "sauce" else [section]
        missing = missing_requirements(order_after, check_sections)

        transaction["confirm_validation"] = {
            "requested": True,
            "allowed": not missing,
            "missing": missing,
            "reason": "ready" if not missing else "missing_required",
        }

        return {
            "order": order_after,
            "execution": execution_after,
            "recommendation": recommendation_after,
            "transaction": transaction,
        }

    if action == "cancel_order":
        # 전체 주문 취소는 지원하지 않고 현재 주문·물리 상태를 그대로 유지함
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
            cancel_reason = "robot_already_started"
        elif has_order_value:
            cancel_reason = "order_already_selected"
        else:
            cancel_reason = "empty_order"

        transaction["cancel_validation"] = {
            "requested": True,
            "blocked": True,
            "needs_confirmation": False,
            "reason": cancel_reason,
        }

        return {
            "order": order_after,
            "execution": execution_after,
            "recommendation": recommendation_after,
            "transaction": transaction,
        }


    # 나머지 Tool은 각 단계에서 별도로 처리한다.
    if action not in ("set_order", "set_order_and_confirm"):
        return {
            "order": order_after,
            "execution": execution_after,
            "recommendation": recommendation_after,
            "transaction": transaction,
        }

    changes = copy.deepcopy(tool_call["changes"])

    # restriction은 다음 단계에서 별도로 처리한다.
    menu_changes = {
        field: changes[field]
        for field in ("sauce", "noodle_type", "noodle_portion", "toppings")
        if field in changes
    }

    clean, invalid = validate_menu_changes(menu_changes)
    transaction["invalid_value"].extend(invalid)

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

    editable = {}
    # editable에 들어간 값만 마지막 apply 단계까지 살아남음

    # scalar와 topping을 같은 편집 가능성 검사에 넣는다.
    entries = [
        (field, None, clean[field])
        for field in ("sauce", "noodle_type", "noodle_portion")
        if field in clean
    ]

    entries.extend(
        ("toppings", item, amount)
        for item, amount in clean.get("toppings", {}).items()
    )

    for field, item, value in entries:
        key = f"toppings.{item}" if field == "toppings" else field
        target_section = physical_section_for(field, item)

        # 면 양도 실제 면 작업과 함께 잠긴다.
        if field in ("noodle_type", "noodle_portion"):
            locked_classes = set(NOODLE_TYPES)
        elif field == "sauce":
            locked_classes = set(SAUCES)
        else:
            locked_classes = {item}

        locked = (
            active_class in locked_classes
            or bool(completed_classes.intersection(locked_classes))
        )

        blocked = {"path": key, "value": value}

        if locked:
            transaction["blocked_active_or_completed"].append(blocked)
            continue

        # 현재보다 앞선 물리 section은 이미 지나갔으므로 수정하지 않는다.
        if SECTION_ORDER.index(target_section) < SECTION_ORDER.index(section):
            transaction["blocked_past_section"].append(blocked)
            continue

        if field == "toppings":
            editable.setdefault("toppings", {})[item] = value
        else:
            editable[field] = value

    applied, unchanged = apply_menu_changes(order_after, editable)

    transaction["accepted"].extend(applied)
    transaction["ignored_same_value"].extend(unchanged)

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

    transaction["invalid_value"].extend(invalid_restrictions)

    restriction_applied, restriction_unchanged = (
        apply_restriction_changes(order_after, clean_restrictions)
    )

    transaction["restriction_applied"].extend(restriction_applied)
    transaction["restriction_unchanged"].extend(restriction_unchanged)

    newly_enabled = {
        (change["target"], change["reason"])
        for change in restriction_applied
        if change["enabled"]
    }

    # 새 restriction과 현재 주문의 충돌을 검사한다.
    for conflict in order_restriction_conflicts(order_after):
        key = conflict["key"]
        item = conflict["item"]
        restriction = conflict["restriction"]

        # 기존 dislike는 소프트 제약이다.
        # 사용자가 해당 재료를 직접 다시 선택하면 허용한다.
        restriction_key = (
            restriction["target"],
            restriction["reason"],
        )

        if (
            restriction["reason"] == "dislike"
            and restriction_key not in newly_enabled
        ):
            continue

        if key == "noodle_type":
            target_section = "noodle"
            locked_classes = set(NOODLE_TYPES)

        elif key == "sauce":
            target_section = "sauce"
            locked_classes = set(SAUCES)

        else:
            target_section = physical_section_for("toppings", item)
            locked_classes = {item}

        locked = (
            active_class in locked_classes
            or bool(completed_classes.intersection(locked_classes))
        )

        past = (
            SECTION_ORDER.index(target_section)
            < SECTION_ORDER.index(section)
        )

        # 이미 물리적으로 시작됐거나 지나간 재료는 order에서도 지우지 않는다.
        if locked or past:
            transaction["kept_physical_conflict"].append(conflict)
            continue

        if key == "sauce":
            order_after["sauce"] = None

        elif key == "noodle_type":
            order_after["noodle_type"] = None

        else:
            del order_after["toppings"][item]

        if key in selected:
            selected.remove(key)

        # 적용 직후 restriction으로 빠졌다면 최종 accepted로 표시하지 않는다.
        if key in transaction["accepted"]:
            transaction["accepted"].remove(key)

        transaction["blocked_constraint"].append(conflict)

    # 개별 restriction 반영 후 현재 section의 모든 선택지가 사라지면
    # 다음 턴의 일반 confirm으로 넘기지 않고 즉시 skip 또는 skip 재확인으로 연결한다.
    unavailable_skip = build_unavailable_section_skip(order_after, execution_after)

    if unavailable_skip is not None:
        if unavailable_skip["needs_confirmation"]:
            transaction["section_skip"] = unavailable_skip
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
            transaction["section_skip"] = unavailable_skip

        transaction["order_after"] = copy.deepcopy(order_after)
        execution_after["selected"] = execution_after.get("selected", selected)

        return {
            "order": order_after,
            "execution": execution_after,
            "recommendation": recommendation_after,
            "transaction": transaction,
        }

    # 변경과 즉시 시작을 같이 요청했을 때는 전체 검증을 통과해야 확정한다.
    if action == "set_order_and_confirm":
        check_sections = ["noodle"] if section == "sauce" else [section]
        missing = missing_requirements(order_after, check_sections)
        has_problem = bool(
            transaction["invalid_value"]
            or transaction["blocked_active_or_completed"]
            or transaction["blocked_past_section"]
            or transaction["blocked_constraint"]
            or transaction["kept_physical_conflict"]
        )

        if missing:
            confirm_reason = "missing_required"
        elif has_problem:
            confirm_reason = "partial_or_blocked_change"
        else:
            confirm_reason = "ready"

        transaction["confirm_validation"] = {
            "requested": True,
            "allowed": not missing and not has_problem,
            "missing": missing,
            "reason": confirm_reason,
        }



    transaction["order_after"] = copy.deepcopy(order_after)
    execution_after["selected"] = selected

    return {
        "order": order_after,
        "execution": execution_after,
        "transaction": transaction,
    }

def build_selection_next_step(order: dict, execution: dict) -> str | None:
    # 자유응답에 흐름 결정을 맡기지 않고 현재 section의 바로 다음 행동을 확정한다.
    # 로봇 작업 중에는 사용자 입력을 받지 않으므로 선택 질문을 덧붙이지 않는다.
    if execution.get("active_task") is not None or execution.get("task_queue"):
        return None

    section = execution.get("section")

    if section == "noodle":
        missing = missing_requirements(order, ["noodle"]).get("noodle", [])
        field_labels = {
            "sauce": "소스",
            "noodle_type": "면 종류",
            "noodle_portion": "면 양",
        }

        if missing:
            missing_text = ", ".join(field_labels[field] for field in missing)
            return f"다음으로 선택할 항목은 {missing_text}예요. 원하는 내용을 말씀해 주세요."

        return "면 선택이 끝났어요. 이대로 면 담기를 시작하려면 진행해 달라고 말씀해 주세요."

    section_options = {
        "veggie": ("야채", VEGGIES),
        "meat": ("육류", MEATS),
        "extra": ("추가 재료", EXTRAS),
    }

    if section in section_options:
        section_label, options = section_options[section]
        selected_items = [item for item in options if item in order["toppings"]]
        available_items = [
            item for item in options
            if strongest_restriction_for(order, item) is None
        ]
        remaining_items = [
            item for item in available_items
            if item not in selected_items
        ]

        if not selected_items:
            if not available_items:
                return f"현재 선택 가능한 {section_label}가 없어요. {section_label} 단계를 제외할지 말씀해 주세요."

            if len(available_items) == 1:
                return f"{section_label}로는 {available_items[0]}가 남아 있어요. 어느 정도 양으로 넣을까요?"

            options_text = "나 ".join(available_items)
            return f"다음은 {section_label} 선택이에요. {options_text} 중 원하는 재료와 양을 말씀해 주세요."

        amount_labels = {"low": "적게", "normal": "보통", "high": "많이"}
        selected_text = ", ".join(
            f"{item} {amount_labels[order['toppings'][item]]}"
            for item in selected_items
        )

        if remaining_items:
            remaining_text = "나 ".join(remaining_items)
            return f"현재 {section_label} 선택은 {selected_text}예요. {remaining_text}도 추가하거나, 이대로 담기를 진행할까요?"

        return f"현재 {section_label} 선택은 {selected_text}예요. 이대로 {section_label} 담기를 진행할까요?"

    if section == "lid":
        return "다음은 뚜껑 닫기예요. 진행해 달라고 말씀해 주세요."

    if section == "sauce":
        if order.get("sauce") is None:
            return "다음으로 소스를 선택해 주세요."

        return "이대로 소스 담기를 시작하려면 진행해 달라고 말씀해 주세요."

    return None


def build_policy_reply(state: AgentState) -> dict:
    # 같은 transaction에서 동일한 재료·이유를 여러 번 안내하지 않는다.
    # 입력: validate_transaction이 확정한 사실 / 반환: Python 고정 문장 또는 None
    # 모델 원문을 다시 보지 않아서 안전·물리 차단 결과를 말로 뒤집지 못하게 함
    transaction = state["transaction"]
    safety_items = []
    dietary_items = []
    dislike_items = []
    seen = set()

    for conflict in transaction["kept_physical_conflict"]:
        item = conflict["item"]
        reason = conflict["restriction"]["reason"]
        notice_key = (item, reason)

        if notice_key in seen:
            continue

        seen.add(notice_key)

        if reason in ("allergy", "cannot_eat"):
            safety_items.append(item)
        elif reason == "dietary_rule":
            dietary_items.append(item)
        elif reason == "dislike":
            dislike_items.append(item)
        else:
            raise ValueError(f"restriction reason 이상함 : {reason}")

    messages = []

    if safety_items:
        items = ", ".join(safety_items)
        messages.append(f"{items} 재료는 이미 담기 시작해서 철회할 수 없습니다. 섭취에 유의하시기 바랍니다.")

    if dietary_items:
        items = ", ".join(dietary_items)
        messages.append(f"{items} 재료는 이미 담기 시작해서 제외할 수 없습니다. 이후 재료부터 해당 식단 조건을 적용하겠습니다.")

    if dislike_items:
        items = ", ".join(dislike_items)
        messages.append(f"{items} 재료는 이미 담기 시작해서 취소할 수 없어요.")

    # 전체 취소는 모델을 호출하지 않고 가능 여부에 맞는 문장으로 답한다.
    cancel = transaction["cancel_validation"]

    if cancel["requested"]:
        if cancel["reason"] == "empty_order":
            messages.append("아직 선택된 주문이 없어요.")
        elif cancel["reason"] == "order_already_selected":
            messages.append("전체 주문 취소는 지원하지 않아요. 바꾸고 싶은 메뉴를 말씀해 주세요.")
        elif cancel["reason"] == "robot_already_started":
            messages.append("이미 재료를 담기 시작해서 전체 주문을 취소할 수 없어요. 아직 담지 않은 재료는 변경할 수 있어요.")
        else:
            raise ValueError(f"전체 취소 reason 이상함 : {cancel['reason']}")

    # 필수값 누락이나 일부 차단이 있으면 로봇을 시작하지 않았다고 명확히 안내한다.
    confirm = transaction["confirm_validation"]

    if confirm["requested"] and not confirm["allowed"]:
        missing_values = []

        for values in confirm["missing"].values():
            missing_values.extend(values)

        field_labels = {
            "sauce": "소스",
            "noodle_type": "면 종류",
            "noodle_portion": "어느 정도 양을 드실지",
        }
        missing_labels = [field_labels.get(value, value) for value in missing_values]

        if missing_labels:
            messages.append(f"아직 {', '.join(missing_labels)} 선택이 필요해서 작업을 시작하지 않았어요.")
        else:
            messages.append("일부 요청을 반영할 수 없어 아직 작업을 시작하지 않았어요.")

    # Tool JSON을 읽지 못한 경우에도 Reply 모델이 내용을 추측하지 않는다.
    if transaction["action"] == "error" and not messages:
        messages.append("요청을 정확히 이해하지 못했어요. 다시 말씀해 주세요.")

    # 일반 주문 변경 결과도 검증된 transaction만 보고 안내한다.
    action = transaction["action"]
    confirm_failed = (
        action == "set_order_and_confirm"
        and transaction["confirm_validation"]["requested"]
        and not transaction["confirm_validation"]["allowed"]
    )

    if action == "set_order" or confirm_failed:
        accepted = transaction["accepted"]
        invalid = transaction["invalid_value"]
        blocked_active = transaction["blocked_active_or_completed"]
        blocked_past = transaction["blocked_past_section"]
        blocked_constraint = transaction["blocked_constraint"]
        restriction_applied = transaction["restriction_applied"]

        if accepted:
            if invalid or blocked_active or blocked_past or blocked_constraint:
                messages.append("가능한 주문 변경만 반영했어요.")
            else:
                messages.append("요청한 주문 변경을 반영했어요.")

        if blocked_constraint:
            if restriction_applied:
                messages.append("식이 조건을 반영했고, 아직 담지 않은 충돌 재료는 주문에서 제외했어요.")
            else:
                messages.append("식이 조건과 충돌한 항목은 주문에 반영하지 않았어요.")

        elif restriction_applied and not transaction["kept_physical_conflict"]:
            messages.append("요청한 식이 조건을 반영했어요.")

        if blocked_active:
            messages.append("이미 담기 시작한 항목은 변경할 수 없어요.")

        if blocked_past:
            messages.append("이미 작업이 끝난 단계의 항목은 변경할 수 없어요.")

        if invalid:
            messages.append("메뉴에 없거나 형식이 맞지 않는 요청은 반영하지 않았어요.")

        if not messages and transaction["ignored_same_value"]:
            messages.append("이미 같은 내용으로 선택되어 있어요.")

    # 추천 범위 질문·추천안 생성·추천 확정 결과를 검증값으로 안내한다.
    recommendation = transaction["recommendation_validation"]

    if action == "recommend_order":
        if recommendation["needs_scope"]:
            messages.append("현재 단계만 추천할까요, 아니면 남은 주문 전체를 추천할까요?")

        elif recommendation["proposal_created"]:
            amount_labels = {"low": "적게", "normal": "보통", "high": "많이"}
            order_before = transaction["order_before"]
            clean = recommendation["clean"]
            selected_parts = []
            recommended_parts = []

            if order_before["sauce"]:
                selected_parts.append(f"{order_before['sauce']} 소스")

            if order_before["noodle_type"]:
                selected_parts.append(order_before["noodle_type"])

            if order_before["noodle_portion"]:
                selected_parts.append(f"면 양 {amount_labels[order_before['noodle_portion']]}")

            for item, amount in order_before["toppings"].items():
                selected_parts.append(f"{item} {amount_labels[amount]}")

            if clean.get("sauce"):
                recommended_parts.append(f"{clean['sauce']} 소스")

            if clean.get("noodle_type"):
                recommended_parts.append(clean["noodle_type"])

            if clean.get("noodle_portion"):
                recommended_parts.append(f"면 양 {amount_labels[clean['noodle_portion']]}")

            for item, amount in clean.get("toppings", {}).items():
                recommended_parts.append(f"{item} {amount_labels[amount]}")

            selected_text = ", ".join(selected_parts)
            recommended_text = ", ".join(recommended_parts)

            if selected_text:
                messages.append(f"이미 고르신 메뉴는 {selected_text}입니다. 여기에 {recommended_text} 조합을 추천해드려요. 이대로 할까요?")
            else:
                messages.append(f"{recommended_text} 조합을 추천해드려요. 이대로 할까요?")

        elif recommendation["blocked"]:
            messages.append("현재 제한 조건을 지키면 추천할 수 있는 재료가 없어요. 추천 범위나 제외 조건을 바꿔 말씀해 주세요.")

        elif recommendation["dropped"] and all(item.get("reason") == "already_selected" for item in recommendation["dropped"]):
            messages.append("요청한 추천 범위의 메뉴는 이미 모두 선택되어 있어요.")

        else:
            messages.append("추천 범위에 맞는 새 조합을 만들지 못했어요. 추천할 단계나 원하는 맛을 다시 말씀해 주세요.")

    elif action == "confirm_section" and recommendation["proposal_confirmed"]:
        messages.append("추천한 내용을 주문에 반영했어요.")

    elif action == "confirm_section" and (recommendation["dropped"] or recommendation["blocked"]):
        messages.append("추천안이 현재 주문이나 제한 조건과 충돌해서 반영하지 않았어요.")

        # section 전체 제외는 dislike 재확인과 안전 제약 즉시 적용을 구분한다.
    if action == "refuse_section":
        section_skip = transaction["section_skip"]

        if section_skip is None:
            messages.append("현재 단계 전체 제외 요청을 정확히 이해하지 못했어요. 원하는 재료를 다시 말씀해 주세요.")
        elif section_skip["needs_confirmation"]:
            messages.append("현재 단계 재료를 모두 제외하고 다음 단계로 넘어갈까요?")
        elif section_skip["applied"]:
            messages.append("현재 단계 재료를 제외 요청에 반영했어요.")


    return {
        "policy_reply": " ".join(messages) if messages else None,
    }

def build_graph(call_model):
    # 모델은 자연어를 Tool JSON으로 해석만 한다.
    # 흐름: 모델 해석 → Python 검증·반영 → 검증 결과로 답변 작성
    # compile된 graph는 node 결과를 합친 state를 반환하고 ROS 상태 반영은 llm_node가 담당함
    def interpret_tool(state: AgentState) -> dict:
        # call_model이 None이면 parser_status만 invalid로 넘겨 다음 node에서 안전하게 처리
        tool_call = call_model(state, state["user_text"])

        return {
            "tool_call": tool_call,
            "parser_status": "ok" if tool_call is not None else "invalid",
        }

    graph = StateGraph(AgentState)

    graph.add_node("interpret_tool", interpret_tool)
    graph.add_node("validate_transaction", validate_transaction)
    graph.add_node("build_policy_reply", build_policy_reply)

    graph.add_edge(START, "interpret_tool")
    graph.add_edge("interpret_tool", "validate_transaction")
    graph.add_edge("validate_transaction", "build_policy_reply")
    graph.add_edge("build_policy_reply", END)

    return graph.compile()
