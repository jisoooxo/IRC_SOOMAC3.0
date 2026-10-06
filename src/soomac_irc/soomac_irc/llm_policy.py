"""자연어 모델 출력과 실제 로봇 상태 사이의 마지막 안전 경계이다.

사용자가 무슨 뜻으로 말했는지는 Decision·Recommendation 모델이 판단한다.
이 파일은 모델이 만든 후보를 현재 주문에 반영할 수 있는지, 실제 ROS 작업으로
실행할 수 있는지만 메뉴 domain·물리 단계·섭취 제한을 기준으로 검사한다.
"""
import copy

from soomac_irc.agent_contract import empty_order_patch
from soomac_irc.domain import (
    AMOUNTS,
    NOODLE_TYPES,
    RESTRICTION_CATEGORY_ITEMS,
    RESTRICTION_PRIORITY,
    RESTRICTION_REASONS,
    SAUCES,
    SECTION_ORDER,
    TOPPINGS,
)

SAFETY_RESTRICTION_REASONS = ("allergy", "cannot_eat", "dietary_rule")
SUPPORTED_RESTRICTION_TARGETS = set(SAUCES) | set(NOODLE_TYPES) | set(TOPPINGS) | set(RESTRICTION_CATEGORY_ITEMS)


#################### section과 실제 주문 field 연결 ####################

def allowed_order_fields(section: str) -> list[str]:
    # 현재 section에서 수정 가능한 주문 key만 반환한다.
    # 이 목록은 past/future 판정과 현재 section 변경 감지에서 같이 사용한다.
    if section == "noodle":
        return ["sauce", "noodle_type", "noodle_portion"]
    if section == "veggie":
        return ["toppings.양파", "toppings.버섯"]
    if section == "meat":
        return ["toppings.소시지", "toppings.게살"]
    if section == "extra":
        return ["toppings.치즈", "toppings.페퍼론치노"]
    return []


def physical_section_for_order_key(key: str) -> str | None:
    # 주문 key 하나가 로봇의 어느 물리 section에 속하는지 역으로 찾는다.
    if key in ("noodle_type", "noodle_portion"):
        return "noodle"
    if key == "sauce":
        return "sauce"
    if key in ("toppings.양파", "toppings.버섯"):
        return "veggie"
    if key in ("toppings.소시지", "toppings.게살"):
        return "meat"
    if key in ("toppings.치즈", "toppings.페퍼론치노"):
        return "extra"
    return None


def section_execution_items(order: dict, section: str) -> list[dict]:
    # 확정 주문을 현재 section에서 실제로 담을 작업 목록으로 바꾼다.
    # 아직 필수값이 부족한 면 section은 빈 목록을 반환해 실행하지 않는다.
    if section == "noodle":
        if order["noodle_type"] is None or order["noodle_portion"] is None:
            return []
        return [{"item": order["noodle_type"], "amount": order["noodle_portion"]}]
    if section == "sauce":
        return [] if order["sauce"] is None else [{"item": order["sauce"], "amount": None}]
    if section == "lid":
        return [{"item": "뚜껑", "amount": None}]

    items = []
    for key in allowed_order_fields(section):
        item = key.split(".", 1)[1]
        amount = order["toppings"].get(item)
        if amount is not None and amount != "none":
            items.append({"item": item, "amount": amount})
    return items


def protected_order_keys(robot_state: dict) -> list[str]:
    """이미 지나갔거나 실행 중·완료되어 되돌릴 수 없는 주문 key를 만든다."""
    current_section = robot_state.get("section")
    if current_section not in SECTION_ORDER:
        return []

    current_index = SECTION_ORDER.index(current_section)
    keys = [
        "noodle_type", "noodle_portion",
        "toppings.양파", "toppings.버섯",
        "toppings.소시지", "toppings.게살",
        "toppings.치즈", "toppings.페퍼론치노",
        "sauce",
    ]
    protected = []
    for key in keys:
        section = physical_section_for_order_key(key)
        if section in SECTION_ORDER and SECTION_ORDER.index(section) < current_index:
            protected.append(key)

    # section 순서만으로 놓칠 수 있는 현재 작업과 완료 작업도 보호 목록에 합친다.
    tasks = list(robot_state.get("completed_tasks") or [])
    if robot_state.get("active_task") is not None:
        tasks.append(robot_state["active_task"])
    for task in tasks:
        item = task.get("class")
        if item in NOODLE_TYPES:
            for key in ("noodle_type", "noodle_portion"):
                if key not in protected:
                    protected.append(key)
        elif item in TOPPINGS:
            key = f"toppings.{item}"
            if key not in protected:
                protected.append(key)
        elif item in SAUCES and "sauce" not in protected:
            protected.append("sauce")
    return protected


def compact_order_patch(patch: dict) -> dict:
    # Response나 pending에 넘길 때 의미 없는 null·빈 toppings를 제거한다.
    compact = {
        field: copy.deepcopy(patch[field])
        for field in ("sauce", "noodle_type", "noodle_portion")
        if patch.get(field) is not None
    }
    if patch.get("toppings"):
        compact["toppings"] = copy.deepcopy(patch["toppings"])
    return compact


def changed_order_keys_from_patch(patch: dict) -> list[str]:
    # patch에 실제로 들어온 값만 평평한 key 목록으로 바꾼다.
    # restriction 충돌이 이번 사용자 요청 때문인지 구분할 때 사용한다.
    changed = []
    for field in ("sauce", "noodle_type", "noodle_portion"):
        if patch.get(field) is not None:
            changed.append(field)
    for topping in (patch.get("toppings") or {}):
        changed.append(f"toppings.{topping}")
    return changed


#################### 모델 주문 후보 검증과 적용 ####################

def validate_order_patch(patch: dict, robot_state: dict) -> dict:
    """모델 주문 후보를 반영 가능 값과 거절 사유로 나눈다.

    사용자 문장을 다시 읽거나 의미를 추측하지 않는다. 지원 메뉴인지, 값이 올바른지,
    이미 되돌릴 수 없는 물리 단계인지만 순서대로 검사한다.
    """
    accepted = empty_order_patch()
    unsupported = []
    invalid = []
    protected = []
    protected_keys = set(protected_order_keys(robot_state))

    # scalar 세 필드는 같은 순서로 검사하되, 이유를 잃지 않도록 별도 목록에 기록한다.
    sauce = patch.get("sauce")
    if sauce is not None:
        if sauce == "none":
            if "sauce" in protected_keys:
                protected.append({"field": "sauce", "value": sauce})
            else:
                accepted["sauce"] = sauce
        elif sauce not in SAUCES:
            unsupported.append({"field": "sauce", "value": sauce})
        elif "sauce" in protected_keys:
            protected.append({"field": "sauce", "value": sauce})
        else:
            accepted["sauce"] = sauce

    noodle_type = patch.get("noodle_type")
    if noodle_type is not None:
        if noodle_type == "none":
            if "noodle_type" in protected_keys:
                protected.append({"field": "noodle_type", "value": noodle_type})
            else:
                accepted["noodle_type"] = noodle_type
        elif noodle_type not in NOODLE_TYPES:
            unsupported.append({"field": "noodle_type", "value": noodle_type})
        elif "noodle_type" in protected_keys:
            protected.append({"field": "noodle_type", "value": noodle_type})
        else:
            accepted["noodle_type"] = noodle_type

    noodle_portion = patch.get("noodle_portion")
    if noodle_portion is not None:
        if noodle_portion == "none":
            if "noodle_portion" in protected_keys:
                protected.append({"field": "noodle_portion", "value": noodle_portion})
            else:
                accepted["noodle_portion"] = noodle_portion
        elif noodle_portion not in AMOUNTS:
            invalid.append({"field": "noodle_portion", "value": noodle_portion})
        elif "noodle_portion" in protected_keys:
            protected.append({"field": "noodle_portion", "value": noodle_portion})
        else:
            accepted["noodle_portion"] = noodle_portion

    # toppings는 메뉴명과 양을 모두 검사한다. 하나가 실패해도 다른 정상 값은 보존한다.
    for item, amount in (patch.get("toppings") or {}).items():
        field = f"toppings.{item}"
        if item not in TOPPINGS:
            unsupported.append({"field": field, "value": amount, "item": item})
        elif amount not in (*AMOUNTS, "none"):
            invalid.append({"field": field, "value": amount})
        elif field in protected_keys:
            protected.append({"field": field, "value": amount, "item": item})
        else:
            accepted["toppings"][item] = amount

    return {
        "accepted_patch": accepted,
        "unsupported": unsupported,
        "invalid": invalid,
        "protected": protected,
    }


def apply_order_patch(order: dict, patch: dict) -> dict:
    # 검증을 통과한 patch만 주문 복사본에 반영한다.
    # none은 실제 값으로 저장하지 않고 scalar는 None, topping은 key 삭제로 적용한다.
    updated = copy.deepcopy(order)
    for field in ("sauce", "noodle_type", "noodle_portion"):
        if patch.get(field) is not None:
            updated[field] = None if patch[field] == "none" else patch[field]

    # 면 종류를 지운 턴에는 면 양도 의미가 없으므로 최종 주문의 일관성을 함께 맞춘다.
    if patch.get("noodle_type") == "none":
        updated["noodle_portion"] = None
    for item, amount in (patch.get("toppings") or {}).items():
        if amount == "none":
            updated["toppings"].pop(item, None)
        else:
            updated["toppings"][item] = amount
    return updated


#################### 섭취 제한과 취향 변경 ####################

def validate_restriction_options(order: dict, options: list[dict]) -> dict:
    """구조화된 제한 변경을 검증한다. 사용자 문장의 의미는 다시 판단하지 않는다."""
    existing = {(r.get("target"), r.get("reason")) for r in order.get("restrictions", [])}
    accepted = []
    unsupported = []
    invalid = []
    unchanged = []
    newly_added = []
    newly_removed = []

    # 같은 턴에 add/remove가 연속으로 나와도 앞선 결과를 반영하도록 existing을 즉시 갱신한다.
    for option in options:
        if not isinstance(option, dict):
            invalid.append({"value": copy.deepcopy(option), "reason": "not_object"})
            continue
        target = option.get("target")
        reason = option.get("reason")
        action = option.get("action")
        if target not in SUPPORTED_RESTRICTION_TARGETS:
            unsupported.append({"target": target, "reason": reason, "action": action})
            continue
        if reason not in RESTRICTION_REASONS or action not in ("add", "remove"):
            invalid.append({"target": target, "reason": reason, "action": action})
            continue

        key = (target, reason)
        if action == "add" and key in existing:
            unchanged.append({"target": target, "reason": reason, "action": action})
            continue
        if action == "remove" and key not in existing:
            unchanged.append({"target": target, "reason": reason, "action": action})
            continue

        clean = {"target": target, "reason": reason, "action": action}
        accepted.append(clean)
        if action == "add":
            newly_added.append({"target": target, "reason": reason})
            existing.add(key)
        else:
            newly_removed.append({"target": target, "reason": reason})
            existing.discard(key)

    return {
        "accepted": accepted,
        "unsupported": unsupported,
        "invalid": invalid,
        "unchanged": unchanged,
        "newly_added": newly_added,
        "newly_removed": newly_removed,
    }


def apply_restriction_options(order: dict, options: list[dict]) -> dict:
    # 검증을 통과한 제한만 주문 복사본에 추가하거나 제거한다.
    updated = copy.deepcopy(order)
    restrictions = updated.setdefault("restrictions", [])
    for option in options:
        entry = {"target": option["target"], "reason": option["reason"]}
        if option["action"] == "add":
            if entry not in restrictions:
                restrictions.append(entry)
        else:
            restrictions[:] = [
                r for r in restrictions
                if not (r.get("target") == entry["target"] and r.get("reason") == entry["reason"])
            ]
    return updated


def apply_preference_options(preferences: list[dict], options: list[dict]) -> list[dict]:
    # 자유 취향은 메뉴 restriction과 분리해서 단순 add/remove 목록으로 관리한다.
    updated = copy.deepcopy(preferences)
    for option in options:
        entry = {"value": option["value"]}
        if option["action"] == "add":
            if entry not in updated:
                updated.append(entry)
        else:
            updated[:] = [p for p in updated if p.get("value") != option["value"]]
    return updated


def restriction_priority(reason: str) -> int:
    # 여러 제한이 같은 재료를 막으면 allergy처럼 우선순위가 높은 이유를 사용자에게 보여준다.
    try:
        return RESTRICTION_PRIORITY.index(reason)
    except ValueError:
        return len(RESTRICTION_PRIORITY)


def strongest_restriction_for_item(order: dict, item: str) -> dict | None:
    # 재료 하나에 걸린 제한 중 가장 강한 제한 하나를 찾는다.
    strongest = None
    for restriction in order.get("restrictions", []):
        blocked = RESTRICTION_CATEGORY_ITEMS.get(restriction["target"], (restriction["target"],))
        if item not in blocked:
            continue
        if strongest is None or restriction_priority(restriction["reason"]) < restriction_priority(strongest["reason"]):
            strongest = restriction
    return copy.deepcopy(strongest)


def find_restriction_conflicts(order: dict) -> list[dict]:
    # 현재 선택된 메뉴 전체를 훑어 활성 restriction과 충돌하는 주문 key를 만든다.
    selected = []
    if order.get("sauce") is not None:
        selected.append(("sauce", order["sauce"]))
    if order.get("noodle_type") is not None:
        selected.append(("noodle_type", order["noodle_type"]))
    for item in order.get("toppings", {}):
        selected.append((f"toppings.{item}", item))

    conflicts = []
    for key, item in selected:
        restriction = strongest_restriction_for_item(order, item)
        if restriction is not None:
            conflicts.append({"key": key, "item": item, "restriction": restriction})
    return conflicts


def _remove_order_key(order: dict, key: str) -> None:
    # 충돌한 주문 key 하나만 제거한다. 면 종류를 지우면 양도 함께 의미가 없어지므로 같이 지운다.
    if key == "sauce":
        order["sauce"] = None
    elif key == "noodle_type":
        order["noodle_type"] = None
        order["noodle_portion"] = None
    elif key == "noodle_portion":
        order["noodle_portion"] = None
    elif key.startswith("toppings."):
        order["toppings"].pop(key.split(".", 1)[1], None)


def _selected_order_items(order: dict) -> list[tuple[str, str]]:
    # scalar와 toppings를 같은 (주문 key, 실제 메뉴) 모양으로 평평하게 만든다.
    selected = []
    if order.get("sauce") is not None:
        selected.append(("sauce", order["sauce"]))
    if order.get("noodle_type") is not None:
        selected.append(("noodle_type", order["noodle_type"]))
    for item in order.get("toppings", {}):
        selected.append((f"toppings.{item}", item))
    return selected


def enforce_restrictions(
    order: dict,
    *,
    newly_added: list[dict],
    changed_order_keys: list[str],
    robot_state: dict,
) -> dict:
    """확정 주문에 restriction 우선순위를 적용한다.

    - allergy·cannot_eat·dietary_rule은 명시적으로 해제하기 전까지 항상 주문보다 우선한다.
    - 이번 턴에 새로 추가한 제한은 dislike까지 포함해 아직 수정 가능한 선택을 제거한다.
    - 과거 dislike는 사용자가 나중에 직접 다시 주문하면 덮어쓸 수 있다.
    - 이미 실행 중이거나 완료된 물리 작업은 restriction이 생겨도 주문 기록을 되돌리지 않는다.
    """
    updated = copy.deepcopy(order)
    snapshot = copy.deepcopy(order)
    protected = set(protected_order_keys(robot_state))
    changed = set(changed_order_keys)

    blocked = []
    physical_conflicts = []
    requested_conflicts = []
    seen = set()

    # 충돌 하나를 처리하는 공통 경로이다. 같은 충돌은 한 번만 기록한다.
    def enforce(conflict: dict):
        marker = (
            conflict["key"],
            conflict["restriction"]["target"],
            conflict["restriction"]["reason"],
        )
        if marker in seen:
            return
        seen.add(marker)
        entry = copy.deepcopy(conflict)
        if conflict["key"] in protected:
            physical_conflicts.append(entry)
            return
        _remove_order_key(updated, conflict["key"])
        blocked.append(entry)
        if conflict["key"] in changed:
            requested_conflicts.append(entry)

    # 새로 선언된 제한은 기존 제한과 우선순위가 같아도 이번 턴의 편집 가능한 선택보다 우선한다.
    selected = _selected_order_items(snapshot)
    for restriction in newly_added:
        blocked_items = RESTRICTION_CATEGORY_ITEMS.get(restriction["target"], (restriction["target"],))
        for key, item in selected:
            if item in blocked_items:
                enforce({"key": key, "item": item, "restriction": copy.deepcopy(restriction)})

    # 기존 안전 제한은 계속 강제한다. 기존 dislike는 사용자의 새 직접 주문을 허용하기 위해 제외한다.
    for conflict in find_restriction_conflicts(snapshot):
        if conflict["restriction"]["reason"] in SAFETY_RESTRICTION_REASONS:
            enforce(conflict)

    return {
        "order": updated,
        "blocked": blocked,
        "requested_conflicts": requested_conflicts,
        "physical_conflicts": physical_conflicts,
    }


def missing_current_section(order: dict, section: str) -> list[str]:
    # 필수 선택이 있는 noodle section만 누락값을 검사한다. 나머지 section은 빈 선택으로 skip 가능하다.
    if section != "noodle":
        return []
    return [field for field in ("noodle_type", "noodle_portion", "sauce") if order.get(field) is None]


#################### Response와 기록에 넘길 변경 사실 계산 ####################

def build_applied_changes(previous_session: dict, current_session: dict) -> dict:
    # 이전 session과 현재 session의 확정값 차이만 계산한다.
    previous_order = previous_session["order"]
    current_order = current_session["order"]
    order_changes = {}
    for field in ("sauce", "noodle_type", "noodle_portion"):
        if previous_order.get(field) != current_order.get(field):
            order_changes[field] = current_order.get(field)
    for item in TOPPINGS:
        before = previous_order.get("toppings", {}).get(item)
        after = current_order.get("toppings", {}).get(item)
        if before != after:
            order_changes[f"toppings.{item}"] = after

    restriction_added = [copy.deepcopy(r) for r in current_order.get("restrictions", []) if r not in previous_order.get("restrictions", [])]
    restriction_removed = [copy.deepcopy(r) for r in previous_order.get("restrictions", []) if r not in current_order.get("restrictions", [])]
    preference_added = [copy.deepcopy(p) for p in current_session.get("preferences", []) if p not in previous_session.get("preferences", [])]
    preference_removed = [copy.deepcopy(p) for p in previous_session.get("preferences", []) if p not in current_session.get("preferences", [])]
    return {
        "order_changes": order_changes,
        "restriction_added": restriction_added,
        "restriction_removed": restriction_removed,
        "preference_added": preference_added,
        "preference_removed": preference_removed,
    }


def build_future_changes(applied_changes: dict, robot_state: dict) -> list[dict]:
    # 이번 변경 중 현재 section보다 뒤에서 실행할 값만 따로 표시한다.
    current_section = robot_state.get("section")
    if current_section not in SECTION_ORDER:
        return []
    current_index = SECTION_ORDER.index(current_section)
    result = []
    for field, value in applied_changes["order_changes"].items():
        section = physical_section_for_order_key(field)
        if section in SECTION_ORDER and SECTION_ORDER.index(section) > current_index:
            result.append({"field": field, "value": value})
    return result


def build_turn_action_event(applied_changes: dict, future_changes: list[dict]) -> dict | None:
    # 실제 변경이 있었던 턴만 action_history에 남길 event를 만든다.
    if not any(applied_changes.values()):
        return None
    event = {"type": "turn_applied", **copy.deepcopy(applied_changes)}
    if future_changes:
        event["future_changes"] = copy.deepcopy(future_changes)
    return event


def current_section_changed(applied_changes: dict, section: str) -> bool:
    # 이번 턴에 현재 section의 주문값이 하나라도 바뀌었는지 확인한다.
    allowed = set(allowed_order_fields(section))
    return bool(set(applied_changes.get("order_changes", {})) & allowed)


def validate_recommendation_proposal(session: dict, proposal: dict, robot_state: dict) -> dict:
    # 추천도 일반 주문과 같은 domain·물리 검사를 통과해야 pending 후보가 될 수 있다.
    validation = validate_order_patch(proposal, robot_state)
    accepted = validation["accepted_patch"]

    # 추천이 사용자가 이미 확정한 값을 조용히 덮어쓰지 못하게 한다.
    for field in ("sauce", "noodle_type", "noodle_portion"):
        if accepted.get(field) is not None and session["order"].get(field) is not None:
            validation["protected"].append({"field": field, "value": accepted[field], "reason": "already_selected"})
            accepted[field] = None
    for item in list(accepted.get("toppings", {})):
        if item in session["order"].get("toppings", {}):
            validation["protected"].append({"field": f"toppings.{item}", "value": accepted["toppings"][item], "reason": "already_selected"})
            accepted["toppings"].pop(item, None)

    temp_order = apply_order_patch(session["order"], accepted)
    conflict_keys = {c["key"] for c in find_restriction_conflicts(temp_order)}

    safe = copy.deepcopy(accepted)
    rejected = list(validation["unsupported"]) + list(validation["invalid"]) + list(validation["protected"])
    for key in conflict_keys:
        if key.startswith("toppings."):
            item = key.split(".", 1)[1]
            if item in safe["toppings"]:
                rejected.append({"field": key, "reason": "restriction_conflict"})
                safe["toppings"].pop(item, None)
        elif safe.get(key) is not None:
            rejected.append({"field": key, "reason": "restriction_conflict"})
            safe[key] = None

    return {
        "proposal": safe,
        "accepted_fields": changed_order_keys_from_patch(safe),
        "rejected_fields": rejected,
    }
