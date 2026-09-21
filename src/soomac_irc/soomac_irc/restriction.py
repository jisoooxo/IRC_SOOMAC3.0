from soomac_irc.order import ALL_TOPPINGS, NOODLE_TYPES, RESTRICTION_CATEGORY_ITEMS, RESTRICTION_REASONS, SAUCES


# 제약은 검증 → 적용 → 실제 메뉴와 충돌 조회 순서로 처리한다.


"""
맛살                 → canonical 메뉴 아님
diet                 → reason 이름이 폐기됨
enabled=1            → bool이 아님
설명 같은 추가 필드    → 최소 Tool 계약 위반

이런거 잡는 함수 ㅇㅇ 즉, 필드 3개(target, reason, enabled)만 있고, target이 메뉴에 있는지, reason이 RESTRICTION_REASONS에 있는지, enabled가 bool인지 확인
"""

def validate_restriction_changes(changes: list) -> tuple[list[dict], list[dict]]: # 검증된 restriction을 추가하거나 철회
    # changes: Tool이 출력한 restriction_changes 목록
    # 반환: (적용 후보, 탈락 기록). 같은 제약의 추가·철회가 같이 오면 둘 다 버림
    if not isinstance(changes, list):
        return [], [{"path": "restriction_changes", "value": changes, "reason": "invalid_value"}]

    candidates = {} # 같은 target+reason 명령을 함께 검사
    clean = []      # 검증과 중복·모순 검사를 통과한 restriction
    dropped = []    # 형식 오류, 중복 또는 서로 모순된 restriction

    for index, change in enumerate(changes):
        path = f"restriction_changes[{index}]"

        if not isinstance(change, dict) or set(change) != {"target", "reason", "enabled"}:
            dropped.append({"path": path, "value": change, "reason": "invalid_value"})
            continue

        target = change["target"]
        reason = change["reason"]
        enabled = change["enabled"]

        if not isinstance(target, str) or not isinstance(reason, str) or not isinstance(enabled, bool):
            dropped.append({"path": path, "value": change, "reason": "invalid_value"})
            continue

        target = target.strip()
        valid_target = target in SAUCES or target in NOODLE_TYPES or target in ALL_TOPPINGS or target in RESTRICTION_CATEGORY_ITEMS

        if not valid_target or reason not in RESTRICTION_REASONS:
            dropped.append({"path": path, "value": change, "reason": "invalid_value"})
            continue

        clean_change = {"target": target, "reason": reason, "enabled": enabled}
        restriction_key = (target, reason)

        if restriction_key not in candidates:
            candidates[restriction_key] = []

        candidates[restriction_key].append({"path": path, "value": clean_change})

    for same_restriction in candidates.values():
        enabled_values = {entry["value"]["enabled"] for entry in same_restriction}

        # 같은 제약을 추가하고 철회하는 명령이 한 번에 들어오면 둘 다 적용하지 않음
        if len(enabled_values) > 1:
            for entry in same_restriction:
                dropped.append({"path": entry["path"], "value": entry["value"], "reason": "conflicting_value"})
            continue

        clean.append(same_restriction[0]["value"])

        # 같은 명령이 반복되면 첫 번째만 적용
        for entry in same_restriction[1:]:
            dropped.append({"path": entry["path"], "value": entry["value"], "reason": "duplicate_value"})

    return clean, dropped


def apply_restriction_changes(order: dict, changes: list[dict]) -> tuple[list[dict], list[dict]]: # 검증된 restriction을 추가하거나 같은 target+reason만 철회
    # 입력 order 자체를 바꾸고 (실제 변경, 이미 같은 상태)을 나눠서 반환
    applied = []
    unchanged = []

    for change in changes:
        restriction = {"target": change["target"], "reason": change["reason"]}

        if change["enabled"]:
            if restriction in order["restrictions"]: # 이미 있는 restriction을 다시 추가 X
                unchanged.append(change)
                continue

            order["restrictions"].append(restriction) # 제약 추가
            applied.append(change) # 제약 적용
            continue

        if restriction not in order["restrictions"]:
            unchanged.append(change)
            continue

        order["restrictions"].remove(restriction) # 제약 철회
        applied.append(change)

    return applied, unchanged


def restriction_items_for(target: str) -> list[str]: # 유제품 같은 범주를 실제 메뉴 목록으로 바꿈
    # restriction 범주를 실제로 차단할 메뉴 목록으로 바꿈.
    # copy를 반환해서 호출자가 범주 정본을 실수로 수정하지 못하게 함
    if target in RESTRICTION_CATEGORY_ITEMS:
        return RESTRICTION_CATEGORY_ITEMS[target].copy()

    if target in SAUCES or target in NOODLE_TYPES or target in ALL_TOPPINGS:
        return [target]

    return []


def matching_restrictions_for(order: dict, item: str) -> list[dict]: # 재료 하나에 걸린 restriction을 전부 찾음
    # 실제 메뉴 하나에 적용되는 활성 restriction을 모두 반환
    # 범주 제약과 개별 재료 제약이 겹치면 둘 다 들어옴
    matches = []

    for restriction in order["restrictions"]:
        blocked_items = restriction_items_for(restriction["target"])

        if item in blocked_items:
            matches.append(restriction.copy())

    return matches


def strongest_restriction_for(order: dict, item: str) -> dict | None: # 여러 restriction 중 가장 강한 하나를 고름
    # 같은 메뉴에 걸린 restriction 중 가장 강한 한 건을 반환
    # 안전 안내가 취향 안내에 묻히면 안 돼서 allergy부터 직접 순회함
    matches = matching_restrictions_for(order, item)

    for reason in ("allergy", "cannot_eat", "dietary_rule", "dislike"): # 이순서 ㅇㅇ
        for restriction in matches:
            if restriction["reason"] == reason:
                return restriction

    return None


def order_restriction_conflicts(order: dict) -> list[dict]:
    # 현재 주문에 선택된 메뉴 중 restriction과 충돌하는 항목을 반환한다.
    # 검사만 하고 order는 안 바꿈. 실제 삭제·유지는 agent가 물리 진행 상태를 보고 결정
    conflicts = []

    if order["sauce"] is not None:
        restriction = strongest_restriction_for(order, order["sauce"])

        if restriction is not None:
            conflicts.append({"key": "sauce", "item": order["sauce"], "restriction": restriction})

    if order["noodle_type"] is not None:
        restriction = strongest_restriction_for(order, order["noodle_type"])

        if restriction is not None:
            conflicts.append({"key": "noodle_type", "item": order["noodle_type"], "restriction": restriction})

    for topping in order["toppings"]:
        restriction = strongest_restriction_for(order, topping)

        if restriction is not None:
            conflicts.append({"key": f"toppings.{topping}", "item": topping, "restriction": restriction})

    return conflicts
