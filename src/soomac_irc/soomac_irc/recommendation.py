import copy

from soomac_irc.order_change import apply_menu_changes, validate_menu_changes
from soomac_irc.order import ALL_TOPPINGS, NOODLE_TYPES, SAUCES, physical_section_for
from soomac_irc.restriction import strongest_restriction_for


# 추천안은 일반 주문 검증을 재사용한 뒤 추천 전용 제외·제약 규칙을 추가로 검사한다.


def validate_recommended_order(order: dict, recommendation: dict) -> tuple[dict, list[dict], list[dict]]:
    # adapter 추천에서 이미 고른 값·제외 재료·restriction 충돌만 제거하고 정상 추천을 살리고, 입력 order는 변경하지 않음
    # recommendation: recommend_order Tool의 changes 딕셔너리
    # 이 함수는 추천안을 검사만 하며 order는 수정하지 않는다.
    # 반환: (살린 추천, 형식·중복 탈락, restriction 차단)
    dropped = [] # 잘못됐거나 이미 선택되어 추천에서 제외한 값
    blocked = [] # 활성 restriction과 충돌하여 제외한 값

    if not isinstance(recommendation, dict):
        return {}, [{"path": "recommendation", "value": recommendation, "reason": "invalid_value"}], []

    # 양은 메뉴가 아니므로 허용 메뉴 목록에 넣지 않는다.
    allowed_menu = SAUCES + NOODLE_TYPES + ALL_TOPPINGS
    excluded = recommendation.get("excluded", [])
    excluded_items = []

    if not isinstance(excluded, list):
        dropped.append({"path": "excluded", "value": excluded, "reason": "invalid_value"})
    else:
        for index, item in enumerate(excluded):
            path = f"excluded[{index}]"

            if not isinstance(item, str) or item not in allowed_menu:
                dropped.append({"path": path, "value": item, "reason": "invalid_value"})
                continue

            if item in excluded_items:
                dropped.append({"path": path, "value": item, "reason": "duplicate_value"})
                continue

            excluded_items.append(item)

    recommended_order = recommendation.get("recommended_order")

    if not isinstance(recommended_order, dict) or not recommended_order:
        dropped.append({"path": "recommended_order", "value": recommended_order, "reason": "invalid_value"})
        return {}, dropped, blocked

    clean, invalid_values = validate_menu_changes(recommended_order)

    for invalid in invalid_values:
        invalid["path"] = f"recommended_order.{invalid['path']}"
        dropped.append(invalid)

    for field in ("sauce", "noodle_type", "noodle_portion"):
        if field not in clean:
            continue

        value = clean[field]

        # 이미 사용자가 고른 항목은 새 추천처럼 다시 제안하지 않는다.
        if order[field] is not None:
            dropped.append({"path": field, "value": value, "reason": "already_selected"})
            del clean[field]
            continue

        # 양은 메뉴가 아니므로 excluded와 restriction 검사 대상이 아니다.
        if field == "noodle_portion":
            continue

        if value in excluded_items:
            dropped.append({"path": field, "value": value, "reason": "excluded"})
            del clean[field]
            continue

        restriction = strongest_restriction_for(order, value)

        if restriction is not None:
            blocked.append({"path": field, "value": value, "restriction": restriction})
            del clean[field]

    for topping, amount in list(clean.get("toppings", {}).items()):
        path = f"toppings.{topping}"

        # 추천은 추가 proposal이므로 none 삭제 명령을 허용하지 않는다.
        if amount == "none":
            dropped.append({"path": path, "value": amount, "reason": "invalid_recommendation_amount"})
            del clean["toppings"][topping]
            continue

        if topping in order["toppings"]:
            dropped.append({"path": path, "value": amount, "reason": "already_selected"})
            del clean["toppings"][topping]
            continue

        if topping in excluded_items:
            dropped.append({"path": path, "value": amount, "reason": "excluded"})
            del clean["toppings"][topping]
            continue

        restriction = strongest_restriction_for(order, topping)

        if restriction is not None:
            blocked.append({"path": path, "value": amount, "restriction": restriction})
            del clean["toppings"][topping]

    if "toppings" in clean and not clean["toppings"]:
        del clean["toppings"]

    return clean, dropped, blocked

def menu_change_keys(changes: dict) -> set[str]:
    # 입력 메뉴 변경에서 sauce, toppings.양파 같은 비교용 key만 추출한다.
    # 입력 변경 없음. revise_pending_recommendation에서 호출한다.
    keys = {field for field in ("sauce", "noodle_type", "noodle_portion") if field in changes}
    # update() 함수는 딕셔너리(Dictionary)나 세트(Set) 같은 자료구조에서 값을 추가하거나 수정할 때 사용하는 메서드로,
    # 맨 뒤에 들어간다고 보면 된다.
    keys.update(f"toppings.{item}" for item in changes.get("toppings", {}))
    return keys


def build_revised_menu_changes(recommended_changes: dict, clean_changes: dict) -> dict:
    # 검증된 수정값을 추천 메뉴 복사본에 적용한다. 실제 주문과 원본 추천은 변경하지 않는다.
    # 반환: 수정된 추천 메뉴. revise_pending_recommendation에서 호출한다.
    revised_changes = copy.deepcopy(recommended_changes)
    for field in ("sauce", "noodle_type", "noodle_portion"):
        if field in clean_changes:
            revised_changes[field] = clean_changes[field]

    if "toppings" in clean_changes:
        revised_toppings = revised_changes.setdefault("toppings", {})
        for item, amount in clean_changes["toppings"].items():
            if amount == "none":
                revised_toppings.pop(item, None)
            else:
                revised_toppings[item] = amount
        if not revised_toppings:
            revised_changes.pop("toppings", None)
    return revised_changes


def revise_pending_recommendation(order: dict, recommendation: dict, menu_changes: dict) -> dict:
    # 확인 대기 중인 추천안만 수정한다.
    # 실제 주문은 바꾸지 않고 추천 proposal 복사본만 변경한다.
    # 입력: 주문, 추천 상태, 메뉴 수정값. 반환: 기존 revision 결과. Order Policy에서 호출한다.
    if recommendation.get("phase") != "confirming":
        return {"applied": False, "reason": "no_pending_proposal"}

    request = recommendation.get("request")
    recommended_changes = recommendation.get("recommended_changes")
    confirmed_sections = recommendation.get("confirmed_sections", [])
    if not isinstance(request, dict):
        return {"applied": False, "reason": "invalid_pending_request"}
    if not isinstance(recommended_changes, dict):
        return {"applied": False, "reason": "invalid_pending_changes"}
    if not isinstance(confirmed_sections, list):
        return {"applied": False, "reason": "invalid_confirmed_sections"}

    clean_changes, invalid = validate_menu_changes(menu_changes)
    if invalid or not clean_changes:
        return {"applied": False, "reason": "invalid_revision", "invalid": invalid}

    proposal_keys = menu_change_keys(recommended_changes)
    requested_keys = menu_change_keys(clean_changes)
    # 현재 추천안에 없는 재료는 실제 주문 변경인지
    # 추천안 수정인지 알 수 없으므로 자동 처리하지 않는다.
    if not requested_keys or not requested_keys.issubset(proposal_keys):
        # issubset() 함수(메서드)는 어떤 집합이 다른 집합의 부분집합인지 확인하여 참(True)이나 거짓(False)을 반환
        return {"applied": False, "reason": "target_not_in_proposal", "requested_keys": sorted(requested_keys)}

    revised_changes = build_revised_menu_changes(recommended_changes, clean_changes)
    if not revised_changes:
        return {"applied": False, "reason": "proposal_became_empty"}

    recommendation_input = copy.deepcopy(request)
    recommendation_input["recommended_order"] = copy.deepcopy(revised_changes)
    clean, dropped, blocked = validate_recommended_order(order, recommendation_input)

    # 일부 항목만 통과하면 의도하지 않은 추천안이 될 수 있으므로
    # 자동으로 부분 반영하지 않는다.
    if dropped or blocked or not clean:
        return {"applied": False, "reason": "revision_blocked", "clean": clean, "dropped": dropped, "blocked": blocked}

    # 실제 주문이 아니라 복사한 proposal만 변경한다.
    proposal = copy.deepcopy(order)
    applied, unchanged = apply_menu_changes(proposal, clean)
    if not applied:
        return {"applied": False, "reason": "revision_has_no_new_value", "unchanged": unchanged}

    covered_sections = []
    for key in applied:
        if key in ("sauce", "noodle_type", "noodle_portion"):
            covered_section = "noodle"
        else:
            item = key.split(".", 1)[1]
            covered_section = physical_section_for("toppings", item)
        if covered_section not in covered_sections:
            covered_sections.append(covered_section)

    return {
        "applied": True,
        "reason": "revised",
        "clean": clean,
        "dropped": [],
        "blocked": [],
        "recommendation": {
            "phase": "confirming",
            "request": copy.deepcopy(request),
            "proposal": proposal,
            "recommended_changes": copy.deepcopy(clean),
            "covered_sections": covered_sections,
            "confirmed_sections": copy.deepcopy(confirmed_sections),
        },
    }
