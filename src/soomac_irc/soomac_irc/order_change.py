from soomac_irc.order import ALL_TOPPINGS, AMOUNTS, NOODLE_TYPES, SAUCES, TOPPING_AMOUNTS


# 일반 주문 변경은 메뉴·양 검증을 통과한 뒤 주문 복사본에 적용한다.


def validate_menu_changes(changes: dict) -> tuple[dict, list[dict]]:
    # 검증된 메뉴·양을 실제 order에 추가·수정·삭제
    # changes: Tool 출력에서 메뉴와 양 관련 필드를 분리한 값
    # 반환: (통과한 값, 탈락한 값). 여기서는 order를 안 건드림
    if not isinstance(changes, dict):
        return {}, [{"path": "changes", "value": changes, "reason": "invalid_value"}]

    clean = {}   # 메뉴와 양 검사를 통과한 값
    dropped = [] # 메뉴에 없거나 형식이 잘못된 값

    for field, value in changes.items():
        if field == "sauce":
            if value in SAUCES:
                clean[field] = value
            else:
                dropped.append({"path": field, "value": value, "reason": "invalid_value"})

        elif field == "noodle_type":
            if value in NOODLE_TYPES:
                clean[field] = value
            else:
                dropped.append({"path": field, "value": value, "reason": "invalid_value"})

        elif field == "noodle_portion":
            if value in AMOUNTS:
                clean[field] = value
            else:
                dropped.append({"path": field, "value": value, "reason": "invalid_value"})

        elif field == "toppings":
            if not isinstance(value, dict):
                dropped.append({"path": field, "value": value, "reason": "invalid_value"})
                continue

            clean_toppings = {}

            for topping, amount in value.items():
                if topping in ALL_TOPPINGS and amount in TOPPING_AMOUNTS:
                    clean_toppings[topping] = amount
                else:
                    dropped.append({"path": f"toppings.{topping}", "value": amount, "reason": "invalid_value"})

            if clean_toppings:
                clean[field] = clean_toppings

        else:
            dropped.append({"path": field, "value": value, "reason": "invalid_value"})

    return clean, dropped


def apply_menu_changes(order: dict, changes: dict) -> tuple[list[str], list[str]]:
    # 검증된 메뉴·양을 실제 order에 추가·수정·삭제
    # 검증된 메뉴 변경을 order에 반영하고 검증된 key를 반환
    # 주의: 입력 order 자체를 바꿈. agent에서는 deepcopy한 order_after만 넘겨야 함
    applied = []
    unchanged = []

    for field in ("sauce", "noodle_type", "noodle_portion"):
        if field not in changes:
            continue

        if order[field] == changes[field]:
            unchanged.append(field)
            continue

        order[field] = changes[field]
        applied.append(field)

    for topping, amount in changes.get("toppings", {}).items():
        key = f"toppings.{topping}"

        # none은 아직 담지 않은 토핑을 주문에서 빼달라는 의미
        if amount == "none":
            if topping not in order["toppings"]:
                unchanged.append(key)
                continue

            del order["toppings"][topping]
            applied.append(key)
            continue

        if order["toppings"].get(topping) == amount:
            unchanged.append(key)
            continue

        order["toppings"][topping] = amount
        applied.append(key)

    return applied, unchanged
