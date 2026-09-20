import copy

"""
함수 순서 : 정본 → 검증 → 반영 → 정책 → ROS 출력
"""


# 1. 메뉴, 주문·section 기본 기능
# 사용자가 선택 가능한 실제 메뉴

SAUCES = ["오일", "토마토", "크림"]

NOODLE_TYPES = ["얇은면", "넓은면"]

VEGGIES = ["양파", "버섯"]

MEATS = ["소시지", "게살"]

EXTRAS = ["치즈", "페퍼론치노"]

ALL_TOPPINGS = VEGGIES + MEATS + EXTRAS # topping field에 들어가는 거


# 양과 로봇 반복 횟수

AMOUNTS = ["low", "normal", "high"]

TOPPING_AMOUNTS = ["none", "low", "normal", "high"]

AMOUNT_TO_COUNT = {"low": 1, "normal": 2, "high": 3}

MIN_ITEM_NUM = 1




SECTION_ORDER = ["noodle", "veggie", "meat", "extra", "lid", "sauce"] # 순서 ㅇㅇ

SECTION_LABELS = {
    "noodle": "면",
    "veggie": "야채",
    "meat": "육류",
    "extra": "추가 재료",
    "lid": "뚜껑 닫기",
    "sauce": "소스",
}


# restriction 규약 - 호불호, 못먹어, 알러지, 식이제약(종교 포함)

RESTRICTION_REASONS = [
    "dislike",
    "cannot_eat",
    "allergy",
    "dietary_rule",
]

# 범주를 사용자가 직접 말했을 때만 실제 메뉴로 확장
RESTRICTION_CATEGORY_ITEMS = {
    "유제품": ["크림", "치즈"],
    "갑각류": ["게살"],
    "육류": ["소시지"],
    "비건": ["소시지"],
}


# 주문에는 음식 선택과 활성 restriction만 저장

DEFAULT_ORDER = {
    "sauce": None,
    "noodle_type": None,
    "noodle_portion": None,
    "toppings": {}, # 토핑 -> {topping_name: amount}로 들어가고, 야채 육류 엑스트라 ㅇㅇ
    "restrictions": [], # 제약
}


def new_order() -> dict: # 새로운 주문 dictionary. 그냥 copy해버리면 값 공유 되어버려서 deepcopy로 객체 만듬.
    # 반환: 다른 주문과 내부 list·dict를 공유하지 않는 빈 주문
    return copy.deepcopy(DEFAULT_ORDER)


"""
next_section("noodle")  # "veggie"
next_section("veggie")  # "meat"
next_section("extra")   # "lid"
next_section("lid")     # "sauce"
next_section("sauce")   # None

이런식으로 다음꺼 반환
"""

def next_section(section: str) -> str | None:     # 현재 하고 있는 section 다음 행동을 반환 예시는 상단에 ㅇㅇ
    # 입력 section이 정본 순서에 없으면 조용히 넘기지 않고 바로 막음
    if section not in SECTION_ORDER:
        raise ValueError(f"섹션 이상함 : {section}")

    current_index = SECTION_ORDER.index(section)
    next_index = current_index + 1

    if next_index >= len(SECTION_ORDER):
        return None

    return SECTION_ORDER[next_index]





# 주문 구조(order field)를 보면 sauce, noodle_type은 필드만 보면 되지만, toppings는 내부 재료(item까지) 봐야 섹션을 알 수 있음
# 주문 필드나 토핑이 실제로 어느 로봇 section에서 처리되는지 반환

"""
소시지 → meat
치즈 → extra
크림 소스 → sauce
"""

def physical_section_for(field: str, item: str | None = None) -> str | None:
    # 주문값이 실제로 로봇에 투입되는 section을 반환
    # 소스는 처음 선택하지만 실제 작업은 마지막 sauce section
    # 반환값은 주문받는 단계가 아니라 로봇이 실제로 투입하는 단계임

    if field in ("noodle_type", "noodle_portion"):
        return "noodle"

    if field == "sauce":
        return "sauce"

    if field != "toppings":
        return None

    if item in VEGGIES:
        return "veggie"

    if item in MEATS:
        return "meat"

    if item in EXTRAS:
        return "extra"

    return None



################################################################################################################################################

# 3. Tool 출력 검증



def validate_menu_changes(changes: dict) -> tuple[dict, list[dict]]: # 검증된 메뉴·양을 실제 order에 추가·수정·삭제
    # changes: Tool 출력에서 메뉴 관련 필드만 분리한 값
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


#################################################################################################################################################

# 4. 검증된 값을 주문에 반영


def apply_menu_changes(order: dict, changes: dict) -> tuple[list[str], list[str]]: # 검증된 메뉴·양을 실제 order에 추가·수정·삭제

    # 그냥 changes 검증 ㅇㅇ

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



def apply_restriction_changes(order: dict, changes: list[dict]) -> tuple[list[dict], list[dict]]:     # 검증된 restriction을 추가하거나 같은 target+reason만 철회

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


##################################################################################################################################################

# 4. Restriction 조회와 추천안 검증


def restriction_items_for(target: str) -> list[str]: # 유제품 같은 범주를 실제 메뉴 목록으로 바꿈
    # restriction 범주를 실제로 차단할 메뉴 목록으로 바꿈.
    # copy를 반환해서 호출자가 범주 정본을 실수로 수정하지 못하게 함
    if target in RESTRICTION_CATEGORY_ITEMS:
        return RESTRICTION_CATEGORY_ITEMS[target].copy()

    if target in SAUCES or target in NOODLE_TYPES or target in ALL_TOPPINGS:
        return [target]

    return []

def matching_restrictions_for(order: dict, item: str) -> list[dict]:  # 재료 하나에 걸린 restriction을 전부 찾음
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





def validate_recommended_order(order: dict, recommendation: dict) -> tuple[dict, list[dict], list[dict]]:
    #adapter 추천에서 이미 고른 값·제외 재료·restriction 충돌만 제거하고 정상 추천을 살리고, 입력 order는 변경하지 않음
    # recommendation: recommend_order Tool의 changes 딕셔너리
    # 이 함수는 추천안을 검사만 하며 order는 수정하지 않는다.
    # 반환: (살린 추천, 형식·중복 탈락, restriction 차단)
    dropped = [] # 잘못됐거나 이미 선택되어 추천에서 제외한 값
    blocked = [] # 활성 restriction과 충돌하여 제외한 값

    if not isinstance(recommendation, dict):
        return {}, [{"path": "recommendation", "value": recommendation, "reason": "invalid_value"}], []

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


###################################################################################################################################################

# 5. 주문 완성도 검사와 ROS plan 변환


def missing_requirements(order: dict, sections: list[str]) -> dict[str, list[str]]:
    # 지정한 section별로 아직 선택하지 않은 필수값만 반환. (확정하기 전에 빠진 필수 선택을 section별로 찾음)
    # extra·lid·sauce는 여기서 새 필수값을 요구하지 않으므로 빈 결과가 정상임
    result = {}

    for section in sections:
        if section not in SECTION_ORDER:
            raise ValueError(f"섹션 이상함 : {section}")

        missing = []

        if section == "noodle":
            for field in ("sauce", "noodle_type", "noodle_portion"):
                if order[field] is None:
                    missing.append(field)

        elif section == "veggie":
            picked_count = sum(item in order["toppings"] for item in VEGGIES)
            if picked_count < MIN_ITEM_NUM:
                missing.append("야채 재료")

        elif section == "meat":
            picked_count = sum(item in order["toppings"] for item in MEATS)
            if picked_count < MIN_ITEM_NUM:
                missing.append("육류 재료")

        if missing:
            result[section] = missing

    return result



# 주문값을 로봇에 보낼 class, repeat_count 목록으로 변환
def build_section_plan(order: dict, section: str) -> list[dict]:
    # 현재 section에서 로봇이 실행할 내부 작업 목록을 만듬
    # 반환 payload는 {class, repeat_count}. order를 바꾸거나 발행까지 하진 않음
    if section not in SECTION_ORDER:
        raise ValueError(f"섹션 이상함 : {section}")

    if section == "noodle":
        noodle_type = order["noodle_type"]
        noodle_portion = order["noodle_portion"]

        if noodle_type is None or noodle_portion not in AMOUNT_TO_COUNT:
            return []

        return [{"class": noodle_type, "repeat_count": AMOUNT_TO_COUNT[noodle_portion]}]

    if section == "sauce":
        if order["sauce"] is None:
            return []

        return [{"class": order["sauce"], "repeat_count": 1}]

    if section == "lid":
        return [{"class": "뚜껑", "repeat_count": 1}]

    section_toppings = []

    if section == "veggie":
        section_toppings = VEGGIES
    elif section == "meat":
        section_toppings = MEATS
    elif section == "extra":
        section_toppings = EXTRAS

    tasks = []

    for topping in section_toppings:
        amount = order["toppings"].get(topping)

        if amount in AMOUNT_TO_COUNT:
            tasks.append({"class": topping, "repeat_count": AMOUNT_TO_COUNT[amount]})

    return tasks
