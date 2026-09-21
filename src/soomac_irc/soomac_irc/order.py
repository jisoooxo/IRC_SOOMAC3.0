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

MIN_ITEM_NUM = 1 # 최소 주문이지만 현재는 크게 의미는 없음.


SECTION_ORDER = ["noodle", "veggie", "meat", "extra", "lid", "sauce"] # 순서 ㅇㅇ

SECTION_LABELS = {"noodle": "면","veggie": "야채", "meat": "육류","extra": "추가 재료","lid": "뚜껑 닫기","sauce": "소스"}


# restriction 규약 - 호불호, 못먹어, 알러지, 식이제약(종교 포함)

RESTRICTION_REASONS = ["dislike","cannot_eat","allergy","dietary_rule"]


# 알러지 제약 키워드

RESTRICTION_CATEGORY_ITEMS = {"유제품": ["크림", "치즈"],"갑각류": ["게살"],"육류": ["소시지"], "비건": ["소시지"]}


# 주문에는 음식 선택과 활성 restriction만 저장

DEFAULT_ORDER = {
    "sauce": None,
    "noodle_type": None,
    "noodle_portion": None,
    "toppings": {}, # 토핑 -> {topping_name: amount}로 들어가고, 야채 육류 엑스트라 ㅇㅇ
    "restrictions": [], # 제약
}


def new_order() -> dict:
    # 새로운 주문 dictionary. 그냥 copy해버리면 값 공유 되어버려서 deepcopy로 객체 만듬.
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

def next_section(section: str) -> str | None: # 현재 하고 있는 section 다음 행동을 반환 예시는 상단에 ㅇㅇ
    # 입력 section이 정본 순서에 없으면 조용히 넘기지 않고 바로 막음
    if section not in SECTION_ORDER:
        raise ValueError(f"섹션 이상함 : {section}")

    current_index = SECTION_ORDER.index(section)
    next_index = current_index + 1

    if next_index >= len(SECTION_ORDER):
        return None # None을 반환하면 마지막 작업?

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

# 주문 완성도 검사와 ROS plan 변환


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
