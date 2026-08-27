import copy


# 사용자가 선택 가능한 메뉴

SAUCES = ["오일", "토마토", "크림",]

NOODLE_TYPES = ["얇은면","넓적면",]

VEGGIES = ["양파", "버섯",]

MEATS = ["소시지", "게살",]

EXTRAS = ["치즈","페퍼론치노",]

ALL_TOPPINGS = VEGGIES + MEATS + EXTRAS

AMOUNTS = ["low", "normal", "high",]

TOPPING_AMOUNTS = ["none", "low", "normal", "high",]

# 양에 따라 로봇이 몇 번 집을지 결정한다.
AMOUNT_TO_COUNT = {"low": 1, "normal": 2, "high": 3,}

MIN_ITEM_NUM = 1


# fields : 해당 섹션에서 사용자가 결정할 일반 주문
# items : 해당 섹션에서 선택할 수 있는 토핑
# required : 섹션을 끝내기 전에 반드시 채워야하는 일반 주문값
# min items : 최소로 선택해야 하는 토핑 개수
# 얘네는 함수에서 직접 검사(중첩 dictionary 어려워잉)


SECTION_ORDER = ["noodle", "veggie", "meat", "extra", "sauce", "lid"]

SECTION_LABELS = {
    "noodle": "면",
    "veggie": "야채",
    "meat": "육류",
    "extra": "추가 재료",
    "sauce": "소스",
    "lid": "뚜껑 닫기",
}



# reply, 화면에서 사용

FIELD_LABELS = {
    "sauce": "소스",
    "noodle_type": "면 종류",
    "noodle_portion": "양",
}

# 필드 내에서 미리 가져갈 수 있는거 ㅇㅇ 이건 따로 지정할 수 있음

EARLY_TOPPINGS = {"noodle": ["치즈", "페퍼론치노"]}

"""
# 여러 단계에서 조기 선택 허용
EARLY_TOPPINGS = {
    "noodle": ["치즈", "페퍼론치노"],
    "veggie": ["소시지"],
}
"""

# 알레르기·식이 제약이 차단할 재료

CONSTRAINT_BLOCKS = {
    "갑각류": {"toppings": ["게살"]},
    "유제품": {"toppings": ["치즈"], "sauce": ["크림"]},
    "육류": {"toppings": ["소시지"]},
    "비건": {"toppings": ["소시지", "게살", "치즈"], "sauce": ["크림"]},
}

# 사용자가 재료 이름으로 제약을 말했을 때 표준 이름으로 변환. 만약 게살 못먹어 이러면 CONSTRAIMT ALIASES로 매핑
CONSTRAINT_ALIASES = {"게살": "갑각류", "치즈": "유제품", "크림": "유제품"}


# DEFAULT_ORDER -> 이 포멧대로 채움

DEFAULT_ORDER = {
    # 소스·면 종류·면 양은 각각 값 하나만 가지므로 Tool changes와 동일한 이름으로 최상위에 저장한다.
    "sauce": None,
    "noodle_type": None,
    "noodle_portion": None,

    # 사용자가 고른 토핑만 저장
    # 예: {"양파": "normal", "치즈": "high"}
    "toppings": {},

    # 알러지나 식이 제약을 저장
    # 예: ["갑각류", "유제품"]
    "constraints": [],
}
def new_order() -> dict:
    # 손님이 사용할 빈 주문 반환(처음에)

    order = copy.deepcopy(DEFAULT_ORDER)

    return order


def next_section(section: str) -> str | None:
    # 다음 섹션 반환, 마지막(lid)면 None을 반환
    if section not in SECTION_ORDER:
        raise ValueError(f"섹션 이상함 : {section}")

    current_index = SECTION_ORDER.index(section)  # str 받아서 SECTION 필드에서 빼먹음
    next_index = current_index + 1

    if next_index >= len(SECTION_ORDER):
        return None

    return SECTION_ORDER[next_index]


def missing_required(order: dict, section: str) -> list[str]:
    # 현재 섹션 전 빠진거 검사. 반환된게 비어 있으면 섹션 끝낼 수 있고
    # list에 값이 있으면 해당 값을 다시 물어봐야 함

    if section not in SECTION_ORDER:
        raise ValueError(f"섹션 이상함 : {section}")

    # section : noodle, veggie, meat, extra, sauce, lid
    missing = []

    # 상단의 required 필요한 필드 검사
    if section == "noodle":
        for field in ["sauce", "noodle_type", "noodle_portion"]:
            current_value = order.get(field)

            if current_value is None:
                missing.append(field) 

    # min item 검사
    # 필요한 섹션만 직접 검사한다.
    elif section == "veggie":
        picked_items = []

        # 내부 아이템 반환
        for item in VEGGIES:
            if item in order["toppings"]:  # 면, 소스 이런거 제외하고
                picked_items.append(item)

        if len(picked_items) < MIN_ITEM_NUM: # 1개 이하면 ㅇㅇ 이건 상단 파라미터화 해야징
            missing.append("야채 재료")

    elif section == "meat":
        picked_items = []

        for item in MEATS:
            if item in order["toppings"]:
                picked_items.append(item)

        if len(picked_items) < MIN_ITEM_NUM:
            missing.append("육류 재료")

    return missing # 비어있으면 섹션 끝



def validate_delta(delta: dict, section: str) -> tuple[dict, list[str]]: # 튜플로 고정

    # 모델이 제안한 변경값 중 허용된 것만 남김(할루시 방지)

    if section not in SECTION_ORDER:
        raise ValueError(f"섹션 이상함 : {section}")

    if not isinstance(delta, dict):
        return {}, [f"changes={delta}"]

    clean = {}
    dropped = []

    # 현재 섹셙에서 선택할 수 있는 토핑(섹션 별로 복사 ㄱㄱ)

    allowed_toppings = []

    if section == "veggie":
        allowed_toppings = VEGGIES.copy()

    elif section == "meat":
        allowed_toppings = MEATS.copy()

    elif section == "extra":
        allowed_toppings = EXTRAS.copy()

    # 현재 단계보다 일찍 선택할 수 있는 토핑
    if section in EARLY_TOPPINGS:
        allowed_toppings += EARLY_TOPPINGS[section]


    for field, value in delta.items(): # key, value(모델이 바꾼거)

        # 면 섹션일때 소스랑 면 종류랑 인분 정함

        if field == "sauce":
            if section == "noodle" and value in SAUCES: # value가 기존 상단 소스에 있으면 ㅇㅇ
                clean[field] = value

            else:
                dropped.append(f"{field}={value}")


        elif field == "noodle_type":
            if section == "noodle" and value in NOODLE_TYPES:
                clean[field] = value

            else:
                dropped.append(f"{field}={value}")


        elif field == "noodle_portion":
            if section == "noodle" and value in AMOUNTS:
                clean[field] = value
            else:
                dropped.append(f"{field}={value}")

        elif field == "toppings": # 토핑은 딕셔너리 형태로 들어옴
            if not isinstance(value, dict):
                dropped.append(f"{field}={value}")
                continue

            clean_toppings = {}

            for topping, amount in value.items(): # 토핑이랑 인분
                if topping in allowed_toppings and amount in TOPPING_AMOUNTS:
                    clean_toppings[topping] = amount
                else:
                    dropped.append(f"toppings.{topping}={amount}")

            if clean_toppings:
                clean[field] = clean_toppings

        elif field == "constraints": # 제약은 리스트로
            if not isinstance(value, list):
                dropped.append(f"{field}={value}")
                continue

            clean_constraints = []

            for constraint in value:
                if not isinstance(constraint, str):
                    dropped.append(f"constraints={constraint}")
                    continue

                # 먼저 흔한 재료명은 alias로 빠르게 표준화
                constraint_name = constraint.strip()

                if constraint_name in CONSTRAINT_ALIASES:
                    constraint_name = CONSTRAINT_ALIASES[constraint_name]

                # alias에 없으면 모델이 판단해서 준 이름을 그대로 검사
                if constraint_name not in CONSTRAINT_BLOCKS:
                    dropped.append(f"constraints={constraint}")

                elif constraint_name not in clean_constraints:
                    clean_constraints.append(constraint_name)

            if clean_constraints:
                clean[field] = clean_constraints

        else:
            # 정해진 주문 필드 외에는 버림
            dropped.append(f"{field}={value}")

    return clean, dropped


def apply_delta(order: dict, clean: dict) -> list[str]:
    # 허용된 변경값을 order에 넣음
    changed = []

    # order꺼를 빼서 changed에 넣어서 반환(섹션을)

    if "sauce" in clean and order["sauce"] != clean["sauce"]: 

        # 걸러진거에서 order랑 다르면 교체

        order["sauce"] = clean["sauce"]

        changed.append("sauce")

    if "noodle_type" in clean and order["noodle_type"] != clean["noodle_type"]:

        order["noodle_type"] = clean["noodle_type"]

        changed.append("noodle_type")

    if "noodle_portion" in clean and order["noodle_portion"] != clean["noodle_portion"]:

        order["noodle_portion"] = clean["noodle_portion"]

        changed.append("noodle_portion")

    if "toppings" in clean:
        for topping, amount in clean["toppings"].items():
            # none은 이미 선택한 토핑을 빼달라는 의미
            if amount == "none":
                if topping in order["toppings"]:

                    del order["toppings"][topping]

                    changed.append(f"toppings.{topping}")

            elif order["toppings"].get(topping) != amount:

                order["toppings"][topping] = amount

                changed.append(f"toppings.{topping}")

    if "constraints" in clean:
        # 제약은 덮어쓰지 않고 계속 누적

        for constraint in clean["constraints"]:

            if constraint not in order["constraints"]:

                order["constraints"].append(constraint)

                changed.append(f"constraints.{constraint}")

    return changed


def enforce_constraints(order: dict) -> list[str]:
    # 주문에 들어간 알레르기·식이 제약을 실제 재료에 적용
    # 여기 changed는 제약 때문에 다시 빠진 항목
    # 호출부에서는 blocked로 받아서 사용자 요청 변경과 구분함
    changed = []

    for constraint in order["constraints"]:
        blocked = CONSTRAINT_BLOCKS.get(constraint)

        # 모르는 제약은 validate_delta에서 걸러지지만 혹시 들어오면 건너뜀
        if blocked is None:
            continue

        # 해당 제약에서 금지한 토핑 제거
        if "toppings" in blocked:
            for topping in blocked["toppings"]:

                if topping in order["toppings"]: # blocked된 topping이 오더에 있으면 삭제

                    del order["toppings"][topping] 
                    changed.append(f"toppings.{topping}")

        # 해당 제약에서 금지한 소스 제거
        if "sauce" in blocked and order["sauce"] in blocked["sauce"]:
            order["sauce"] = None
            changed.append("sauce")

    return changed

def build_section_plan(order: dict, section: str) -> list[dict]:
    # 현재 섹션에서 로봇이 실행할 작업만 리스트로 만듬(최종 필터링본)
    if section not in SECTION_ORDER:
        raise ValueError(f"섹션 이상함 : {section}")

    # 면은 면 종류와 양을 이용해서 작업 하나 생성
    if section == "noodle":
        noodle_type = order["noodle_type"]
        noodle_portion = order["noodle_portion"]

        if noodle_type is None or noodle_portion not in AMOUNT_TO_COUNT:
            return []

        return [{
            "class": noodle_type,
            "repeat_count": AMOUNT_TO_COUNT[noodle_portion],
        }]

    # 소스는 처음에 선택하지만 실제 투입은 sauce 섹션에서 함
    if section == "sauce":
        if order["sauce"] is None:
            return []

        return [{
            "class": order["sauce"],
            "repeat_count": 1,
        }]

    # 뚜껑은 차후 VLM 최종 확인 뒤 별도 제어로 닫음
    if section == "lid":
        return [
            {
                "class": "뚜껑",
                "repeat_count": 1,
            }
        ]

    # 현재 토핑 섹션에 해당하는 재료 목록
    section_toppings = []

    if section == "veggie":
        section_toppings = VEGGIES
    elif section == "meat":
        section_toppings = MEATS
    elif section == "extra":
        section_toppings = EXTRAS

    tasks = []

    # 메뉴에 정의된 순서대로 로봇 작업 생성
    for topping in section_toppings:
        amount = order["toppings"].get(topping)

        if amount in AMOUNT_TO_COUNT:
            task = {
                "class": topping,
                "repeat_count": AMOUNT_TO_COUNT[amount],
            }

            tasks.append(task)

    return tasks
