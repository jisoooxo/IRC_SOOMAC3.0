"""
# llm_policy.py는 모델 출력을 직접 믿지 않고, 로봇의 물리 상태 기준으로 한번 더 계산
# 파일은 field/추천 -> semantic 검사 -> restriction -> 물리 보호 -> Response 데이터 -> 실행 policy 순서로 봄

"""
import copy

from soomac_irc.agent_contract import new_decision
from soomac_irc.domain import (
    NOODLE_TYPES, ORDER_COLLECTION_TARGETS, RESTRICTION_CATEGORY_ITEMS, RESTRICTION_PRIORITY,
    SAUCES, SECTION_ORDER, TOPPINGS,
)

# Python positive gate와 Decision prompt가 동일한 실행 확정 표현을 사용한다.
EXPLICIT_COMMIT_PHRASES = (
    "바로 진행해",
    "이대로 진행해",
    "바로 시작해",
    "담기 시작해",
    "주문 확정해",
    "실행해줘",
    "바로시작",
    "바로시작해줘",
    "바로시작해",
)
# 질문·제안·부정 표현이 있으면 positive phrase가 포함돼도 commit으로 올리지 않는다.
BLOCKED_COMMIT_MARKERS = (
    "?",
    "해도돼",
    "해도될까",
    "해도괜찮",
    "해볼까",
    "할까",
    "하지마",
    "하지말",
    "하지않",
)

# Decision prompt에 보여줄 질문·제안·부정의 완성 문장 예시이다.
BLOCKED_COMMIT_EXAMPLES = (
    "바로 진행해도 돼?",
    "진행해볼까?",
    "진행하지 마",
    "시작하지 마",
)

# 메뉴 근거는 domain 메뉴명과 발화의 자모 편집거리 비율로만 판단한다. 0은 같은 글자, 1은 전혀 다른 글자이다.
# 기준값은 2026-10-05 로그 실측값이다. STT 오인식은 0.17~0.33, 모델이 지어낸 메뉴는 0.67 이상이었다.
MENU_GROUNDED_MAX_DISTANCE = 0.35 # 이하이면 발화에 나온 메뉴로 인정
MENU_UNRELATED_MIN_DISTANCE = 0.65 # 이상이면 발화와 무관한 메뉴로 판단. 사이 구간은 사용자에게 확인

SUPPORTED_MENU_NAMES = (*SAUCES, *NOODLE_TYPES, *TOPPINGS, *RESTRICTION_CATEGORY_ITEMS.keys(), *ORDER_COLLECTION_TARGETS.keys())

SAFETY_RESTRICTION_REASONS = ("allergy", "cannot_eat", "dietary_rule") # 해제에 더 엄격한 근거가 필요한 제한

HANGUL_SYLLABLE_BASE = 0xAC00
HANGUL_SYLLABLE_COUNT = 11172
# ################ 주문 field 위치, 현재·미래 시점, 추천 가능 범위 ###############
def order_key_timing(key: str, robot_state: dict)-> str:
    # 주문 field가 로봇 기준으로 과거, 현재, 미래인지 계산
    # return은 past, current, future이고, 우리 규약과 response에 끼워넣음

    if key in protected_order_keys(robot_state): 
        return "past"

    field_section = physical_section_for_order_key(key) # 이 주문 field가 실제로 투입되는 물리 section
    # domain에는 있는데 section mapping이 빠진 field가 들어와도 여기서 바로 죽지는 않게 둠
    if field_section is None:
        return "future"

    
    field_index = SECTION_ORDER.index(field_section) # 대상 field의 물리 section 순번
    current_index = SECTION_ORDER.index(robot_state["section"]) # 현재 로봇 section 순번

    if field_index == current_index:
        return "current"

    return "past" if field_index < current_index else "future" 

def allowed_order_fields(section: str) -> list[str]:
    # 여기 section은 실제 투입 위치가 아니라 사용자한테 주문받는 대화 section임
    # 한 section에 속하는 주문 field를 반환

    if section == "noodle":
        return ["sauce", "noodle_type", "noodle_portion"]

    if section == "veggie":
        return ["toppings.양파", "toppings.버섯"]

    if section == "meat":
        return ["toppings.소시지", "toppings.게살"]

    if section == "extra":
        return ["toppings.치즈", "toppings.페퍼론치노"]

    return []

def section_execution_items(order: dict, section: str) -> list[dict]:
    # 현재 물리 section에서 실행할 주문값을 반환한다. 노드 task 생성과 Response 입력이 같은 계산을 쓴다.
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
        topping = key.split(".", 1)[1]
        amount = order["toppings"].get(topping)

        if amount is not None and amount != "none":
            items.append({"item": topping, "amount": amount})

    return items

def recommendation_allowed_fields(scope: str, current_section: str) -> list[str]:
    # scope=current/remaining/all을 대화 section 목록으로 바꾼 뒤 field를 평탄화함
    # 추천 범위에서 후보로 만들 수 있는 현재·미래 field를 계산한다.

    order_sections = ("noodle", "veggie", "meat", "extra") # 추천 가능한 대화 section 순서

    if scope == "current":
        sections = (current_section,) # 현재 대화 section만 추천
    elif scope == "all":
        sections = order_sections # 전체 대화 section 추천
    elif current_section in order_sections:
        sections = order_sections[order_sections.index(current_section):] # 현재부터 남은 section 추천
    else:
        sections = () # 주문 section이 아니면 추천 대상 없음

    fields = [] # scope 안에서 추천 가능한 canonical field 목록

    for section in sections:
        for field in allowed_order_fields(section):
            if field not in fields:
                fields.append(field)

    return fields


def validate_recommendation_proposal(session: dict, proposal: dict, robot_state: dict, allowed_fields: list[str]) -> dict:
    # allowed 필드는 위에서 나옴
    # 실제 반환은 항상 dict임: proposal은 통과값, accepted_fields는 통과 field, rejected_fields는 field/reason 목록
    # 추천값의 범위, 물리적 시점, 기존 선택, 제약 충돌 검사용도 ㅇㅇ
    # return은 항상 안전 추천과 통과/거절 field를 담은 dict

    safe_proposal = new_decision()["order_patch"] # 검사를 통과한 추천값만 담을 빈 주문 patch
    protected = set(protected_order_keys(robot_state)) # 이미 지나갔거나 실행돼 추천 불가능한 field
    accepted_fields = [] # 추천값 중 실제 적용 가능한 field 목록
    rejected_fields = [] # 거절한 추천 field와 거절 reason 목록

    for field in ("sauce", "noodle_type", "noodle_portion"):
        value = proposal[field] # Recommendation Agent가 제안한 scalar 값

        if value is None:
            continue

        if field not in allowed_fields:
            rejected_fields.append({"field": field, "reason": "outside_scope"}) # field=거절 대상, reason=추천 scope 밖
        elif field in protected:
            rejected_fields.append({"field": field, "reason": "physical_state"}) # 이미 투입 중이거나 완료된 field
        elif session["order"][field] is not None:
            rejected_fields.append({"field": field, "reason": "already_selected"}) # 사용자가 이미 직접 고른 field
        elif strongest_restriction_for_item(session["order"], value) is not None:
            rejected_fields.append({"field": field, "reason": "restriction_conflict"}) # 활성 restriction과 충돌하는 추천
        else:
            safe_proposal[field] = value
            accepted_fields.append(field)

    for topping, amount in proposal["toppings"].items():
        field = f"toppings.{topping}" # 토핑 이름을 canonical field 이름으로 변환

        if amount == "none":
            rejected_fields.append({"field": field, "reason": "proposal_cannot_delete"}) # 추천 Agent의 기존 메뉴 삭제는 금지
        elif field not in allowed_fields:
            rejected_fields.append({"field": field, "reason": "outside_scope"}) # field=거절 대상, reason=추천 scope 밖
        elif field in protected:
            rejected_fields.append({"field": field, "reason": "physical_state"}) # 이미 투입 중이거나 완료된 field
        elif topping in session["order"]["toppings"]:
            rejected_fields.append({"field": field, "reason": "already_selected"}) # 사용자가 이미 직접 고른 field
        elif strongest_restriction_for_item(session["order"], topping) is not None:
            rejected_fields.append({"field": field, "reason": "restriction_conflict"}) # 활성 restriction과 충돌하는 추천
        else:
            safe_proposal["toppings"][topping] = amount
            accepted_fields.append(field)

    return {
        "proposal": safe_proposal, # 검사를 통과한 추천 주문 patch
        "accepted_fields": accepted_fields, # 실제 proposal에 남은 field 목록
        "rejected_fields": rejected_fields, # 거절한 field와 reason 목록
    }

    # 같은 소스도 대화에서는 noodle에서 받지만 실제 로봇 투입 위치는 sauce일 수 있음

def physical_section_for_order_key(key: str) -> str | None:
    # 주문 field가 실제로 투입되는 section을 반환

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

#################### Decision semantic 검사 -> 틀린 field만 repair 대상으로 넘김 ##########################3
def find_invalid_order_fields(decision: dict, robot_state: dict) -> list[str]:
    # 정상 Schema와 mapping이 맞으면 항상 빈 list가 나와야함. robot_state는 현재 호출 계약 유지용
    # 메뉴 field가 아닌 의미 오류를 찾음
    # return은 잘못된 field

    patch = decision["order_patch"] # 이번 발화에서 바꾸려는 주문값
    invalid = [] # semantic 검사를 통과하지 못한 field 위치 목록

    for field in ("sauce", "noodle_type", "noodle_portion"):
        if patch[field] is not None and physical_section_for_order_key(field) is None:
            invalid.append(field)

    for topping in patch["toppings"]:
        field = f"toppings.{topping}" # 토핑 이름을 canonical field 이름으로 변환

        if physical_section_for_order_key(field) is None:
            invalid.append(field)

    return invalid

# 제약 충돌 확인(queries)
def find_invalid_queries(decision: dict) -> list[str]:
    # Schema만으로 표현하기 어려운 query type-target 조합만 여기서 확인함
    # XGrammar가 허용한 query 구조 안에서 type-target 조합만 추가로 검사
    invalid = [] # semantic 검사를 통과하지 못한 field 위치 목록
    scalar_targets = {"sauce", "noodle_type", "noodle_portion"} # order_field query가 가리킬 수 있는 값
    restriction_targets = {*SAUCES, *NOODLE_TYPES, *TOPPINGS, *RESTRICTION_CATEGORY_ITEMS.keys()} # restriction_status가 가리킬 수 있는 메뉴·카테고리

    for index, query in enumerate(decision["queries"]):
        query_type = query["type"] # 사용자가 물어본 상태 종류
        target = query.get("target") # 질문 대상 field·메뉴. 필요 없는 query면 None

        if query_type in (
            "order_status",
            "recommendation_status",
            "robot_status",
            "robot_completed",
            "robot_failure",
        ):
            valid = target is None # 이 query 종류는 target이 없어야 정상
        elif query_type == "order_field":
            valid = target in scalar_targets # scalar 주문 field만 허용
        elif query_type == "order_item":
            valid = target in TOPPINGS # 실제 토핑 이름만 허용
        elif query_type == "restriction_status":
            valid = target is None or target in restriction_targets # 전체 조회 또는 지원 대상 조회만 허용
        else:
            valid = False # 정의되지 않은 query type은 거절

        if not valid:
            invalid.append(f"queries[{index}]")

    return invalid

def decision_has_task_semantics(decision: dict) -> bool:
    # mentions와 route 자체는 대화 문맥이고 주문·상태 조회·실행 semantic에는 포함하지 않는다.
    patch = decision["order_patch"]
    has_order_patch = (
        patch["sauce"] is not None
        or patch["noodle_type"] is not None
        or patch["noodle_portion"] is not None
        or bool(patch["toppings"])
    )

    return bool(
        has_order_patch
        or decision["restriction_options"]
        or decision["preference_options"]
        or decision["recommendation"]["action"] != "none"
        or decision["commit"]
        or decision["confirmation"] != "none"
        or decision["queries"]
    )


def find_route_consistency_errors(decision: dict) -> list[str]:
    # general의 mutation과 task/mixed의 빈 의미를 찾아 Decision repair 대상으로 돌린다.
    route = decision["route"]
    has_task_semantics = decision_has_task_semantics(decision)
    has_task_intent = has_task_semantics or decision["understanding"] == "clarify"
    errors = []

    if route == "general" and has_task_semantics:
        errors.append("general_has_task_semantics")
    elif route == "mixed" and not has_task_intent:
        errors.append("mixed_missing_task_semantics")
    elif route == "task" and not has_task_intent:
        errors.append("task_missing_task_semantics")

    return errors

# user_text와 mention의 공백 차이만 제거한다. alias·유사어·형태소 추측은 하지 않는다.
def _compact_surface_text(value: str) -> str:
    return "".join(value.split())

def has_explicit_commit_intent(user_text: str) -> bool:
    # 범용 자연어 parser가 아니라 합의된 실행 표현만 허용하는 좁은 positive gate이다.
    # 질문·제안·부정 marker를 먼저 검사한 뒤 명시적 실행 표현을 확인한다.
    compact_user_text = _compact_surface_text(user_text)

    if any(
        marker in compact_user_text
        for marker in BLOCKED_COMMIT_MARKERS
    ):
        return False

    return any(
        _compact_surface_text(phrase) in compact_user_text
        for phrase in EXPLICIT_COMMIT_PHRASES
    )

def _jamo_units(text: str) -> list:
    # 한글 음절을 초성·중성·종성으로 나눠 STT의 받침·모음 오인식을 작은 거리로 계산한다.
    units = []

    for char in text:
        offset = ord(char) - HANGUL_SYLLABLE_BASE

        if 0 <= offset < HANGUL_SYLLABLE_COUNT:
            units.append(("initial", offset // 588))
            units.append(("medial", (offset % 588) // 28))

            if offset % 28:
                units.append(("final", offset % 28))
        else:
            units.append(("char", char))

    return units

def _edit_distance(left: list, right: list) -> int:
    previous = list(range(len(right) + 1))

    for left_index, left_unit in enumerate(left, 1):
        current = [left_index]

        for right_index, right_unit in enumerate(right, 1):
            current.append(min(
                previous[right_index] + 1,
                current[right_index - 1] + 1,
                previous[right_index - 1] + (left_unit != right_unit),
            ))

        previous = current

    return previous[-1]

def menu_surface_match(user_text: str, menu_name: str) -> tuple[float, str]:
    # menu_name과 가장 비슷한 발화 구간과 그 자모 편집거리 비율을 반환한다.
    # 메뉴명 길이 -1 ~ +1 글자 구간만 비교하므로 긴 문장 전체와 비교해 거리가 희석되지 않는다.
    text = _compact_surface_text(user_text)
    target = _compact_surface_text(menu_name)
    target_units = _jamo_units(target)

    if not target_units:
        return 1.0, ""

    best = (1.0, "")

    for length in range(max(1, len(target) - 1), len(target) + 2):
        for start in range(len(text) - length + 1):
            span = text[start:start + length]
            distance = _edit_distance(_jamo_units(span), target_units) / len(target_units)

            if distance < best[0]:
                best = (distance, span)

    return best

def menu_surface_distance(user_text: str, menu_name: str) -> float:
    return menu_surface_match(user_text, menu_name)[0]

def menu_grounding_distance(user_text: str, menu_name: str) -> float:
    # 그 구간이 다른 메뉴명에 더 가까우면 이 메뉴의 근거가 아니다. 예: '넓은면' 구간은 얇은면(0.2)의 근거가 아니다.
    distance, span = menu_surface_match(user_text, menu_name)

    for other in SUPPORTED_MENU_NAMES:
        if other != menu_name and menu_surface_distance(span, other) < distance:
            return 1.0

    return distance

def closest_menu_name(surface: str) -> tuple[str | None, float]:
    # 사용자 표면형과 가장 가까운 domain 메뉴명을 찾는다. 짧은 표면형('면')은 메뉴명 안에서도 찾는다.
    best_name = None
    best_distance = 1.0

    for name in SUPPORTED_MENU_NAMES:
        distance = min(
            menu_surface_distance(surface, name),
            menu_surface_distance(name, surface),
        )

        if distance < best_distance:
            best_name = name
            best_distance = distance

    return best_name, best_distance

def canonical_mentions(mentions: list[str]) -> list[str]:
    # focus에 저장할 mention을 가까운 domain 메뉴명으로 바꾼다. 애매하면 사용자 표면형 그대로 둔다.
    canonical = []

    for mention in mentions:
        name, distance = closest_menu_name(mention)
        value = name if name is not None and distance <= MENU_GROUNDED_MAX_DISTANCE else mention

        if value not in canonical:
            canonical.append(value)

    return canonical

def unsupported_mentions(mentions: list[str]) -> list[str]:
    # 어떤 domain 메뉴의 표면 변형으로도 볼 수 없는 사용자 표면형만 '제공하지 않는 메뉴' 후보로 반환한다.
    # 짧은 단어는 거리 비율이 쉽게 0.5 근처로 가므로 무관 기준(0.65)이 아니라 인정 기준(0.35)으로 판단한다.
    return [
        mention for mention in mentions
        if closest_menu_name(mention)[1] > MENU_GROUNDED_MAX_DISTANCE
    ]

def _menu_references(decision: dict) -> list[dict]:
    # Decision 안에서 메뉴명을 가리키는 값만 모은다. 양(low/normal/high)은 발화 근거 검사 대상이 아니다.
    references = []
    patch = decision["order_patch"]

    for field in ("sauce", "noodle_type"):
        if patch[field] is not None:
            references.append({"kind": field, "target": patch[field]})

    for topping in patch["toppings"]:
        references.append({"kind": "topping", "target": topping})

    for index, option in enumerate(decision["restriction_options"]):
        references.append({
            "kind": f"restriction_{option['action']}",
            "target": option["target"],
            "index": index,
        })

    return references

def grade_menu_grounding(
    user_text: str,
    decision: dict,
    context_targets: list[str] | None = None,
) -> dict:
    # 모델이 낸 메뉴명이 이번 발화에 근거가 있는지 자모 거리로만 판정한다.
    # context_targets는 지시어·확인 질문·안전 확인처럼 발화 밖 상태가 이미 확정한 대상이다.
    # 반환: uncertain=사용자에게 확인할 값, ungrounded=근거 없는 값
    context = set(context_targets or [])
    # 실행 게이트와 같은 질문·부정 표지를 재사용한다. "게살 먹어도 돼?"가 알러지를 풀지 않게 한다.
    compact_user_text = _compact_surface_text(user_text)
    is_question_or_negative = any(marker in compact_user_text for marker in BLOCKED_COMMIT_MARKERS)

    for collection, items in ORDER_COLLECTION_TARGETS.items():
        if menu_surface_distance(user_text, collection) <= MENU_GROUNDED_MAX_DISTANCE:
            context.update(items)

    uncertain = []
    ungrounded = []

    for reference in _menu_references(decision):
        if (
            reference["kind"] == "restriction_remove"
            and is_question_or_negative
            and decision["restriction_options"][reference["index"]]["type"] in SAFETY_RESTRICTION_REASONS
        ):
            ungrounded.append(reference)
            continue

        if reference["target"] in context:
            continue

        distance = menu_grounding_distance(user_text, reference["target"])

        if reference["kind"] == "restriction_add":
            # 제한 등록은 잘못 들어가도 더 안전한 방향이므로 무관한 값만 버린다.
            if distance >= MENU_UNRELATED_MIN_DISTANCE:
                ungrounded.append(reference)
        elif reference["kind"] == "restriction_remove":
            # 제한 해제는 안전과 직결되므로 확실히 들린 대상만 허용한다.
            if distance > MENU_GROUNDED_MAX_DISTANCE:
                ungrounded.append(reference)
        elif distance >= MENU_UNRELATED_MIN_DISTANCE:
            ungrounded.append(reference)
        elif distance > MENU_GROUNDED_MAX_DISTANCE:
            uncertain.append(reference)

    return {"uncertain": uncertain, "ungrounded": ungrounded}

def decision_targets_are_direct(user_text: str, decision: dict) -> bool:
    # Decision의 메뉴 대상이 모두 이번 발화에 근거가 있으면 지시어("둘 다", "그거")를 다시 해석할 필요가 없다.
    references = _menu_references(decision)
    return bool(references) and all(
        menu_grounding_distance(user_text, reference["target"]) <= MENU_GROUNDED_MAX_DISTANCE
        for reference in references
    )

def compact_order_patch(patch: dict) -> dict:
    # Response에 넘길 때 비어 있는 필드를 뺀다. 빈 필드를 질문 대상으로 오해하지 않게 한다.
    compact = {field: patch[field] for field in ("sauce", "noodle_type", "noodle_portion") if patch.get(field) is not None}

    if patch.get("toppings"):
        compact["toppings"] = copy.deepcopy(patch["toppings"])

    return compact

def drop_menu_references(decision: dict, references: list[dict]) -> dict:
    # 근거가 없거나 확인이 필요한 값만 빼고 같은 발화의 나머지 semantic은 보존한다.
    cleaned = copy.deepcopy(decision)
    dropped_restrictions = set()

    for reference in references:
        if reference["kind"] in ("sauce", "noodle_type"):
            cleaned["order_patch"][reference["kind"]] = None
        elif reference["kind"] == "topping":
            cleaned["order_patch"]["toppings"].pop(reference["target"], None)
        else:
            dropped_restrictions.add(reference["index"])

    cleaned["restriction_options"] = [
        option for index, option in enumerate(cleaned["restriction_options"])
        if index not in dropped_restrictions
    ]
    return cleaned

def grounded_mentions(user_text: str, mentions: list[str]) -> list[str]:
    # Decision이 낸 mention 중 현재 사용자 발화에 비슷한 구간이 있는 값만 순서대로 보존한다.
    grounded = []

    for mention in mentions:
        if not isinstance(mention, str) or not _compact_surface_text(mention):
            continue

        if menu_surface_distance(user_text, mention) <= MENU_GROUNDED_MAX_DISTANCE:
            grounded.append(copy.deepcopy(mention))

    return grounded


def build_safe_clarify_decision(decision: dict, user_text: str) -> dict:
    # repair가 재실패하면 첫 Decision의 route와 실제 발화에 grounded된 mentions만 보존한다.
    # new_decision() 기본값을 사용하므로 모든 mutation/query semantic은 fail-closed로 폐기된다.
    safe = new_decision()
    safe["route"] = decision["route"]
    safe["mentions"] = grounded_mentions(
        user_text,
        decision["mentions"],
    )
    safe["understanding"] = "clarify"
    return safe

"""
파이썬의 all() 함수는 반복 가능한 객체(iterable)의 모든 요소가 참(True)인지 확인하고, 모두 참일 때만 True를 반환하는 내장 함수
"""

def reference_targets_are_supported(resolved_targets: list[str]) -> bool:
    # Decision JSON을 확장하지 않고 focus에서 해석된 문자열이 현재 domain 정본에 있는지만 검사한다.
    # alias나 유사어 보정은 하지 않으므로 햄, 페퍼로니, 떡볶이 등은 그대로 unsupported가 된다.
    supported_targets = {*SAUCES,*NOODLE_TYPES,*TOPPINGS,*RESTRICTION_CATEGORY_ITEMS.keys()}

    return all(target in supported_targets for target in resolved_targets)



def reference_targets_match_decision(
    decision: dict,
    resolved_targets: list[str],
) -> bool:
    # resolved target은 Python이 focus에서 계산한 지시어 대상이다.
    # 같은 발화에서 사용자가 직접 말한 새 대상은 decision["mentions"]로 별도 허용한다.
    # 예: focus=["치즈"], "그거 빼고 버섯 많이"는 치즈와 버섯이 함께 있어야 정상이다.
    supported_targets = {
        *SAUCES,
        *NOODLE_TYPES,
        *TOPPINGS,
        *RESTRICTION_CATEGORY_ITEMS.keys(),
    }

    if not reference_targets_are_supported(resolved_targets):
        return False

    patch = decision["order_patch"]
    decision_targets = []

    for field in ("sauce", "noodle_type"):
        if patch[field] is not None:
            decision_targets.append(patch[field])

    decision_targets.extend(patch["toppings"].keys())

    for option in decision["restriction_options"]:
        decision_targets.append(option["target"])

    for query in decision["queries"]:
        target = query.get("target")

        if target in supported_targets:
            decision_targets.append(target)

    unique_decision_targets = set(decision_targets)
    unique_resolved_targets = set(resolved_targets)

    # 현재 user_text에서 직접 언급했다고 Decision이 기록한 supported target만 추가 대상으로 허용한다.
    explicit_supported_targets = {
        mention
        for mention in decision["mentions"]
        if mention in supported_targets
    }
    allowed_targets = unique_resolved_targets | explicit_supported_targets

    # reference 대상은 전부 semantic에 있어야 하고,
    # 그 외 semantic target은 현재 발화의 직접 mention으로 확인된 값만 허용한다.
    return (
        unique_resolved_targets.issubset(unique_decision_targets)
        and unique_decision_targets.issubset(allowed_targets)
    )

####################### restriction 우선순위와 주문 충돌 계산 ################################33
def restriction_priority(reason: str) -> int:
    # restriction reason을 안전 우선순위를 숫자로 변환(알러지, 식이제약, 단순 기호 등)
    return RESTRICTION_PRIORITY.index(reason)

def strongest_restriction_for_item(order: dict, item: str) -> dict | None:
    # 한 메뉴에 적용되는 restriction 중 가장 강한 하나를 찾음(중복 방지 용도)

    strongest = None # 현재까지 찾은 restriction 중 우선순위가 가장 높은 값

    for restriction in order["restrictions"]:
        blocked_items = RESTRICTION_CATEGORY_ITEMS.get(restriction["target"], (restriction["target"],)) # 카테고리 restriction을 실제 메뉴 목록으로 펼친 값

        if item not in blocked_items:
            continue

        if strongest is None or restriction_priority(restriction["reason"]) < restriction_priority(strongest["reason"]):
            strongest = restriction

    return copy.deepcopy(strongest)

def find_restriction_conflicts(order: dict) -> list[dict]:
    # dislike까지 충돌 목록에는 남기되, 실제 실행 차단 여부는 evaluate_runtime_policy에서 reason별로 나눔
    # 최종 주문과 restriction이 충돌하는 메뉴를 찾음

    selected_items = [] # 현재 주문에서 실제 선택된 field와 메뉴 묶음

    if order["sauce"] is not None:
        selected_items.append(("sauce", order["sauce"]))

    if order["noodle_type"] is not None:
        selected_items.append(("noodle_type", order["noodle_type"]))

    for topping in order["toppings"]:
        selected_items.append((f"toppings.{topping}", topping))

    conflicts = [] # 발견한 주문-restriction 충돌 목록

    for key, item in selected_items:
        restriction = strongest_restriction_for_item(order, item) # 이 메뉴에 적용되는 가장 강한 restriction

        if restriction is not None:
            conflicts.append({"key": key,"item": item, "restriction": restriction}) # key=주문 field, item=메뉴, restriction=충돌 제한

    return conflicts


######################## 이미 실행됐거나 지나간 주문값 보호 ############################################### -> 이미 한거 빼고 싶어 이럴때 ㅇㅇ
def changed_order_keys(decision: dict) -> list[str]:
    # dict이 수정하려는 주문 key를 반환 -> 나중에 덮어 쓸거 ㅇㅇ 크게는 그냥 소스, 면, 토핑류

    changed = [] # 이번 Decision이 수정하려는 canonical 주문 field 목록
    patch = decision["order_patch"] # 이번 발화에서 바꾸려는 주문값

    for field in ("sauce", "noodle_type", "noodle_portion"):
        if patch[field] is not None:
            changed.append(field)

    for topping in patch["toppings"]:
        changed.append(f"toppings.{topping}")

    return changed


def protected_order_keys(robot_state: dict) -> list[str]:
    # 과거 section + active task + completed task를 합쳐서 수정 금지 key 목록으로 만듦
    # 지나갔거나 active·completed인 주문 field를 찾아서 수정 못하게 막음.

    protected = [] # 과거·active·completed라 수정하면 안 되는 field 목록
    current_index = SECTION_ORDER.index(robot_state["section"]) # 현재 로봇 section 순번

    order_keys = [ # 물리 section을 추적하는 전체 canonical 주문 field
        "noodle_type",
        "noodle_portion",
        "toppings.양파",
        "toppings.버섯",
        "toppings.소시지",
        "toppings.게살",
        "toppings.치즈",
        "toppings.페퍼론치노",
        "sauce",
    ]

    for key in order_keys:
        field_section = physical_section_for_order_key(key)

        if SECTION_ORDER.index(field_section) < current_index:
            protected.append(key)

    tasks = list(robot_state["completed_tasks"]) # 이미 완료된 task 복사본. 아래에서 active task도 합침

    if robot_state["active_task"] is not None:
        tasks.append(robot_state["active_task"])

    for task in tasks:
        item = task["class"] # task가 실제로 투입한 메뉴 이름

        if item in NOODLE_TYPES:
            if "noodle_type" not in protected:
                protected.append("noodle_type")
            if "noodle_portion" not in protected:
                protected.append("noodle_portion")

        elif item in TOPPINGS:
            key = f"toppings.{item}"

            if key not in protected:
                protected.append(key)

        elif item in SAUCES and "sauce" not in protected:
            protected.append("sauce")

    return protected



def restore_protected_order_values(previous_session: dict, candidate_session: dict, robot_state: dict) -> dict:
    # candidate에 섞인 과거·active·completed 변경만 이전 canonical 값으로 복구함
    # 원본 session은 안건드리고 복구한 복사본을 반환함
    restored = copy.deepcopy(candidate_session) # 보호값만 되돌릴 candidate session 복사본
    previous_order = previous_session["order"] # 이번 턴 적용 전 canonical 주문
    restored_order = restored["order"] # 보호값을 복구할 candidate 주문

    for key in protected_order_keys(robot_state):
        if key.startswith("toppings."):
            item = key.split(".", 1)[1] # toppings. 접두사를 뺀 실제 토핑 이름

            if item in previous_order["toppings"]:
                restored_order["toppings"][item] = previous_order["toppings"][item]
            else:
                restored_order["toppings"].pop(item, None)

        else:
            restored_order[key] = copy.deepcopy(previous_order[key])

    return restored


def find_new_physical_restriction_conflicts(previous_session: dict, candidate_session: dict, robot_state: dict) -> list[dict]:
    # 기존 restriction 전체가 아니라 이번 턴에 새로 추가된 강한 restriction만 검사함
    # 이번 턴에 추가된 안전 restriction과 이미 투입된 task의 충돌을 찾는다 -> 이미 담았으니 섭취에 유의하세요 용도

    previous_restrictions = previous_session["order"]["restrictions"] # 이번 턴 전부터 있던 restriction
    added_restrictions = [] # 이번 턴에 새로 추가된 강한 restriction만 모음

    for restriction in candidate_session["order"]["restrictions"]:
        if restriction not in previous_restrictions and restriction["reason"] in ("allergy", "cannot_eat", "dietary_rule"):
            added_restrictions.append(restriction)

    tasks = list(robot_state["completed_tasks"]) # 이미 완료된 task 복사본. 아래에서 active task도 합침

    if robot_state["active_task"] is not None:
        tasks.append(robot_state["active_task"])

    conflicts = [] # 발견한 주문-restriction 충돌 목록
    seen_items = [] # 같은 투입 메뉴 충돌을 중복 경고하지 않기 위한 목록

    for task in tasks:
        item = task["class"] # task가 실제로 투입한 메뉴 이름

        if item in seen_items:
            continue

        for restriction in added_restrictions:
            blocked_items = RESTRICTION_CATEGORY_ITEMS.get(restriction["target"], (restriction["target"],))

            if item not in blocked_items:
                continue

            if item in NOODLE_TYPES:
                key = "noodle_type" # 면 task가 대응되는 canonical field
            elif item in TOPPINGS:
                key = f"toppings.{item}" # 토핑 task가 대응되는 canonical field
            elif item in SAUCES:
                key = "sauce" # 소스 task가 대응되는 canonical field
            else:
                continue

            conflicts.append({"key": key, "item": item,"restriction": strongest_restriction_for_item(candidate_session["order"], item)}) # 이미 투입된 field·메뉴·새 제한
            seen_items.append(item)
            break

    return conflicts

# 위는 바로 아래 용도임 ㅇㅇ
######################### Response와 action_history에 넘길 결과 정리 ###################################3
def build_completed_restriction_warning(conflicts: list[dict]) -> str:
    # 이미 투입된 재료와 새 restriction이 충돌할 때 쓰는 고정 안전 문장
    items = [] # 사용자에게 경고할 이미 투입된 메뉴 이름

    for conflict in conflicts:
        if conflict["item"] not in items:
            items.append(conflict["item"])

    item_text = ", ".join(items) # 경고 문장에 넣을 메뉴 이름 문자열
    return f"{item_text} 재료는 이미 담긴 상태입니다. 등록한 식이 제약과 충돌하므로 섭취하지 않도록 주의해 주세요."


def build_completed_modification_warning(blocked_changes: list[str]) -> str:
    # 빼고싶어 방어 용도 ㅇㅇ 
    field_names = { # canonical field를 사용자용 한국어 이름으로 바꾸는 표
        "sauce": "소스",
        "noodle_type": "면 종류",
        "noodle_portion": "면 양",
        "toppings.양파": "양파",
        "toppings.버섯": "버섯",
        "toppings.소시지": "소시지",
        "toppings.게살": "게살",
        "toppings.치즈": "치즈",
        "toppings.페퍼론치노": "페퍼론치노",
    }
    blocked_items = [field_names.get(key, key) for key in blocked_changes] # 변경이 막힌 field의 사용자용 이름
    blocked_text = ", ".join(blocked_items) # 거절 문장에 넣을 항목 문자열

    return f"{blocked_text} 항목은 작업이 시작됐거나 선택 단계가 지나 지금 변경할 수 없어요. 실제로 담긴 재료와는 구분해 주세요."


def build_applied_changes(previous_session: dict, current_session: dict) -> dict:
    # 반환 field: order_changes, restriction_added/removed, preference_added/removed
    # 차이 비교 용도

    previous_order = previous_session["order"] # 이번 턴 적용 전 canonical 주문
    current_order = current_session["order"] # 이번 턴 적용 후 canonical 주문
    order_changes = {} # 실제 값이 달라진 canonical field와 변경값

    for field in ("sauce", "noodle_type", "noodle_portion"):
        if previous_order[field] != current_order[field]:
            order_changes[field] = current_order[field]

    for topping in TOPPINGS:
        previous_amount = previous_order["toppings"].get(topping) # 변경 전 토핑 양. 없으면 None
        current_amount = current_order["toppings"].get(topping) # 변경 후 토핑 양. 삭제됐으면 None

        if previous_amount != current_amount:
            order_changes[f"toppings.{topping}"] = current_amount

    restriction_added = [] # 이번 턴에 새로 등록된 restriction
    restriction_removed = [] # 이번 턴에 철회된 restriction

    for restriction in current_order["restrictions"]:
        if restriction not in previous_order["restrictions"]:
            restriction_added.append(copy.deepcopy(restriction))

    for restriction in previous_order["restrictions"]:
        if restriction not in current_order["restrictions"]:
            restriction_removed.append(copy.deepcopy(restriction))

    preference_added = [] # 이번 턴에 새로 등록된 자유 취향
    preference_removed = [] # 이번 턴에 철회된 자유 취향

    for preference in current_session["preferences"]:
        if preference not in previous_session["preferences"]:
            preference_added.append(copy.deepcopy(preference))

    for preference in previous_session["preferences"]:
        if preference not in current_session["preferences"]:
            preference_removed.append(copy.deepcopy(preference))

    return {
        "order_changes": order_changes, # 실제 반영된 주문 변경
        "restriction_added": restriction_added, # 추가된 제한사항
        "restriction_removed": restriction_removed, # 삭제된 제한사항
        "preference_added": preference_added, # 추가된 자유 취향
        "preference_removed": preference_removed, # 삭제된 자유 취향
    }


def build_already_set(previous_session: dict, decision: dict) -> dict:
    # 이번 요청값 중 턴 시작 전에 이미 같은 값이던 항목. 변경이 없을 때 Response가 말할 근거로 쓴다.
    previous_order = previous_session["order"]
    patch = decision["order_patch"]
    order = {}

    for field in ("sauce", "noodle_type", "noodle_portion"):
        if patch[field] is not None and patch[field] == previous_order[field]:
            order[field] = patch[field]

    for topping, amount in patch["toppings"].items():
        if amount != "none" and previous_order["toppings"].get(topping) == amount:
            order[f"toppings.{topping}"] = amount

    restrictions = [
        {"target": option["target"], "reason": option["type"]}
        for option in decision["restriction_options"]
        if option["action"] == "add"
        and {"target": option["target"], "reason": option["type"]} in previous_order["restrictions"]
    ]

    if not order and not restrictions:
        return {}

    return {"order": order, "restrictions": restrictions}

def build_future_changes(applied_changes: dict, robot_state: dict) -> list[dict]:
    # 실제 반영된 주문 변경 중 미래 section에서 실행할 값만 분리

    future_changes = [] # 반영은 됐지만 미래 section에서 실행할 field와 값

    for field, value in applied_changes["order_changes"].items():
        if order_key_timing(field, robot_state) == "future":
            future_changes.append({"field": field, "value": value}) # field=미래 주문 key, value=저장된 변경값

    return future_changes


def build_next_prompt(session: dict, decision: dict, policy: dict, robot_state: dict) -> dict | None:
    # 반환 dict의 type이 Response가 다음에 물어볼 질문 종류가 됨. 물어볼 게 없으면 None
    # Response가 이어 말할 다음 질문의 의미를 Python에서 결정한다.

    pending = session["pending_confirmation"] # 이전 턴에서 사용자 확인을 기다리는 정보 또는 None

    if pending is not None and pending["type"] == "restriction_conflict":
        conflicts = pending.get("conflicts", []) # 확인이 필요한 restriction 충돌 목록
        target = conflicts[0]["item"] if conflicts else None # 사용자에게 먼저 확인할 충돌 메뉴
        return {
            "type": "confirmation_required", # 사용자 승인/거절이 필요한 질문
            "reason": "restriction_conflict", # 안전 restriction 해제 여부를 확인
            "target": target, # 확인 질문에 넣을 충돌 메뉴
        }

    if pending is not None and pending["type"] == "preselected_section":
        return {
            "type": "future_confirmation", # 미래 section에 미리 고른 값 확인
            "section": pending["section"], # 미리 선택된 값이 실행될 section
            "items": copy.deepcopy(pending["items"]), # 사용자에게 확인할 미리 선택된 메뉴
        }

    if policy["execute"]:
        return None

    if policy["reason"] in (
        "understanding",
        "ambiguous_reference",
        "unsupported_reference",
        "physical_state",
        "robot_busy",
        "completed_restriction_conflict",
        "confirmation_rejected",
    ):
        return None

    if session["recommendation"]["phase"] == "proposed":
        return {
            "type": "confirmation_required", # 사용자 승인/거절이 필요한 질문
            "reason": "recommendation_proposal", # 생성된 추천안을 쓸지 확인
            "target": None, # 특정 메뉴 하나가 아니라 추천안 전체가 대상
        }

    if robot_state["section"] == "noodle":
        order = session["order"] # 다음 질문을 정할 때 보는 현재 canonical 주문

        for field in ("noodle_type", "noodle_portion", "sauce"):
            if order[field] is None:
                return {
                    "type": "missing_field", # 실행 전에 빠진 주문값 질문
                    "target": field, # 사용자에게 물어볼 누락 field
                }

    if robot_state["section"] in ("veggie", "meat", "extra") and decision["recommendation"]["action"] == "none":
        selected = False # 현재 토핑 section에 선택된 메뉴가 있는지 여부

        for key in allowed_order_fields(robot_state["section"]):
            topping = key.split(".", 1)[1] # canonical field에서 실제 토핑 이름만 분리

            if topping in session["order"]["toppings"]:
                selected = True
                break

        if not selected and session["recommendation"]["phase"] == "idle":
            return {
                "type": "recommendation_offer", # 비어 있는 section 추천 제안
                "scope": robot_state["section"], # 추천을 제안할 현재 section
            }

    # 이번 턴에 현재 section 주문을 바꿨고 바로 담을 수 있으면 실행 여부를 묻는다. 질문의 존재는 state가 정한다.
    section = robot_state["section"]
    items = section_execution_items(session["order"], section)
    changed_here = set(changed_order_keys(decision)) & set(allowed_order_fields(section))

    if (
        policy["reason"] == "state_update_only"
        and items
        and changed_here
        and not missing_current_section(session["order"], section)
    ):
        return {
            "type": "execution_offer", # 반영한 현재 section을 바로 담을지 확인
            "section": section,
            "targets": [entry["item"] for entry in items],
        }

    return None


def build_turn_action_event(applied_changes: dict, future_changes: list[dict]) -> dict | None:
    # full session 대신 이번 턴에 실제 바뀐 값만 작은 event로 기록함
    # 실제 적용된 결과만 다음 턴에서 참조할 작은 기록화(풀 세션 X)

    has_changes = bool( # 실제 반영된 변경이 하나라도 있는지 여부
        applied_changes["order_changes"]
        or applied_changes["restriction_added"]
        or applied_changes["restriction_removed"]
        or applied_changes["preference_added"]
        or applied_changes["preference_removed"]
    )

    if not has_changes:
        return None

    event = { # 다음 턴 문맥에 남길 최소 변경 기록
        "type": "turn_applied", # 실제 변경이 반영된 한 턴 기록
        "order_changes": copy.deepcopy(applied_changes["order_changes"]), # 반영된 주문 변경
        "restriction_added": copy.deepcopy(applied_changes["restriction_added"]), # 추가된 제한사항
        "restriction_removed": copy.deepcopy(applied_changes["restriction_removed"]), # 삭제된 제한사항
        "preference_added": copy.deepcopy(applied_changes["preference_added"]), # 추가된 취향
        "preference_removed": copy.deepcopy(applied_changes["preference_removed"]), # 삭제된 취향
    }

    if future_changes:
        event["future_changes"] = copy.deepcopy(future_changes) # 미래 section에서 실행할 변경만 선택적으로 기록

    return event

############################ 현재 section 실행 가능 여부를 마지막으로 결정 ############################
def missing_current_section(order: dict, section: str) -> list[str]:
    # 현재 section 실행에 필요한 선택값을 반환하지만, 나머지 토핑은 넘어갈 수 있음.

    if section != "noodle":
        return []

    missing = [] # 현재 section 실행 전에 더 받아야 하는 필수 field

    for field in ("sauce", "noodle_type", "noodle_portion"):
        if order[field] is None:
            missing.append(field)

    return missing

def evaluate_runtime_policy(session: dict, decision: dict, robot_state: dict, physical_conflicts: list[dict] | None = None) -> dict:
    # 검사 순서가 우선순위임: 이해 실패 -> 물리 충돌 -> 변경 금지 -> 안전 충돌 -> 실행 의도 -> 필수값 -> 로봇 상태
    # 반환 dict의 각 field 의미는 아래 key 바로 옆에 적음. session은 직접 수정하지 않음

    if decision["understanding"] == "clarify":
        return {
            "status": "clarify", # 사용자에게 추가 설명이 필요한 상태
            "reason": "understanding", # Decision 의미를 안전하게 특정하지 못함
            "execute": False, # 이번 턴 로봇 실행 금지
            "conflicts": [], # 충돌 없음
        }

    changed = changed_order_keys(decision) # 이번 발화가 수정하려고 한 주문 field
    protected = protected_order_keys(robot_state) # 물리적으로 이미 수정할 수 없는 주문 field
    blocked_changes = [] # changed와 protected가 겹친 field

    for key in changed:
        if key in protected:
            blocked_changes.append(key)

    if physical_conflicts:
        return {
            "status": "warning", # 상태는 보존하지만 사용자 안전 경고 필요
            "reason": "completed_restriction_conflict", # 이미 투입된 메뉴와 새 제한이 충돌
            "execute": False, # 이번 턴 로봇 실행 금지
            "blocked_changes": blocked_changes, # 되돌린 과거·실행완료 field
            "conflicts": physical_conflicts, # 이미 투입된 메뉴와 새 제한 충돌
        }

    if blocked_changes:
        return {
            "status": "blocked", # 현재 물리 상태에서 요청 수행 불가
            "reason": "physical_state", # 이미 지나갔거나 실행된 주문값 변경
            "execute": False, # 이번 턴 로봇 실행 금지
            "blocked_changes": blocked_changes, # 되돌린 과거·실행완료 field
            "conflicts": [], # 충돌 없음
        }

    # commit=False여도 canonical 주문 자체가 강한 restriction과 충돌하면 먼저 HITL로 보냄
    # 안전 우선이지만 기존 충돌이 남아있으면 query-only 턴도 막을 수 있어서 과검사 후보는 여기임
    conflicts = find_restriction_conflicts(session["order"]) # 현재 주문 전체의 restriction 충돌
    safety_conflicts = [] # 알레르기·섭취불가·식단규칙 충돌만 추린 목록

    for conflict in conflicts:
        reason = conflict["restriction"]["reason"] # 해당 충돌의 안전 등급

        if reason in ("allergy", "cannot_eat", "dietary_rule") and conflict["key"] not in protected:
            safety_conflicts.append(conflict)

    if safety_conflicts:
        return {
            "status": "hitl", # 사람의 안전 확인이 필요한 상태
            "reason": "restriction_conflict", # 안전 restriction 해제 여부를 확인
            "execute": False, # 이번 턴 로봇 실행 금지
            "conflicts": safety_conflicts, # 사용자 확인이 필요한 강한 안전 충돌
        }

    if decision["confirmation"] == "reject":
        return {
            "status": "pass", # 정책상 정상 처리된 상태
            "reason": "confirmation_rejected", # 사용자가 보류된 실행을 거절
            "execute": False, # 이번 턴 로봇 실행 금지
            "conflicts": conflicts, # 현재 주문에서 발견된 전체 restriction 충돌
        }

    if not decision["commit"]:
        return {
            "status": "pass", # 정책상 정상 처리된 상태
            "reason": "state_update_only", # 상태만 바꾸고 로봇 실행 요청은 없음
            "execute": False, # 이번 턴 로봇 실행 금지
            "conflicts": conflicts, # 현재 주문에서 발견된 전체 restriction 충돌
        }

    missing = missing_current_section(session["order"], robot_state["section"]) # 지금 실행하려면 부족한 필수 주문 field

    if missing:
        return {
            "status": "clarify", # 사용자에게 추가 설명이 필요한 상태
            "reason": "missing_order", # 현재 section 필수 주문값 부족
            "execute": False, # 이번 턴 로봇 실행 금지
            "missing": missing, # 사용자에게 추가로 받을 필수 field
            "conflicts": conflicts, # 현재 주문에서 발견된 전체 restriction 충돌
        }

    if robot_state["active_task"] is not None or robot_state["task_queue"]:
        return {
            "status": "blocked", # 현재 물리 상태에서 요청 수행 불가
            "reason": "robot_busy", # active task나 대기 queue가 존재
            "execute": False, # 이번 턴 로봇 실행 금지
            "conflicts": conflicts, # 현재 주문에서 발견된 전체 restriction 충돌
        }

    return {
        "status": "pass", # 정책 검사를 통과한 상태
        "reason": "execution_allowed", # 모든 검사를 통과해 실행 가능
        "execute": True, # 이번 턴 로봇 실행 허용
        "conflicts": conflicts, # 실행 판단 때 참고한 전체 restriction 충돌
    }
