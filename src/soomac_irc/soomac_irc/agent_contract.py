import copy

from soomac_irc.domain import AMOUNTS, NOODLE_TYPES, RESTRICTION_CATEGORIES, RESTRICTION_REASONS, SAUCES, TOPPINGS


"""
# JSON Schema 문법
type: 값의 자료형. object=dict, array=list, string=str, boolean=bool
properties: object 안에서 사용할 수 있는 key와 각 key의 Schema
enum: 모델이 생성할 수 있는 값을 목록 안으로 제한
required: 항상 출력해야 하는 key. 이 목록에 없으면 해당 key는 생략 가능
additionalProperties=False: properties에 없는 key 생성을 막음
items: array 안에 들어갈 각 원소의 Schema
"""


# For Recommendation Agent(전체 추천용)

ORDER_PATCH_SCHEMA = {
    "type": "object", # dict
    "properties": { # dict 내부의 key, value
        "sauce": {"type": ["string", "null"], "enum": [*SAUCES, None]}, # 문자열 타입, 소스 중 하나로 제한하거나, 아예 안내놓던가(이미 있는 경우 같은거)
        "noodle_type": {"type": ["string", "null"], "enum": [*NOODLE_TYPES, None]},
        "noodle_portion": {"type": ["string", "null"], "enum": [*AMOUNTS, None]},
        "toppings": {
            "type": "object", # 내부는 dict
            "properties": {item: {"type": "string", "enum": [*AMOUNTS, "none"]} for item in TOPPINGS},
            "additionalProperties": False # properties에 없는 키 생성 X
        }, # item은 string으로(양파, 버섯, 치즈, 페퍼론치노, 게살 등)

    },
    "required": ["sauce", "noodle_type", "noodle_portion", "toppings"], # 무조건 있어야하는거 -> properties 내부의 모든 필드 ㅇㅇ
    "additionalProperties": False,
}

# 즉 위는 지금 넣던지 말던지 아무튼 요구하는 properties인 sauce, noodle, noodle_portion, toppings를 무조건 요구하고 나머지는 내보내지 않는 것을 원칙으로 하는 스키마임


# Decision 모델이 실제 발화에 나타난 주문값만 출력하도록 하는 스키마

SPARSE_ORDER_SCHEMA = {
    "type": "object", # dict
    "properties": {
        "sauce": {"type": "string", "enum": list(SAUCES)},
        "noodle_type": {"type": "string", "enum": list(NOODLE_TYPES)},
        "noodle_portion": {"type": "string", "enum": list(AMOUNTS)},
        "toppings":{
            "type": "object", # 딕셔너리 중첩. key는 properties, value는 또 딕셔너리 value안도 dictionary
            "properties": {item: {"type": "string", "enum": [*AMOUNTS, "none"]} for item in TOPPINGS},
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}

# 요구하는건 dictionary(변화한 properties) -> sauce, noodle type, noodle portion, toppings(얘는 당연히 여러겠지용)

DECISION_SCHEMA = {
    "type": "object", # dict
    "properties": {
        "order": SPARSE_ORDER_SCHEMA, # 변화한거
        "restrictions": { # 제한
            "type": "array", # 제한은 여러개가 한 번에 들어올 수 있으니까 list
            "items": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "enum": [*SAUCES, *NOODLE_TYPES, *TOPPINGS, *RESTRICTION_CATEGORIES]}, # 제한은 여러개 가능하게
                    "reason": {"type": "string", "enum": list(RESTRICTION_REASONS)},
                    "action": {"type": "string", "enum": ["add", "remove"]}, # 넣거나 빼거나 ㅇㅇ
                },
                "required": ["target", "reason", "action"], # 제한 대상, 이유, 넣거나 빼거나
                "additionalProperties": False,
            },
        },
        "preferences": { # 취향
            "type": "array", # 취향도 여러개 가능
            "items": {
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                    "action": {"type": "string", "enum": ["add", "remove"]},
                },
                "required": ["value", "action"], # 취향 추가 or 삭제 ㅇㅇ
                "additionalProperties": False,
            },
        },
        "recommendation": { # 추천
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["request", "revise", "cancel", "select"]}, # 새 추천 요청(처음 추천 받았을 때), 기존에 받은 추천 수정, 추천 취소, 기존 추천 선택
                "scope": {"type": "string", "enum": ["current", "remaining", "all"]}, # 지금 세션, 남아 있는거, 그리고 전부
                "criteria": {"type": "string"}, # 판단 범위(취향 같은거) ex : 매운 기준 이런 느낌
            },
            "required": ["action"], # 추천이 들어오면 뭘 할지는 무조건 있어야함
            "additionalProperties": False,
        },
        "commit": {"type": "boolean"}, # 바로 시작 or not(True면 로봇 실행)
        "confirmation": {"type": "string", "enum": ["accept", "reject"]}, # 안전 관련 철회
        "queries": { # 질문 ㅇㅇ. 현재 주문이나 로봇 상태를 실제로 질문 했을 때만(여러 질문이 들어올 수 있어서 복수로 선택하도록 list화)
            "type": "array", # 한 문장에 질문이 여러개일 수도 있음
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["order_status", "order_field", "order_item", "restriction_status", "recommendation_status", "robot_status", "robot_completed", "robot_failure"]},
                    "target": {"type": "string", "enum": ["sauce", "noodle_type", "noodle_portion", *SAUCES, *NOODLE_TYPES, *TOPPINGS, *RESTRICTION_CATEGORIES]},
                },
                "required": ["type"], 
                "additionalProperties": False,
            },
        },
        "clarify": {"type": "boolean"}, 
    },
    # sparse 출력이라 top-level required는 없음 -> 이번 발화랑 상관없는 값은 아예 안내보내도 됨
    "additionalProperties": False,
}

"""
결론적으로 decision model은 
변화한 메뉴,
제한, 
취향, 
추천(넣었다 뺐다 가능)
바로 시작
제한 관련 철회
작업 상태에 대한 질문
의도 결정 - 여기서 의도 결정에서 understanding : clarify(다시 질문해야함), ok(정상처리). 
(지시 대상이 여러 개이거나, 취소 범위가 불분명, 원래 상태가 무엇인지, 지원하지 않는 값을 반환해야 할 때 등)
"""

"""
추천안 사용      → recommendation.action="select"
추천안 취소      → recommendation.action="cancel"
안전 확인 승인   → confirmation="accept"
안전 확인 거절   → confirmation="reject"
"""

# 추천 Agent는 완성된 추천안이랑 추천 이유를 같이 내보냄
RECOMMENDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "proposal": ORDER_PATCH_SCHEMA,
        "reason_tags": {"type": "array", "items": {"type": "string"}}, # 이유
    },
    "required": ["proposal", "reason_tags"], # 추천, 추천 이유
    "additionalProperties": False,
}


# Response Agent는 최종 답변 하나만 내보내면 됨
RESPONSE_SCHEMA = {
    "type": "object", # 최종 답변
    "properties": {"reply": {"type": "string"}},
    "required": ["reply"],
    "additionalProperties": False,
}


def new_decision() -> dict:
    # 모델은 sparse하게 내보내니까 기존 LangGraph가 먹을 수 있는 full 기본값을 먼저 만듦
    return {
        "understanding": "ok",
        "order_patch": {"sauce": None, "noodle_type": None, "noodle_portion": None, "toppings": {}},
        "restriction_options": [],
        "preference_options": [],
        "recommendation": {"action": "none", "scope": "current", "criteria": None},
        "commit": False,
        "confirmation": "none",
        "queries": [],
    }

# 위의 DECISION SCHEMA 결과를 기존 LangGraph용 full Decision으로 바꿔주는 부분


def normalize_decision(sparse_decision: dict) -> dict:
    # 아래에서는 sparse_decision 원본을 안건드리고 decision에 필요한 값만 복사함
    decision = new_decision()

    # 의미를 하나로 못정한 경우 -> 주문 반영이나 로봇 실행을 막는 understanding 상태로 변환
    if sparse_decision.get("clarify") is True:
        decision["understanding"] = "clarify"

    # order 자체가 없으면 빈 dict로 받아서 아래 로직을 그냥 통과시킴
    sparse_order = sparse_decision.get("order", {})

    # scalar 값 3개는 실제로 모델이 출력한 필드만 기존 기본값 위에 덮음
    for field in ("sauce", "noodle_type", "noodle_portion"):
        if field in sparse_order:
            decision["order_patch"][field] = sparse_order[field]

    # toppings는 중첩 dict라 원본이랑 메모리를 공유하지 않게 깊은 복사함
    if "toppings" in sparse_order:
        decision["order_patch"]["toppings"] = copy.deepcopy(sparse_order["toppings"])

    # Schema에서는 reason인데 기존 내부 계약은 type을 쓰고 있어서 key 이름만 바꿈
    for restriction in sparse_decision.get("restrictions", []):
        decision["restriction_options"].append({
            "target": restriction["target"],
            "type": restriction["reason"],
            "action": restriction["action"],
        })

    # preferences가 없으면 빈 list, 있으면 add/remove 목록을 그대로 복사
    decision["preference_options"] = copy.deepcopy(sparse_decision.get("preferences", []))

    # recommendation이 있을 때만 action을 바꾸고, optional인 scope/criteria는 나온 것만 덮음
    if "recommendation" in sparse_decision:
        sparse_recommendation = sparse_decision["recommendation"] # 모델이 바꾼거 덮어 씀
        decision["recommendation"]["action"] = sparse_recommendation["action"]

        if "scope" in sparse_recommendation:
            decision["recommendation"]["scope"] = sparse_recommendation["scope"]

        if "criteria" in sparse_recommendation:
            decision["recommendation"]["criteria"] = sparse_recommendation["criteria"]

    # 명시적으로 출력된 경우만 덮음 -> 없으면 기본값 False 유지
    if "commit" in sparse_decision:
        decision["commit"] = sparse_decision["commit"]

    # pending confirmation에 답한 경우만 덮음 -> 없으면 기본값 none 유지
    if "confirmation" in sparse_decision:
        decision["confirmation"] = sparse_decision["confirmation"]

    # 질문이 없으면 빈 list, 있으면 여러 query를 그대로 깊은 복사함
    decision["queries"] = copy.deepcopy(sparse_decision.get("queries", []))

    # 여기서부터 기존 LangGraph는 sparse 여부 신경 안쓰고 항상 같은 구조로 받음
    return decision
