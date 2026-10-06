import copy

from soomac_irc.domain import AMOUNTS, RESTRICTION_REASONS


# 자연어 의미는 Decision 모델이 최대한 그대로 꺼내고, 실제 반영 가능 여부는 Python이 마지막에 검사한다.
# 그래서 "햄", "라면"처럼 지원하지 않는 메뉴도 여기서는 문자열로 표현할 수 있게 둔다.
# 지원 메뉴·물리 상태 검사는 실제 state 반영과 ROS 실행 직전 llm_policy.py에서 처리한다.


#################### Decision이 출력할 주문 변경 형식 ####################

FLEXIBLE_ORDER_SCHEMA = {
    "type": "object",
    "properties": {
        # scalar의 none은 선택 해제 의미이다. 실제 주문에는 문자열로 저장하지 않는다.
        "sauce": {"type": "string"},
        "noodle_type": {"type": "string"},
        "noodle_portion": {"type": "string"},
        "toppings": {
            "type": "object",
            "additionalProperties": {
                "type": "string",
                "enum": [*AMOUNTS, "none"],
            },
        },
    },
    "additionalProperties": False,
}

# 추천 모델은 완성된 proposal을 내므로 비어 있는 scalar 값도 null로 포함한다.
# Decision의 sparse order와 모양이 비슷하지만 required 여부가 다르니 스키마를 따로 둔다.
ORDER_PATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "sauce": {"type": ["string", "null"]},
        "noodle_type": {"type": ["string", "null"]},
        "noodle_portion": {"type": ["string", "null"]},
        "toppings": {
            "type": "object",
            "additionalProperties": {
                "type": "string",
                "enum": [*AMOUNTS, "none"],
            },
        },
    },
    "required": ["sauce", "noodle_type", "noodle_portion", "toppings"],
    "additionalProperties": False,
}

DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "route": {"type": "string", "enum": ["task", "general", "mixed"]},
        "order": FLEXIBLE_ORDER_SCHEMA,
        "restrictions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                    "reason": {"type": "string", "enum": list(RESTRICTION_REASONS)},
                    "action": {"type": "string", "enum": ["add", "remove"]},
                },
                "required": ["target", "reason", "action"],
                "additionalProperties": False,
            },
        },
        "preferences": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                    "action": {"type": "string", "enum": ["add", "remove"]},
                },
                "required": ["value", "action"],
                "additionalProperties": False,
            },
        },
        # Decision은 추천 조건을 중간 JSON으로 다시 번역하지 않는다.
        # 추천 모델이 사용자 원문과 최근 대화를 직접 읽고 구체적인 추천안을 만든다.
        "recommendation": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["request", "revise"]},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
        "commit": {"type": "boolean"},
        "confirmation": {"type": "string", "enum": ["accept", "reject"]},
        "clarify": {"type": "boolean"},
    },
    "required": ["route"],
    "additionalProperties": False,
}

RECOMMENDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "proposal": ORDER_PATCH_SCHEMA,
        "reason_tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["proposal", "reason_tags"],
    "additionalProperties": False,
}

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"reply": {"type": "string"}},
    "required": ["reply"],
    "additionalProperties": False,
}


def empty_order_patch() -> dict:
    # 내부 로직은 sparse 여부를 반복 검사하지 않도록 항상 같은 주문 patch 모양을 사용한다.
    return {
        "sauce": None,
        "noodle_type": None,
        "noodle_portion": None,
        "toppings": {},
    }


def new_decision() -> dict:
    # 모델 출력은 sparse하지만 Graph 내부에서는 모든 key가 있는 full Decision을 사용한다.
    # route의 기본값은 task로 둬서 잘못된 빈 출력이 안전 검증 경로를 우회하지 않게 한다.
    return {
        "route": "task",
        "understanding": "ok",
        "order_patch": empty_order_patch(),
        "restriction_options": [],
        "preference_options": [],
        "recommendation": {"action": "none"},
        "commit": False,
        "confirmation": "none",
    }


def normalize_decision(sparse_decision: dict) -> dict:
    # 모델이 출력한 sparse JSON을 Graph가 바로 읽을 수 있는 고정 모양으로 바꾼다.
    # 원본 객체와 state가 list/dict를 공유하지 않도록 중첩 값은 모두 깊은 복사한다.
    decision = new_decision()
    decision["route"] = sparse_decision.get("route", "task")

    # 의미를 확정하지 못한 경우 주문 변경과 실행을 막는 내부 상태로 바꾼다.
    if sparse_decision.get("clarify") is True:
        decision["understanding"] = "clarify"

    # 이번 턴에 실제로 나온 주문 필드만 기본 patch 위에 덮는다.
    order = sparse_decision.get("order") or {}
    for field in ("sauce", "noodle_type", "noodle_portion"):
        if field in order:
            decision["order_patch"][field] = copy.deepcopy(order[field])
    if isinstance(order.get("toppings"), dict):
        decision["order_patch"]["toppings"] = copy.deepcopy(order["toppings"])

    # 제한과 취향은 add/remove 동작을 그대로 보존하고 실제 적용은 policy에 맡긴다.
    decision["restriction_options"] = copy.deepcopy(
        sparse_decision.get("restrictions", [])
    )
    decision["preference_options"] = copy.deepcopy(
        sparse_decision.get("preferences", [])
    )

    # 추천은 요청인지 수정인지까지만 Decision이 정한다.
    recommendation = sparse_decision.get("recommendation")
    if isinstance(recommendation, dict) and recommendation.get("action") in ("request", "revise"):
        decision["recommendation"] = {"action": recommendation["action"]}

    # commit/confirmation은 모델의 자연어 판단을 그대로 내부 값으로 옮긴다.
    # 실제 실행 가능 여부는 Graph와 policy가 별도로 검사한다.
    decision["commit"] = bool(sparse_decision.get("commit", False))
    decision["confirmation"] = sparse_decision.get("confirmation", "none")
    return decision
