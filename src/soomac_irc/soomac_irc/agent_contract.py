import copy
from typing import TypedDict

from soomac_irc.domain import AMOUNTS, RESTRICTION_REASONS


# 자연어 의미는 Decision 모델이 최대한 그대로 꺼내고, 실제 반영 가능 여부는 Python이 마지막에 검사한다.
# 그래서 "햄", "라면"처럼 지원하지 않는 메뉴도 여기서는 문자열로 표현할 수 있게 둔다.
# 지원 메뉴·물리 상태 검사는 실제 state 반영과 ROS 실행 직전 llm_policy.py에서 처리한다.


#################### 현재 runtime contract 식별자 ####################

# 과거 runtime 로그와 현재 자연어 우선 계약의 로그를 dataset 생성 시 구분한다.
RUNTIME_LOG_SCHEMA_VERSION = 2
RUNTIME_CONTRACT_VERSION = "natural_multiturn_v3"


#################### 외부 모델 출력과 내부 runtime 타입 ####################

class ExternalDecision(TypedDict, total=False):
    """Decision Agent가 실제로 출력하는 sparse JSON이다."""

    route: str
    order: dict
    restrictions: list[dict]
    preferences: list[dict]
    recommendation: dict
    commit: bool
    confirmation: str
    clarify: bool


class OrderPatch(TypedDict):
    sauce: str | None
    noodle_type: str | None
    noodle_portion: str | None
    toppings: dict[str, str]


class Restriction(TypedDict):
    target: str
    reason: str


class Preference(TypedDict):
    value: str


class OrderState(TypedDict):
    sauce: str | None
    noodle_type: str | None
    noodle_portion: str | None
    toppings: dict[str, str]
    restrictions: list[Restriction]


class ExecutionPending(TypedDict, total=False):
    type: str
    source: str
    section: str
    targets: list[str]
    items: dict[str, str]
    candidate: OrderPatch


class RecommendationPending(TypedDict, total=False):
    type: str
    candidate: OrderPatch
    reason_tags: list[str]


Pending = ExecutionPending | RecommendationPending


class SessionState(TypedDict):
    # 확정된 주문 사실과 자연어 대화 기억만 session에 보관한다.
    # 아직 확정하지 않은 실행·추천은 pending 하나에서 종류로 구분한다.
    order: OrderState
    preferences: list[Preference]
    pending: Pending | None
    history: list[dict]
    action_history: list[dict]


class RobotState(TypedDict, total=False):
    section: str
    task_queue: list[dict]
    active_task: dict | None
    completed_tasks: list[dict]
    robot_started: bool
    section_transition: dict


class NormalizedDecision(TypedDict):
    """normalize_decision() 이후 Graph에서만 사용하는 고정 모양이다."""

    route: str
    understanding: str
    order_patch: OrderPatch
    restriction_options: list[dict]
    preference_options: list[dict]
    recommendation: dict
    commit: bool
    confirmation: str


class PolicyIssues(TypedDict):
    unsupported: list[dict]
    invalid: list[dict]
    protected: list[dict]
    restriction_conflicts: list[dict]
    physical_conflicts: list[dict]


class Policy(TypedDict, total=False):
    status: str
    reason: str
    execute: bool
    conflicts: list[dict]
    issues: PolicyIssues
    unsupported: list[dict]
    protected: list[dict]
    invalid: list[dict]
    restriction_blocked: list[dict]
    restriction_unchanged: list[dict]
    rejected_fields: list[dict]
    missing: list[str]


class RecommendationResult(TypedDict, total=False):
    proposal: OrderPatch
    reason_tags: list[str]
    accepted_fields: list[str]
    rejected_fields: list[dict]


class TurnState(TypedDict, total=False):
    session: SessionState
    previous_session: SessionState
    user_text: str
    robot_state: RobotState
    decision: NormalizedDecision | None
    recommendation_result: RecommendationResult | None
    policy: Policy | None
    reply: str | None


class ResponseInput(TypedDict):
    user_text: str
    recent_history: list[dict]
    confirmed_order: OrderState
    preferences: list[Preference]
    pending: Pending | None
    policy: Policy
    applied_this_turn: dict
    future_changes: list[dict]
    recommendation_result: RecommendationResult | None
    next_prompt: dict | None
    robot_state: RobotState
    recent_action_history: list[dict]
    execution_authorized: bool
    starting_now: list[dict]


class DuplicateDecisionKeyError(ValueError):
    """외부 sparse Decision JSON에 같은 key가 두 번 나온 경우이다."""

    pass


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


def empty_order_patch() -> OrderPatch:
    # 내부 로직은 sparse 여부를 반복 검사하지 않도록 항상 같은 주문 patch 모양을 사용한다.
    return {
        "sauce": None,
        "noodle_type": None,
        "noodle_portion": None,
        "toppings": {},
    }


def new_order_state() -> OrderState:
    # 실제 주문의 기준값이다. 추천 후보나 거절된 값은 이 dict에 넣지 않는다.
    return {
        "sauce": None,
        "noodle_type": None,
        "noodle_portion": None,
        "toppings": {},
        "restrictions": [],
    }


def new_session_state() -> SessionState:
    # 한 주문 대화가 시작될 때 만드는 최소 session 구조이다.
    return {
        "order": new_order_state(),
        "preferences": [],
        "pending": None,
        "history": [],
        "action_history": [],
    }


def new_decision() -> NormalizedDecision:
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


def normalize_decision(sparse_decision: ExternalDecision | dict) -> NormalizedDecision:
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


def new_turn_state(
    session: SessionState,
    user_text: str,
    robot_state: RobotState | dict,
) -> TurnState:
    # Graph는 복사본에서만 동작한다. 중간 단계가 실패해도 실제 Node state가 오염되지 않는다.
    session_copy = copy.deepcopy(session)
    return {
        "session": session_copy,
        "previous_session": copy.deepcopy(session_copy),
        "user_text": user_text,
        "robot_state": copy.deepcopy(robot_state),
        "decision": None,
        "recommendation_result": None,
        "policy": None,
        "reply": None,
    }
