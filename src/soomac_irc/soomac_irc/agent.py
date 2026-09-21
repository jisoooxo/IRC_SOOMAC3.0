import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from soomac_irc.order_policy import evaluate_order_policy
from soomac_irc.reply import build_policy_reply


class ToolCall(TypedDict):
    # call_model이 만든 한 턴의 최소 Tool 계약
    call_id: str
    name: str
    changes: dict


class AgentState(TypedDict, total=False):
    # llm_node와 llm_debug가 한 턴마다 복사해서 전달하는 공용 상태
    user_text: str
    order: dict
    execution: dict
    recommendation: dict
    history: list[dict]
    action_history: list[dict]

    # Tool 해석과 Python Order Policy 결과
    tool_call: ToolCall | None
    parser_status: str
    turn_result: dict
    policy_reply: str | None


def classify_confirmation_intent(user_text: str, context: str) -> bool | None:
    # 모델의 Tool 선택과 독립적으로 원문에 명시적 동의·거절가 있는지 검사한다.
    if not isinstance(user_text, str) or context not in ("start", "section_skip"):
        return None

    normalized = re.sub(r"[\s.,!?~'”“\"`]+", "", user_text.lower())

    if not normalized:
        return None

    negative_markers = (
        "아니",
        "안돼",
        "안해",
        "하지마",
        "하지말",
        "말고",
        "취소",
        "안진행",
        "진행안",
        "안시작",
        "시작안",
        "담지마",
        "넘어가지마",
        "넘어가지말",
        "안넘어가",
        "제외하지마",
        "제외하지말",
        "안제외",
        "빼지마",
        "빼지말",
        "계속고를",
        "그대로둘",
    )

    if any(marker in normalized for marker in negative_markers):
        return False

    short_affirmatives = {
        "네", "예", "응", "어", "그래", "좋아", "맞아", "오케이", "ok",
        "진행", "시작", "확정",
    }
    common_positive_markers = (
        "그렇게해",
        "이대로해",
        "그대로해",
        "확정해",
    )
    context_positive_markers = {
        "start": (
            "진행해",
            "진행하자",
            "바로진행",
            "시작해",
            "시작하자",
            "담아줘",
            "담아주세요",
            "담기시작",
        ),
        "section_skip": (
            "넘어가",
            "제외해",
            "빼줘",
            "진행해",
        ),
    }
    ambiguous_markers = (
        "진행해도될까",
        "시작해도될까",
        "진행할까",
        "시작할까",
        "진행해야하나",
        "시작해야하나",
        "해도될까",
        "하는게맞",
        "괜찮을까",
        "생각해볼",
        "고민중",
    )

    if any(marker in normalized for marker in ambiguous_markers):
        return None

    if normalized in short_affirmatives:
        return True

    positive_markers = common_positive_markers + context_positive_markers[context]

    if any(marker in normalized for marker in positive_markers):
        return True

    return None


def build_graph(call_model):
    # 모델은 사용자 발화를 Tool JSON으로 해석만 한다.
    # Python이 원문의 진행 의도를 별도로 판정한 뒤 Order Policy와 Reply를 실행한다.
    def interpret_tool(state: AgentState) -> dict:
        tool_call = call_model(state, state["user_text"])

        return {
            "tool_call": tool_call,
            "parser_status": "ok" if tool_call is not None else "invalid",
        }

    def apply_order_policy(state: AgentState) -> dict:
        confirmation_intent = classify_confirmation_intent(
            state.get("user_text"),
            "start",
        )

        return evaluate_order_policy(
            state,
            confirmation_intent,
        )

    graph = StateGraph(AgentState)

    graph.add_node("interpret_tool", interpret_tool)
    graph.add_node("apply_order_policy", apply_order_policy)
    graph.add_node("build_policy_reply", build_policy_reply)

    graph.add_edge(START, "interpret_tool")
    graph.add_edge("interpret_tool", "apply_order_policy")
    graph.add_edge("apply_order_policy", "build_policy_reply")
    graph.add_edge("build_policy_reply", END)

    return graph.compile()
