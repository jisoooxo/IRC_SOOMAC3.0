import re

from soomac_irc.agent_contract import new_decision
from soomac_irc.llm_langgraph import Decision, SessionState


ACCEPT_CONFIRMATIONS = {
    "응",
    "네",
    "예",
    "그래",
    "좋아",
    "오케이",
    "ok",
}

REJECT_CONFIRMATIONS = {
    "아니",
    "아니요",
    "싫어",
    "취소",
    "안돼",
    "안해",
}


def pre_decision_override(text: str, session: SessionState, robot_state: dict) -> Decision | None:
    # 정확한 단답만 deterministic하게 처리한다. 복합 발화는 항상 모델이 전체 의미를 본다.
    normalized = re.sub(r"""[\s.,!?~'"`]+""", "", text.lower())
    pending = session["pending_confirmation"]

    if pending is not None:
        if normalized in ACCEPT_CONFIRMATIONS:
            decision = new_decision()
            decision["confirmation"] = "accept"
            return decision

        if normalized in REJECT_CONFIRMATIONS:
            decision = new_decision()
            decision["confirmation"] = "reject"
            return decision

        return None

    if session["recommendation"]["phase"] == "proposed":
        if normalized in ACCEPT_CONFIRMATIONS:
            decision = new_decision()
            decision["recommendation"]["action"] = "select"
            return decision

        if normalized in REJECT_CONFIRMATIONS:
            decision = new_decision()
            decision["recommendation"]["action"] = "cancel"
            return decision

    return None


def post_decision_override(text: str, decision: Decision, session: SessionState, robot_state: dict) -> Decision:
    # 실제 모델 회귀검사에서 반복 실패한 케이스만 나중에 한곳에서 수정한다.
    # 현재는 known failure가 없으므로 모델 Decision을 그대로 반환한다.
    return decision
