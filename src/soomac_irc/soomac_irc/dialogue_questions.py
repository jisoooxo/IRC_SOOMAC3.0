"""로봇이 실제로 물은 질문의 수명·대상을 관리한다. 자연어를 주문으로 변환하지 않는다."""

import copy

from soomac_irc.domain import NOODLE_TYPES, SAUCES

SECTION_CHOICES = {
    "veggie": ["양파", "버섯"], "meat": ["소시지", "게살"],
    "extra": ["치즈", "페퍼론치노"],
}
SECTION_NAMES = {"noodle": "면", "veggie": "야채", "meat": "육류",
                 "extra": "추가 재료", "lid": "뚜껑", "sauce": "소스"}

def question_for_section(section, turn):
    if section not in SECTION_CHOICES:
        return None
    return {"type": "choice", "targets": list(SECTION_CHOICES[section]),
            "section": section, "history_turn": turn}


def active_question(session, section):
    question = session.get("pending_question")
    if not question or question.get("section") != section:
        return None
    # 대기 중인 질문은 포커스 TTL과 다르다. 답변/교체/작업 시작 때 닫는다.
    return copy.deepcopy(question)


def question_from_prompt(prompt, section, turn, order):
    """Response에 넘긴 질문의 의미를 저장한다. 생성된 문장을 재해석하지 않는다."""
    if not prompt or prompt.get("type") not in (
        "choice", "amount", "missing_field", "menu_confirmation",
    ):
        # 안전 확인과 추천 확인은 기존 pending_confirmation/recommendation이 소유한다.
        return None
    question = copy.deepcopy(prompt)
    question.update(section=section, history_turn=turn)
    if question["type"] == "missing_field":
        field = question["target"]
        question["fields"] = [field]
        question["targets"] = {
            "noodle_type": list(NOODLE_TYPES),
            "sauce": list(SAUCES),
            "noodle_portion": [order["noodle_type"]] if order["noodle_type"] else [],
        }.get(field, [])
    question.setdefault("targets", [])
    return question


def question_after_turn(session, decision, policy, next_prompt, section, turn, mentions):
    """질문은 하나만 유지한다. 턴 수가 아닌 처리 결과로 소비/교체한다."""
    if policy["execute"] or session["pending_confirmation"] is not None:
        return None
    if session["recommendation"]["phase"] == "proposed":
        return None
    question = active_question(session, section)
    if decision["confirmation"] == "reject":
        question = None
    if question and mentions and set(mentions).isdisjoint(question["targets"]):
        question = None  # 다른 대상을 명시한 새 주제는 오래된 질문보다 우선한다.
    if question and policy["reason"] != "menu_confirmation":
        patch = decision["order_patch"]
        answered = any(patch.get(field) is not None for field in question.get("fields", []))
        answered = answered or bool(set(patch["toppings"]) & set(question["targets"]))
        answered = answered or any(
            patch[field] is not None and patch[field] in question["targets"]
            for field in ("noodle_type", "sauce")
        )
        if answered and (policy["status"] in ("pass", "warning") or policy["reason"] == "missing_order"):
            question = None
    if next_prompt is not None:
        return question_from_prompt(next_prompt, section, turn, session["order"])
    return question
