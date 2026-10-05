"""주문 사실은 확정된 state/policy로만 말한다. 자연어 의도는 여기서 해석하지 않는다."""

from soomac_irc.dialogue_questions import SECTION_CHOICES, SECTION_NAMES, question_for_section
from soomac_irc.domain import NOODLE_TYPES, SAUCES
from soomac_irc.llm_policy import (
    build_completed_modification_warning, build_completed_restriction_warning,
    reference_targets_are_supported,
)

AMOUNT_WORDS = {"low": "적게", "normal": "보통", "high": "많이"}
FIELD_NAMES = {"sauce": "소스", "noodle_type": "면 종류", "noodle_portion": "면 양"}


def describe_patch(patch):
    parts = []
    for key in ("sauce", "noodle_type", "noodle_portion"):
        value = patch.get(key)
        if value is not None:
            parts.append(f"{FIELD_NAMES[key]}: {AMOUNT_WORDS.get(value, value)}")
    for target, amount in patch.get("toppings", {}).items():
        parts.append(f"{target}: {'제외' if amount == 'none' else AMOUNT_WORDS.get(amount, amount)}")
    return ", ".join(parts)


def render_task_result(session, decision, policy, changes, next_prompt, robot, user_text=""):
    """(사실 기반 응답, 이번에 실제 물은 질문)을 반환한다."""
    reason = policy["reason"]
    section = robot["section"]
    turn = len(session["history"]) // 2 + 1
    targets = policy.get("reference_targets", [])
    question = None
    parts = []

    # 이 안내는 변경 반영 여부와 무관하게 상태상 거절/보류를 먼저 알린다.
    if reason == "physical_state":
        return build_completed_modification_warning(policy["blocked_changes"]), None
    if reason == "completed_restriction_conflict":
        return build_completed_restriction_warning(policy["conflicts"]), None
    if reason == "robot_busy":
        return "지금 작업 중이에요. 현재 작업을 마친 뒤 진행할 수 있어요.", None
    if reason == "unsupported_reference":
        return f"말씀하신 {', '.join(targets)} 메뉴는 현재 제공하지 않아요.", None
    if reason == "ambiguous_reference":
        if targets and all(not reference_targets_are_supported([t]) for t in targets):
            return f"말씀하신 {', '.join(targets)} 메뉴는 현재 제공하지 않아요.", None
        if len(targets) > 1:
            question = {"type": "choice", "targets": targets, "section": section, "history_turn": turn}
            return f"{' 또는 '.join(targets)} 중 어떤 것을 말씀하시는 건가요?", question
        question = question_for_section(section, turn)
        if question:
            return f"대상을 확인할게요. {' 또는 '.join(question['targets'])} 중 무엇을 원하시나요?", question
        return "어떤 메뉴를 말씀하시는지 이름을 다시 알려주세요.", None
    if reason == "understanding":
        mentions = decision.get("mentions", [])
        unsupported = [m for m in mentions if not reference_targets_are_supported([m])]
        if unsupported and not any("면" in m for m in unsupported):
            return f"말씀하신 {', '.join(unsupported)} 메뉴는 현재 제공하지 않아요. 주문에는 반영하지 않았어요.", None
        question = question_for_section(section, turn)
        if section == "noodle":
            question = {"type": "choice", "targets": list(NOODLE_TYPES), "section": section, "history_turn": turn}
        if question:
            return f"아직 주문에 반영하지 않았어요. {' 또는 '.join(question['targets'])} 중 원하는 메뉴를 말씀해 주세요.", question
        return "요청을 확실히 이해하지 못해 반영하지 않았어요. 원하는 메뉴와 양을 다시 말씀해 주세요.", None
    if reason == "confirmation_rejected":
        return "확인을 취소했어요. 보류된 요청은 실행하지 않았어요.", None

    patch = {"toppings": {}}
    for key, value in changes["order_changes"].items():
        if key.startswith("toppings."):
            patch["toppings"][key.split(".", 1)[1]] = value or "none"
        else:
            patch[key] = value
    description = describe_patch(patch)
    if description:
        parts.append(f"주문에 반영했어요({description}).")
    for restriction in changes["restriction_added"]:
        parts.append(f"{restriction['target']} 제한을 등록했어요.")
    for restriction in changes["restriction_removed"]:
        parts.append(f"{restriction['target']} 제한을 해제했어요.")

    if reason == "menu_confirmation":
        question = session.get("pending_question")
        parts.append(f"{describe_patch(question['proposal'])}을 말씀하신 건가요? 이 항목은 아직 반영하지 않았어요. 맞으면 네, 아니면 메뉴를 다시 말씀해 주세요.")
        return " ".join(parts), question

    if policy["execute"]:
        if section in SECTION_CHOICES:
            selected = [t for t in SECTION_CHOICES[section] if t in session["order"]["toppings"]]
            parts.append(f"{', '.join(selected)} 담기를 시작할게요." if selected else
                         f"{SECTION_NAMES[section]}는 선택하지 않아 건너뛸게요.")
        else:
            parts.append(f"{SECTION_NAMES.get(section, section)} 담기를 시작할게요.")
        return " ".join(parts), None

    for query in decision["queries"]:
        kind = query["type"]
        target = query.get("target")
        if kind in ("order_status", "order_field", "order_item"):
            if kind == "order_field":
                value = session["order"].get(target)
                parts.append(f"{FIELD_NAMES.get(target, target)}: {AMOUNT_WORDS.get(value, value) if value else '미선택'}이에요.")
            elif kind == "order_item":
                value = session["order"]["toppings"].get(target)
                parts.append(f"{target}: {AMOUNT_WORDS.get(value, '미선택')}이에요.")
            else:
                parts.append(f"현재 주문은 {describe_patch(session['order']) or '아직 선택 전'}이에요.")
        elif kind == "restriction_status":
            names = [r["target"] for r in session["order"]["restrictions"]]
            parts.append(f"등록한 제한은 {', '.join(names)}이에요." if names else "등록한 재료 제한은 없어요.")
        elif kind == "robot_completed":
            completed = [task["class"] for task in robot["completed_tasks"]]
            parts.append(f"완료한 재료는 {', '.join(completed)}이에요." if completed else "아직 완료한 작업은 없어요.")
        elif kind == "robot_failure":
            events = [e for e in session["action_history"] if e.get("type") == "vlm_result" and e.get("verdict") in ("fail", "uncertain")]
            parts.append(f"최근 {events[-1]['class']} 작업에서 시각 확인이 되지 않았어요." if events else "기록된 시각 확인 실패는 없어요.")
        elif kind == "robot_status" and robot.get("active_task"):
            parts.append(f"지금 {robot['active_task']['class']} 작업 중이에요.")
        elif kind in ("robot_status", "recommendation_status"):
            parts.append(f"현재 {SECTION_NAMES.get(section, section)} 선택 단계예요.")

    if next_prompt:
        kind = next_prompt["type"]
        if kind == "missing_field":
            field = next_prompt["target"]
            options = {"noodle_type": list(NOODLE_TYPES), "sauce": list(SAUCES),
                       "noodle_portion": ["적게", "보통", "많이"]}[field]
            parts.append(f"{FIELD_NAMES[field]}은 {', '.join(options)} 중 어떻게 할까요?")
            question = {"type": "field", "field": field, "targets": options,
                        "section": section, "history_turn": turn}
        elif kind == "recommendation_offer":
            question = question_for_section(section, turn)
            if question:
                parts.append(f"{', '.join(question['targets'])} 중 원하는 재료와 양을 말씀해 주세요. 원하지 않으면 다음 단계라고 해주세요.")
        elif kind == "confirmation_required":
            if next_prompt["reason"] == "restriction_conflict":
                parts.append(f"{next_prompt['target']} 주문이 등록한 제한과 충돌해 보류했어요. 해당 제한을 해제할까요?")
            else:
                proposal = session["recommendation"]["last_proposal"]["proposal"]
                parts.append(f"추천은 {describe_patch(proposal)}이에요. 이 추천을 주문에 반영할까요?")
        elif kind == "future_confirmation":
            parts.append(f"앞서 고른 {', '.join(next_prompt['items'])}를 그대로 담을까요?")
    if not parts:
        parts.append("현재 주문을 유지했어요. 실행하려면 바로 진행해라고 말씀해 주세요.")
    return " ".join(parts), question
