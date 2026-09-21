import copy
import re
from collections.abc import Callable

from soomac_irc.order import (
    EXTRAS,
    MEATS,
    SECTION_LABELS,
    VEGGIES,
    missing_requirements,
)
from soomac_irc.restriction import strongest_restriction_for

# 자유응답 모델이 말하면 안 되는 주문 반영·로봇 진행 표현
OPERATIONAL_CLAIMS = (
    "반영했",
    "확정했",
    "제외했",
    "제외할게",
    "제외하고",
    "넣어드릴게",
    "담기를 시작",
    "담기가 끝",
    "담기가 완료",
    "작업을 시작",
    "작업을 마쳤",
    "다음은",
    "다음 단계",
    "차례입니다",
    "진행할게",
    "진행하겠",
    "선택해 주",
    "말씀해 주",
    "골라 주",
)


def build_selection_next_step(order: dict, execution: dict) -> str | None:
    # 자유응답에 흐름 결정을 맡기지 않고 현재 section의 바로 다음 행동을 확정한다.
    # 로봇 작업 중에는 사용자 입력을 받지 않으므로 선택 질문을 덧붙이지 않는다.
    if execution.get("active_task") is not None or execution.get("task_queue"):
        return None

    section = execution.get("section")

    if section == "noodle":
        missing = missing_requirements(order, ["noodle"]).get("noodle", [])
        field_labels = {
            "sauce": "소스",
            "noodle_type": "면 종류",
            "noodle_portion": "면 양",
        }

        if missing:
            missing_text = ", ".join(field_labels[field] for field in missing)
            return f"다음으로 선택할 항목은 {missing_text}예요. 원하는 내용을 말씀해 주세요."

        return "면 선택이 끝났어요. 이대로 면 담기를 시작하려면 진행해 달라고 말씀해 주세요."

    section_options = {
        "veggie": ("야채", VEGGIES),
        "meat": ("육류", MEATS),
        "extra": ("추가 재료", EXTRAS),
    }

    if section in section_options:
        section_label, options = section_options[section]
        selected_items = [item for item in options if item in order["toppings"]]
        available_items = [
            item for item in options
            if strongest_restriction_for(order, item) is None
        ]
        remaining_items = [
            item for item in available_items
            if item not in selected_items
        ]

        if not selected_items:
            if not available_items:
                return f"현재 선택 가능한 {section_label}가 없어요. {section_label} 단계를 제외할지 말씀해 주세요."

            if len(available_items) == 1:
                return f"{section_label}로는 {available_items[0]}가 남아 있어요. 어느 정도 양으로 넣을까요?"

            options_text = "나 ".join(available_items)
            return f"다음은 {section_label} 선택이에요. {options_text} 중 원하는 재료와 양을 말씀해 주세요."

        amount_labels = {"low": "적게", "normal": "보통", "high": "많이"}
        selected_text = ", ".join(
            f"{item} {amount_labels[order['toppings'][item]]}"
            for item in selected_items
        )

        if remaining_items:
            remaining_text = "나 ".join(remaining_items)
            return f"현재 {section_label} 선택은 {selected_text}예요. {remaining_text}도 추가하거나, 이대로 담기를 진행할까요?"

        return f"현재 {section_label} 선택은 {selected_text}예요. 이대로 {section_label} 담기를 진행할까요?"

    if section == "lid":
        return "다음은 뚜껑 닫기예요. 진행해 달라고 말씀해 주세요."

    if section == "sauce":
        if order.get("sauce") is None:
            return "다음으로 소스를 선택해 주세요."

        return "이대로 소스 담기를 시작하려면 진행해 달라고 말씀해 주세요."

    return None


def build_policy_reply(state: dict) -> dict:
    # 같은 turn_result에서 동일한 재료·이유를 여러 번 안내하지 않는다.
    # 입력: evaluate_order_policy이 확정한 사실 / 반환: Python 고정 문장 또는 None
    # 모델 원문을 다시 보지 않아서 안전·물리 차단 결과를 말로 뒤집지 못하게 함
    turn_result = state["turn_result"]
    safety_items = []
    dietary_items = []
    dislike_items = []
    seen = set()

    for conflict in turn_result["kept_physical_conflict"]:
        item = conflict["item"]
        reason = conflict["restriction"]["reason"]
        notice_key = (item, reason)

        if notice_key in seen:
            continue

        seen.add(notice_key)

        if reason in ("allergy", "cannot_eat"):
            safety_items.append(item)
        elif reason == "dietary_rule":
            dietary_items.append(item)
        elif reason == "dislike":
            dislike_items.append(item)
        else:
            raise ValueError(f"restriction reason 이상함 : {reason}")

    messages = []

    if safety_items:
        items = ", ".join(safety_items)
        messages.append(f"{items} 재료는 이미 담기 시작해서 철회할 수 없습니다. 섭취에 유의하시기 바랍니다.")

    if dietary_items:
        items = ", ".join(dietary_items)
        messages.append(f"{items} 재료는 이미 담기 시작해서 제외할 수 없습니다. 이후 재료부터 해당 식단 조건을 적용하겠습니다.")

    if dislike_items:
        items = ", ".join(dislike_items)
        messages.append(f"{items} 재료는 이미 담기 시작해서 취소할 수 없어요.")

    # 전체 취소는 모델을 호출하지 않고 가능 여부에 맞는 문장으로 답한다.
    cancel = turn_result["cancel_validation"]

    if cancel["requested"]:
        if cancel["reason"] == "empty_order":
            messages.append("아직 선택된 주문이 없어요.")
        elif cancel["reason"] == "order_already_selected":
            messages.append("전체 주문 취소는 지원하지 않아요. 바꾸고 싶은 메뉴를 말씀해 주세요.")
        elif cancel["reason"] == "robot_already_started":
            messages.append("이미 재료를 담기 시작해서 전체 주문을 취소할 수 없어요. 아직 담지 않은 재료는 변경할 수 있어요.")
        else:
            raise ValueError(f"전체 취소 reason 이상함 : {cancel['reason']}")

    # 필수값 누락이나 일부 차단이 있으면 로봇을 시작하지 않았다고 명확히 안내한다.
    confirm = turn_result["confirm_validation"]

    if confirm["requested"] and not confirm["allowed"]:
        if confirm["reason"] == "explicit_rejection":
            messages.append("진행하지 않을게요. 현재 선택은 그대로 유지했어요.")

        elif confirm["reason"] == "explicit_confirmation_required":
            messages.append("진행 의사를 확인하지 못해 작업을 시작하지 않았어요. 시작하려면 진행해 달라고 말씀해 주세요.")

        missing_values = []

        for values in confirm["missing"].values():
            missing_values.extend(values)

        field_labels = {
            "sauce": "소스",
            "noodle_type": "면 종류",
            "noodle_portion": "어느 정도 양을 드실지",
        }
        missing_labels = [field_labels.get(value, value) for value in missing_values]

        if confirm["reason"] in (
            "explicit_rejection",
            "explicit_confirmation_required",
        ):
            pass
        elif missing_labels:
            messages.append(f"아직 {', '.join(missing_labels)} 선택이 필요해서 작업을 시작하지 않았어요.")
        else:
            messages.append("일부 요청을 반영할 수 없어 아직 작업을 시작하지 않았어요.")

    # Tool JSON을 읽지 못한 경우에도 Reply 모델이 내용을 추측하지 않는다.
    if turn_result["action"] == "error" and not messages:
        messages.append("요청을 정확히 이해하지 못했어요. 다시 말씀해 주세요.")

    # 일반 주문 변경 결과도 검증된 turn_result만 보고 안내한다.
    action = turn_result["action"]
    confirm_failed = (
        action == "set_order_and_confirm"
        and turn_result["confirm_validation"]["requested"]
        and not turn_result["confirm_validation"]["allowed"]
    )

    if action == "set_order" or confirm_failed:
        accepted = turn_result["accepted"]
        invalid = turn_result["invalid_value"]
        blocked_active = turn_result["blocked_active_or_completed"]
        blocked_past = turn_result["blocked_past_section"]
        blocked_constraint = turn_result["blocked_constraint"]
        restriction_applied = turn_result["restriction_applied"]

        if accepted:
            if invalid or blocked_active or blocked_past or blocked_constraint:
                messages.append("가능한 주문 변경만 반영했어요.")
            else:
                messages.append("요청한 주문 변경을 반영했어요.")

        if blocked_constraint:
            if restriction_applied:
                messages.append("식이 조건을 반영했고, 아직 담지 않은 충돌 재료는 주문에서 제외했어요.")
            else:
                messages.append("식이 조건과 충돌한 항목은 주문에 반영하지 않았어요.")

        elif restriction_applied and not turn_result["kept_physical_conflict"]:
            messages.append("요청한 식이 조건을 반영했어요.")

        if blocked_active:
            messages.append("이미 담기 시작한 항목은 변경할 수 없어요.")

        if blocked_past:
            messages.append("이미 작업이 끝난 단계의 항목은 변경할 수 없어요.")

        if invalid:
            messages.append("메뉴에 없거나 형식이 맞지 않는 요청은 반영하지 않았어요.")

        if not messages and turn_result["ignored_same_value"]:
            messages.append("이미 같은 내용으로 선택되어 있어요.")

    # 추천 범위 질문·추천안 생성·추천 확정 결과를 검증값으로 안내한다.
    # 추천 범위 질문·추천안 생성·추천 확정 결과를 검증값으로 안내한다.
    recommendation = turn_result["recommendation_validation"]

    if (
        action in ("set_order", "set_order_and_confirm")
        and recommendation.get("proposal_revised")
    ):
        clean = recommendation["clean"]

        amount_labels = {
            "low": "적게",
            "normal": "보통",
            "high": "많이",
        }
        revised_parts = []

        if clean.get("sauce"):
            revised_parts.append(
                f"{clean['sauce']} 소스"
            )

        if clean.get("noodle_type"):
            revised_parts.append(
                clean["noodle_type"]
            )

        if clean.get("noodle_portion"):
            revised_parts.append(
                f"면 양 {amount_labels[clean['noodle_portion']]}"
            )

        for item, amount in clean.get("toppings", {}).items():
            revised_parts.append(
                f"{item} {amount_labels[amount]}"
            )

        messages.append(
            f"추천안을 {', '.join(revised_parts)} 조합으로 수정했어요. "
            "이대로 할까요?"
        )

    elif (
        action in ("set_order", "set_order_and_confirm")
        and recommendation.get("revision_reason")
    ):
        if (
            recommendation["revision_reason"]
            == "mixed_menu_and_restriction"
        ):
            messages.append(
                "추천안 수정과 식이·알레르기 조건을 "
                "한 번에 처리하지 않았어요. "
                "안전 조건을 먼저 따로 말씀해 주세요."
            )
        else:
            messages.append(
                "현재 추천안에서 어떤 항목을 바꿀지 "
                "정확히 확인하지 못했어요. "
                "추천안에 있는 재료와 변경 내용을 "
                "다시 말씀해 주세요."
            )

    elif action == "recommend_order":
        if recommendation["needs_scope"]:
            messages.append("현재 단계만 추천할까요, 아니면 남은 주문 전체를 추천할까요?")

        elif recommendation["proposal_created"]:
            amount_labels = {"low": "적게", "normal": "보통", "high": "많이"}
            order_before = turn_result["order_before"]
            clean = recommendation["clean"]
            selected_parts = []
            recommended_parts = []

            if order_before["sauce"]:
                selected_parts.append(f"{order_before['sauce']} 소스")

            if order_before["noodle_type"]:
                selected_parts.append(order_before["noodle_type"])

            if order_before["noodle_portion"]:
                selected_parts.append(f"면 양 {amount_labels[order_before['noodle_portion']]}")

            for item, amount in order_before["toppings"].items():
                selected_parts.append(f"{item} {amount_labels[amount]}")

            if clean.get("sauce"):
                recommended_parts.append(f"{clean['sauce']} 소스")

            if clean.get("noodle_type"):
                recommended_parts.append(clean["noodle_type"])

            if clean.get("noodle_portion"):
                recommended_parts.append(f"면 양 {amount_labels[clean['noodle_portion']]}")

            for item, amount in clean.get("toppings", {}).items():
                recommended_parts.append(f"{item} {amount_labels[amount]}")

            selected_text = ", ".join(selected_parts)
            recommended_text = ", ".join(recommended_parts)

            if selected_text:
                messages.append(f"이미 고르신 메뉴는 {selected_text}입니다. 여기에 {recommended_text} 조합을 추천해드려요. 이대로 할까요?")
            else:
                messages.append(f"{recommended_text} 조합을 추천해드려요. 이대로 할까요?")

        elif recommendation["blocked"]:
            messages.append("현재 제한 조건을 지키면 추천할 수 있는 재료가 없어요. 추천 범위나 제외 조건을 바꿔 말씀해 주세요.")

        elif recommendation["dropped"] and all(item.get("reason") == "already_selected" for item in recommendation["dropped"]):
            messages.append("요청한 추천 범위의 메뉴는 이미 모두 선택되어 있어요.")

        else:
            messages.append("추천 범위에 맞는 새 조합을 만들지 못했어요. 추천할 단계나 원하는 맛을 다시 말씀해 주세요.")

    elif action == "confirm_section" and recommendation["proposal_confirmed"]:
        messages.append("추천한 내용을 주문에 반영했어요.")

    elif action == "confirm_section" and (recommendation["dropped"] or recommendation["blocked"]):
        messages.append("추천안이 현재 주문이나 제한 조건과 충돌해서 반영하지 않았어요.")

        # section 전체 제외는 dislike 재확인과 안전 제약 즉시 적용을 구분한다.
    if action == "refuse_section":
        section_skip = turn_result["section_skip"]

        if section_skip is None:
            messages.append("현재 단계 전체 제외 요청을 정확히 이해하지 못했어요. 원하는 재료를 다시 말씀해 주세요.")
        elif section_skip["needs_confirmation"]:
            messages.append("현재 단계 재료를 모두 제외하고 다음 단계로 넘어갈까요?")
        elif section_skip["applied"]:
            messages.append("현재 단계 재료를 제외 요청에 반영했어요.")


    return {
        "policy_reply": " ".join(messages) if messages else None,
    }

def sanitize_free_reply(reply: str) -> tuple[str, list[str]]:
    # 자유응답 모델은 일반 질문의 답만 말한다.
    # 주문 반영·현재 상태·다음 행동을 주장하는 문장과 되묻는 문장은 제거한다.
    if not isinstance(reply, str):
        raise TypeError("자유응답은 문자열이어야 함")

    sentences = re.split(
        r"(?<=[.!?])\s+",
        reply.strip(),
    )

    safe_sentences = []
    removed_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        has_operational_claim = any(
            claim in sentence
            for claim in OPERATIONAL_CLAIMS
        )

        if "?" in sentence or has_operational_claim:
            removed_sentences.append(sentence)
            continue

        safe_sentences.append(sentence)

    return " ".join(safe_sentences).strip(), removed_sentences


def finalize_reply(reply: str) -> str:
    # 같은 문장이 반복되면 한 번만 남기되 필요한 문장은 길이 때문에 자르지 않는다.
    # UI 표기는 사용자 결정에 따라 '예요'를 '에요'로 통일한다.
    if not isinstance(reply, str):
        raise TypeError("최종 응답은 문자열이어야 함")

    reply = reply.strip().replace("예요", "에요")

    if not reply:
        return ""

    sentences = re.split(
        r"(?<=[.!?])\s+",
        reply,
    )

    unique_sentences = []
    normalized_sentences = set()

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        normalized = re.sub(
            r"[\s.,!?~'”“\"`]+",
            "",
            sentence,
        )

        if not normalized or normalized in normalized_sentences:
            continue

        normalized_sentences.add(normalized)
        unique_sentences.append(sentence)

    return " ".join(unique_sentences)


def build_turn_reply(user_text: str, state_before: dict, result: dict, generate_reply: Callable[[dict, str], str], *,
warning: Callable[[str], None] | None = None, scene_reply: Callable[[], str] | None = None,) -> str | None:
    # 한 턴의 최종 응답을 정책 응답 + 자유응답 + 다음 행동 순서로 조립한다.
    # 주문·진행 사실은 turn_result와 Python policy_reply만 말할 수 있다.
    turn_result = result.get("turn_result")
    policy_reply = result.get("policy_reply")

    if not isinstance(turn_result, dict):
        raise ValueError("Reply를 만들 turn_result가 없음")

    order_after = result.get("order")
    execution_after = result.get("execution")
    recommendation_after = result.get("recommendation")

    if not isinstance(order_after, dict):
        raise ValueError("Reply를 만들 order가 없음")

    if not isinstance(execution_after, dict):
        raise ValueError("Reply를 만들 execution이 없음")

    if not isinstance(recommendation_after, dict):
        raise ValueError("Reply를 만들 recommendation이 없음")

    reply_state = copy.deepcopy(state_before)
    reply_state["order"] = copy.deepcopy(order_after)
    reply_state["execution"] = copy.deepcopy(execution_after)
    reply_state["recommendation"] = copy.deepcopy(
        recommendation_after
    )

    next_step = build_selection_next_step(
        order_after,
        execution_after,
    )

    def append_next_step(reply: str) -> str:
        if not next_step:
            return reply

        if not reply:
            return next_step

        return f"{reply} {next_step}"

    if policy_reply is not None:
        if not isinstance(policy_reply, str) or not policy_reply.strip():
            raise ValueError("policy_reply 형식이 잘못됨")

        policy_reply = policy_reply.strip()
        action = turn_result.get("action")
        confirm = turn_result.get("confirm_validation", {})
        recommendation = turn_result.get(
            "recommendation_validation",
            {},
        )
        section_skip = turn_result.get("section_skip")

        # 이미 질문하거나 작업 시작·단계 이동이 확정된 응답에는
        # 별도의 다음 선택 안내를 붙이지 않는다.
        policy_already_asks_user = (
            action == "recommend_order" and (recommendation.get("needs_scope")or recommendation.get("proposal_created"))) or (
            action == "refuse_section"
            and isinstance(section_skip, dict)
            and section_skip.get("needs_confirmation")
        )

        work_will_start = (
            confirm.get("requested")
            and confirm.get("allowed")
        ) or recommendation.get("proposal_confirmed")

        section_will_advance = (
            isinstance(section_skip, dict)
            and section_skip.get("applied")
        )

        if (
            policy_already_asks_user
            or work_will_start
            or section_will_advance
        ):
            return finalize_reply(policy_reply)

        policy_already_guides_user = any(
            marker in policy_reply
            for marker in (
                "?",
                "말씀해 주세요",
                "다시 말씀해 주세요",
            )
        )

        if policy_already_guides_user:
            return finalize_reply(policy_reply)

        return finalize_reply(
            append_next_step(policy_reply)
        )

    action = turn_result.get("action")

    if action == "respond":
        free_reply = generate_reply(
            reply_state,
            user_text,
        )

        if not isinstance(free_reply, str):
            raise TypeError("자유응답 모델 결과가 문자열이 아님")

        safe_reply, removed_sentences = sanitize_free_reply(
            free_reply,
        )

        if removed_sentences and warning is not None:
            warning(
                "자유응답의 주문 흐름 문장을 제거함 : "
                f"{free_reply}"
            )

        # 자유응답의 운영 주장만 제거하고 문장 수는 강제로 제한하지 않는다.
        safe_reply = finalize_reply(safe_reply)

        if not safe_reply:
            fallback = (
                next_step
                or "현재 주문에서 원하시는 내용을 말씀해 주세요."
            )
            return finalize_reply(fallback)

        return finalize_reply(
            append_next_step(safe_reply)
        )

    if action == "describe_scene":
        if scene_reply is None:
            reply = (
                "LLM 디버깅 모드에는 카메라가 연결되어 있지 않아요."
            )
        else:
            reply = scene_reply()

            if not isinstance(reply, str) or not reply.strip():
                reply = "카메라 화면을 정확히 설명하지 못했어요."

        return finalize_reply(
            append_next_step(reply)
        )

    confirm = turn_result.get("confirm_validation", {})

    if confirm.get("requested") and confirm.get("allowed"):
        section = execution_after.get("section")

        if section not in SECTION_LABELS:
            raise ValueError(
                f"Reply section 이상함 : {section}"
            )

        return finalize_reply(
            f"{SECTION_LABELS[section]} 선택을 확정했어요."
        )

    return None
