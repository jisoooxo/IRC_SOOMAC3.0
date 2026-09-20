import copy
import re
from collections.abc import Callable

from soomac_irc.agent_v7 import build_selection_next_step
from soomac_irc.order_v7 import SECTION_LABELS


REPLY_GUARD_VERSION = "debug_v1.20260920"


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
    "추가됐",
    "추가되었",
    "저장됐",
    "저장되었",
    "담는 중",
    "담고 있",
    "작업은 완료",
    "작업이 완료",
    "단계로 넘어",
)

MENU_MARKERS = (
    "오일", "토마토", "크림", "얇은면", "넓은면", "면", "양파", "버섯",
    "소시지", "게살", "치즈", "페퍼론치노", "소스", "야채", "육류", "재료",
)

USER_OPERATION_MARKERS = (
    "적게", "보통", "많이", "빼", "제외", "추가", "넣", "담", "바꿔", "변경",
    "진행", "시작", "확정", "추천", "골라", "선택해", "취소",
)

RESTRICTION_MARKERS = ("알레르기", "못먹", "싫어", "안먹")

REPLY_OPERATION_MARKERS = (
    "추가", "저장", "반영", "변경", "바뀌", "확정", "처리", "제외", "취소",
    "담", "넣", "진행", "시작", "완료", "끝", "넘어", "선택", "차례",
)

REPLY_WORKFLOW_MARKERS = MENU_MARKERS + ("주문", "작업", "단계", "공정", "로봇")


def is_operational_user_text(user_text: str) -> bool:
    """respond 오분류 시 자유응답을 막아야 할 주문 동작 표현인지 보수적으로 판정한다."""
    if not isinstance(user_text, str):
        return False

    normalized = re.sub(r"[\s\u200b-\u200d\ufeff]+", "", user_text.lower())
    has_menu = any(re.sub(r"\s+", "", marker) in normalized for marker in MENU_MARKERS)
    has_operation = any(re.sub(r"\s+", "", marker) in normalized for marker in USER_OPERATION_MARKERS)
    has_restriction = any(marker in normalized for marker in RESTRICTION_MARKERS)
    return has_operation or (has_menu and has_restriction)


def _is_operational_reply_sentence(sentence: str) -> bool:
    normalized = re.sub(r"[\s\u200b-\u200d\ufeff]+", "", sentence)
    normalized_claims = (
        re.sub(r"\s+", "", claim)
        for claim in OPERATIONAL_CLAIMS
    )

    if any(claim in normalized for claim in normalized_claims):
        return True

    has_workflow_subject = any(marker in normalized for marker in REPLY_WORKFLOW_MARKERS)
    has_operation = any(marker in normalized for marker in REPLY_OPERATION_MARKERS)
    return has_workflow_subject and has_operation


def sanitize_free_reply(reply: str) -> tuple[str, list[str]]:
    """자유응답에서 검증되지 않은 주문·진행 문장을 제거한다."""
    if not isinstance(reply, str):
        raise TypeError("자유응답은 문자열이어야 함")

    sentences = re.split(r"(?<=[.!?])\s+", reply.strip())
    kept = []
    removed = []

    for sentence in sentences:
        if not sentence:
            continue

        if "?" in sentence or _is_operational_reply_sentence(sentence):
            removed.append(sentence)
        else:
            kept.append(sentence)

    return " ".join(kept).strip(), removed


def build_turn_reply(
    user_text: str,
    state_before: dict,
    result: dict,
    generate_reply: Callable[[dict, str], str],
    *,
    warning: Callable[[str], None] | None = None,
    scene_reply: Callable[[], str] | None = None,
) -> str | None:
    """검증된 transaction과 제한된 자유응답을 합쳐 최종 문장을 만든다."""
    transaction = result.get("transaction")
    policy_reply = result.get("policy_reply")

    if not isinstance(transaction, dict):
        raise ValueError("Reply를 만들 transaction이 없음")

    order_after = result.get("order")
    execution_after = result.get("execution")
    recommendation_after = result.get("recommendation")

    if not isinstance(order_after, dict) or not isinstance(execution_after, dict):
        raise ValueError("Reply를 만들 order 또는 execution이 없음")

    if not isinstance(recommendation_after, dict):
        raise ValueError("Reply를 만들 recommendation이 없음")

    reply_state = copy.deepcopy(state_before)
    reply_state["order"] = copy.deepcopy(order_after)
    reply_state["execution"] = copy.deepcopy(execution_after)
    reply_state["recommendation"] = copy.deepcopy(recommendation_after)
    next_step = build_selection_next_step(order_after, execution_after)

    def append_next_step(reply: str) -> str:
        return f"{reply} {next_step}" if next_step else reply

    if policy_reply is not None:
        if not isinstance(policy_reply, str) or not policy_reply.strip():
            raise ValueError("policy_reply 형식이 잘못됨")

        policy_reply = policy_reply.strip()
        action = transaction.get("action")
        confirm = transaction.get("confirm_validation", {})
        recommendation = transaction.get("recommendation_validation", {})
        section_skip = transaction.get("section_skip")
        policy_already_asks_user = (
            action == "recommend_order"
            and (
                recommendation.get("needs_scope")
                or recommendation.get("proposal_created")
            )
        ) or (
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

        if policy_already_asks_user or work_will_start or section_will_advance:
            return policy_reply

        policy_already_guides_user = any(marker in policy_reply for marker in (
            "?",
            "말씀해 주세요",
            "다시 말씀해 주세요",
        ))

        if policy_already_guides_user:
            return policy_reply

        return append_next_step(policy_reply)

    action = transaction.get("action")

    if action == "respond":
        if is_operational_user_text(user_text):
            if warning is not None:
                warning("respond Tool이 주문 동작 표현을 포함해 자유응답 생성을 차단함")

            return next_step or "주문 변경이나 진행 요청을 다시 명확하게 말씀해 주세요."

        free_reply = generate_reply(reply_state, user_text).strip()
        safe_reply, removed = sanitize_free_reply(free_reply)

        if removed and warning is not None:
            warning(f"자유응답의 주문 흐름 문장을 제거함 : {free_reply}")

        if not safe_reply:
            return next_step or "현재 주문에서 원하시는 내용을 말씀해 주세요."

        return append_next_step(safe_reply)

    if action == "describe_scene":
        if scene_reply is None:
            return append_next_step("LLM 디버깅 모드에는 카메라가 연결되어 있지 않아요.")

        return append_next_step(scene_reply())

    confirm = transaction.get("confirm_validation", {})

    if confirm.get("requested") and confirm.get("allowed"):
        section = execution_after.get("section")

        if section not in SECTION_LABELS:
            raise ValueError(f"Reply section 이상함 : {section}")

        return f"{SECTION_LABELS[section]} 선택을 확정했어요."

    return None
