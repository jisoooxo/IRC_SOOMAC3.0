"""v2 dataset migration에서만 쓰는 과거 지시어 해석기이다.

현재 runtime은 이 로직을 사용하지 않는다. Decision Agent가 recent_history와
pending을 직접 읽기 전 생성된 dataset을 재검사할 때만 보존한다.
"""
FOCUS_MAX_TURN_GAP = 3

# 이 표현 목록은 v2 dataset validator의 과거 positive gate를 재현할 때만 사용한다.
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


def focus_event_is_fresh(event: dict | None, current_history_turn: int) -> bool:
    if not isinstance(event, dict):
        return False

    event_history_turn = event.get("history_turn")
    if not isinstance(event_history_turn, int) or not isinstance(current_history_turn, int):
        return False

    turn_gap = current_history_turn - event_history_turn
    return 0 <= turn_gap <= FOCUS_MAX_TURN_GAP


def _compact_reference_text(user_text: str) -> str:
    return "".join(user_text.strip().split())


def build_reference_context(
    user_text: str,
    focus: dict,
    current_history_turn: int,
    question: dict | None = None,
) -> dict:
    """과거 v2 dataset의 reference_context를 원래 규칙대로 복원한다."""
    text = _compact_reference_text(user_text)

    if "아까그거" in text or "전에말한거" in text:
        recent = focus.get("recent", [])
        event = recent[0] if recent else None
        reference_kind = "single"
    elif "둘다" in text or "두개다" in text:
        event = focus.get("current")
        reference_kind = "plural"
    elif "첫번째거" in text:
        event = focus.get("current")
        reference_kind = "first"
    elif "두번째거" in text:
        event = focus.get("current")
        reference_kind = "second"
    elif any(reference in text for reference in ("방금그거", "그거", "이거")):
        event = focus.get("current")
        reference_kind = "single"
    else:
        return {"status": "none", "targets": []}

    uses_question = bool(
        question
        and "아까그거" not in text
        and "전에말한거" not in text
    )
    if uses_question:
        event = {
            "mentions": question.get("targets", []),
            "history_turn": question["history_turn"],
        }

    if not isinstance(event, dict):
        return {"status": "missing", "targets": []}

    if not uses_question and not focus_event_is_fresh(event, current_history_turn):
        return {"status": "stale", "targets": []}

    targets = list(dict.fromkeys(event.get("mentions", [])))
    if not targets:
        return {"status": "missing", "targets": []}

    if reference_kind == "plural":
        if len(targets) == 2:
            return {"status": "resolved", "targets": targets}
        return {"status": "ambiguous", "targets": targets}

    if reference_kind == "first":
        if len(targets) >= 1:
            return {"status": "resolved", "targets": [targets[0]]}
        return {"status": "ambiguous", "targets": targets}

    if reference_kind == "second":
        if len(targets) >= 2:
            return {"status": "resolved", "targets": [targets[1]]}
        return {"status": "ambiguous", "targets": targets}

    if len(targets) == 1:
        return {"status": "resolved", "targets": targets}
    return {"status": "ambiguous", "targets": targets}
