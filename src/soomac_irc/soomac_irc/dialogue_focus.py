import copy


# focus는 주문을 해석하거나 수정하지 않고, 사용자가 최근 직접 말한 대상만 기억한다.
# 대화 턴 번호는 별도로 만들지 않고 SessionState.history의 완료된 user/assistant 쌍을 그대로 사용한다.
FOCUS_RECENT_LIMIT = 3
FOCUS_MAX_TURN_GAP = 3


def new_dialogue_focus() -> dict:
    # current는 가장 최근 mention event, recent는 그 이전 event를 최신순으로 보관한다.
    return {
        "current": None,
        "recent": [],
    }


def update_dialogue_focus(focus: dict, mentions: list[str], history_turn: int) -> dict:
    # mentions의 의미·지원 여부·주문 field를 해석하지 않고 Decision 출력과 등장 순서를 그대로 저장한다.
    # graph가 확정한 SessionState와 list를 공유하지 않도록 focus와 mentions를 모두 복사한다.
    if not isinstance(history_turn, int) or history_turn < 1:
        raise ValueError("history_turn은 1 이상의 int여야 함")

    updated = copy.deepcopy(focus)
    updated.setdefault("current", None)
    updated.setdefault("recent", [])

    if not mentions:
        # 직접 mention이 없는 턴은 current를 교체하지 않는다.
        # TTL은 caller가 넘기는 현재 history 턴과 event의 history_turn 차이로 계산한다.
        return updated

    previous_current = updated["current"]

    if previous_current is not None:
        updated["recent"].insert(0, copy.deepcopy(previous_current))
        updated["recent"] = updated["recent"][:FOCUS_RECENT_LIMIT]

    updated["current"] = {
        "mentions": copy.deepcopy(mentions),
        "history_turn": history_turn,
    }
    return updated


def remove_focus_mentions(focus: dict, mentions_to_remove: list[str]) -> dict:
    # focus는 task 의미를 해석하지 않고, caller가 넘긴 문자열과 정확히 같은 mention만 제거한다.
    # mention이 0개가 된 event는 참조할 대상이 없으므로 남기지 않는다. 빈 current를 resolver가 잡는 것을 막는다.
    # current가 비면 None으로 두고 recent를 current로 당기지 않는다. '그거'가 더 오래된 대상을 가리키지 않게 한다.
    updated = copy.deepcopy(focus)
    remove_set = set(mentions_to_remove)

    def without_removed(event):
        if not isinstance(event, dict):
            return None
        mentions = [m for m in event.get("mentions", []) if m not in remove_set]
        return {**event, "mentions": mentions} if mentions else None

    updated["current"] = without_removed(updated.get("current"))
    updated["recent"] = [
        event for event in (without_removed(e) for e in updated.get("recent", []))
        if event is not None
    ]

    return updated


def focus_event_is_fresh(event: dict | None, current_history_turn: int) -> bool:
    # 기존 SessionState.history 기준으로 event가 단독 지시어에 사용할 만큼 최근인지 확인한다.
    if not isinstance(event, dict):
        return False

    event_history_turn = event.get("history_turn")

    if not isinstance(event_history_turn, int) or not isinstance(current_history_turn, int):
        return False

    turn_gap = current_history_turn - event_history_turn
    return 0 <= turn_gap <= FOCUS_MAX_TURN_GAP


def current_focus_mentions(focus: dict, current_history_turn: int) -> list[str]:
    # caller가 focus 내부 list를 수정하지 못하도록 항상 복사본을 반환한다.
    current = focus.get("current")

    if not focus_event_is_fresh(current, current_history_turn):
        return []

    return copy.deepcopy(current.get("mentions", []))


def previous_focus_mentions(focus: dict, current_history_turn: int) -> list[str]:
    # "아까 그거"는 current 바로 이전 event만 사용하고 더 오래된 event를 임의 선택하지 않는다.
    recent = focus.get("recent", [])

    if not recent:
        return []

    previous = recent[0]

    if not focus_event_is_fresh(previous, current_history_turn):
        return []

    return copy.deepcopy(previous.get("mentions", []))

def _compact_reference_text(user_text: str) -> str:
    # STT가 띄어쓰기를 다르게 보내도 같은 reference 표현으로 비교하기 위해 공백만 제거한다.
    # 문장 전체 의미를 Python regex로 해석하는 게 아니라 이미 지원하는 지시어 표면형만 통일한다.
    return "".join(user_text.strip().split())


def build_reference_context(user_text: str,focus: dict,current_history_turn: int, question: dict | None = None) -> dict:
    # Decision output field가 아니라 model input과 graph 한 턴에서만 사용하는 Python 내부값이다.
    # status:
    # - none: reference 표현 자체가 없음
    # - resolved: target이 안전하게 하나 또는 둘로 확정됨
    # - ambiguous: 후보는 있지만 요구한 개수/순서 조건을 만족하지 않음
    # - missing: 사용할 focus event나 mention이 없음
    # - stale: focus event는 있지만 TTL을 넘김
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

    # 질문에 대한 답은 로봇이 실제 제시한 선택지를 참조한다.
    # '아까 그거'는 의도적으로 사용자 과거 mention 경로를 유지한다.
    uses_question = bool(question and "아까그거" not in text and "전에말한거" not in text)
    if uses_question:
        event = {"mentions": question.get("targets", []),
                 "history_turn": question["history_turn"]}

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


def resolve_focus_reference(user_text: str,focus: dict,current_history_turn: int) -> list[str] | None:
    # 기존 caller 계약은 유지한다.
    # []는 reference 없음, list[str]은 정상 해석, None은 ambiguous/missing/stale이다.
    context = build_reference_context(
        user_text,
        focus,
        current_history_turn,
    )

    if context["status"] == "none":
        return []

    if context["status"] == "resolved":
        return copy.deepcopy(context["targets"])

    return None
