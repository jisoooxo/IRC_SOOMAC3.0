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
    # event 자체는 남겨서 recent[0]이 더 오래된 event로 당겨지는 것을 막는다.
    updated = copy.deepcopy(focus)
    remove_set = set(mentions_to_remove)

    current = updated.get("current")

    if isinstance(current, dict):
        current["mentions"] = [
            mention for mention in current.get("mentions", [])
            if mention not in remove_set
        ]

    for event in updated.get("recent", []):
        if not isinstance(event, dict):
            continue

        event["mentions"] = [
            mention for mention in event.get("mentions", [])
            if mention not in remove_set
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