import pytest
import torch

from soomac_irc import input_budget
from soomac_irc.input_budget import InputBudgetError, check_static_budget, fit_recent_history, input_token_limit


class CharProcessor:
    """user 메시지 글자 수를 토큰 수로 보는 가짜 processor(system은 고정 100토큰)."""

    def apply_chat_template(self, messages, **_kwargs):
        return {"input_ids": torch.zeros((1, 100 + len(messages[1]["content"])), dtype=torch.long)}


def _history(turns):
    rows = []
    for i in range(turns):
        rows += [{"role": "user", "content": f"u{i}" * 10}, {"role": "assistant", "content": f"a{i}" * 10}]
    return rows


def test_limits_reserve_output_tokens():
    assert input_budget.SERVING_MAX_MODEL_LEN == 10240
    assert input_token_limit(1024) == 9216
    assert input_token_limit(512) == 9728


def test_under_limit_input_is_unchanged():
    model_input = {"recent_history": _history(3), "message": "hi"}
    original = [dict(m) for m in model_input["recent_history"]]
    messages, inputs, dropped = fit_recent_history(CharProcessor(), "SYS", model_input, 10_000, "T")
    assert dropped == 0
    assert model_input["recent_history"] == original
    assert messages[0]["content"] == "SYS"


def test_over_limit_drops_oldest_turns_until_it_fits():
    model_input = {"recent_history": _history(10), "message": "hi"}
    full = CharProcessor().apply_chat_template(
        [{}, {"content": __import__("json").dumps(model_input, ensure_ascii=False, separators=(",", ":"))}]
    )["input_ids"].shape[1]
    messages, inputs, dropped = fit_recent_history(CharProcessor(), "SYS", model_input, full - 60, "T")
    assert dropped > 0 and dropped % 2 == 0
    assert inputs["input_ids"].shape[1] <= full - 60
    # 남은 기록은 가장 최근 쪽이다.
    assert model_input["recent_history"][-1]["content"] == "a9" * 10


def test_raises_when_nothing_left_to_drop():
    with pytest.raises(InputBudgetError, match="뺄 대화 기록이 없음"):
        fit_recent_history(CharProcessor(), "SYS", {"recent_history": [], "message": "x" * 50}, 120, "T")


def test_static_budget_check_fails_fast_when_prompt_too_long():
    check_static_budget(CharProcessor(), "SYS", 100 + 2 + input_budget.DYNAMIC_INPUT_RESERVE_TOKENS, "T")
    with pytest.raises(InputBudgetError, match="system prompt"):
        check_static_budget(CharProcessor(), "SYS", 100 + 2 + input_budget.DYNAMIC_INPUT_RESERVE_TOKENS - 1, "T")


def test_decision_raises_input_budget_error_when_history_is_gone(monkeypatch):
    from soomac_irc import decision_model

    monkeypatch.setattr(decision_model, "DECISION_INPUT_MAX_TOKENS", 120)
    session = {"history": [], "order": {}, "preferences": {}, "pending": None}
    with pytest.raises(InputBudgetError, match="Decision 입력 token 초과"):
        decision_model.build_decision_inputs(session, "x" * 50, {}, CharProcessor(), system_prompt="SYS")


def test_odd_history_with_leading_greeting_keeps_user_first():
    # 시작 안내(assistant 단독) + 대화 턴 → 잘라도 남은 기록은 user로 시작한다.
    history = [{"role": "assistant", "content": "g" * 30}] + _history(5)
    model_input = {"recent_history": history, "message": "hi"}
    full = 100 + len(__import__("json").dumps(model_input, ensure_ascii=False, separators=(",", ":")))
    empty = 100 + len(__import__("json").dumps({"recent_history": [], "message": "hi"}, separators=(",", ":")))
    for limit in range(full - 1, empty - 1, -7):
        trial = {"recent_history": [dict(m) for m in history], "message": "hi"}
        fit_recent_history(CharProcessor(), "SYS", trial, limit, "T")
        kept = trial["recent_history"]
        assert not kept or kept[0]["role"] == "user"
        assert kept == history[len(history) - len(kept):]
