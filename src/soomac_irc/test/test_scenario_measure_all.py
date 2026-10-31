"""The measuring harness must not stop or repair state on semantic mismatches."""
import copy
import io
import json

import pytest

import manual_scenario_stack_probe as probe


@pytest.mark.parametrize("measure_all, expected_turns", [(False, 1), (True, 6)])
def test_semantic_failure_does_not_stop_measure_all_or_force_robot_stage(monkeypatch, measure_all, expected_turns):
    class StubCall:
        def __init__(self):
            self.trace_events = []

    class NoMutationGraph:
        def invoke(self, state):
            session = copy.deepcopy(state["session"])
            session["history"] += [{"role": "user", "content": state["user_text"]},
                                   {"role": "assistant", "content": "계측용 응답"}]
            return {"session": session, "decision": {"commit": False}, "reply": "계측용 응답",
                    "policy": {"status": "pass", "execute": False, "reason": "state_update_only"}}

    monkeypatch.setattr("soomac_irc.decision_model.make_call_decision", lambda *_args, **_kw: StubCall())
    monkeypatch.setattr("soomac_irc.recommendation_model.make_call_recommendation", lambda *_args, **_kw: StubCall())
    monkeypatch.setattr("soomac_irc.response_model.make_call_response", lambda *_args, **_kw: StubCall())
    monkeypatch.setattr("soomac_irc.llm_langgraph.build_graph", lambda *_args: NoMutationGraph())
    stream = io.StringIO()
    summary = probe.run_episode(True, None, None, stream, measure_all=measure_all)
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert len(rows) == summary["observed_turns"] == expected_turns
    assert all(row["robot_before"]["section"] == "noodle" for row in rows)
    assert all(row["after"]["order"]["toppings"] == {} for row in rows)
    assert summary["plans"] == [] and summary["passed"] is False
    assert len(rows[-1]["after"]["history"]) == 1 + expected_turns * 2
    if measure_all:
        assert rows[-1]["expected_section"] == "extra"
        assert rows[-1]["checks"]["section_matches"] is False
        assert rows[-1]["expected_order"]["toppings"] == {
            "양파": "low", "버섯": "normal", "소시지": "high", "치즈": "low"}
        assert summary["unreached"] == []
