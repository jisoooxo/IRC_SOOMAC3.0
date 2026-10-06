"""자연어 우선 runtime 계약의 회귀 테스트이다.

고정된 모의 모델 출력을 실제 LangGraph에 넣어 state와 policy 결과를 확인한다.
삭제한 dialogue_focus·pending_question·pending_confirmation 기반 테스트를 대신한다.
"""
import copy
import json
from pathlib import Path

import xgrammar as xgr

from soomac_irc.agent_contract import DECISION_SCHEMA, normalize_decision
from soomac_irc.decision_model import DECISION_KEYS_ANY_ORDER
from soomac_irc.llm_langgraph import (
    DuplicateDecisionKeyError,
    build_graph,
    build_preselected_section_confirmation,
    new_session_state,
    new_turn_state,
)


def robot_state(section="veggie", *, completed=None, active=None, queue=None, started=False):
    return {
        "section": section,
        "task_queue": list(queue or []),
        "active_task": copy.deepcopy(active),
        "completed_tasks": copy.deepcopy(completed or []),
        "robot_started": started,
    }


def compiled_for(sparse_decision, recommendation=None):
    def call_decision(_session, _text, _robot_state, _repair):
        if isinstance(sparse_decision, Exception):
            raise sparse_decision
        return normalize_decision(copy.deepcopy(sparse_decision))

    def call_recommendation(*_args, **_kwargs):
        return recommendation or {
            "proposal": {
                "sauce": None,
                "noodle_type": None,
                "noodle_portion": None,
                "toppings": {},
            },
            "reason_tags": [],
        }

    def call_response(*_args, **_kwargs):
        return "ok"

    return build_graph(call_decision, call_recommendation, call_response)


def run(session, decision, state, text="테스트"):
    return compiled_for(decision).invoke(new_turn_state(session, text, state))


def test_schema_and_session_use_the_v3_contract():
    properties = DECISION_SCHEMA["properties"]
    assert "mentions" not in properties
    assert "queries" not in properties
    assert "cancel" not in properties
    assert "oneOf" not in DECISION_SCHEMA
    assert "maxProperties" not in DECISION_SCHEMA
    assert "cancel" not in normalize_decision({"route": "task"})
    assert set(new_session_state()) == {
        "order", "preferences", "pending", "history", "action_history"
    }


def test_xgrammar_accepts_shuffled_keys_but_keeps_schema_strict():
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo(["a"]))
    grammar = compiler.compile_json_schema(
        DECISION_SCHEMA,
        any_order=DECISION_KEYS_ANY_ORDER,
    )

    accepted_values = [
        {
            "commit": True,
            "preferences": [{"action": "add", "value": "매콤하게"}],
            "order": {
                "toppings": {"양파": "high"},
                "noodle_portion": "normal",
                "noodle_type": "넓은면",
                "sauce": "토마토",
            },
            "route": "task",
        },
        {
            "restrictions": [
                {"action": "add", "reason": "allergy", "target": "치즈"}
            ],
            "order": {"toppings": {"치즈": "none"}},
            "route": "task",
        },
    ]
    rejected_values = [
        {"route": "task", "unknown": True},
        {"route": "task", "cancel": True},
        {"route": "task", "commit": "true"},
        {"route": "task", "order": {"toppings": {"양파": "huge"}}},
    ]

    for value in accepted_values:
        matcher = xgr.GrammarMatcher(grammar, terminate_without_stop_token=True)
        raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        assert matcher.accept_string(raw)
        assert matcher.is_completed()

    for value in rejected_values:
        matcher = xgr.GrammarMatcher(grammar, terminate_without_stop_token=True)
        raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        assert not matcher.accept_string(raw) or not matcher.is_completed()


def test_one_decision_applies_multiple_fields_and_semantics_together():
    decision = {
        "route": "task",
        "order": {
            "sauce": "토마토",
            "noodle_type": "넓은면",
            "noodle_portion": "high",
            "toppings": {
                "양파": "high",
                "버섯": "low",
                "소시지": "normal",
            },
        },
        "restrictions": [
            {"target": "치즈", "reason": "dislike", "action": "add"},
            {"target": "게살", "reason": "cannot_eat", "action": "add"},
        ],
        "preferences": [
            {"value": "매콤하게", "action": "add"},
            {"value": "푸짐하게", "action": "add"},
        ],
    }

    result = run(new_session_state(), decision, robot_state(section="noodle"))
    order = result["session"]["order"]

    assert order["sauce"] == "토마토"
    assert order["noodle_type"] == "넓은면"
    assert order["noodle_portion"] == "high"
    assert order["toppings"] == {
        "양파": "high",
        "버섯": "low",
        "소시지": "normal",
    }
    assert order["restrictions"] == [
        {"target": "치즈", "reason": "dislike"},
        {"target": "게살", "reason": "cannot_eat"},
    ]
    assert result["session"]["preferences"] == [
        {"value": "매콤하게"},
        {"value": "푸짐하게"},
    ]


def test_unsupported_candidate_is_preserved_then_blocked():
    result = run(
        new_session_state(),
        {"route": "task", "order": {"toppings": {"햄": "high"}}},
        robot_state(),
    )
    assert result["policy"]["reason"] == "unsupported"
    assert result["policy"]["execute"] is False
    assert result["session"]["order"]["toppings"] == {}
    assert result["policy"]["unsupported"][0]["item"] == "햄"


def test_valid_fields_survive_when_other_fields_are_rejected():
    decision = {
        "route": "task",
        "order": {
            "sauce": "토마토",
            "noodle_type": "라면",
            "noodle_portion": "normal",
            "toppings": {"양파": "high", "햄": "low"},
        },
    }
    result = run(new_session_state(), decision, robot_state(section="noodle"))

    assert result["policy"]["reason"] == "unsupported"
    assert result["policy"]["execute"] is False
    assert result["session"]["order"]["sauce"] == "토마토"
    assert result["session"]["order"]["noodle_type"] is None
    assert result["session"]["order"]["noodle_portion"] == "normal"
    assert result["session"]["order"]["toppings"] == {"양파": "high"}


def test_completed_item_cannot_be_removed():
    session = new_session_state()
    session["order"]["toppings"]["양파"] = "normal"
    result = run(
        session,
        {"route": "task", "order": {"toppings": {"양파": "none"}}},
        robot_state(
            section="veggie",
            completed=[{"class": "양파", "repeat_count": 2}],
            started=True,
        ),
    )
    assert result["policy"]["reason"] == "physical_state"
    assert result["session"]["order"]["toppings"]["양파"] == "normal"


def test_active_scalar_selection_cannot_be_removed():
    session = new_session_state()
    session["order"].update({"noodle_type": "넓은면", "noodle_portion": "high"})
    result = run(
        session,
        {
            "route": "task",
            "order": {"noodle_type": "none", "noodle_portion": "none"},
        },
        robot_state(
            section="noodle",
            active={"class": "넓은면", "repeat_count": 3},
            started=True,
        ),
    )
    assert result["policy"]["reason"] == "physical_state"
    assert result["session"]["order"]["noodle_type"] == "넓은면"
    assert result["session"]["order"]["noodle_portion"] == "high"


def test_specific_and_current_section_selections_can_be_removed():
    session = new_session_state()
    session["order"]["toppings"]["양파"] = "high"
    session["order"]["toppings"]["버섯"] = "low"

    one_removed = run(
        session,
        {"route": "task", "order": {"toppings": {"양파": "none"}}},
        robot_state(section="veggie"),
    )
    assert one_removed["session"]["order"]["toppings"] == {"버섯": "low"}

    all_removed = run(
        session,
        {
            "route": "task",
            "order": {"toppings": {"양파": "none", "버섯": "none"}},
        },
        robot_state(section="veggie"),
    )
    assert all_removed["session"]["order"]["toppings"] == {}


def test_scalar_none_becomes_canonical_none_and_clears_noodle_pair():
    session = new_session_state()
    session["order"].update({
        "sauce": "토마토",
        "noodle_type": "넓은면",
        "noodle_portion": "high",
    })
    result = run(
        session,
        {"route": "task", "order": {"sauce": "none", "noodle_type": "none"}},
        robot_state(section="noodle"),
    )
    assert result["session"]["order"]["sauce"] is None
    assert result["session"]["order"]["noodle_type"] is None
    assert result["session"]["order"]["noodle_portion"] is None


def test_remove_all_and_commit_allows_optional_section_skip():
    session = new_session_state()
    session["order"]["toppings"] = {"양파": "high", "버섯": "low"}
    result = run(
        session,
        {
            "route": "task",
            "order": {"toppings": {"양파": "none", "버섯": "none"}},
            "commit": True,
        },
        robot_state(section="veggie"),
    )
    assert result["session"]["order"]["toppings"] == {}
    assert result["policy"]["reason"] == "execution_allowed"
    assert result["policy"]["execute"] is True


def test_past_section_removal_is_blocked_without_state_change():
    session = new_session_state()
    session["order"]["toppings"]["양파"] = "high"
    result = run(
        session,
        {"route": "task", "order": {"toppings": {"양파": "none"}}},
        robot_state(section="meat", started=True),
    )
    assert result["policy"]["reason"] == "physical_state"
    assert result["session"]["order"]["toppings"]["양파"] == "high"


def test_mixed_removal_addition_and_preference_all_survive():
    session = new_session_state()
    session["order"]["toppings"] = {"양파": "high", "버섯": "low"}
    session["order"]["restrictions"].append({"target": "치즈", "reason": "allergy"})
    session["preferences"].append({"value": "담백하게"})
    result = run(
        session,
        {
            "route": "task",
            "order": {
                "toppings": {
                    "양파": "none",
                    "버섯": "none",
                    "소시지": "high",
                },
            },
            "preferences": [{"value": "푸짐하게", "action": "add"}],
        },
        robot_state(section="veggie"),
    )
    assert result["session"]["order"]["toppings"] == {"소시지": "high"}
    assert result["session"]["order"]["restrictions"] == [
        {"target": "치즈", "reason": "allergy"}
    ]
    assert result["session"]["preferences"] == [
        {"value": "담백하게"},
        {"value": "푸짐하게"},
    ]


def test_reset_topics_and_handlers_remain_in_runtime_node():
    node_path = Path(__file__).parents[1] / "soomac_irc" / "llm_langgraph_node.py"
    source = node_path.read_text(encoding="utf-8")
    assert '"/ui/reset"' in source
    assert '"/llm/reset"' in source
    assert "def _process_reset(" in source
    assert "def _process_finish(" in source


def test_new_allergy_removes_editable_future_selection():
    session = new_session_state()
    session["order"]["toppings"]["치즈"] = "normal"
    result = run(
        session,
        {
            "route": "task",
            "restrictions": [
                {"target": "치즈", "reason": "allergy", "action": "add"}
            ],
        },
        robot_state(section="veggie"),
    )
    assert "치즈" not in result["session"]["order"]["toppings"]
    assert {"target": "치즈", "reason": "allergy"} in result["session"]["order"]["restrictions"]


def test_new_allergy_after_completed_item_preserves_physical_truth():
    session = new_session_state()
    session["order"]["toppings"]["치즈"] = "normal"
    result = run(
        session,
        {
            "route": "task",
            "restrictions": [
                {"target": "치즈", "reason": "allergy", "action": "add"}
            ],
        },
        robot_state(
            section="sauce",
            completed=[{"class": "치즈", "repeat_count": 2}],
            started=True,
        ),
    )
    assert result["policy"]["reason"] == "completed_restriction_conflict"
    assert result["session"]["order"]["toppings"]["치즈"] == "normal"


def test_existing_allergy_blocks_direct_order_without_override_pending():
    session = new_session_state()
    session["order"]["restrictions"].append({"target": "치즈", "reason": "allergy"})
    result = run(
        session,
        {"route": "task", "order": {"toppings": {"치즈": "normal"}}},
        robot_state(section="extra"),
    )
    assert result["policy"]["reason"] == "restriction_conflict"
    assert "치즈" not in result["session"]["order"]["toppings"]
    assert result["session"]["pending"] is None


def test_old_dislike_can_be_explicitly_overridden():
    session = new_session_state()
    session["order"]["restrictions"].append({"target": "치즈", "reason": "dislike"})
    result = run(
        session,
        {"route": "task", "order": {"toppings": {"치즈": "low"}}},
        robot_state(section="extra"),
    )
    assert result["session"]["order"]["toppings"]["치즈"] == "low"


def test_current_update_creates_execution_pending_then_accept_executes():
    session = new_session_state()
    first = run(
        session,
        {"route": "task", "order": {"toppings": {"양파": "normal"}}},
        robot_state(section="veggie"),
    )
    assert first["session"]["pending"] == {
        "type": "execution",
        "source": "current_update",
        "section": "veggie",
        "targets": ["양파"],
    }

    accepted = run(
        first["session"],
        {"route": "task", "confirmation": "accept"},
        robot_state(section="veggie"),
    )
    assert accepted["policy"]["reason"] == "execution_allowed"
    assert accepted["policy"]["execute"] is True
    assert accepted["session"]["pending"] is None


def test_status_turn_keeps_execution_pending():
    session = new_session_state()
    session["order"]["toppings"]["양파"] = "normal"
    session["pending"] = {
        "type": "execution",
        "source": "current_update",
        "section": "veggie",
        "targets": ["양파"],
    }
    result = run(
        session,
        {"route": "task"},
        robot_state(section="veggie"),
        text="지금까지 뭐했어?",
    )
    assert result["session"]["pending"]["type"] == "execution"
    assert result["policy"]["execute"] is False


def test_recommendation_stays_pending_until_acceptance():
    session = new_session_state()
    recommendation = {
        "proposal": {
            "sauce": None,
            "noodle_type": None,
            "noodle_portion": None,
            "toppings": {"양파": "normal"},
        },
        "reason_tags": ["담백함"],
    }
    graph = compiled_for(
        {"route": "task", "recommendation": {"action": "request"}},
        recommendation,
    )
    proposed = graph.invoke(new_turn_state(session, "추천해줘", robot_state()))
    assert proposed["session"]["order"]["toppings"] == {}
    assert proposed["session"]["pending"]["type"] == "recommendation"

    accepted = run(
        proposed["session"],
        {"route": "task", "confirmation": "accept"},
        robot_state(),
    )
    assert accepted["session"]["order"]["toppings"] == {"양파": "normal"}


def test_preselected_future_choice_uses_execution_pending():
    session = new_session_state()
    session["order"]["toppings"]["소시지"] = "high"
    pending = build_preselected_section_confirmation(session, "meat")
    assert pending["type"] == "execution"
    assert pending["source"] == "preselected"
    assert pending["targets"] == ["소시지"]


def test_commit_is_model_owned_but_runtime_checks_missing_and_busy():
    ready = new_session_state()
    ready["order"].update({
        "sauce": "토마토",
        "noodle_type": "넓은면",
        "noodle_portion": "normal",
    })
    allowed = run(ready, {"route": "task", "commit": True}, robot_state(section="noodle"))
    assert allowed["policy"]["reason"] == "execution_allowed"
    assert allowed["policy"]["execute"] is True

    missing = run(new_session_state(), {"route": "task", "commit": True}, robot_state(section="noodle"))
    assert missing["policy"]["reason"] == "missing_order"
    assert missing["policy"]["execute"] is False

    busy = run(
        ready,
        {"route": "task", "commit": True},
        robot_state(section="noodle", active={"class": "넓은면", "repeat_count": 2}),
    )
    assert busy["policy"]["reason"] == "robot_busy"
    assert busy["policy"]["execute"] is False


def test_duplicate_decision_key_falls_back_without_mutating_state():
    session = new_session_state()
    graph = compiled_for(DuplicateDecisionKeyError("duplicate route"))
    result = graph.invoke(new_turn_state(session, "양파 넣어줘", robot_state()))
    assert result["policy"]["reason"] == "understanding"
    assert result["policy"]["execute"] is False
    assert result["session"]["order"] == session["order"]
