"""GL04 controlled-state contrasts, not actual LLM comprehension tests.

Keep these separate from closed-loop model diagnostics: the decisions below are
oracle inputs and never prove that a model interpreted natural-language history.
"""
import copy
import json

import pytest

from soomac_irc.agent_contract import empty_order_patch, new_session_state, new_turn_state, normalize_decision
from soomac_irc.llm_langgraph import _canonicalize_structured_flavor_preference, build_graph
from soomac_irc.llm_policy import apply_order_patch, build_recommendation_scope, validate_recommendation_proposal


@pytest.mark.parametrize("action", ["add", "remove"])
def test_spicy_cannot_eat_keeps_its_strong_restriction_meaning(action):
    # Compound oracle: already resolved order, restriction and recommendation.
    # Python must not turn the strong restriction into a soft preference.
    restriction = {"target": "매운맛", "reason": "cannot_eat", "action": action}
    decision = normalize_decision({
        "route": "task", "restrictions": [restriction],
        "order": {"toppings": {"소시지": "high"}},
        "recommendation": {"action": "request"},
    })
    before = copy.deepcopy(decision)
    migrated = _canonicalize_structured_flavor_preference(decision)
    assert restriction in migrated["restriction_options"]
    assert migrated["order_patch"] == before["order_patch"]
    assert migrated["recommendation"] == before["recommendation"]
    assert decision == before


@pytest.mark.parametrize("amount", ["low", "high"])
def test_already_confirmed_quantity_is_unchanged_by_an_unrelated_patch(amount):
    # Exclusion has already been resolved by an oracle Decision. No natural-
    # language keyword or implicit normal reset belongs in this state operation.
    order = new_session_state()["order"]
    order["toppings"] = {"소시지": amount, "게살": "normal"}
    before = copy.deepcopy(order)
    patch = empty_order_patch()
    patch["toppings"] = {"게살": "none"}
    updated = apply_order_patch(order, patch)
    assert updated["toppings"] == {"소시지": amount}
    assert order == before


def run_oracle(session, sparse, section="extra", proposal=None, captures=None, text="controlled oracle turn", robot=None):
    def decision(*_args):
        decision.last_external_output = copy.deepcopy(sparse)
        return normalize_decision(copy.deepcopy(sparse))

    def recommendation(*_args):
        return {"proposal": copy.deepcopy(proposal or empty_order_patch()), "reason_tags": ["검수 후보"]}

    def response(*args):
        if captures is not None:
            captures.append(copy.deepcopy(args))
        return "ok"

    robot = copy.deepcopy(robot) if robot is not None else {
        "section": section, "task_queue": [], "active_task": None,
        "completed_tasks": [], "robot_started": False,
    }
    graph = build_graph(decision, recommendation, response)
    return graph.invoke(new_turn_state(session, text, robot)), robot


def test_strong_spicy_add_compares_the_entire_session_and_preserves_input():
    session = new_session_state()
    session["order"]["sauce"] = "오일"
    session["order"]["toppings"] = {"소시지": "high"}
    session["preferences"] = [{"value": "버섯 좋아함"}]
    session["history"] = [{"role": "user", "content": "소시지는 많이"}]
    before = copy.deepcopy(session)
    restriction = {"target": "매운맛", "reason": "cannot_eat"}
    result, robot = run_oracle(session, {"route": "task", "restrictions": [{**restriction, "action": "add"}]})
    expected = copy.deepcopy(before)
    expected["order"]["restrictions"] = [restriction]
    expected["history"] += [{"role": "user", "content": "controlled oracle turn"},
                            {"role": "assistant", "content": "ok"}]
    expected["action_history"] += [{
        "type": "turn_applied", "order_changes": {}, "restriction_added": [restriction],
        "restriction_removed": [], "preference_added": [], "preference_removed": [],
    }]
    assert result["session"] == expected
    assert session == before
    assert result["robot_state"] == robot
    assert result["policy"]["execute"] is False


def test_spicy_dislike_keeps_legacy_soft_preference_behavior():
    result, _ = run_oracle(new_session_state(), {"route": "task", "restrictions": [
        {"target": "매운맛", "reason": "dislike", "action": "add"},
    ]})
    assert result["session"]["preferences"] == [{"value": "맵지 않게"}]
    assert result["session"]["order"]["restrictions"] == []


def test_strong_spicy_scope_and_recommendation_exclude_pepper():
    session = new_session_state()
    session["order"]["restrictions"] = [{"target": "매운맛", "reason": "cannot_eat"}]
    robot = {"section": "extra", "task_queue": [], "active_task": None, "completed_tasks": []}
    extra = next(entry for entry in build_recommendation_scope(session, robot)["sections"] if entry["section"] == "extra")
    assert extra["options"] == ["치즈"]
    proposal = empty_order_patch()
    proposal["toppings"] = {"페퍼론치노": "high", "치즈": "low"}
    validated = validate_recommendation_proposal(session, proposal, robot)
    assert validated["proposal"]["toppings"] == {"치즈": "low"}


def test_strong_spicy_removal_requires_confirmation_and_rejection_keeps_it():
    session = new_session_state()
    restriction = {"target": "매운맛", "reason": "cannot_eat"}
    session["order"]["restrictions"] = [restriction]
    session["order"]["toppings"] = {"소시지": "high"}
    result, _ = run_oracle(session, {"route": "task", "restrictions": [{**restriction, "action": "remove"}]})
    assert result["session"]["order"] == session["order"]
    assert result["session"]["pending"]["type"] == "restriction_removal"
    assert result["policy"]["execute"] is False
    rejected, _ = run_oracle(result["session"], {"route": "task", "confirmation": "reject"})
    assert rejected["session"]["order"] == session["order"]
    assert rejected["session"]["pending"] is None
    assert rejected["policy"]["execute"] is False


def test_spicy_restricted_order_cannot_execute_or_remove_other_quantities():
    session = new_session_state()
    session["order"]["restrictions"] = [{"target": "매운맛", "reason": "cannot_eat"}]
    session["order"]["toppings"] = {"소시지": "high", "치즈": "low"}
    result, _ = run_oracle(session, {"route": "task", "order": {"toppings": {"페퍼론치노": "high"}}, "commit": True})
    assert result["session"]["order"] == session["order"]
    assert result["policy"]["execute"] is False


@pytest.mark.parametrize("recommendation", [False, True])
@pytest.mark.parametrize("amount", ["low", "normal", "high", "none", "많이", "huge", None])
def test_quantity_schema_agrees_with_domain_and_actual_grammar(recommendation, amount):
    import xgrammar as xgr
    from soomac_irc.agent_contract import DECISION_SCHEMA, RECOMMENDATION_SCHEMA

    schema = RECOMMENDATION_SCHEMA if recommendation else DECISION_SCHEMA
    proposal = {**empty_order_patch(), "noodle_portion": amount}
    payload = {"proposal": proposal, "reason_tags": []} if recommendation else {
        "route": "task", "order": {"noodle_portion": amount},
    }
    grammar = xgr.GrammarCompiler(xgr.TokenizerInfo(["a"])).compile_json_schema(schema, any_order=True)
    matcher = xgr.GrammarMatcher(grammar, terminate_without_stop_token=True)
    accepted = matcher.accept_string(json.dumps(payload, ensure_ascii=False)) and matcher.is_completed()
    assert accepted is (amount in ("low", "normal", "high", "none") or (recommendation and amount is None))


def test_clarification_preserves_old_pending_but_does_not_ask_to_execute_it():
    session = new_session_state()
    session["order"]["toppings"] = {"양파": "high", "버섯": "high"}
    session["pending"] = {"type": "execution", "section": "veggie", "targets": ["양파", "버섯"]}
    captures = []
    result, _ = run_oracle(session, {"route": "task", "clarify": True}, section="veggie", captures=captures)
    assert result["session"]["order"] == session["order"]
    assert result["session"]["pending"] == {**session["pending"], "question_type": "clarification"}
    assert result["policy"]["execute"] is False
    assert captures[0][6] == {"type": "clarification"}


@pytest.mark.parametrize("first_confirmation", ["accept", "reject"])
def test_multiple_strong_removals_are_not_lost_between_confirmation_turns(first_confirmation):
    session = new_session_state()
    crab = {"target": "게살", "reason": "allergy"}
    dairy = {"target": "유제품", "reason": "cannot_eat"}
    session["order"]["restrictions"] = [crab, dairy]
    result, _ = run_oracle(session, {"route": "task", "restrictions": [
        {**crab, "action": "remove"}, {**dairy, "action": "remove"},
    ]})
    assert result["session"]["order"]["restrictions"] == [crab, dairy]
    assert result["session"]["pending"]["candidate"] == crab
    result, _ = run_oracle(result["session"], {"route": "task", "confirmation": first_confirmation})
    assert result["session"]["order"]["restrictions"] == ([dairy] if first_confirmation == "accept" else [crab, dairy])
    assert result["session"]["pending"]["candidate"] == dairy
    assert result["policy"]["reason"] == "restriction_removal_confirmation_required"
    assert result["policy"]["execute"] is False
    result, _ = run_oracle(result["session"], {"route": "task", "confirmation": "reject"})
    assert result["session"]["order"]["restrictions"] == ([dairy] if first_confirmation == "accept" else [crab, dairy])
    assert result["session"]["pending"] is None


def field_diff(before, after, path=""):
    """Type-sensitive full recursive diff; missing and null are distinct."""
    if type(before) is not type(after):
        return [{"path": path, "before": before, "after": after, "change": "type"}]
    if isinstance(before, dict):
        changes = []
        for key in sorted(before.keys() | after.keys()):
            child = f"{path}.{key}" if path else key
            if key not in before or key not in after:
                changes.append({"path": child, "before_present": key in before,
                                "after_present": key in after, "before": before.get(key), "after": after.get(key)})
            else:
                changes.extend(field_diff(before[key], after[key], child))
        return changes
    if isinstance(before, list):
        changes = []
        for index in range(max(len(before), len(after))):
            child = f"{path}[{index}]"
            if index >= len(before) or index >= len(after):
                changes.append({"path": child, "before_present": index < len(before),
                                "after_present": index < len(after),
                                "before": before[index] if index < len(before) else None,
                                "after": after[index] if index < len(after) else None})
            else:
                changes.extend(field_diff(before[index], after[index], child))
        return changes
    return [] if before == after else [{"path": path, "before": before, "after": after}]


EPISODE_TRACES = []


def checked_turn(session, sparse, text, *, expected_order=None, expected_preferences=None,
                 expected_pending="unchanged", expected_event=None, execute=False,
                 section="meat", proposal=None, expected_reason=None):
    # Expectations are supplied before invoking the graph. Never copy actual
    # order/preferences/pending into the oracle or repair the next turn with gold.
    before = copy.deepcopy(session)
    expected = copy.deepcopy(before)
    if expected_order is not None:
        expected["order"] = copy.deepcopy(expected_order)
    if expected_preferences is not None:
        expected["preferences"] = copy.deepcopy(expected_preferences)
    if expected_pending != "unchanged":
        expected["pending"] = copy.deepcopy(expected_pending)
    if expected_event is not None:
        expected["action_history"].append(copy.deepcopy(expected_event))
    result, robot = run_oracle(session, sparse, section=section, proposal=proposal, text=text)
    # Actual wording is retained only for the required append, not used as a
    # semantic oracle. Fixed-path Response wording belongs to GL05/actual eval.
    expected["history"] += [{"role": "user", "content": text}, {"role": "assistant", "content": result["reply"]}]
    assert result["session"] == expected
    assert result["previous_session"] == before
    assert session == before
    assert result["robot_state"] == robot
    assert result["policy"]["execute"] is execute
    if expected_reason is not None:
        assert result["policy"]["reason"] == expected_reason
    EPISODE_TRACES.append({
        "mode": "controlled_oracle_NOT_model_comprehension", "text": text,
        "raw_oracle_decision": copy.deepcopy(sparse), "normalized_decision": result["decision"],
        "before_session": before, "expected_after_session": expected,
        "actual_after_session": copy.deepcopy(result["session"]),
        "actual_field_diff": field_diff(before, result["session"]),
        "expected_field_diff": field_diff(before, expected), "robot_state": robot,
        "robot_field_diff": field_diff(robot, result["robot_state"]),
        "policy": result["policy"], "reply": result["reply"], "ROS_publishers": False,
    })
    return result["session"]


def event(changes=None, restriction_added=None, preference_added=None, preference_removed=None):
    return {"type": "turn_applied", "order_changes": changes or {},
            "restriction_added": restriction_added or [], "restriction_removed": [],
            "preference_added": preference_added or [], "preference_removed": preference_removed or []}


@pytest.mark.parametrize("amount", ["normal", "high"])
@pytest.mark.parametrize("focus", ["양파", "버섯"])
def test_mt01_two_prior_facts_control_reference_and_quantity_preservation(amount, focus):
    session = new_session_state()
    selected = {"양파": amount, "버섯": amount}
    order = {**session["order"], "toppings": selected}
    pending = {"type": "execution", "source": "current_update", "section": "veggie", "targets": ["양파", "버섯"]}
    session = checked_turn(session, {"route": "task", "order": {"toppings": selected}},
                           f"양파랑 버섯은 둘 다 {amount} 양으로 골라줘", section="veggie",
                           expected_order=order, expected_pending=pending,
                           expected_event=event({f"toppings.{key}": amount for key in selected}))
    session = checked_turn(session, {"route": "general"}, "로봇도 쉬는 날 있어?", section="veggie")
    session = checked_turn(session, {"route": "task"}, f"아까 고른 것 중에 {focus} 얘기야", section="veggie")
    order = copy.deepcopy(order)
    order["toppings"][focus] = "low"
    checked_turn(session, {"route": "task", "order": {"toppings": {focus: "low"}}},
                 "그것만 조금으로, 나머지는 그대로", section="veggie", expected_order=order,
                 expected_event=event({f"toppings.{focus}": "low"}))


@pytest.mark.parametrize("excluded", ["게살", "소시지"])
@pytest.mark.parametrize("amount", ["low", "high"])
def test_mt02_cumulative_restrictions_and_candidate_amount_are_both_preserved(excluded, amount):
    session = new_session_state()
    spicy = {"target": "매운맛", "reason": "cannot_eat"}
    allergy = {"target": excluded, "reason": "allergy"}
    order = {**session["order"], "restrictions": [spicy]}
    session = checked_turn(session, {"route": "task", "restrictions": [{**spicy, "action": "add"}]},
                           "매운 것은 못 먹어", expected_order=order, expected_event=event(restriction_added=[spicy]))
    order = copy.deepcopy(order)
    order["restrictions"].append(allergy)
    session = checked_turn(session, {"route": "task", "restrictions": [{**allergy, "action": "add"}]},
                           f"그리고 {excluded} 알레르기도 있어", expected_order=order,
                           expected_event=event(restriction_added=[allergy]))
    session = checked_turn(session, {"route": "general"}, "오늘은 정신없이 바빴어")
    allowed = "소시지" if excluded == "게살" else "게살"
    proposal = {**empty_order_patch(), "toppings": {allowed: amount}}
    # GL05 preserves this test double's reason instead of replacing it with section tags.
    # All before/after amount, restriction, pending and execution oracles remain pinned.
    reasons = ["검수 후보"]
    pending = {"type": "recommendation", "candidate": proposal, "reason_tags": reasons}
    session = checked_turn(session, {"route": "task", "recommendation": {"action": "request"}},
                           f"그거 빼고 육류를 {amount} 양으로 추천해줘", proposal=proposal, expected_pending=pending)
    order = copy.deepcopy(order)
    order["toppings"] = {allowed: amount}
    checked_turn(session, {"route": "task", "confirmation": "accept", "commit": True},
                 "응, 나머지 그걸로 바로 담아줘", expected_order=order, expected_pending=None,
                 expected_event=event({f"toppings.{allowed}": amount}), execute=True)


@pytest.mark.parametrize("context", ["recommendation", "execution", "clarification", "restriction_removal"])
def test_mt03_same_reply_does_not_collapse_different_question_purposes(context):
    session = new_session_state()
    if context == "recommendation":
        proposal = {**empty_order_patch(), "toppings": {"소시지": "high"}}
        session["pending"] = {"type": "recommendation", "candidate": proposal, "reason_tags": []}
        session["history"] = [{"role": "assistant", "content": "소시지를 많이 추천해요. 선택할까요?"}]
        order = {**session["order"], "toppings": {"소시지": "high"}}
        checked_turn(session, {"route": "task", "confirmation": "accept"}, "응, 그걸로",
                     expected_order=order, expected_event=event({"toppings.소시지": "high"}),
                     expected_pending={"type": "execution", "source": "current_update", "section": "meat", "targets": ["소시지"]})
    elif context in ("execution", "clarification"):
        session["order"]["toppings"] = {"소시지": "high"}
        session["pending"] = {"type": "execution", "section": "meat", "targets": ["소시지"]}
        session["history"] = [{"role": "assistant", "content": "소시지를 많이 담을까요?"}]
        if context == "clarification":
            pending = {**session["pending"], "question_type": "clarification"}
            session = checked_turn(session, {"route": "task", "clarify": True}, "아까 거를 그거 말고", expected_pending=pending)
            checked_turn(session, {"route": "task", "clarify": True}, "응, 그걸로", expected_reason="understanding")
        else:
            checked_turn(session, {"route": "task", "confirmation": "accept"}, "응, 그걸로",
                         expected_pending=None, execute=True)
    else:
        crab = {"target": "게살", "reason": "allergy"}
        session["order"]["restrictions"] = [crab]
        session["pending"] = {"type": "restriction_removal", "candidate": crab}
        session["history"] = [{"role": "assistant", "content": "게살 알레르기 제한은 유지할까요?"}]
        # A positive answer to retaining a restriction rejects the removal.
        checked_turn(session, {"route": "task", "confirmation": "reject"}, "응, 그걸로", expected_pending=None)


@pytest.mark.parametrize("amount", ["low", "high"])
@pytest.mark.parametrize("route", ["general", "mixed"])
def test_mt04_wrong_assistant_history_is_preserved_without_order_corruption(amount, route):
    session = new_session_state()
    session["order"]["toppings"] = {"소시지": amount}
    wrong = "high" if amount == "low" else "low"
    session["history"] = [
        {"role": "user", "content": f"소시지는 {amount}로 해줘"},
        {"role": "assistant", "content": f"소시지는 {wrong}로 골랐어요"},
    ]
    session = checked_turn(session, {"route": route}, "응 근데 오늘 날씨는 어떨까?")
    checked_turn(session, {"route": "task"}, "아까 내가 고른 양 그대로야. 아직 담지는 마")


@pytest.mark.parametrize("withdrawn", [False, True])
@pytest.mark.parametrize("selected", [False, True])
def test_mt05_preference_revocation_does_not_cancel_preselected_future_order(withdrawn, selected):
    session = new_session_state()
    session["order"]["toppings"] = {"치즈": "high"} if selected else {}
    preference = {"value": "매콤하게"}
    session = checked_turn(session, {"route": "task", "preferences": [{**preference, "action": "add"}]},
                           "매운 맛이 좋아", expected_preferences=[preference],
                           expected_event=event(preference_added=[preference]))
    session = checked_turn(session, {"route": "general"}, "다음 재료는 천천히 고를게")
    if withdrawn:
        session = checked_turn(session, {"route": "task", "preferences": [{**preference, "action": "remove"}]},
                               "아까 매운 맛이 좋다는 취향은 취소할래", expected_preferences=[],
                               expected_event=event(preference_removed=[preference]))
    checked_turn(session, {"route": "task"}, "이제 추가 재료 차례네. 내 주문은 그대로지?", section="extra")


def test_field_diff_distinguishes_absence_null_none_and_boolean_types():
    assert field_diff({}, {"amount": None})[0]["before_present"] is False
    assert field_diff({"amount": None}, {"amount": "none"})[0]["change"] == "type"
    assert field_diff({"execute": False}, {"execute": 0})[0]["change"] == "type"


def test_two_separated_prior_user_facts_reach_actual_serializer_without_mutation():
    from soomac_irc.decision_model import build_decision_model_input
    session = new_session_state()
    session["history"] = [
        {"role": "user", "content": "소시지는 많이 먹을래"},
        {"role": "assistant", "content": "소시지는 많이 골랐어요. 아직 담지는 않았어요"},
        {"role": "user", "content": "오늘은 좀 피곤하네"},
        {"role": "assistant", "content": "바쁜 하루였나 봐요"},
        {"role": "user", "content": "게살은 알레르기가 있어"},
        {"role": "assistant", "content": "게살은 제외할게요"},
    ]
    session["order"]["toppings"] = {"소시지": "high"}
    session["order"]["restrictions"] = [{"target": "게살", "reason": "allergy"}]
    before = copy.deepcopy(session)
    delivered = build_decision_model_input(session, "그거 빼고 나머지만", {"section": "meat"})
    assert delivered["recent_history"] == before["history"]
    assert delivered["order"] == before["order"]
    delivered["recent_history"][0]["content"] = "mutated input copy"
    assert session == before


@pytest.mark.parametrize("proposal_toppings,execute", [
    ({"치즈": "low"}, True), ({"소시지": "low"}, False),
    ({"소시지": "low", "치즈": "low"}, False),
    ({"소시지": "high"}, True), ({"소시지": "high", "치즈": "low"}, True),
])
def test_current_commit_and_new_recommendation_preserve_independent_authority(proposal_toppings, execute):
    session = new_session_state()
    session["order"]["toppings"] = {"소시지": "high"}
    before = copy.deepcopy(session)
    proposal = {**empty_order_patch(), "toppings": proposal_toppings}
    result, _ = run_oracle(session, {"route": "task", "commit": True, "recommendation": {"action": "request"}},
                            section="meat", proposal=proposal)
    assert result["policy"]["execute"] is execute
    assert result["session"]["order"] == before["order"]
    expected_toppings = {key: value for key, value in proposal_toppings.items() if before["order"]["toppings"].get(key) != value}
    if expected_toppings:
        assert result["session"]["pending"]["candidate"] == {**proposal, "toppings": expected_toppings}
    else:
        assert result["session"]["pending"] is None
    assert session == before


def test_new_recommendation_commit_without_current_selection_never_executes_candidate():
    result, _ = run_oracle(new_session_state(), {"route": "task", "commit": True, "recommendation": {"action": "request"}},
                            section="meat", proposal={**empty_order_patch(), "toppings": {"소시지": "high"}})
    assert result["policy"]["execute"] is False
    assert result["session"]["order"]["toppings"] == {}


@pytest.mark.parametrize("candidate,section,expected", [
    ({**empty_order_patch(), "toppings": {"치즈": "low"}}, "meat", True),
    ({**empty_order_patch(), "sauce": "크림"}, "noodle", True),
    ({**empty_order_patch(), "toppings": {"소시지": "low"}}, "meat", False),
    ({**empty_order_patch(), "toppings": {"양파": "low"}}, "meat", False),
    ({**empty_order_patch(), "toppings": {"치즈": "none"}}, "meat", False),
    ({**empty_order_patch(), "toppings": {"햄": "high"}}, "meat", False),
    ({**empty_order_patch(), "toppings": {"치즈": "huge"}}, "meat", False),
    (empty_order_patch(), "meat", False),
    ({**empty_order_patch(), "toppings": {"치즈": "low"}}, "unknown", False),
    ({**empty_order_patch(), "toppings": {"치즈": "low"}}, "extra", False),
])
def test_only_nonempty_valid_future_candidates_are_independent_of_current_execution(candidate, section, expected):
    from soomac_irc.llm_policy import is_future_recommendation_pending
    pending = {"type": "recommendation", "candidate": candidate}
    assert is_future_recommendation_pending(pending, section) is expected
    assert is_future_recommendation_pending({"type": "restriction_removal", "candidate": candidate}, section) is False


@pytest.mark.parametrize("blocker", ["busy", "empty", "restriction"])
def test_future_pending_never_creates_execution_authority_or_bypasses_safety(blocker):
    session = new_session_state()
    session["order"]["toppings"] = {"소시지": "high"} if blocker != "empty" else {}
    if blocker == "restriction":
        session["order"]["restrictions"] = [{"target": "소시지", "reason": "allergy"}]
    session["pending"] = {"type": "recommendation", "candidate": {**empty_order_patch(), "toppings": {"치즈": "low"}}}
    robot = {"section": "meat", "task_queue": [], "active_task": None, "completed_tasks": [], "robot_started": False}
    if blocker == "busy":
        robot["active_task"] = {"class": "소시지", "repeat_count": 3}
    result, _ = run_oracle(session, {"route": "task", "commit": True}, robot=robot)
    assert result["policy"]["execute"] is False


def test_partial_recommendation_choice_at_unchanged_confirmed_amount_preserves_commit():
    session = new_session_state()
    session["order"]["toppings"] = {"소시지": "high"}
    session["pending"] = {"type": "recommendation", "candidate": {
        **empty_order_patch(), "toppings": {"소시지": "low", "치즈": "low"},
    }, "reason_tags": []}
    result, _ = run_oracle(session, {"route": "task", "confirmation": "reject", "order": {
        "toppings": {"소시지": "high"},
    }, "commit": True}, section="meat")
    assert result["policy"]["execute"] is True
    assert result["session"]["pending"] is None
    assert result["session"]["order"] == session["order"]


def test_clarification_pending_cannot_be_implicitly_accepted_but_explicit_commit_remains_available():
    session = new_session_state()
    session["order"]["toppings"] = {"소시지": "high"}
    session["pending"] = {"type": "execution", "section": "meat", "targets": ["소시지"], "question_type": "clarification"}
    result, _ = run_oracle(session, {"route": "task", "confirmation": "accept"}, section="meat")
    assert result["decision"]["understanding"] == "clarify"
    assert result["policy"]["execute"] is False
    assert result["session"]["order"] == session["order"]
    assert result["session"]["pending"] == session["pending"]
    explicit, _ = run_oracle(session, {"route": "task", "commit": True}, section="meat")
    assert explicit["policy"]["execute"] is True
    assert explicit["session"]["pending"] is None


def test_resolved_current_order_closes_clarification_without_inventing_execution():
    session = new_session_state()
    session["order"]["toppings"] = {"소시지": "high"}
    session["pending"] = {"type": "execution", "section": "meat", "targets": ["소시지"], "question_type": "clarification"}
    result, _ = run_oracle(session, {"route": "task", "order": {"toppings": {"소시지": "low"}}}, section="meat")
    assert result["session"]["order"]["toppings"] == {"소시지": "low"}
    assert "question_type" not in result["session"]["pending"]
    assert result["policy"]["execute"] is False


def test_new_sauce_recommendation_waits_before_automatic_lid_sauce_chain():
    session = new_session_state()
    session["order"]["sauce"] = "토마토"
    session["order"]["toppings"] = {"치즈": "high"}
    result, _ = run_oracle(session, {"route": "task", "commit": True, "recommendation": {"action": "request"}},
                            section="extra", proposal={**empty_order_patch(), "sauce": "크림"})
    assert result["policy"]["execute"] is False
    assert result["session"]["pending"]["candidate"]["sauce"] == "크림"
    assert result["session"]["order"] == session["order"]


def test_dataset_auditor_matches_question_purpose_and_deferred_pending_contracts():
    import importlib.util
    from pathlib import Path
    source = Path(__file__).resolve().parents[3] / "training/learning_first_v43/audit_dataset_quality.py"
    spec = importlib.util.spec_from_file_location("gl04_dataset_contract_auditor", source)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    order = new_session_state()["order"]
    order["toppings"] = {"소시지": "high"}
    pending = {"type": "execution", "source": "current_update", "section": "meat",
               "targets": ["소시지"], "question_type": "clarification"}
    row = {"id": "gl04-question", "input": {"order": order, "pending": pending}}
    failures = []
    audit.validate_pending("decision", "test", row, failures)
    assert not failures
    assert audit.expected_pending_prompt(pending) == {"type": "clarification"}
    assert audit.expected_response_next_prompt({"policy": {"reason": "understanding"}, "pending": None}) == {"type": "clarification"}
    pending["question_type"] = False
    audit.validate_pending("decision", "test", row, failures)
    assert any(entry["issue"] == "pending_question_type" for entry in failures)
    row["input"]["pending"] = {"type": "restriction_removal", "candidate": {"target": "게살", "reason": "allergy"},
                               "deferred": [{"target": "매운맛", "reason": "cannot_eat"}]}
    failures = []
    audit.validate_pending("decision", "test", row, failures)
    assert not failures


def test_current_amount_revision_does_not_silently_discard_future_sauce_candidate():
    session = new_session_state()
    session["order"]["sauce"] = "토마토"
    session["order"]["toppings"] = {"치즈": "high"}
    session["pending"] = {"type": "recommendation", "candidate": {
        **empty_order_patch(), "sauce": "크림",
    }, "reason_tags": []}
    result, _ = run_oracle(session, {"route": "task", "commit": True, "order": {"toppings": {"치즈": "low"}}}, section="extra")
    assert result["policy"]["execute"] is False
    assert result["session"]["pending"] == session["pending"]
    assert result["session"]["order"]["toppings"] == {"치즈": "low"}
    assert result["session"]["order"]["sauce"] == "토마토"
