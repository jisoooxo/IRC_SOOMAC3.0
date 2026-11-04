"""Menu-wide speech regression, CPU only; no generated-model quality claim."""
import copy

import pytest

from soomac_irc.agent_contract import new_session_state
from soomac_irc.llm_langgraph import (
    build_execution_started_reply,
    build_preselected_confirmation_reply,
)


@pytest.mark.parametrize("section,name,object_name", [
    ("noodle", "얇은면", "얇은면을"),
    ("noodle", "넓은면", "넓은면을"),
    ("veggie", "양파", "양파를"),
    ("veggie", "버섯", "버섯을"),
    ("meat", "소시지", "소시지를"),
    ("meat", "게살", "게살을"),
    ("extra", "치즈", "치즈를"),
    ("extra", "페퍼론치노", "페퍼론치노를"),
])
@pytest.mark.parametrize("amount,phrase", [
    ("low", "적은 양으로"), ("normal", "보통 양으로"), ("high", "많이"),
])
def test_single_item_all_menu_portions(section, name, object_name, amount, phrase):
    session = new_session_state()
    session["order"].update(sauce="크림", noodle_type="얇은면", noodle_portion="normal")
    if section == "noodle":
        session["order"].update(noodle_type=name, noodle_portion=amount)
    else:
        session["order"]["toppings"] = {name: amount}
    before = copy.deepcopy(session)
    assert build_execution_started_reply(session, section) == f"{object_name} {phrase} 담을게요."
    assert session == before
    if section != "noodle":
        pending = {"items": {name: amount}}
        assert build_preselected_confirmation_reply(pending) == f"미리 고르신 {object_name} {phrase} 담을까요?"
        assert pending == {"items": {name: amount}}


@pytest.mark.parametrize("section,left,right", [
    ("veggie", "양파", "버섯"),
    ("meat", "소시지", "게살"),
    ("extra", "치즈", "페퍼론치노"),
])
@pytest.mark.parametrize("amounts,wording", [
    (("low", "low"), "{joined} 적은 양으로"),
    (("normal", "normal"), "{joined} 보통 양으로"),
    (("high", "high"), "{joined} 많이"),
    (("low", "normal"), "{left} 적게, {right} 보통 양으로"),
    (("low", "high"), "{left} 적게, {right} 많이"),
    (("normal", "low"), "{left} 보통 양으로, {right} 적게"),
    (("normal", "high"), "{left} 보통 양으로, {right} 많이"),
    (("high", "low"), "{left} 많이, {right} 적게"),
    (("high", "normal"), "{left} 많이, {right} 보통 양으로"),
])
def test_all_pair_portions_and_preselected_question(section, left, right, amounts, wording):
    particles = {
        "veggie": ("양파와 버섯을", "양파는", "버섯은"),
        "meat": ("소시지와 게살을", "소시지는", "게살은"),
        "extra": ("치즈와 페퍼론치노를", "치즈는", "페퍼론치노는"),
    }
    joined, left_topic, right_topic = particles[section]
    expected = wording.format(joined=joined, left=left_topic, right=right_topic)
    session = new_session_state()
    # Deliberately reversed insertion order: execution speech must follow the actual plan.
    session["order"]["toppings"] = {right: amounts[1], left: amounts[0]}
    before = copy.deepcopy(session)
    assert build_execution_started_reply(session, section) == expected + " 담을게요."
    assert session == before
    pending = {"items": {left: amounts[0], right: amounts[1]}}
    assert build_preselected_confirmation_reply(pending) == "미리 고르신 " + expected + " 담을까요?"


def test_excluded_and_other_section_items_are_not_announced():
    session = new_session_state()
    session["order"]["toppings"] = {"양파": "low", "버섯": "none", "치즈": "high"}
    assert build_execution_started_reply(session, "veggie") == "양파를 적은 양으로 담을게요."
    with pytest.raises(ValueError, match="시작 안내를 만들 작업이 없음"):
        build_execution_started_reply(session, "meat")


def test_preselected_legacy_candidate_and_empty_fallback():
    pending = {"candidate": {"toppings": {"버섯": "low", "양파": "low"}}}
    assert build_preselected_confirmation_reply(pending) == "미리 고르신 버섯과 양파를 적은 양으로 담을까요?"
    assert build_preselected_confirmation_reply({}) == "미리 고르신 재료를 담을까요?"
