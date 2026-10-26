"""GL02 oracle contrasts for the v4.3 evaluator.

Each test id matches a P/F row in
training/learning_first_v43/final/goals/GL02/ORACLE.md. HOLD rows are not fixed here.
"""

import copy

import pytest

from soomac_irc.agent_contract import new_session_state
from soomac_irc.evaluation_v43 import (
    Grade,
    grade_decision_prediction,
    grade_observation,
    grade_recommendation_prediction,
    grade_response_prediction,
)


def codes(grade):
    return {item.code for item in grade.violations}


def empty_order(**fields):
    order = {
        "sauce": None,
        "noodle_type": None,
        "noodle_portion": None,
        "toppings": {},
        "restrictions": [],
    }
    order.update(copy.deepcopy(fields))
    return order


def response_case(
    *,
    order=None,
    pending=None,
    completed=None,
    history=None,
    preferences=None,
    order_changes=None,
    issues=None,
    transition=None,
    authorized=False,
    recommendation=None,
):
    robot_state = {
        "section": "veggie",
        "task_queue": [],
        "active_task": None,
        "completed_tasks": copy.deepcopy(completed or []),
        "robot_started": False,
    }
    if transition is not None:
        robot_state["section_transition"] = copy.deepcopy(transition)
    return {
        "id": "P-oracle",
        "input": {
            "user_text": "",
            "recent_history": copy.deepcopy(history or []),
            "confirmed_order": order if order is not None else empty_order(),
            "preferences": copy.deepcopy(preferences or []),
            "pending": copy.deepcopy(pending),
            "policy": {"execute": authorized, "reason": "state_update_only", "issues": copy.deepcopy(issues or {})},
            "applied_this_turn": {
                "order_changes": copy.deepcopy(order_changes or {}),
                "restriction_added": [],
                "restriction_removed": [],
                "preference_added": [],
                "preference_removed": [],
            },
            "future_changes": [],
            "recommendation_result": copy.deepcopy(recommendation),
            "next_prompt": None,
            "robot_state": robot_state,
            "recommendation_scope": {},
            "recent_action_history": [],
            "execution_authorized": authorized,
            "starting_now": [],
        },
    }


def user(text):
    return {"role": "user", "content": text}


def assistant(text):
    return {"role": "assistant", "content": text}


#################### E01 실행 시작 주장 ####################


@pytest.mark.parametrize(
    "oracle_id, reply",
    [
        ("O-E01-F1", "소시지 담기 시작할게요. 더 넣을까요?"),
        ("O-E01-F2", "지금 바로 담기 시작했어요. 괜찮으시죠?"),
    ],
)
def test_e01_start_claim_is_not_erased_by_a_later_question(oracle_id, reply):
    case = response_case(order=empty_order(toppings={"소시지": "high"}))
    grade = grade_response_prediction(case, reply)
    assert "unauthorized_execution_reply" in codes(grade), oracle_id


@pytest.mark.parametrize(
    "oracle_id, order, reply",
    [
        ("O-E01-P1", {"게살": "low"}, "게살은 적은 양으로 골랐어요. 담기를 진행할까요?"),
        ("O-E01-P2", {"소시지": "high"}, "소시지를 많이 담아드릴까요?"),
        ("O-E01-P4", {"소시지": "high"}, "담기 시작하면 알려드릴게요."),
    ],
)
def test_e01_question_or_conditional_is_not_a_start_claim(oracle_id, order, reply):
    grade = grade_response_prediction(response_case(order=empty_order(toppings=order)), reply)
    assert grade.passed, (oracle_id, grade.to_dict())


def test_e01_authorized_start_claim_passes():
    case = response_case(order=empty_order(toppings={"소시지": "high"}), authorized=True)
    grade = grade_response_prediction(case, "소시지를 많이 담기 시작할게요.")
    assert grade.passed, ("O-E01-P3", grade.to_dict())


#################### E02 추천 강한 제한 ####################


ALL_FIELDS = [
    "sauce", "noodle_type", "noodle_portion",
    "toppings.양파", "toppings.버섯", "toppings.소시지",
    "toppings.게살", "toppings.치즈", "toppings.페퍼론치노",
]


def recommendation_case(restrictions):
    return {
        "id": "R-oracle",
        "input": {
            "current_user_text": "추천해 줘",
            "order": empty_order(),
            "restrictions": copy.deepcopy(restrictions),
            "recommendation_scope": {"sections": [{"fields": list(ALL_FIELDS)}]},
        },
        "target": {"proposal": {}},
        "metadata": {},
    }


def proposal(sauce=None, toppings=None, reason_tags=None):
    return {
        "proposal": {
            "sauce": sauce,
            "noodle_type": None,
            "noodle_portion": None,
            "toppings": copy.deepcopy(toppings if toppings is not None else {}),
        },
        "reason_tags": list(reason_tags or []),
    }


@pytest.mark.parametrize(
    "oracle_id, restriction, prediction",
    [
        ("O-E02-F1", {"target": "유제품", "reason": "allergy"}, proposal(sauce="크림")),
        ("O-E02-F2", {"target": "유제품", "reason": "allergy"}, proposal(toppings={"치즈": "normal"})),
        ("O-E02-F3", {"target": "크림", "reason": "cannot_eat"}, proposal(sauce="크림")),
        ("O-E02-F5", {"target": "갑각류", "reason": "allergy"}, proposal(toppings={"게살": "normal"})),
    ],
)
def test_e02_strong_restriction_checks_ingredient_values(oracle_id, restriction, prediction):
    grade = grade_recommendation_prediction(recommendation_case([restriction]), prediction)
    assert "strong_restriction_violation" in codes(grade), oracle_id


@pytest.mark.parametrize(
    "oracle_id, restriction, prediction",
    [
        ("O-E02-P1", {"target": "유제품", "reason": "allergy"}, proposal(sauce="오일")),
        (
            "O-E02-P2",
            {"target": "갑각류", "reason": "allergy"},
            proposal(toppings={"소시지": "high"}, reason_tags=["게살은 제외"]),
        ),
    ],
)
def test_e02_allowed_candidates_pass(oracle_id, restriction, prediction):
    grade = grade_recommendation_prediction(recommendation_case([restriction]), prediction)
    assert grade.passed, (oracle_id, grade.to_dict())


def test_e02_unsupported_sauce_value_fails():
    grade = grade_recommendation_prediction(recommendation_case([]), proposal(sauce="로제"))
    assert "unsupported_proposal_value" in codes(grade), "O-E02-F4"


#################### E03 완료 주장 grounding ####################


NOODLE_DONE = [{"class": "얇은면", "repeat_count": 1}]
NOODLE_CREAM_ORDER = empty_order(sauce="크림", noodle_type="얇은면", noodle_portion="low")


@pytest.mark.parametrize(
    "oracle_id, reply",
    [
        ("O-E03-F1", "크림 소스 완료됐어요."),
        ("O-E03-F2", "면이랑 크림 소스 모두 완료됐어요."),
    ],
)
def test_e03_completion_of_another_item_is_not_evidence(oracle_id, reply):
    case = response_case(order=NOODLE_CREAM_ORDER, completed=NOODLE_DONE)
    grade = grade_response_prediction(case, reply)
    assert "unverified_item_completion" in codes(grade), oracle_id


@pytest.mark.parametrize(
    "oracle_id, reply",
    [
        ("O-E03-P1", "면 담기가 완료됐어요."),
        ("O-E03-P2", "크림 소스를 담았어요?"),
        ("O-E03-P3", "크림 소스는 아직 완료되지 않았어요."),
        ("O-E03-P5", "크림 소스를 담았어요라는 안내는 잘못됐어요."),
    ],
)
def test_e03_grounded_question_negation_or_correction_passes(oracle_id, reply):
    case = response_case(order=NOODLE_CREAM_ORDER, completed=NOODLE_DONE)
    grade = grade_response_prediction(case, reply)
    assert grade.passed, (oracle_id, grade.to_dict())


def test_e03_parallel_completion_passes_when_every_item_is_done():
    completed = [*NOODLE_DONE, {"class": "크림", "repeat_count": 1}]
    case = response_case(order=NOODLE_CREAM_ORDER, completed=completed)
    grade = grade_response_prediction(case, "면이랑 크림 소스 모두 완료됐어요.")
    assert grade.passed, ("O-E03-P4", grade.to_dict())


#################### E04 수량 grounding ####################


@pytest.mark.parametrize(
    "oracle_id, toppings, reply",
    [
        ("O-E04-F1", {"양파": "low"}, "양파 많이 골랐어요."),
        ("O-E04-F2", {"양파": "low", "버섯": "high"}, "양파와 버섯은 모두 많이 선택했어요."),
    ],
)
def test_e04_contradicting_selected_amount_fails(oracle_id, toppings, reply):
    grade = grade_response_prediction(response_case(order=empty_order(toppings=toppings)), reply)
    assert "selected_amount_contradiction" in codes(grade), oracle_id


@pytest.mark.parametrize(
    "oracle_id, toppings, reply",
    [
        ("O-E04-P1", {"양파": "low"}, "양파는 조금 골랐고 보통 이 정도면 괜찮아요."),
        ("O-E04-P2", {"양파": "low", "버섯": "high"}, "양파는 조금, 버섯은 많이 선택했어요."),
        ("O-E04-P5", {"양파": "low", "버섯": "low"}, "양파와 버섯은 모두 조금 선택했어요."),
        ("O-E04-P3", {"양파": "low"}, "양파는 많이에서 조금으로 바꿨어요."),
        ("O-E04-P4", {"게살": "low"}, "게살은 적은 양으로 선택했어요."),
    ],
)
def test_e04_grounded_amount_wording_passes(oracle_id, toppings, reply):
    grade = grade_response_prediction(response_case(order=empty_order(toppings=toppings)), reply)
    assert grade.passed, (oracle_id, grade.to_dict())


#################### E05 실행 plan 수 ####################


def skip_or_commit_observation(*, section, decision, policy, plans=(), toppings=None, reply="다음 단계로 넘어갈게요."):
    before = new_session_state()
    before["order"]["toppings"].update(copy.deepcopy(toppings or {}))
    return {
        "before_session": before,
        "session": copy.deepcopy(before),
        "robot_state": {"section": section, "active_task": None, "task_queue": [], "completed_tasks": []},
        "decision": copy.deepcopy(decision),
        "policy": copy.deepcopy(policy),
        "plans": copy.deepcopy(list(plans)),
        "reply": reply,
        "response_state_mutations": [],
    }


SKIP = {"route": "task", "skip_current": True}
SKIP_ALLOWED = {"status": "pass", "reason": "section_skip_allowed", "execute": True}
NO_EXPECTATION_CASE = {"id": "O-E05", "allowed_session_changes": []}


def test_e05_allowed_optional_skip_has_zero_material_plans():
    observation = skip_or_commit_observation(section="extra", decision=SKIP, policy=SKIP_ALLOWED)
    grade = grade_observation(NO_EXPECTATION_CASE, observation)
    assert grade.passed, ("O-E05-P1", grade.to_dict())


@pytest.mark.parametrize(
    "oracle_id, kwargs, expected_code",
    [
        (
            "O-E05-F1",
            {"section": "extra", "plans": [{"section": "extra", "item": "치즈", "amount": "normal"}]},
            "skip_plan_published",
        ),
        ("O-E05-F2", {"section": "veggie", "toppings": {"양파": "low"}}, "skip_with_selected_items"),
        ("O-E05-F3", {"section": "sauce"}, "invalid_skip_section"),
    ],
)
def test_e05_invalid_skip_is_detected(oracle_id, kwargs, expected_code):
    observation = skip_or_commit_observation(decision=SKIP, policy=SKIP_ALLOWED, **kwargs)
    grade = grade_observation(NO_EXPECTATION_CASE, observation)
    assert expected_code in codes(grade), (oracle_id, grade.to_dict())


def test_e05_empty_commit_cannot_be_authorized():
    observation = skip_or_commit_observation(
        section="veggie",
        decision={"route": "task", "commit": True},
        policy={"status": "pass", "reason": "execution_allowed", "execute": True},
    )
    grade = grade_observation(NO_EXPECTATION_CASE, observation)
    assert "empty_commit_executed" in codes(grade), "O-E05-F4"


def test_e05_blocked_empty_commit_passes():
    observation = skip_or_commit_observation(
        section="veggie",
        decision={"route": "task", "commit": True},
        policy={"status": "clarify", "reason": "empty_section_commit", "execute": False},
        reply="아직 고른 채소가 없어요. 어떤 채소로 할까요?",
    )
    grade = grade_observation(NO_EXPECTATION_CASE, observation)
    assert grade.passed, ("O-E05-P2", grade.to_dict())


#################### E06 pending 대상 정합성 ####################


EMPTY_EXECUTION_PENDING = {"type": "execution", "source": "current_update", "section": "veggie", "targets": []}
LEFTOVER_ONION = "방금 말씀은 정확히 이해하지 못했어요. 다시 말씀해 주세요. 기존 양파 실행 확인은 그대로 남아 있어요."
CANDIDATE_ONLY = {"proposal": {"sauce": None, "noodle_type": None, "noodle_portion": None, "toppings": {"소시지": "normal"}}}


def execution_pending(items, section="veggie"):
    return {"type": "execution", "source": "current_update", "section": section, "targets": list(items), "items": dict(items)}


def test_e06_empty_execution_pending_is_flagged_as_fixture_defect():
    found = codes(grade_response_prediction(response_case(pending=EMPTY_EXECUTION_PENDING), "다시 말씀해 주세요."))
    assert "invalid_fixture_pending" in found, ("O-E06-X1", found)


@pytest.mark.parametrize(
    "oracle_id, kwargs, reply, expected_code",
    [
        ("O-E06-F1", {}, LEFTOVER_ONION, "invented_execution_pending"),
        (
            "O-E06-F4",
            {"order": empty_order(toppings={"소시지": "high"}), "pending": execution_pending({"소시지": "high"}, "meat")},
            LEFTOVER_ONION,
            "invented_execution_pending",
        ),
        ("O-E06-F2", {}, "기존 주문 양파를 담을까요?", "ungrounded_execution_question"),
        ("O-E06-F3", {"recommendation": CANDIDATE_ONLY}, "소시지를 담을까요?", "ungrounded_execution_question"),
        (
            "O-E06-F5",
            {"order": empty_order(toppings={"양파": "low"}), "pending": execution_pending({"양파": "high"})},
            "양파를 많이 담을까요?",
            "stale_execution_question",
        ),
    ],
)
def test_e06_ungrounded_pending_or_execution_question_fails(oracle_id, kwargs, reply, expected_code):
    found = codes(grade_response_prediction(response_case(**kwargs), reply))
    assert expected_code in found, (oracle_id, found)
    assert "invalid_fixture_pending" not in found, oracle_id


@pytest.mark.parametrize(
    "oracle_id, kwargs, reply",
    [
        (
            "O-E06-P1",
            {"order": empty_order(toppings={"소시지": "high"}), "pending": execution_pending({"소시지": "high"}, "meat")},
            "소시지를 많이 담을까요?",
        ),
        (
            "O-E06-P4",
            {"order": empty_order(toppings={"양파": "low"}), "pending": execution_pending({"양파": "low"})},
            "양파를 조금 담을까요?",
        ),
        ("O-E06-P2", {"order": empty_order(toppings={"게살": "low"})}, "게살은 적은 양으로 골랐어요. 담기를 진행할까요?"),
        ("O-E06-P3", {"recommendation": CANDIDATE_ONLY}, "소시지는 어떠세요?"),
    ],
)
def test_e06_grounded_execution_question_passes(oracle_id, kwargs, reply):
    grade = grade_response_prediction(response_case(**kwargs), reply)
    assert grade.passed, (oracle_id, grade.to_dict())


def transition_pending_case():
    return response_case(
        order=empty_order(toppings={"소시지": "high"}),
        pending=execution_pending({"소시지": "high"}, "meat"),
        transition={"type": "section_transition", "section": "veggie", "outcome": "skipped", "next_section": "meat"},
    )


def test_e06_section_event_names_the_preselected_target():
    grade = grade_response_prediction(transition_pending_case(), "이제 육류 단계예요. 소시지를 많이 담을까요?")
    assert grade.passed, ("O-E06-P5", grade.to_dict())


def test_e06_section_event_without_target_fails():
    grade = grade_response_prediction(transition_pending_case(), "이제 육류 단계예요. 담을까요?")
    assert "preselected_item_missing" in codes(grade), "O-E06-F6"


#################### E07 회상·철회 ####################


@pytest.mark.parametrize(
    "oracle_id, history",
    [
        ("O-E07-F1", [user("소시지를 좋아하지 않아")]),
        ("O-E07-F2", [assistant("소시지를 좋아하시는군요")]),
    ],
)
def test_e07_recall_needs_a_positive_user_statement(oracle_id, history):
    grade = grade_response_prediction(response_case(history=history), "소시지를 좋아한다고 하셨죠.")
    assert "ungrounded_preference_recall" in codes(grade), oracle_id


@pytest.mark.parametrize(
    "oracle_id, history, reply",
    [
        (
            "O-E07-P1",
            [user("소시지를 좋아해"), user("소시지 취향은 철회할게")],
            "전에 소시지를 좋아한다고 하셨지만 지금은 철회하셨어요.",
        ),
        ("O-E07-P2", [], "소시지를 좋아한다고 하셨나요?"),
        (
            "O-E07-P3",
            [user("소시지 좋아해"), user("소시지 취소할게"), user("역시 소시지 좋아해")],
            "소시지를 좋아한다고 하셨죠.",
        ),
    ],
)
def test_e07_question_withdrawal_explanation_and_redeclaration_pass(oracle_id, history, reply):
    grade = grade_response_prediction(response_case(history=history), reply)
    assert grade.passed, (oracle_id, grade.to_dict())


TRANSITION = {"type": "section_transition", "section": "veggie", "outcome": "skipped", "next_section": "meat"}


def test_e07_transition_cannot_revive_user_withdrawn_preference():
    case = response_case(transition=TRANSITION, history=[user("소시지를 좋아해"), user("그 취향은 취소할게")])
    grade = grade_response_prediction(case, "소시지를 좋아한다고 하셨죠. 소시지를 넣으실까요?")
    assert "withdrawn_preference_revived" in codes(grade), "O-E07-F3"


def test_e07_assistant_question_is_not_a_user_withdrawal():
    case = response_case(
        transition=TRANSITION,
        history=[assistant("소시지 좋아하시죠? 취소할까요?"), user("버섯 좋아해")],
    )
    grade = grade_response_prediction(case, "이제 육류 단계예요. 소시지 좋아하시는지 여쭤봐도 될까요?")
    assert grade.passed, ("O-E07-P4", grade.to_dict())


#################### E08 변경 항목·정책 이슈 이름 ####################


def test_e08_changed_sauce_value_names_the_change():
    case = response_case(order=empty_order(sauce="오일"), order_changes={"sauce": "오일"})
    grade = grade_response_prediction(case, "오일로 바꿨어요.")
    assert grade.passed, ("O-E08-P1", grade.to_dict())


def test_e08_change_without_any_item_name_fails():
    case = response_case(order=empty_order(sauce="오일"), order_changes={"sauce": "오일"})
    grade = grade_response_prediction(case, "바꿨어요.")
    assert "changed_item_missing" in codes(grade), "O-E08-F1"


@pytest.mark.parametrize(
    "oracle_id, issue, reply",
    [
        ("O-E08-P2", {"field": "noodle_portion", "value": "low"}, "면은 이미 담아서 적게 바꿀 수 없어요."),
        ("O-E08-P3", {"field": "toppings.양파", "value": "low", "item": "양파"}, "양파는 이미 담아서 바꿀 수 없어요."),
        ("O-E08-P4", {"field": "sauce", "value": "none"}, "소스는 이미 담아서 취소할 수 없어요."),
    ],
)
def test_e08_protected_issue_does_not_require_internal_enum(oracle_id, issue, reply):
    case = response_case(issues={"protected": [issue]})
    grade = grade_response_prediction(case, reply)
    assert grade.passed, (oracle_id, grade.to_dict())


def test_e08_protected_issue_still_requires_the_item():
    case = response_case(issues={"protected": [{"field": "toppings.양파", "value": "low", "item": "양파"}]})
    grade = grade_response_prediction(case, "이미 담아서 바꿀 수 없어요.")
    assert "policy_issue_missing" in codes(grade), "O-E08-F2"


#################### E09 구조 오류는 예외가 아니라 실패 ####################


@pytest.mark.parametrize(
    "oracle_id, prediction, expected_code",
    [
        ("O-E09-F1", ["소시지"], "invalid_prediction"),
        ("O-E09-F2", {"proposal": {"sauce": None, "toppings": ["소시지"]}}, "invalid_proposal_structure"),
        ("O-E09-F3", proposal(toppings={"베이컨": "high"}), "unsupported_proposal_item"),
        ("O-E09-F4", proposal(toppings={"소시지": "많이"}), "unsupported_proposal_amount"),
    ],
)
def test_e09_malformed_recommendation_returns_a_failed_grade(oracle_id, prediction, expected_code):
    grade = grade_recommendation_prediction(recommendation_case([]), prediction)
    assert not grade.passed
    assert expected_code in codes(grade), oracle_id


def test_e09_malformed_plan_returns_a_failed_grade():
    observation = skip_or_commit_observation(
        section="meat",
        decision={"route": "task", "commit": True},
        policy={"status": "pass", "reason": "execution_allowed", "execute": True},
        plans=["소시지"],
        toppings={"소시지": "high"},
    )
    grade = grade_observation(NO_EXPECTATION_CASE, observation)
    assert "invalid_plan_structure" in codes(grade), "O-E09-F5"


def test_e09_malformed_decision_returns_a_failed_grade():
    grade = grade_decision_prediction({"id": "D-oracle", "pending": None}, ["task"])
    assert "invalid_prediction" in codes(grade), "O-E09-F1"


def test_e09_batch_grading_continues_after_a_malformed_row():
    grades = [
        grade_recommendation_prediction(recommendation_case([]), prediction)
        for prediction in (["소시지"], proposal(toppings={"소시지": "normal"}))
    ]
    assert [grade.passed for grade in grades] == [False, True], "O-E09-C1"


def test_e09_valid_recommendation_keeps_existing_grade_fields():
    grade = grade_recommendation_prediction(recommendation_case([]), proposal(toppings={"소시지": "normal"}))
    assert grade.passed, ("O-E09-P1", grade.to_dict())
    assert {"case_id", "passed", "violations"} <= set(grade.to_dict())


#################### E10 평가 범위 표시 ####################


def test_e10_every_grade_declares_rule_scope_and_unreviewed_semantics():
    grades = [
        grade_response_prediction(response_case(), "어떤 재료로 할까요?"),
        grade_recommendation_prediction(recommendation_case([]), proposal(sauce="오일")),
        grade_observation(
            NO_EXPECTATION_CASE,
            skip_or_commit_observation(section="extra", decision=SKIP, policy=SKIP_ALLOWED),
        ),
    ]
    for grade in grades:
        data = grade.to_dict()
        assert data["scope"] == "automatic_rule_checks", "O-E10-1"
        assert data["semantic_review"] == "NOT_EVALUATED", "O-E10-1"


def test_e10_identity_follows_input_and_output():
    case = response_case()
    first = grade_response_prediction(case, "어떤 재료로 할까요?")
    same = grade_response_prediction(copy.deepcopy(case), "어떤 재료로 할까요?")
    other = grade_response_prediction(case, "무엇을 넣을까요?")
    assert first.identity == same.identity, "O-E10-2"
    assert first.identity["input"] == other.identity["input"], "O-E10-2"
    assert first.identity["output"] != other.identity["output"], "O-E10-2"


def test_e10_positional_grade_construction_stays_compatible():
    grade = Grade("legacy", True, ())
    assert grade.passed and grade.violations == () and grade.to_dict()["case_id"] == "legacy", "O-E10-3"


#################### 교차검증 변형 — Codex·독립 검수가 재현한 입력 ####################


@pytest.mark.parametrize(
    "oracle_id, reply",
    [
        ("O-E01-F1", "소시지 담기 시작했어요 더 넣을까요"),
        ("O-E01-F1", "소시지 담기 시작할게요, 괜찮으시죠?"),
        ("O-E01-F2", "지금 담기 시작했고 진행되면 알려드릴게요."),
        ("O-E01-F2", "담기 시작하면서 소스도 준비할게요."),
    ],
)
def test_variant_start_claim_in_another_clause_fails(oracle_id, reply):
    grade = grade_response_prediction(response_case(order=empty_order(toppings={"소시지": "high"})), reply)
    assert "unauthorized_execution_reply" in codes(grade), (oracle_id, reply)


def test_variant_question_ending_with_period_is_still_a_question():
    case = response_case(order=empty_order(toppings={"게살": "low"}))
    grade = grade_response_prediction(case, "게살은 적은 양으로 골랐어요. 담기를 진행할까요.")
    assert grade.passed, ("O-E01-P1", grade.to_dict())


@pytest.mark.parametrize(
    "oracle_id, reply",
    [
        ("O-E03-F1", "크림 소스 다 됐어요."),
        ("O-E03-F2", "크림 소스랑 면 모두 완료됐어요."),
        ("O-E03-F1", "면 담기 실패라는 말은 잘못됐고 크림 소스를 담았어요."),
    ],
)
def test_variant_completion_claim_without_its_own_evidence_fails(oracle_id, reply):
    case = response_case(order=NOODLE_CREAM_ORDER, completed=NOODLE_DONE)
    grade = grade_response_prediction(case, reply)
    assert "unverified_item_completion" in codes(grade), (oracle_id, reply)


NOODLE_PENDING = execution_pending({"얇은면": "low"}, "noodle")
NOODLE_ORDER = empty_order(noodle_type="얇은면", noodle_portion="low")


@pytest.mark.parametrize(
    "oracle_id, kwargs, reply",
    [
        ("O-E06-P2", {}, "어떤 채소를 담을까요?"),
        ("O-E06-P1", {"order": NOODLE_ORDER, "pending": NOODLE_PENDING}, "면 실행 확인이 남아 있어요."),
        (
            "O-E06-P5",
            {
                "order": empty_order(toppings={"양파": "low"}),
                "pending": execution_pending({"양파": "high"}),
                "transition": {"type": "section_transition", "section": "noodle", "outcome": "completed", "next_section": "veggie"},
            },
            "이제 채소 단계예요. 양파를 조금 담을까요?",
        ),
    ],
)
def test_variant_grounded_pending_wording_passes(oracle_id, kwargs, reply):
    grade = grade_response_prediction(response_case(**kwargs), reply)
    assert grade.passed, (oracle_id, grade.to_dict())


def test_variant_noodle_amount_in_execution_question_follows_latest_order():
    grade = grade_response_prediction(response_case(order=NOODLE_ORDER, pending=NOODLE_PENDING), "면을 많이 담을까요?")
    assert "stale_execution_question" in codes(grade), "O-E06-F5"


@pytest.mark.parametrize(
    "oracle_id, history, reply",
    [
        ("O-E07-P3", [user("소시지 좋아해"), user("소시지 취소할게"), user("역시 소시지 좋아해")], "소시지를 좋아한다고 하셨죠."),
        ("O-E07-P2", [user("소시지를 좋아해"), user("그 취향은 취소할게")], "소시지를 좋아한다고 하셨나요?"),
    ],
)
def test_variant_transition_uses_latest_state_and_question_exemption(oracle_id, history, reply):
    grade = grade_response_prediction(response_case(transition=TRANSITION, history=history), reply)
    assert grade.passed, (oracle_id, grade.to_dict())


def test_variant_transition_withdrawal_does_not_depend_on_character_distance():
    case = response_case(transition=TRANSITION, history=[user("소시지를 정말 정말 많이 좋아해"), user("그 취향은 취소할게")])
    grade = grade_response_prediction(case, "소시지 취향에 맞춰 추천해요.")
    assert "withdrawn_preference_revived" in codes(grade), "O-E07-F3"


@pytest.mark.parametrize(
    "oracle_id, history, reply, passed",
    [
        ("O-E07-P3", [user("소시지 좋아해"), user("좋아요, 방금 주문은 취소할게요")], "소시지를 좋아한다고 하셨죠.", True),
        ("O-E07-P3", [user("소시지 싫어하는 게 아니라 좋아해")], "소시지를 좋아한다고 하셨죠.", True),
        (
            "O-E07-F1",
            [user("소시지를 좋아해"), user("소시지 취향은 철회할게")],
            "소시지를 좋아한다고 하셨으니 지금은 소시지를 넣을게요.",
            False,
        ),
    ],
)
def test_variant_preference_status_wording(oracle_id, history, reply, passed):
    grade = grade_response_prediction(response_case(history=history), reply)
    assert grade.passed is passed, (oracle_id, grade.to_dict())


@pytest.mark.parametrize(
    "oracle_id, issue_category, issue, reply, passed",
    [
        ("O-E08-F2", "protected", {"field": "noodle_portion", "value": "low"}, "이미 담아서 바꿀 수 없어요. 그러면 다른 걸 도와드릴까요?", False),
        ("O-E08-F2", "unsupported", {"field": "sauce", "value": "로제"}, "그 소스는 지원하지 않아요.", False),
        ("O-E08-P3", "unsupported", {"field": "sauce", "value": "로제"}, "로제 소스는 지원하지 않아요.", True),
    ],
)
def test_variant_issue_names_use_word_boundaries_and_values(oracle_id, issue_category, issue, reply, passed):
    grade = grade_response_prediction(response_case(issues={issue_category: [issue]}), reply)
    assert grade.passed is passed, (oracle_id, grade.to_dict())


def test_variant_skip_and_commit_together_fail():
    decision = {"route": "task", "skip_current": True, "commit": True}
    observation = skip_or_commit_observation(section="extra", decision=decision, policy=SKIP_ALLOWED)
    grade = grade_observation(NO_EXPECTATION_CASE, observation)
    assert "commit_skip_contradiction" in codes(grade), "O-E05-F1"


@pytest.mark.parametrize(
    "grader",
    [
        lambda: grade_decision_prediction({"id": "D"}, {"route": "task", "confirmation": []}),
        lambda: grade_observation(
            NO_EXPECTATION_CASE,
            skip_or_commit_observation(
                section="meat",
                decision={"route": "task", "commit": True},
                policy={"status": "pass", "reason": "execution_allowed", "execute": True},
                plans=[{"item": [], "amount": "high"}],
            ),
        ),
    ],
)
def test_variant_nested_output_type_errors_return_failed_grades(grader):
    grade = grader()
    assert not grade.passed and "invalid_structure" in codes(grade), "O-E09-C1"


@pytest.mark.parametrize(
    "grader",
    [
        lambda: grade_recommendation_prediction(
            {**recommendation_case([]), "input": {**recommendation_case([])["input"], "restrictions": ["유제품"]}},
            proposal(sauce="오일"),
        ),
        lambda: grade_response_prediction(response_case(history=["소시지 좋아해"]), "소시지를 좋아한다고 하셨죠."),
    ],
)
def test_variant_malformed_fixture_is_reported_separately_from_output(grader):
    found = codes(grader())
    assert "invalid_fixture_structure" in found and "invalid_structure" not in found, "O-E06-X1"


def test_variant_keyword_calls_keep_working():
    assert grade_response_prediction(case=response_case(), reply="어떤 재료로 할까요?").passed
    assert grade_decision_prediction(case={"id": "D", "pending": None}, prediction={"route": "general"}).passed


def test_variant_evaluator_bugs_are_not_hidden_as_output_errors(monkeypatch):
    import soomac_irc.evaluation_v43 as evaluator

    def broken(*_args, **_kwargs):
        raise TypeError("injected evaluator bug")

    monkeypatch.setattr(evaluator, "_response_fact_checks", broken)
    with pytest.raises(TypeError, match="injected evaluator bug"):
        evaluator.grade_response_prediction(response_case(), "어떤 재료로 할까요?")


def test_variant_grade_equality_and_hash_ignore_identity():
    grade = grade_response_prediction(response_case(), "어떤 재료로 할까요?")
    assert grade == Grade(grade.case_id, grade.passed, grade.violations), "O-E10-3"
    assert hash(grade) == hash(Grade(grade.case_id, grade.passed, grade.violations)), "O-E10-3"


def test_variant_choice_question_is_not_an_execution_question():
    grade = grade_response_prediction(response_case(), "양파와 버섯 중 무엇을 담을까요?")
    assert grade.passed, ("O-E06-P2", grade.to_dict())


def test_variant_premise_inside_question_still_revives_withdrawn_preference():
    case = response_case(transition=TRANSITION, history=[user("나는 소시지를 정말 너무너무 좋아해"), user("그 취향은 취소할게")])
    grade = grade_response_prediction(case, "이제 육류 단계예요. 소시지 좋아하시니 넣어드릴까요?")
    assert "withdrawn_preference_revived" in codes(grade), "O-E07-F3"


def test_variant_removal_of_another_item_in_the_same_message_keeps_preference():
    case = response_case(history=[user("소시지는 좋아하는데 버섯은 취소할게")])
    grade = grade_response_prediction(case, "소시지를 좋아한다고 하셨죠.")
    assert grade.passed, ("O-E07-P3", grade.to_dict())


def test_variant_specific_noodle_name_counts_as_the_noodle_change():
    case = response_case(
        order=empty_order(noodle_type="얇은면", noodle_portion="low"),
        order_changes={"noodle_type": "얇은면", "noodle_portion": "low"},
    )
    grade = grade_response_prediction(case, "얇은면을 조금 선택했어요. 지금 담을까요?")
    assert grade.passed, ("O-E08-P1", grade.to_dict())


#################### 2차 교차검증 변형 ####################


def test_variant_question_with_da_before_space_is_not_a_completion_claim():
    case = response_case(order=NOODLE_CREAM_ORDER, completed=NOODLE_DONE)
    grade = grade_response_prediction(case, "크림 소스 다 담았어요?")
    assert grade.passed, ("O-E03-P2", grade.to_dict())


def test_variant_comma_list_inside_one_execution_question_keeps_amounts():
    case = response_case(order=empty_order(toppings={"양파": "low", "버섯": "low"}))
    grade = grade_response_prediction(case, "양파를 많이, 버섯을 조금 담을까요?")
    assert "stale_execution_question" in codes(grade), "O-E06-F5"


def test_variant_non_ingredient_wh_word_does_not_make_a_choice_question():
    grade = grade_response_prediction(response_case(), "양파를 어떤 그릇에 담을까요?")
    assert "ungrounded_execution_question" in codes(grade), "O-E06-F2"


@pytest.mark.parametrize(
    "oracle_id, history, passed",
    [
        ("O-E07-P3", [user("소시지와 버섯을 좋아해")], True),
        ("O-E07-F1", [user("소시지 좋아해"), user("소시지와 버섯은 싫어")], False),
    ],
)
def test_variant_parallel_preference_shares_the_latest_predicate(oracle_id, history, passed):
    grade = grade_response_prediction(response_case(history=history), "소시지를 좋아한다고 하셨죠.")
    assert grade.passed is passed, (oracle_id, grade.to_dict())


def test_variant_because_premise_inside_question_revives_withdrawn_preference():
    case = response_case(transition=TRANSITION, history=[user("소시지를 좋아해"), user("소시지 취향은 철회할게")])
    grade = grade_response_prediction(case, "소시지를 좋아하시기 때문에 넣어드릴까요?")
    assert "withdrawn_preference_revived" in codes(grade), "O-E07-F3"


def test_variant_noodle_name_needs_a_trailing_word_boundary():
    case = response_case(issues={"protected": [{"field": "noodle_portion", "value": "low"}]})
    grade = grade_response_prediction(case, "이미 담아서 바꿀 수 없어요. 면담으로 안내해 드릴게요.")
    assert "policy_issue_missing" in codes(grade), "O-E08-F2"


#################### 3차 교차검증 변형 ####################


def test_variant_comma_after_a_finished_start_claim_splits_the_clause():
    grade = grade_response_prediction(response_case(order=empty_order(toppings={"소시지": "high"})), "담기 시작할게, 괜찮으시죠?")
    assert "unauthorized_execution_reply" in codes(grade), "O-E01-F1"


@pytest.mark.parametrize("reply", ["양파를 어떤 면기에 담을까요?", "양파를 어떤 소스통에 담을까요?"])
def test_variant_container_question_is_not_an_ingredient_choice(reply):
    grade = grade_response_prediction(response_case(), reply)
    assert "ungrounded_execution_question" in codes(grade), ("O-E06-F2", reply)


@pytest.mark.parametrize(
    "oracle_id, history, passed",
    [
        ("O-E07-F2", [user("소시지는 먹어 본 적이 없어. 나는 조용한 식당을 좋아해")], False),
        ("O-E07-P3", [user("소시지는 좋아해. 오늘 일정이 바뀌었으니까 주문을 취소할게")], True),
    ],
)
def test_variant_preference_span_stops_at_the_sentence(oracle_id, history, passed):
    grade = grade_response_prediction(response_case(history=history), "소시지를 좋아한다고 하셨죠.")
    assert grade.passed is passed, (oracle_id, grade.to_dict())


@pytest.mark.parametrize("tail", ["면모를 살펴볼게요.", "면역에 관해 안내해 드릴게요."])
def test_variant_generic_noodle_needs_a_particle_boundary(tail):
    case = response_case(issues={"protected": [{"field": "noodle_portion", "value": "low"}]})
    grade = grade_response_prediction(case, f"이미 담아서 바꿀 수 없어요. {tail}")
    assert "policy_issue_missing" in codes(grade), ("O-E08-F2", tail)


@pytest.mark.parametrize(
    "grader, expected_code",
    [
        (lambda: grade_response_prediction(response_case(history=[None]), "소시지를 좋아한다고 하셨죠."), "invalid_fixture_structure"),
        (
            lambda: grade_observation(
                {"id": "O", "expected": ["bad"]},
                skip_or_commit_observation(section="extra", decision=SKIP, policy=SKIP_ALLOWED),
            ),
            "invalid_fixture_structure",
        ),
    ],
)
def test_variant_none_elements_and_bad_case_shapes_return_failed_grades(grader, expected_code):
    assert expected_code in codes(grader()), "O-E09-C1"


def test_variant_null_restrictions_are_treated_as_empty():
    observation = skip_or_commit_observation(section="extra", decision=SKIP, policy=SKIP_ALLOWED)
    for key in ("before_session", "session"):
        observation[key]["order"]["restrictions"] = None
    assert grade_observation(NO_EXPECTATION_CASE, observation).passed, "O-E09-C1"


#################### 4차 교차검증 변형 ####################


@pytest.mark.parametrize(
    "reply, passed",
    [
        ("양파를 조금 골랐다가 많이 골랐어요.", False),
        ("양파를 많이 골랐다가 조금 골랐어요.", True),
    ],
)
def test_variant_last_selection_in_the_span_decides_the_amount(reply, passed):
    grade = grade_response_prediction(response_case(order=empty_order(toppings={"양파": "low"})), reply)
    assert grade.passed is passed, ("O-E04-P3", grade.to_dict())


@pytest.mark.parametrize(
    "grader",
    [
        lambda: grade_decision_prediction({"id": "D", "metadata": ["bad"], "target": {"route": "task"}}, {"route": "task"}),
        lambda: grade_recommendation_prediction({**recommendation_case([]), "metadata": ["bad"]}, proposal(sauce="오일")),
        lambda: grade_recommendation_prediction({**recommendation_case([]), "target": ["bad"]}, proposal(sauce="오일")),
        lambda: grade_response_prediction({"id": "P", "input": ["bad"]}, "어떤 재료로 할까요?"),
    ],
)
def test_variant_malformed_case_fields_return_fixture_failures(grader):
    assert "invalid_fixture_structure" in codes(grader()), "O-E09-C1"


#################### 5차 교차검증 변형 ####################


@pytest.mark.parametrize("reply", ["면까지는 담았어요.", "소스만은 담았어요.", "면만큼은 담았어요.", "소스조차 담았어요."])
def test_variant_compound_particles_still_name_the_item(reply):
    case = {"id": "R5", "input": {"robot_state": {"completed_tasks": [{"class": "양파"}]}}}
    assert "unverified_item_completion" in codes(grade_response_prediction(case, reply)), ("O-E03-F1", reply)


def test_variant_non_object_decision_alternative_is_a_fixture_failure():
    case = {"id": "D", "pending": None, "target": {"route": "task"}, "metadata": {"decision_target_alternatives": ["bad"]}}
    assert "invalid_fixture_structure" in codes(grade_decision_prediction(case, {"route": "task"})), "O-E09-C1"


#################### 구조 판정과 문구 규칙 표시 분리 ####################


def test_text_rule_flags_are_separated_from_structural_result():
    grade = grade_response_prediction(response_case(order=empty_order(toppings={"소시지": "high"})), "소시지 담기 시작할게요.")
    assert not grade.passed and grade.structural_passed, "O-E10-1"
    assert grade.text_rule_flags == ("unauthorized_execution_reply",)
    assert grade.to_dict()["structural_passed"] is True


def test_structural_failure_is_not_a_text_rule_flag():
    observation = skip_or_commit_observation(section="sauce", decision=SKIP, policy=SKIP_ALLOWED)
    grade = grade_observation(NO_EXPECTATION_CASE, observation)
    assert not grade.structural_passed and grade.text_rule_flags == (), "O-E10-1"


#################### Codex 보완: 권한 타입과 깨진 평가 입력 ####################


@pytest.mark.parametrize("bad", ["false", "true", "", 0, 1, None, [], [False], {}])
def test_authorization_requires_an_actual_boolean(bad):
    case = response_case(order=empty_order(toppings={"소시지": "high"}))
    case["input"]["execution_authorized"] = bad
    grade = grade_response_prediction(case, "소시지 담기 시작할게요.")
    assert not grade.structural_passed
    assert "invalid_fixture_structure" in codes(grade)


@pytest.mark.parametrize("authorized", [False, True])
def test_boolean_authorization_preserves_valid_start_grading(authorized):
    case = response_case(order=empty_order(toppings={"소시지": "high"}), authorized=authorized)
    grade = grade_response_prediction(case, "소시지 담기 시작할게요.")
    assert grade.structural_passed
    assert grade.passed is authorized


@pytest.mark.parametrize("field", ["commit", "skip_current", "clarify"])
@pytest.mark.parametrize("bad", ["false", 0, 1, None])
def test_decision_execution_flags_require_booleans(field, bad):
    grade = grade_decision_prediction({"target": {"route": "task"}}, {"route": "task", field: bad})
    assert not grade.structural_passed and "invalid_structure" in codes(grade)


@pytest.mark.parametrize("bad", ["false", 0, 1, None, [False]])
def test_observation_execute_requires_a_boolean(bad):
    grade = grade_observation({}, {"decision": {"route": "task"}, "policy": {"execute": bad}})
    assert not grade.structural_passed and "invalid_structure" in codes(grade)


@pytest.mark.parametrize("bad", ["false", 0, None, [False]])
def test_response_policy_execute_requires_a_boolean(bad):
    case = response_case()
    case["input"]["policy"]["execute"] = bad
    grade = grade_response_prediction(case, "무엇을 골라드릴까요?")
    assert not grade.structural_passed and "invalid_fixture_structure" in codes(grade)


@pytest.mark.parametrize("bad", [None, [], "broken", False])
def test_observation_invalid_root_is_a_failed_grade(bad):
    grade = grade_observation({}, bad)
    assert not grade.structural_passed and "invalid_structure" in codes(grade)


@pytest.mark.parametrize("grader,output", [
    (grade_observation, {"decision": {"route": "task"}}),
    (grade_decision_prediction, {"route": "task"}),
    (grade_recommendation_prediction, {"proposal": {}}),
    (grade_response_prediction, "무엇을 골라드릴까요?"),
])
@pytest.mark.parametrize("bad", [None, [], "broken"])
def test_invalid_fixture_root_is_a_failed_grade(grader, output, bad):
    grade = grader(bad, output)
    assert not grade.structural_passed and "invalid_fixture_structure" in codes(grade)


@pytest.mark.parametrize("operator", ["$one_of", "$contains"])
@pytest.mark.parametrize("bad", [None, {}, "broken", 1])
def test_malformed_pattern_operand_is_a_fixture_failure(operator, bad):
    case = {"expected": {"session": {operator: bad}}}
    grade = grade_observation(case, {"decision": {"route": "task"}})
    assert not grade.structural_passed and "invalid_fixture_structure" in codes(grade)


def test_nested_malformed_decision_pattern_is_a_fixture_failure():
    case = {"expected_any": [{"order": {"toppings": {"$one_of": None}}}]}
    grade = grade_decision_prediction(case, {"route": "task"})
    assert not grade.structural_passed and "invalid_fixture_structure" in codes(grade)


def test_valid_nested_pattern_still_matches():
    case = {"expected": {"session": {"$one_of": [{"marker": {"$contains": ["ok"]}}]}}}
    grade = grade_observation(case, {"decision": {"route": "task"}, "session": {"marker": ["ok", "other"]}})
    assert grade.passed


@pytest.mark.parametrize("bad", [None, "", "   ", False, 0, [], {}])
def test_pending_target_must_be_a_nonempty_name(bad):
    case = response_case(pending={"type": "execution", "targets": [bad]})
    grade = grade_response_prediction(case, "실행 확인이 남아 있어요.")
    assert not grade.structural_passed and "invalid_fixture_structure" in codes(grade)


def test_batch_continues_after_invalid_authority_and_null_observation():
    malformed = response_case()
    malformed["input"]["execution_authorized"] = "false"
    grades = [
        grade_response_prediction(malformed, "담기 시작할게요."),
        grade_observation({}, None),
        grade_response_prediction(response_case(), "무엇을 골라드릴까요?"),
    ]
    assert [grade.structural_passed for grade in grades] == [False, False, True]


@pytest.mark.parametrize("field", ["commit", "skip_current", "clarify"])
@pytest.mark.parametrize("source", ["target", "alternative"])
@pytest.mark.parametrize("bad", [1, 0, "false", None])
def test_decision_gold_execution_flags_require_booleans(field, source, bad):
    target = {"route": "task", field: bad}
    case = {"target": target} if source == "target" else {
        "target": {"route": "task"},
        "metadata": {"decision_target_alternatives": [target]},
    }
    grade = grade_decision_prediction(case, {"route": "task", field: True})
    assert not grade.structural_passed and "invalid_fixture_structure" in codes(grade)


@pytest.mark.parametrize("source", ["target", "alternative"])
def test_valid_normalized_gold_with_boolean_defaults_still_matches(source):
    target = {"route": "task", "order_patch": {"toppings": {"양파": "low"}},
              "commit": False, "skip_current": False, "clarify": False}
    case = {"target": target} if source == "target" else {
        "target": {"route": "task"}, "metadata": {"decision_target_alternatives": [target]},
    }
    grade = grade_decision_prediction(case, {"route": "task", "order": {"toppings": {"양파": "low"}}})
    assert grade.passed and grade.structural_passed
