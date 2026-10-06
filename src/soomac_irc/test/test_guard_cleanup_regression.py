"""2026-10-05 로그에서 실패한 대화를 모의 Decision으로 재현한다. 모델 품질 시험은 아니다."""

import pytest

pytest.skip(
    "legacy Decision v2 state contract; replacement coverage is in "
    "test_natural_multiturn_contract.py",
    allow_module_level=True,
)

import copy
import unittest

from soomac_irc.agent_contract import new_decision
from soomac_irc.decision_overrides import pre_decision_override
from soomac_irc.llm_langgraph import build_graph, new_session_state, new_turn_state


def robot(section):
    return {"section": section, "active_task": None, "task_queue": [], "completed_tasks": []}


def invoke(text, decision, session=None, section="veggie", recommendation=None, response=None):
    def decide(session, text, robot_state, repair):
        return pre_decision_override(text, session, robot_state) or copy.deepcopy(decision)

    graph = build_graph(
        decide,
        recommendation or (lambda *args: {}),
        response or (lambda *args: "응답"),
    )
    return graph.invoke(new_turn_state(
        copy.deepcopy(session or new_session_state()), text, robot(section),
    ))


def topping_decision(toppings, restrictions=(), mentions=(), commit=False):
    decision = new_decision()
    decision["route"] = "task"
    decision["mentions"] = list(mentions)
    decision["order_patch"]["toppings"] = dict(toppings)
    decision["restriction_options"] = [dict(option) for option in restrictions]
    decision["commit"] = commit
    return decision


ONION_ALLERGY = {"target": "양파", "type": "allergy", "action": "add"}


def onion_conflict_session():
    # 175513 #7: "…알러지가 있고 양파만은 적게" → 양파 알러지 + 양파 적게, 버섯 보통도 함께 요청한 경우
    return invoke(
        "벗어준 알러지가 있고 양파만은 적게 넣고 버섯은 보통",
        topping_decision({"양파": "low", "버섯": "normal"}, [ONION_ALLERGY], ["양파", "버섯"]),
    )


class TestRestrictionConfirmationLifecycle(unittest.TestCase):
    def test_conflict_keeps_restriction_live_and_holds_only_conflicting_item(self):
        result = onion_conflict_session()
        session = result["session"]

        self.assertEqual(result["policy"]["status"], "hitl")
        self.assertIn({"target": "양파", "reason": "allergy"}, session["order"]["restrictions"])
        self.assertEqual(session["order"]["toppings"], {"버섯": "normal"})  # 충돌 없는 변경은 반영
        self.assertEqual(session["pending_confirmation"]["held_patch"]["toppings"], {"양파": "low"})
        self.assertNotIn("candidate_session", session["pending_confirmation"])

    def test_unanswered_confirmation_expires_after_one_turn(self):
        session = onion_conflict_session()["session"]
        result = invoke("버섯은 많이", topping_decision({"버섯": "high"}, mentions=["버섯"]), session)

        self.assertIsNone(result["session"]["pending_confirmation"])
        self.assertEqual(result["policy"]["reason"], "state_update_only")
        self.assertIn({"target": "양파", "reason": "allergy"}, result["session"]["order"]["restrictions"])
        self.assertNotIn("양파", result["session"]["order"]["toppings"])

    def test_general_turn_also_expires_confirmation(self):
        session = onion_conflict_session()["session"]
        decision = new_decision()
        decision["route"] = "general"
        result = invoke("동국대학교가 뭐야", decision, session)

        self.assertIsNone(result["session"]["pending_confirmation"])

    def test_accept_applies_held_item_on_current_state(self):
        session = onion_conflict_session()["session"]
        result = invoke("응", new_decision(), session)

        self.assertEqual(result["session"]["order"]["restrictions"], [])
        self.assertEqual(result["session"]["order"]["toppings"], {"양파": "low", "버섯": "normal"})
        self.assertIsNone(result["session"]["pending_confirmation"])

    def test_reject_drops_held_item_and_keeps_restriction(self):
        session = onion_conflict_session()["session"]
        result = invoke("아니", new_decision(), session)

        self.assertEqual(result["session"]["order"]["toppings"], {"버섯": "normal"})
        self.assertIn({"target": "양파", "reason": "allergy"}, result["session"]["order"]["restrictions"])

    def test_commit_while_waiting_does_not_put_conflicting_item(self):
        session = onion_conflict_session()["session"]
        decision = new_decision()
        decision["route"] = "task"
        decision["commit"] = True
        result = invoke("바로 시작해", decision, session)

        self.assertNotIn("양파", result["session"]["order"]["toppings"])

    def test_spoken_restriction_removal_without_marker_word(self):
        # 175513 #32: "양파 제약 빼자" — 예전에는 '빼'가 해제 단어 목록에 없어 버려졌다.
        session = new_session_state()
        session["order"]["restrictions"] = [{"target": "양파", "reason": "allergy"}]
        decision = new_decision()
        decision["route"] = "task"
        decision["mentions"] = ["양파"]
        decision["restriction_options"] = [{"target": "양파", "type": "allergy", "action": "remove"}]
        result = invoke("아니 양파 제약 빼자", decision, session)

        self.assertEqual(result["session"]["order"]["restrictions"], [])

    def test_restriction_removal_can_refer_to_pending_conflict(self):
        # 175513 #33: "해제하라고" — 대상이 발화에 없어도 지금 확인 중인 충돌 대상이면 근거로 인정한다.
        session = onion_conflict_session()["session"]
        decision = new_decision()
        decision["route"] = "task"
        decision["restriction_options"] = [{"target": "양파", "type": "allergy", "action": "remove"}]
        result = invoke("해제하라고", decision, session)

        self.assertEqual(result["session"]["order"]["restrictions"], [])


class TestStaleRecommendation(unittest.TestCase):
    def test_direct_choice_closes_recommendation(self):
        # 175513 #16~#18: 소시지+게살 추천 → "소시지만 많이" → "바로 시작해"에 게살이 몰래 들어갔다.
        session = new_session_state()
        session["recommendation"].update(phase="proposed", last_proposal={
            "action": "request", "scope": "current", "criteria": None, "reason_tags": [],
            "proposal": {"sauce": None, "noodle_type": None, "noodle_portion": None,
                         "toppings": {"소시지": "high", "게살": "high"}},
        })
        chosen = invoke("소시지만 많이 줘", topping_decision({"소시지": "high"}, mentions=["소시지"]),
                        session, section="meat")
        self.assertEqual(chosen["session"]["recommendation"]["phase"], "idle")

        start = new_decision()
        start["route"] = "task"
        start["commit"] = True
        result = invoke("바로 시작해", start, chosen["session"], section="meat")

        self.assertEqual(result["session"]["order"]["toppings"], {"소시지": "high"})
        self.assertTrue(result["policy"]["execute"])


class TestGroundingKeepsValidParts(unittest.TestCase):
    def test_stt_variant_and_restriction_survive_together(self):
        # 175513 #9: "버섯 알러지가 있는데 양판은 먹을게" — 예전에는 양파 근거 실패로 버섯 알러지까지 버렸다.
        result = invoke(
            "버섯 알러지가 있는데 양판은 먹을게",
            topping_decision({"양파": "low"}, [{"target": "버섯", "type": "allergy", "action": "add"}],
                             ["버섯", "양판"]),
        )

        self.assertEqual(result["session"]["order"]["toppings"], {"양파": "low"})
        self.assertIn({"target": "버섯", "reason": "allergy"}, result["session"]["order"]["restrictions"])

    def test_only_ungrounded_item_is_dropped(self):
        result = invoke(
            "양파 많이 넣어줘",
            topping_decision({"양파": "high", "버섯": "high"}, mentions=["양파"]),
        )

        self.assertEqual(result["session"]["order"]["toppings"], {"양파": "high"})

    def test_stt_variant_is_stored_as_canonical_focus(self):
        # 다음 턴 '그거'가 '양판'이 아니라 양파를 가리켜야 한다.
        result = invoke("양판 보통으로 줘", topping_decision({"양파": "normal"}, mentions=["양판"]))

        self.assertEqual(result["session"]["dialogue_focus"]["current"]["mentions"], ["양파"])

    def test_unknown_menu_is_reported_as_unsupported(self):
        decision = topping_decision({"페퍼론치노": "low"}, mentions=["고추"])
        result = invoke("고추만 담아줘", decision, section="extra")

        self.assertEqual(result["session"]["order"]["toppings"], {})
        self.assertEqual(result["policy"]["reason"], "unsupported_reference")
        self.assertEqual(result["policy"]["reference_targets"], ["고추"])

    def test_unsupported_reply_comes_from_response_agent(self):
        calls = []
        decision = topping_decision({"페퍼론치노": "low"}, mentions=["고추"])
        result = invoke("고추만 담아줘", decision, section="extra",
                        response=lambda *args: calls.append(args[2]) or "모델 응답")

        self.assertEqual(result["reply"], "모델 응답")
        self.assertEqual(calls[0]["reason"], "unsupported_reference")


class TestSectionExecutionItems(unittest.TestCase):
    def test_items_follow_current_section(self):
        from soomac_irc.llm_policy import section_execution_items

        order = new_session_state()["order"]
        order.update(sauce="토마토", noodle_type="얇은면", noodle_portion="low")
        order["toppings"] = {"양파": "normal", "버섯": "none", "소시지": "high"}

        # 18:18 #8: veggie 실행 응답이 '면'을 말했다. Response에는 이 값이 starting_now로 들어간다.
        self.assertEqual(section_execution_items(order, "veggie"), [{"item": "양파", "amount": "normal"}])
        self.assertEqual(section_execution_items(order, "noodle"), [{"item": "얇은면", "amount": "low"}])
        self.assertEqual(section_execution_items(order, "meat"), [{"item": "소시지", "amount": "high"}])
        self.assertEqual(section_execution_items(order, "extra"), [])
        self.assertEqual(section_execution_items(order, "sauce"), [{"item": "토마토", "amount": None}])


class TestAlreadySetFacts(unittest.TestCase):
    def test_repeated_request_is_passed_as_already_set(self):
        # 17:55 #10: 변경 0건인데 응답이 직전 "반영했어요"를 반복했다. 이미 있던 값을 사실로 넘긴다.
        session = new_session_state()
        session["order"]["toppings"] = {"양파": "high"}
        session["order"]["restrictions"] = [{"target": "버섯", "reason": "allergy"}]
        calls = []
        decision = topping_decision(
            {"양파": "high"}, [{"target": "버섯", "type": "allergy", "action": "add"}], ["버섯", "양판"],
        )
        result = invoke("버섯 알러지 있고 양판은 먹을게", decision, session,
                        response=lambda *args: calls.append(args) or "응답")

        policy, applied_changes = calls[0][2], calls[0][3]
        self.assertEqual(policy["already_set"], {
            "order": {"toppings.양파": "high"},
            "restrictions": [{"target": "버섯", "reason": "allergy"}],
        })
        self.assertFalse(any(applied_changes.values()))
        self.assertEqual(result["policy"]["already_set"], policy["already_set"])

    def test_new_value_is_not_already_set(self):
        calls = []
        invoke("양파 많이", topping_decision({"양파": "high"}, mentions=["양파"]),
               response=lambda *args: calls.append(args) or "응답")

        self.assertEqual(calls[0][2]["already_set"], {})


class TestDecisionInputMatchesTraining(unittest.TestCase):
    def test_pending_view_uses_training_shape(self):
        try:
            from soomac_irc.decision_model import decision_pending_view
        except ModuleNotFoundError as error:
            self.skipTest(f"모델 런타임 의존성 없음: {error}")

        pending = onion_conflict_session()["session"]["pending_confirmation"]
        self.assertEqual(
            decision_pending_view(pending),
            {"type": "restriction_conflict", "conflicts": [{"item": "양파"}]},
        )


if __name__ == "__main__":
    unittest.main()
