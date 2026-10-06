"""실제 graph에 모의 Decision을 넣는다. 모델 품질/실물 실행 시험은 아니다."""

import pytest

pytest.skip(
    "legacy Decision v2 state contract; replacement coverage is in "
    "test_natural_multiturn_contract.py",
    allow_module_level=True,
)
import copy
import unittest

from soomac_irc.agent_contract import new_decision
from soomac_irc.agent_prompts import DECISION_SYSTEM
from soomac_irc.decision_overrides import pre_decision_override
from soomac_irc.dialogue_focus import build_reference_context, update_dialogue_focus
from soomac_irc.dialogue_questions import active_question, question_for_section
from soomac_irc.llm_langgraph import build_graph, new_session_state, new_turn_state


def candidate_session(target="게살", amount="normal"):
    session = new_session_state()
    proposal = new_decision()["order_patch"]
    proposal["toppings"][target] = amount
    session["pending_question"] = {
        "type": "menu_confirmation", "targets": [target], "proposal": proposal,
        "section": "meat", "history_turn": 1,
    }
    return session


def invoke(text, decision, session=None, section="meat", robot=None, response=None):
    robot = robot or {"section": section, "active_task": None,
                      "task_queue": [], "completed_tasks": []}
    def decide(session, text, robot, repair):
        return pre_decision_override(text, session, robot) or copy.deepcopy(decision)
    graph = build_graph(decide, lambda *args: {}, response or (lambda *args: "모의 응답"))
    return graph.invoke(new_turn_state(session or new_session_state(), text, robot))


class TestQuestionLifecycle(unittest.TestCase):
    def test_prompt_allows_menu_confirmation_without_safety_confirmation(self):
        self.assertIn("pending_question.type=menu_confirmation", DECISION_SYSTEM)
        self.assertNotIn("pending_confirmation이 없으면 confirmation field를 생략한다", DECISION_SYSTEM)

    def test_accept_with_and_without_reference(self):
        for text in ("응", "응, 그거", "네 그걸로 해주세요"):
            with self.subTest(text=text):
                decision = new_decision()
                decision["confirmation"] = "accept"
                result = invoke(text, decision, candidate_session())
                self.assertEqual(result["session"]["order"]["toppings"], {"게살": "normal"})
                self.assertFalse(result["policy"]["execute"])
                # 후보 질문은 닫히고, 반영된 현재 section에 대한 실행 제안만 남는다. (2026-10-05 execution_offer)
                self.assertEqual(result["session"]["pending_question"]["type"], "execution_offer")

    def test_compound_accept_keeps_explicit_changes(self):
        decision = new_decision()
        decision["confirmation"] = "accept"
        decision["order_patch"]["toppings"] = {"게살": "high", "소시지": "normal"}
        result = invoke("응 그거 많이 하고 소시지도 보통 양", decision, candidate_session())
        self.assertEqual(result["session"]["order"]["toppings"], {"게살": "high", "소시지": "normal"})
        self.assertFalse(result["policy"]["execute"])

    def test_confirmed_removal_does_not_require_repeating_operation(self):
        session = candidate_session(amount="none")
        session["order"]["toppings"]["게살"] = "normal"
        decision = new_decision()
        decision["confirmation"] = "accept"
        result = invoke("응 그거", decision, session)
        self.assertNotIn("게살", result["session"]["order"]["toppings"])

    def test_reject_discards_candidate(self):
        result = invoke("아니", new_decision(), candidate_session())
        self.assertEqual(result["session"]["order"]["toppings"], {})
        self.assertIsNone(result["session"]["pending_question"])
        self.assertFalse(result["policy"]["execute"])

    def test_accept_still_checks_allergy(self):
        session = candidate_session()
        session["order"]["restrictions"] = [{"target": "게살", "reason": "allergy"}]
        result = invoke("응", new_decision(), session)
        self.assertEqual(result["policy"]["status"], "hitl")
        self.assertEqual(result["session"]["order"]["toppings"], {})
        self.assertIsNotNone(result["session"]["pending_confirmation"])
        self.assertIsNone(result["session"]["pending_question"])

    def test_accept_still_checks_active_task(self):
        session = candidate_session(amount="high")
        session["order"]["toppings"]["게살"] = "normal"
        robot = {"section": "meat", "active_task": {"class": "게살", "repeat_count": 2},
                 "task_queue": [], "completed_tasks": []}
        result = invoke("응", new_decision(), session, robot=robot)
        self.assertEqual(result["policy"]["reason"], "physical_state")
        self.assertEqual(result["session"]["order"]["toppings"]["게살"], "normal")
        self.assertFalse(result["policy"]["execute"])

    def test_question_survives_three_turns_but_focus_does_not(self):
        session = candidate_session()
        session["history"] = [{"role": "user", "content": "잡담"},
                              {"role": "assistant", "content": "답변"}] * 5
        session["dialogue_focus"] = update_dialogue_focus(session["dialogue_focus"], ["게살"], 1)
        self.assertEqual(build_reference_context("그거", session["dialogue_focus"], 6)["status"], "stale")
        decision = new_decision()
        decision["confirmation"] = "accept"
        result = invoke("응 그거", decision, session)
        self.assertEqual(result["session"]["order"]["toppings"], {"게살": "normal"})

    def test_choice_answer_consumes_question(self):
        session = new_session_state()
        session["pending_question"] = question_for_section("meat", 0)
        decision = new_decision()
        decision["order_patch"]["toppings"] = {"게살": "normal", "소시지": "normal"}
        result = invoke("두 개 다 보통 양으로 줘", decision, session)
        self.assertEqual(result["session"]["order"]["toppings"], decision["order_patch"]["toppings"])
        # 선택 질문은 닫히고 실행 제안으로 바뀐다. (2026-10-05 execution_offer)
        self.assertEqual(result["session"]["pending_question"]["type"], "execution_offer")

    def test_new_topic_clears_old_candidate(self):
        decision = new_decision()
        decision["route"] = "general"
        decision["mentions"] = ["떡볶이"]
        result = invoke("떡볶이가 뭐야", decision, candidate_session())
        self.assertIsNone(result["session"]["pending_question"])
        decision = new_decision()
        decision["order_patch"]["toppings"]["게살"] = "normal"
        result = invoke("그거 줘", decision, result["session"])
        self.assertEqual(result["policy"]["reason"], "unsupported_reference")
        self.assertEqual(result["session"]["order"]["toppings"], {})

    def test_unrelated_general_turn_without_target_keeps_question(self):
        decision = new_decision()
        decision["route"] = "general"
        result = invoke("고마워", decision, candidate_session())
        self.assertEqual(result["session"]["pending_question"], candidate_session()["pending_question"])

    def test_different_section_invalidates_question(self):
        session = candidate_session()
        self.assertIsNone(active_question(session, "extra"))
        result = invoke("응", new_decision(), session, section="extra")
        self.assertEqual(result["session"]["order"]["toppings"], {})

    def test_planned_question_is_saved_and_replaced_after_answer(self):
        calls = []
        def response(*args):
            calls.append(copy.deepcopy(args[7]))
            return "모의 질문"
        decision = new_decision()
        decision["order_patch"]["noodle_type"] = "얇은면"
        result = invoke("면은 얇은 걸로", decision, section="noodle", response=response)
        question = result["session"]["pending_question"]
        self.assertEqual(calls[-1], {"type": "missing_field", "target": "noodle_portion"})
        self.assertEqual(question["fields"], ["noodle_portion"])
        self.assertEqual(question["targets"], ["얇은면"])
        decision = new_decision()
        decision["order_patch"]["noodle_portion"] = "low"
        result = invoke("조금만", decision, result["session"], "noodle", response=response)
        self.assertEqual(result["session"]["order"]["noodle_portion"], "low")
        self.assertEqual(result["session"]["pending_question"]["target"], "sauce")

    def test_candidate_generated_in_graph_is_same_question_given_to_response(self):
        calls = []
        decision = new_decision()
        decision["order_patch"]["toppings"] = {"게살": "normal"}
        # '개사'는 게살과 자모 거리 0.4라 바로 반영하지 않고 확인한다. ('계살'은 0.2라 바로 반영된다)
        result = invoke("개사 주세요", decision, response=lambda *args: calls.append(args[7]) or "확인 질문")
        self.assertEqual(result["policy"]["reason"], "menu_confirmation")
        self.assertEqual(result["session"]["order"]["toppings"], {})
        self.assertEqual(calls[0]["proposal"], result["session"]["pending_question"]["proposal"])
        decision = new_decision()
        decision["confirmation"] = "accept"
        result = invoke("응 그거", decision, result["session"])
        self.assertEqual(result["session"]["order"]["toppings"], {"게살": "normal"})


if __name__ == "__main__":
    unittest.main()
