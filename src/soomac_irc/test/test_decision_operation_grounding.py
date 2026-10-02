"""P1 target + operation grounding의 위험 사례를 먼저 고정한다."""

import copy
import unittest

from soomac_irc.agent_contract import new_decision
from soomac_irc.llm_langgraph import (
    build_graph,
    new_session_state,
    new_turn_state,
)


ROBOT_STATE = {
    "section": "noodle",
    "active_task": None,
    "task_queue": [],
    "completed_tasks": [],
}


def order_decision(target: str, amount: str, mentions=None) -> dict:
    decision = new_decision()
    decision["route"] = "task"
    decision["mentions"] = copy.deepcopy(mentions or [])
    decision["order_patch"]["toppings"] = {
        target: amount,
    }
    return decision


def invoke_decision(
    user_text: str,
    decision: dict,
    session: dict | None = None,
) -> tuple[dict, dict]:
    initial_session = copy.deepcopy(
        session if session is not None else new_session_state()
    )
    graph = build_graph(
        lambda *args: copy.deepcopy(decision),
        lambda *args: {},
        lambda *args: "응답",
    )
    result = graph.invoke(
        new_turn_state(
            initial_session,
            user_text,
            ROBOT_STATE,
        )
    )
    return initial_session, result


class TestDangerousDecisionContradictions(unittest.TestCase):
    def test_vegetable_collection_does_not_allow_partial_mutation(self):
        vegetable_session = new_session_state()
        vegetable_session["order"]["toppings"] = {
            "양파": "normal",
            "버섯": "normal",
        }
        vegetable_decision = new_decision()
        vegetable_decision["route"] = "task"
        vegetable_decision["mentions"] = ["야채"]
        vegetable_decision["order_patch"]["toppings"] = {
            "양파": "none",
        }

        vegetable_initial, vegetable_result = invoke_decision(
            "야채 빼줘",
            vegetable_decision,
            vegetable_session,
        )
        self.assertEqual(
            vegetable_result["session"]["order"],
            vegetable_initial["order"],
        )
        
    def test_extra_collection_does_not_allow_partial_mutation(self):
        extra_session = new_session_state()
        extra_decision = new_decision()
        extra_decision["route"] = "task"
        extra_decision["mentions"] = ["추가 재료"]
        extra_decision["order_patch"]["toppings"] = {
            "치즈": "high",
        }

        extra_initial, extra_result = invoke_decision(
            "추가 재료 많이 넣어줘",
            extra_decision,
            extra_session,
        )
        self.assertEqual(
            extra_result["session"]["order"],
            extra_initial["order"],
        )
    def test_wrong_target_or_operation_does_not_mutate_order(self):
        cases = [
            {
                "name": "unsupported_ham_becomes_onion",
                "user_text": "햄 많이 넣어줘",
                "decision": order_decision("양파", "high"),
                "initial_toppings": {},
            },
            {
                "name": "pepperoni_becomes_peperoncino",
                "user_text": "페퍼로니 조금 넣어줘",
                "decision": order_decision(
                    "페퍼론치노",
                    "low",
                    ["페퍼로니"],
                ),
                "initial_toppings": {},
            },
            {
                "name": "remove_becomes_add",
                "user_text": "양파 빼줘",
                "decision": order_decision("양파", "high"),
                "initial_toppings": {},
            },
            {
                "name": "add_becomes_remove",
                "user_text": "양파 많이 넣어줘",
                "decision": order_decision("양파", "none"),
                "initial_toppings": {
                    "양파": "normal",
                },
            },
            {
                "name": "general_question_becomes_mutation",
                "user_text": "양파는 뭐야?",
                "decision": order_decision("양파", "normal"),
                "initial_toppings": {},
            },
        ]

        for case in cases:
            with self.subTest(case=case["name"]):
                session = new_session_state()
                session["order"]["toppings"] = copy.deepcopy(
                    case["initial_toppings"]
                )
                initial, result = invoke_decision(
                    case["user_text"],
                    case["decision"],
                    session,
                )

                self.assertEqual(
                    result["session"]["order"],
                    initial["order"],
                )
                self.assertFalse(result["policy"]["execute"])

    def test_question_does_not_remove_restriction(self):
        session = new_session_state()
        session["order"]["restrictions"] = [
            {
                "target": "게살",
                "reason": "allergy",
            }
        ]

        decision = new_decision()
        decision["route"] = "task"
        decision["mentions"] = ["게살"]
        decision["restriction_options"] = [
            {
                "target": "게살",
                "type": "allergy",
                "action": "remove",
            }
        ]

        initial, result = invoke_decision(
            "게살 먹어도 돼?",
            decision,
            session,
        )

        self.assertEqual(
            result["session"]["order"]["restrictions"],
            initial["order"]["restrictions"],
        )
        self.assertFalse(result["policy"]["execute"])

    def test_execution_question_does_not_execute(self):
        session = new_session_state()
        session["order"].update({
            "sauce": "토마토",
            "noodle_type": "얇은면",
            "noodle_portion": "normal",
        })
        session["order"]["toppings"] = {
            "치즈": "normal",
        }
        session["dialogue_focus"]["current"] = {
            "mentions": ["치즈"],
            "history_turn": 1,
        }

        decision = order_decision("치즈", "normal")
        decision["commit"] = True

        _, result = invoke_decision(
            "그거 실행해볼까?",
            decision,
            session,
        )

        self.assertFalse(result["policy"]["execute"])


class TestValidDecisionControls(unittest.TestCase):
    def test_matching_add_and_remove_still_work(self):
        add_session = new_session_state()
        _, added = invoke_decision(
            "양파 많이 넣어줘",
            order_decision("양파", "high"),
            add_session,
        )
        self.assertEqual(
            added["session"]["order"]["toppings"]["양파"],
            "high",
        )

        remove_session = new_session_state()
        remove_session["order"]["toppings"] = {
            "양파": "normal",
        }
        _, removed = invoke_decision(
            "양파 빼줘",
            order_decision("양파", "none"),
            remove_session,
        )
        self.assertNotIn(
            "양파",
            removed["session"]["order"]["toppings"],
        )

    def test_explicit_restriction_removal_still_works(self):
        session = new_session_state()
        session["order"]["restrictions"] = [
            {
                "target": "게살",
                "reason": "allergy",
            }
        ]

        decision = new_decision()
        decision["route"] = "task"
        decision["mentions"] = ["게살"]
        decision["restriction_options"] = [
            {
                "target": "게살",
                "type": "allergy",
                "action": "remove",
            }
        ]

        _, result = invoke_decision(
            "게살 알레르기 제한 해제해줘",
            decision,
            session,
        )
        self.assertEqual(
            result["session"]["order"]["restrictions"],
            [],
        )

    def test_explicit_execution_still_executes(self):
        session = new_session_state()
        session["order"].update({
            "sauce": "토마토",
            "noodle_type": "얇은면",
            "noodle_portion": "normal",
        })
        session["order"]["toppings"] = {
            "치즈": "normal",
        }
        session["dialogue_focus"]["current"] = {
            "mentions": ["치즈"],
            "history_turn": 1,
        }

        decision = order_decision("치즈", "normal")
        decision["commit"] = True

        _, result = invoke_decision(
            "그거 실행해줘",
            decision,
            session,
        )
        self.assertTrue(result["policy"]["execute"])

    def test_complete_collection_mutation_still_works(self):
        vegetable_session = new_session_state()
        vegetable_session["order"]["toppings"] = {
            "양파": "normal",
            "버섯": "normal",
        }
        vegetable_decision = new_decision()
        vegetable_decision["route"] = "task"
        vegetable_decision["mentions"] = ["야채"]
        vegetable_decision["order_patch"]["toppings"] = {
            "양파": "none",
            "버섯": "none",
        }

        _, vegetable_result = invoke_decision(
            "야채 빼줘",
            vegetable_decision,
            vegetable_session,
        )
        self.assertEqual(
            vegetable_result["session"]["order"]["toppings"],
            {},
        )

        extra_decision = new_decision()
        extra_decision["route"] = "task"
        extra_decision["mentions"] = ["추가 재료"]
        extra_decision["order_patch"]["toppings"] = {
            "치즈": "high",
            "페퍼론치노": "high",
        }

        _, extra_result = invoke_decision(
            "추가 재료 많이 넣어줘",
            extra_decision,
        )
        self.assertEqual(
            extra_result["session"]["order"]["toppings"],
            {
                "치즈": "high",
                "페퍼론치노": "high",
            },
        )
    def test_mixed_information_and_mutation_still_works(self):
        decision = order_decision("양파", "high")
        decision["route"] = "mixed"

        _, result = invoke_decision(
            "양파 많이 넣어줘. 그리고 양파는 뭐야?",
            decision,
        )

        self.assertEqual(
            result["session"]["order"]["toppings"]["양파"],
            "high",
        )

if __name__ == "__main__":
    unittest.main()