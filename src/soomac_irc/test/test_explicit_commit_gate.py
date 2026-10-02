"""명시적 실행 의도만 commit=true로 복구하는 Python positive gate 테스트."""

import copy
import unittest

from soomac_irc.agent_contract import new_decision
from soomac_irc.agent_prompts import DECISION_SYSTEM
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


def task_decision() -> dict:
    decision = new_decision()
    decision["route"] = "task"
    return decision


def invoke_decision(
    user_text: str,
    decision: dict,
    session: dict | None = None,
) -> dict:
    initial_session = copy.deepcopy(
        session if session is not None else new_session_state()
    )
    graph = build_graph(
        lambda *args: copy.deepcopy(decision),
        lambda *args: {},
        lambda *args: "응답",
    )
    return graph.invoke(
        new_turn_state(
            initial_session,
            user_text,
            ROBOT_STATE,
        )
    )


class TestExplicitCommitPositiveGate(unittest.TestCase):
    def test_explicit_execution_phrases_raise_commit(self):
        phrases = (
            "바로 진행해",
            "이대로 진행해",
            "담기 시작해",
            "바로 시작해",
            "주문 확정해",
        )

        for user_text in phrases:
            with self.subTest(user_text=user_text):
                result = invoke_decision(
                    user_text,
                    task_decision(),
                )
                self.assertTrue(result["decision"]["commit"])
                self.assertFalse(result["policy"]["execute"])

    def test_order_semantics_are_preserved_when_commit_is_raised(self):
        decision = task_decision()
        decision["mentions"] = ["치즈"]
        decision["order_patch"]["toppings"] = {
            "치즈": "high",
        }

        result = invoke_decision(
            "치즈 많이 넣고 바로 진행해",
            decision,
        )

        self.assertEqual(
            result["session"]["order"]["toppings"],
            {
                "치즈": "high",
            },
        )
        self.assertTrue(result["decision"]["commit"])

    def test_question_proposal_and_negative_do_not_raise_commit(self):
        phrases = (
            "바로 진행해도 돼?",
            "진행해볼까?",
            "진행하지 마",
            "시작하지 마",
            "치즈 넣어줘",
            "치즈 빼줘",
            "치즈로 바꿔줘",
        )

        for user_text in phrases:
            with self.subTest(user_text=user_text):
                result = invoke_decision(
                    user_text,
                    task_decision(),
                )
                self.assertFalse(result["decision"]["commit"])
                self.assertFalse(result["policy"]["execute"])

    def test_ambiguous_reference_still_blocks_execution(self):
        session = new_session_state()
        session["dialogue_focus"]["current"] = {
            "mentions": ["치즈", "버섯"],
            "history_turn": 1,
        }

        result = invoke_decision(
            "그거 바로 진행해",
            task_decision(),
            session,
        )

        # positive gate 자체는 명시적 실행 의도를 복구한다.
        self.assertTrue(result["decision"]["commit"])

        # 이후 reference resolver가 ambiguity를 발견하면 실행은 금지한다.
        self.assertEqual(
            result["decision"]["understanding"],
            "clarify",
        )
        self.assertEqual(
            result["policy"]["reason"],
            "ambiguous_reference",
        )
        self.assertFalse(result["policy"]["execute"])

    def test_decision_prompt_documents_commit_contract(self):
        required_examples = (
            "바로 진행해",
            "이대로 진행해",
            "바로 시작해",
            "담기 시작해",
            "주문 확정해",
            "치즈 많이 넣고 바로 진행해",
            "바로 진행해도 돼?",
            "진행해볼까?",
            "진행하지 마",
            "시작하지 마",
        )

        for example in required_examples:
            with self.subTest(example=example):
                self.assertIn(example, DECISION_SYSTEM)


if __name__ == "__main__":
    unittest.main()