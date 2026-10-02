"""Decision JSON duplicate key를 graph 내부 safe clarify로 격리하는 테스트."""

import copy
import json
import unittest

from soomac_irc.agent_contract import new_decision
from soomac_irc.decision_model import parse_decision
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

DUPLICATE_DECISION_RAW = """
{
  "route": "general",
  "queries": [
    {
      "type": "order_item",
      "target": "치즈",
      "target": "버섯"
    }
  ]
}
"""

CANONICAL_FIELDS = (
    "order",
    "preferences",
    "recommendation",
    "pending_confirmation",
    "dialogue_focus",
    "action_history",
)


def invoke_graph(call_decision, user_text="치즈랑 버섯 중 뭐가 좋아?"):
    initial_session = new_session_state()
    graph = build_graph(
        call_decision,
        lambda *args: {},
        lambda *args: "응답",
    )
    result = graph.invoke(
        new_turn_state(
            copy.deepcopy(initial_session),
            user_text,
            ROBOT_STATE,
        )
    )
    return initial_session, result


class TestDuplicateDecisionKeyFallback(unittest.TestCase):
    def test_duplicate_decision_finishes_as_safe_clarify(self):
        def duplicate_decision(session, user_text, robot_state, repair=None):
            return parse_decision(DUPLICATE_DECISION_RAW, session)

        initial_session, result = invoke_graph(duplicate_decision)

        # A: duplicate key가 graph.invoke 밖으로 전파되지 않는다.
        self.assertIsInstance(result, dict)

        # B: 대화 history 기록 외의 canonical 상태는 전혀 바뀌지 않는다.
        for field in CANONICAL_FIELDS:
            with self.subTest(field=field):
                self.assertEqual(result["session"][field], initial_session[field])

        # invalid JSON의 route나 query를 복구하지 않고 새 safe Decision을 사용한다.
        expected_decision = new_decision()
        expected_decision["understanding"] = "clarify"
        self.assertEqual(result["decision"], expected_decision)

        # C: understanding clarify가 실행보다 먼저 차단한다.
        self.assertFalse(result["policy"]["execute"])
        self.assertEqual(result["policy"]["reason"], "understanding")

        # D: 정상 종료된 턴은 clarification reply와 대화 history를 남긴다.
        self.assertTrue(result["reply"])
        self.assertEqual(len(result["session"]["history"]), 2)

    def test_duplicate_repair_decision_also_falls_back(self):
        invalid_first = new_decision()
        invalid_first["route"] = "general"
        invalid_first["order_patch"]["toppings"] = {"치즈": "normal"}
        call_count = 0

        def duplicate_on_repair(session, user_text, robot_state, repair=None):
            nonlocal call_count
            call_count += 1

            if repair is None:
                return copy.deepcopy(invalid_first)

            return parse_decision(DUPLICATE_DECISION_RAW, session)

        initial_session, result = invoke_graph(
            duplicate_on_repair,
            "치즈 넣어줘",
        )

        self.assertEqual(call_count, 2)
        self.assertEqual(result["decision"]["understanding"], "clarify")
        self.assertFalse(result["policy"]["execute"])

        for field in CANONICAL_FIELDS:
            with self.subTest(field=field):
                self.assertEqual(result["session"][field], initial_session[field])

    def test_normal_decision_parsing_is_unchanged(self):
        # E: 서로 다른 query object의 같은 key는 정상이며 기존 normalize 결과를 유지한다.
        raw = json.dumps(
            {
                "route": "general",
                "mentions": ["치즈", "버섯"],
                "queries": [
                    {"type": "order_item", "target": "치즈"},
                    {"type": "order_item", "target": "버섯"},
                ],
            },
            ensure_ascii=False,
        )
        expected = new_decision()
        expected["route"] = "general"
        expected["mentions"] = ["치즈", "버섯"]
        expected["queries"] = [
            {"type": "order_item", "target": "치즈"},
            {"type": "order_item", "target": "버섯"},
        ]

        self.assertEqual(parse_decision(raw), expected)

    def test_unrelated_value_error_is_not_hidden(self):
        def broken_decision(*args):
            raise ValueError("Decision 입력 token 초과")

        with self.assertRaisesRegex(ValueError, "token 초과"):
            invoke_graph(broken_decision)


if __name__ == "__main__":
    unittest.main()
