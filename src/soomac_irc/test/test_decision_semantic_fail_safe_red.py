"""Decision semantic별 좁은 fail-safe 처리를 구현하기 전 RED/control 테스트."""

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


def complete_noodle_session() -> dict:
    session = new_session_state()
    session["order"].update({
        "sauce": "토마토",
        "noodle_type": "얇은면",
        "noodle_portion": "normal",
    })
    return session


def topping_decision(commit: bool) -> dict:
    decision = new_decision()
    decision["route"] = "task"
    decision["mentions"] = ["치즈"]
    decision["order_patch"]["toppings"] = {
        "치즈": "normal",
    }
    decision["commit"] = commit
    return decision


def restriction_remove_decision(mention: str) -> dict:
    decision = new_decision()
    decision["route"] = "task"
    decision["mentions"] = [mention]
    decision["restriction_options"] = [{
        "target": "게살",
        "type": "allergy",
        "action": "remove",
    }]
    return decision


def invoke_decision(user_text: str, decision: dict, session: dict) -> dict:
    graph = build_graph(
        lambda *args: copy.deepcopy(decision),
        lambda *args: {},
        lambda *args: "정상 일반 응답",
    )
    return graph.invoke(
        new_turn_state(
            copy.deepcopy(session),
            user_text,
            ROBOT_STATE,
        )
    )


class TestHallucinatedCommitRed(unittest.TestCase):
    def test_hallucinated_commit_does_not_execute_and_keeps_order_change(self):
        result = invoke_decision(
            "치즈 넣어줘",
            topping_decision(commit=True),
            complete_noodle_session(),
        )

        # valid order semantic은 보존하되 모델이 환각한 실행 권한만 내려야 한다.
        self.assertEqual(
            result["session"]["order"]["toppings"],
            {"치즈": "normal"},
        )
        self.assertFalse(result["policy"]["execute"])

    def test_explicit_commit_positive_controls_remain_executable(self):
        reference_session = complete_noodle_session()
        reference_session["dialogue_focus"]["current"] = {
            "mentions": ["치즈"],
            "history_turn": 1,
        }

        cases = (
            (
                "그거 실행해줘",
                topping_decision(commit=True),
                reference_session,
            ),
            (
                "이대로 진행해",
                new_decision(),
                complete_noodle_session(),
            ),
            (
                "치즈 넣고 바로 진행해",
                topping_decision(commit=False),
                complete_noodle_session(),
            ),
        )

        for user_text, decision, session in cases:
            with self.subTest(user_text=user_text):
                result = invoke_decision(user_text, decision, session)
                self.assertTrue(result["decision"]["commit"])
                self.assertTrue(result["policy"]["execute"])

        combined = invoke_decision(
            "치즈 넣고 바로 진행해",
            topping_decision(commit=False),
            complete_noodle_session(),
        )
        self.assertEqual(
            combined["session"]["order"]["toppings"],
            {"치즈": "normal"},
        )

    def test_commit_question_and_negative_controls_do_not_execute(self):
        for user_text in ("실행해볼까?", "진행하지 마"):
            with self.subTest(user_text=user_text):
                decision = new_decision()
                decision["route"] = "task"
                decision["commit"] = True
                result = invoke_decision(
                    user_text,
                    decision,
                    complete_noodle_session(),
                )
                self.assertFalse(result["policy"]["execute"])


class TestRestrictionRemoveTargetGroundingRed(unittest.TestCase):
    def setUp(self):
        self.session = new_session_state()
        self.session["order"]["restrictions"] = [{
            "target": "게살",
            "reason": "allergy",
        }]

    def test_wrong_restriction_remove_target_is_preserved(self):
        result = invoke_decision(
            "치즈 알레르기 제한 해제해줘",
            restriction_remove_decision("치즈"),
            self.session,
        )

        self.assertEqual(
            result["session"]["order"]["restrictions"],
            self.session["order"]["restrictions"],
        )
        self.assertFalse(result["policy"]["execute"])

    def test_matching_restriction_remove_target_still_works(self):
        result = invoke_decision(
            "게살 알레르기 제한 해제해줘",
            restriction_remove_decision("게살"),
            self.session,
        )

        self.assertEqual(result["session"]["order"]["restrictions"], [])


class TestHallucinatedMentionFocusRed(unittest.TestCase):
    def general_decision(self, mention: str) -> dict:
        decision = new_decision()
        decision["route"] = "general"
        decision["mentions"] = [mention]
        return decision

    def test_hallucinated_mention_is_not_saved_to_focus(self):
        session = new_session_state()
        result = invoke_decision(
            "양자역학 설명해줘",
            self.general_decision("치즈"),
            session,
        )

        self.assertEqual(result["session"]["dialogue_focus"], session["dialogue_focus"])
        self.assertEqual(result["reply"], "정상 일반 응답")
        self.assertEqual(len(result["session"]["history"]), 2)

    def test_surface_grounded_mentions_are_saved_even_when_unsupported(self):
        cases = (
            ("치즈가 뭐야?", "치즈"),
            ("떡볶이가 뭐야?", "떡볶이"),
        )

        for user_text, mention in cases:
            with self.subTest(user_text=user_text):
                result = invoke_decision(
                    user_text,
                    self.general_decision(mention),
                    new_session_state(),
                )
                self.assertEqual(
                    result["session"]["dialogue_focus"]["current"]["mentions"],
                    [mention],
                )


if __name__ == "__main__":
    unittest.main()
