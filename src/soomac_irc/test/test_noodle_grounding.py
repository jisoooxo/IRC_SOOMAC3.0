"""STT 보정과 면 종류 근거 검사를 실제 graph에 연결한 회귀 테스트."""

import copy
import unittest

from soomac_irc.agent_contract import new_decision
from soomac_irc.llm_langgraph import build_graph, new_session_state, new_turn_state
from soomac_irc.llm_policy import grade_menu_grounding
from soomac_irc.stt_hotword import normalize_stt_text


ROBOT_STATE = {
    "section": "noodle", "active_task": None,
    "task_queue": [], "completed_tasks": [],
}


def noodle_decision(noodle_type):
    decision = new_decision()
    decision["order_patch"]["noodle_type"] = noodle_type
    return decision


def invoke(text, decision, session=None):
    graph = build_graph(
        lambda *args: copy.deepcopy(decision),
        lambda *args: {},
        lambda *args: "응답",
    )
    return graph.invoke(new_turn_state(
        copy.deepcopy(session if session is not None else new_session_state()),
        text, copy.deepcopy(ROBOT_STATE),
    ))


class TestNoodleGrounding(unittest.TestCase):
    def test_raw_uncertain_noodle_is_not_guessed(self):
        for text in ("널적면으로 주세요", "보통면으로 주세요", "보통 양으로 주세요"):
            for noodle in ("얇은면", "넓은면"):
                with self.subTest(text=text, noodle=noodle):
                    decision = noodle_decision(noodle)
                    decision["mentions"] = [noodle]  # 모델의 허위 mentions도 근거가 아니다.
                    result = invoke(text, decision)
                    self.assertIsNone(result["session"]["order"]["noodle_type"])
                    self.assertEqual(result["policy"]["status"], "clarify")
                    self.assertFalse(result["policy"]["execute"])

    def test_normalized_alias_accepts_only_wide_noodle(self):
        text = normalize_stt_text("널적 면으로 주세요")
        for noodle, expected in (("넓은면", "넓은면"), ("얇은면", None)):
            with self.subTest(noodle=noodle):
                result = invoke(text, noodle_decision(noodle))
                self.assertEqual(result["session"]["order"]["noodle_type"], expected)
                self.assertFalse(result["policy"]["execute"])

    def test_canonical_names_with_spaces_remain_valid(self):
        for text, noodle in (("얇은 면으로 주세요", "얇은면"), ("넓은면으로 주세요", "넓은면")):
            with self.subTest(text=text):
                result = invoke(text, noodle_decision(noodle))
                self.assertEqual(result["session"]["order"]["noodle_type"], noodle)

    def test_wrong_noodle_does_not_replace_existing_order_or_execute(self):
        session = new_session_state()
        session["order"].update(sauce="토마토", noodle_type="넓은면", noodle_portion="normal")
        result = invoke("보통면으로 바꾸고 바로 진행해", noodle_decision("얇은면"), session)
        self.assertEqual(result["session"]["order"], session["order"])
        self.assertFalse(result["policy"]["execute"])

    def test_resolved_reference_still_selects_noodle(self):
        session = new_session_state()
        session["dialogue_focus"]["current"] = {"mentions": ["넓은면"], "history_turn": 1}
        result = invoke("그거 줘", noodle_decision("넓은면"), session)
        self.assertEqual(result["session"]["order"]["noodle_type"], "넓은면")

    def test_amount_only_does_not_require_noodle_name(self):
        decision = new_decision()
        decision["order_patch"]["noodle_portion"] = "normal"
        result = invoke("보통 양으로 주세요", decision)
        self.assertIsNone(result["session"]["order"]["noodle_type"])
        self.assertEqual(result["session"]["order"]["noodle_portion"], "normal")

    def test_normalized_full_order_still_executes(self):
        decision = noodle_decision("넓은면")
        decision["order_patch"].update(sauce="토마토", noodle_portion="normal")
        text = normalize_stt_text("토마토 소스에 널적면 보통 양으로 바로 진행해")
        result = invoke(text, decision)
        self.assertEqual(result["session"]["order"]["noodle_type"], "넓은면")
        self.assertTrue(result["policy"]["execute"])

    def test_recommendation_selection_keeps_dedicated_path(self):
        session = new_session_state()
        session["recommendation"].update(phase="proposed", last_proposal={
            "scope": "current", "reason_tags": [],
            "proposal": {"sauce": "토마토", "noodle_type": "넓은면",
                         "noodle_portion": "normal", "toppings": {}},
        })
        decision = new_decision()
        decision["recommendation"]["action"] = "select"
        result = invoke("추천한 대로 할게", decision, session)
        self.assertEqual(result["session"]["order"]["noodle_type"], "넓은면")

    def test_confirmation_keeps_dedicated_path(self):
        session = new_session_state()
        held_patch = new_decision()["order_patch"]
        held_patch.update(sauce="토마토", noodle_type="넓은면", noodle_portion="normal")
        session["pending_confirmation"] = {
            "type": "restriction_conflict", "conflicts": [],
            "held_patch": held_patch, "held_commit": False,
        }
        decision = new_decision()
        decision["confirmation"] = "accept"
        result = invoke("응", decision, session)
        self.assertEqual(result["session"]["order"]["noodle_type"], "넓은면")
        self.assertIsNone(result["session"]["pending_confirmation"])

    def test_guard_reports_specific_error_without_mutating_decision(self):
        decision = noodle_decision("얇은면")
        before = copy.deepcopy(decision)
        grading = grade_menu_grounding("널적면", decision)
        self.assertEqual([r["target"] for r in grading["uncertain"] + grading["ungrounded"]], ["얇은면"])
        self.assertEqual(decision, before)


if __name__ == "__main__":
    unittest.main()
