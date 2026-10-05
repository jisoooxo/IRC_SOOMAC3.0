"""2026-10-05 thin python 리팩토링 회귀 테스트. session_20261005_221135 실패를 모의 Decision으로 재현한다."""

import copy
import unittest

from soomac_irc.agent_contract import new_decision
from soomac_irc.decision_overrides import pre_decision_override
from soomac_irc.dialogue_focus import remove_focus_mentions
from soomac_irc.llm_langgraph import build_graph, new_session_state, new_turn_state
from soomac_irc.llm_policy import section_execution_items


def robot(section):
    return {"section": section, "active_task": None, "task_queue": [], "completed_tasks": []}


def invoke(text, decision, session=None, section="extra", response=None):
    def decide(session, text, robot_state, repair):
        return pre_decision_override(text, session, robot_state) or copy.deepcopy(decision)

    graph = build_graph(decide, lambda *args: {}, response or (lambda *args: "응답"))
    return graph.invoke(new_turn_state(
        copy.deepcopy(session or new_session_state()), text, robot(section),
    ))


def task_decision(toppings=None, mentions=(), commit=False, confirmation="none", restrictions=()):
    decision = new_decision()
    decision["route"] = "task"
    decision["mentions"] = list(mentions)
    decision["order_patch"]["toppings"] = dict(toppings or {})
    decision["restriction_options"] = [dict(option) for option in restrictions]
    decision["commit"] = commit
    decision["confirmation"] = confirmation
    return decision


def peperoncino_candidate_session():
    # 221135 #13: "나는 치즈가 싫어. 패포론 지눈만 줘." → 페퍼론치노 후보 보류
    return invoke(
        "나는 치즈가 싫어. 패포론 지눈만 줘.",
        task_decision({"페퍼론치노": "normal"}, mentions=["치즈", "패포론"],
                      restrictions=[{"target": "치즈", "type": "dislike", "action": "add"}]),
    )["session"]


class TestDirectPluralTargets(unittest.TestCase):
    def test_direct_targets_beat_empty_focus(self):
        # 221135 #10: Decision 정답을 "둘다" resolver가 ambiguous로 덮었다.
        session = new_session_state()
        session["dialogue_focus"]["current"] = {"mentions": [], "history_turn": 1}
        result = invoke(
            "소시지한 계살 둘 다 많이줘",
            task_decision({"소시지": "high", "게살": "high"}, mentions=["소시지", "게살"]),
            session, section="meat",
        )

        self.assertEqual(result["session"]["order"]["toppings"], {"소시지": "high", "게살": "high"})
        self.assertNotEqual(result["policy"]["status"], "clarify")


class TestUnresolvedCandidate(unittest.TestCase):
    def test_candidate_is_pending_first(self):
        session = peperoncino_candidate_session()

        self.assertEqual(session["pending_question"]["type"], "menu_confirmation")
        self.assertEqual(session["order"]["toppings"], {})

    def test_commit_without_accept_does_not_execute_or_drop_candidate(self):
        # 221135 #14: "그럼 그럼 바로 시작해." → 후보를 버리고 빈 extra를 실행해 skip했다.
        session = peperoncino_candidate_session()
        result = invoke("그럼 그럼 바로 시작해.", task_decision(commit=True), session)

        self.assertFalse(result["policy"]["execute"])
        self.assertEqual(result["session"]["pending_question"]["type"], "menu_confirmation")
        self.assertEqual(result["session"]["pending_question"]["proposal"]["toppings"], {"페퍼론치노": "normal"})

    def test_accept_then_start_executes_extra(self):
        session = peperoncino_candidate_session()
        accepted = invoke("응", new_decision(), session)

        self.assertEqual(accepted["session"]["order"]["toppings"], {"페퍼론치노": "normal"})
        result = invoke("바로 시작해", task_decision(commit=True), accepted["session"])
        self.assertTrue(result["policy"]["execute"])

    def test_accept_with_commit_applies_and_executes(self):
        session = peperoncino_candidate_session()
        result = invoke("그래 그걸로 바로 시작해", task_decision(commit=True, confirmation="accept"), session)

        self.assertEqual(result["session"]["order"]["toppings"], {"페퍼론치노": "normal"})
        self.assertTrue(result["policy"]["execute"])


class TestPendingQuestionLifecycle(unittest.TestCase):
    def test_status_query_keeps_section_choice(self):
        # 221135 #9: recommendation_offer가 기존 소시지/게살 선택 질문을 None으로 덮었다.
        session = new_session_state()
        session["pending_question"] = {"type": "choice", "targets": ["소시지", "게살"],
                                       "section": "meat", "history_turn": 1}
        decision = new_decision()
        decision["route"] = "task"
        decision["queries"] = [{"type": "robot_status"}]
        result = invoke("지금까지 뭐 했어?", decision, session, section="meat")

        self.assertEqual(result["session"]["pending_question"]["targets"], ["소시지", "게살"])


class TestFocusCleanup(unittest.TestCase):
    def test_removed_target_leaves_no_empty_event(self):
        focus = {
            "current": {"mentions": ["양파"], "history_turn": 7},
            "recent": [{"mentions": ["양파"], "history_turn": 6}, {"mentions": ["오일"], "history_turn": 4}],
        }
        updated = remove_focus_mentions(focus, ["양파"])

        self.assertIsNone(updated["current"])
        self.assertEqual(updated["recent"], [{"mentions": ["오일"], "history_turn": 4}])


class TestExecutionOffer(unittest.TestCase):
    def test_applied_item_offers_execution_and_yes_executes(self):
        calls = []
        added = invoke("양파 보통으로 추가해줘", task_decision({"양파": "normal"}, mentions=["양파"]),
                       section="veggie", response=lambda *args: calls.append(args[7]) or "응답")

        self.assertEqual(calls[0]["type"], "execution_offer")
        self.assertEqual(added["session"]["pending_question"]["type"], "execution_offer")

        result = invoke("응", new_decision(), added["session"], section="veggie")
        self.assertTrue(result["policy"]["execute"])

    def test_no_offer_while_question_unresolved(self):
        calls = []
        invoke("패포론 지눈만 줘", task_decision({"페퍼론치노": "normal"}, mentions=["패포론"]),
               response=lambda *args: calls.append(args[7]) or "응답")

        self.assertNotEqual((calls[0] or {}).get("type"), "execution_offer")


class TestLidOwnership(unittest.TestCase):
    def test_lid_section_owns_single_cover_task(self):
        order = new_session_state()["order"]
        order["sauce"] = "토마토"

        self.assertEqual(section_execution_items(order, "lid"), [{"item": "뚜껑", "amount": None}])
        self.assertEqual(section_execution_items(order, "sauce"), [{"item": "토마토", "amount": None}])


class TestResponseContract(unittest.TestCase):
    def test_candidate_given_to_response_has_no_empty_fields(self):
        # 221135 #13: null인 sauce/noodle까지 전달돼 Response가 "소스, 면 종류, 면 양은 기본 설정대로…"를 물었다.
        from soomac_irc.llm_policy import compact_order_patch

        session = peperoncino_candidate_session()
        self.assertEqual(compact_order_patch(session["pending_question"]["proposal"]),
                         {"toppings": {"페퍼론치노": "normal"}})


if __name__ == "__main__":
    unittest.main()
