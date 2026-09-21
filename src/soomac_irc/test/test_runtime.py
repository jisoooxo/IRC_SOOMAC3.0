"""버전 없는 운영 모듈의 CPU 회귀 검사. 모델·ROS 노드·로봇은 실행하지 않는다."""

import copy
import unittest

from soomac_irc.agent import build_graph, classify_confirmation_intent
from soomac_irc.order import build_section_plan, missing_requirements, new_order
from soomac_irc.order_change import apply_menu_changes, validate_menu_changes
from soomac_irc.recommendation import validate_recommended_order
from soomac_irc.reply import build_turn_reply
from soomac_irc.validation import validate_agent_result
from soomac_irc.vlm import SAUCE_NAMES, build_vlm_request, build_vlm_spoken_reply, decide_vlm_outcome, parse_vlm_verdict


def new_state(section="noodle"):
    return {
        "order": new_order(),
        "execution": {"section": section, "selected": [], "task_queue": [], "active_task": None, "completed_tasks": [], "skipped_sections": {}, "robot_started": False},
        "recommendation": {"phase": "idle", "confirmed_sections": []},
        "history": [], "action_history": [],
    }


def run_turn(state, text, call):
    state = copy.deepcopy(state)
    state["user_text"] = text
    result = build_graph(lambda current, user_text: copy.deepcopy(call)).invoke(state)
    validate_agent_result(result)
    return result


class TestOrderRuntime(unittest.TestCase):
    def test_order_objects_are_independent(self):
        first, second = new_order(), new_order()
        first["toppings"]["양파"] = "low"
        self.assertEqual(second["toppings"], {})

    def test_portion_is_not_a_menu_item(self):
        clean, dropped = validate_menu_changes({"noodle_portion": "high", "많이": "high", "toppings": {"양파": "low"}})
        self.assertEqual(clean, {"noodle_portion": "high", "toppings": {"양파": "low"}})
        self.assertEqual(dropped[0]["path"], "많이")
        order = new_order()
        apply_menu_changes(order, clean)
        self.assertEqual(order["noodle_portion"], "high")

    def test_required_fields_and_robot_plan(self):
        order = new_order()
        self.assertEqual(missing_requirements(order, ["noodle"]), {"noodle": ["sauce", "noodle_type", "noodle_portion"]})
        order.update({"sauce": "토마토", "noodle_type": "넓은면", "noodle_portion": "high"})
        self.assertEqual(build_section_plan(order, "noodle"), [{"class": "넓은면", "repeat_count": 3}])
        self.assertEqual(build_section_plan(order, "sauce"), [{"class": "토마토", "repeat_count": 1}])

    def test_set_order_preserves_input_and_waits_for_confirmation(self):
        state = new_state()
        before = copy.deepcopy(state)
        changes = {"sauce": "토마토", "noodle_type": "넓은면", "noodle_portion": "high"}
        result = run_turn(state, "토마토 소스, 넓은면, 면 양 많이로 해줘", {"name": "set_order", "changes": changes})
        self.assertEqual(state, before)
        self.assertEqual(set(result["turn_result"]["accepted"]), set(changes))
        self.assertFalse(result["turn_result"]["confirm_validation"]["allowed"])
        confirmed = run_turn(result, "진행해", {"name": "confirm_section", "changes": {}})
        self.assertTrue(confirmed["turn_result"]["confirm_validation"]["allowed"])

    def test_all_eight_tool_paths_and_invalid_output(self):
        calls = [
            {"name": "set_order", "changes": {"sauce": "토마토"}},
            {"name": "set_order_and_confirm", "changes": {"sauce": "토마토"}},
            {"name": "refuse_section", "changes": {"reason": "dislike", "items": ["양파", "버섯"]}},
            {"name": "recommend_order", "changes": {"scope": "ask"}},
            *({"name": name, "changes": {}} for name in ("confirm_section", "cancel_order", "respond", "describe_scene")),
        ]
        for call in calls:
            with self.subTest(tool=call["name"]):
                state = new_state("veggie" if call["name"] == "refuse_section" else "noodle")
                result = run_turn(state, "진행해", call)
                self.assertEqual(result["turn_result"]["action"], call["name"])
        result = run_turn(new_state(), "테스트", None)
        self.assertEqual(result["parser_status"], "invalid")

    def test_robot_start_requires_explicit_intent(self):
        changes = {"sauce": "토마토", "noodle_type": "넓은면", "noodle_portion": "normal"}
        for text in ("토마토 소스 줘", "진행하지마", "진행할까"):
            with self.subTest(text=text):
                result = run_turn(new_state(), text, {"name": "set_order_and_confirm", "changes": changes})
                self.assertFalse(result["turn_result"]["confirm_validation"]["allowed"])
        self.assertIs(classify_confirmation_intent("진행하지마", "start"), False)
        self.assertIsNone(classify_confirmation_intent("진행할까", "start"))

    def test_recommendation_partial_removal_then_acceptance(self):
        original = new_state("veggie")
        proposed = run_turn(original, "야채 추천해줘", {"name": "recommend_order", "changes": {"scope": "current", "recommended_order": {"toppings": {"양파": "normal", "버섯": "normal"}}}})
        revised = run_turn(proposed, "양파만 빼줘", {"name": "set_order", "changes": {"toppings": {"양파": "none"}}})
        self.assertEqual(revised["order"], original["order"])
        self.assertEqual(revised["recommendation"]["recommended_changes"], {"toppings": {"버섯": "normal"}})
        confirmed = run_turn(revised, "진행해", {"name": "confirm_section", "changes": {}})
        self.assertEqual(confirmed["order"]["toppings"], {"버섯": "normal"})
        self.assertEqual(original["order"]["toppings"], {})

    def test_recommendation_blocks_allergy_and_exclusion(self):
        order = new_order()
        order["restrictions"] = [{"target": "유제품", "reason": "allergy"}]
        clean, dropped, blocked = validate_recommended_order(order, {"excluded": ["버섯"], "recommended_order": {"sauce": "크림", "noodle_type": "얇은면", "noodle_portion": "high", "toppings": {"치즈": "normal", "버섯": "low"}}})
        self.assertEqual(clean, {"noodle_type": "얇은면", "noodle_portion": "high"})
        self.assertEqual({item["path"] for item in blocked}, {"sauce", "toppings.치즈"})
        self.assertTrue(any(item["path"] == "toppings.버섯" and item["reason"] == "excluded" for item in dropped))

    def test_new_allergy_keeps_active_and_completed_items(self):
        for physical_state in ("unfilled", "active", "completed"):
            with self.subTest(physical_state=physical_state):
                state = new_state("veggie")
                state["order"]["toppings"] = {"양파": "normal"}
                state["execution"]["selected"] = ["toppings.양파"]
                if physical_state == "active":
                    state["execution"]["active_task"] = {"class": "양파", "repeat_count": 2}
                if physical_state == "completed":
                    state["execution"]["completed_tasks"] = [{"class": "양파", "repeat_count": 2}]
                before = copy.deepcopy(state)
                result = run_turn(state, "양파 알레르기 있어", {"name": "set_order", "changes": {"restriction_changes": [{"target": "양파", "reason": "allergy", "enabled": True}]}})
                self.assertEqual("양파" in result["order"]["toppings"], physical_state != "unfilled")
                self.assertEqual(state, before)

    def test_free_reply_deduplicates_and_drops_untrue_order_claim(self):
        state = new_state()
        result = run_turn(state, "파스타가 뭐야", {"name": "respond", "changes": {}})
        reply = build_turn_reply("파스타가 뭐야", state, result, lambda current, text: "파스타는 이탈리아의 면 요리에요. 파스타는 이탈리아의 면 요리에요. 주문에 반영했어요.")
        self.assertEqual(reply.count("파스타는"), 1)
        self.assertNotIn("반영했", reply)

    def test_boundary_validation_rejects_invalid_state(self):
        result = run_turn(new_state(), "안녕", {"name": "respond", "changes": {}})
        result["execution"]["selected"] = None
        with self.assertRaises(ValueError):
            validate_agent_result(result)


class TestVlmRuntime(unittest.TestCase):
    def test_three_sauces_and_lid_image_order(self):
        for sauce in SAUCE_NAMES:
            request = build_vlm_request(sauce, ["after"], reference_image="reference", comparison_image="before")
            self.assertEqual(request["images"], ["reference", "before", "after"])
            self.assertIn(f"{sauce} 소스", request["user_text"])
        self.assertEqual(build_vlm_request("뚜껑", ["after"], comparison_image="before")["images"], ["before", "after"])

    def test_missing_reference_and_first_ingredient(self):
        self.assertIsNone(build_vlm_request("토마토", ["after"]))
        self.assertIsNone(build_vlm_request("토마토", ["after"], reference_image="reference"))
        self.assertEqual(build_vlm_request("양파", ["after"], reference_image="reference")["images"], ["reference", "after"])

    def test_verdict_requires_final_line(self):
        self.assertEqual(parse_vlm_verdict("확인함\n판정: PASS"), "pass")
        self.assertEqual(parse_vlm_verdict("판정: FAIL\n추가 문장"), "uncertain")

    def test_retry_contract_and_spoken_reason(self):
        retry = decide_vlm_outcome("토마토", "fail", 0, False)
        self.assertEqual(retry["publish_result"], "fail")
        self.assertFalse(retry["allow_spoken_reason"])
        reply = build_vlm_spoken_reply("보이지 않음\n판정: FAIL", retry["policy_reply"], True, retry["allow_spoken_reason"])
        self.assertEqual(reply, retry["policy_reply"])
        fallback = decide_vlm_outcome("토마토", "fail", 1, False)
        self.assertEqual(fallback["publish_result"], "success")
        self.assertFalse(fallback["trusted_pass"])


if __name__ == "__main__":
    unittest.main()
