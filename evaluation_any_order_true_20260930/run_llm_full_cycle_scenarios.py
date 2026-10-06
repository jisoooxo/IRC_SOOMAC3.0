#!/usr/bin/env python3
"""Run the documented ROS full-cycle scenarios through the production LLM graph.

This runner performs no ROS publishing.  It uses the production Decision,
normalization, confirmation, state-application, policy, and Response stages.
"""

import argparse
import copy
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "soomac_irc"
sys.path.insert(0, str(PACKAGE_ROOT))

from soomac_irc import model_runtime
from soomac_irc.decision_model import make_call_decision
from soomac_irc.agent_contract import new_session_state, new_turn_state
from soomac_irc.llm_langgraph import build_graph
from soomac_irc.recommendation_model import make_call_recommendation
from soomac_irc.response_model import make_call_response


DEFAULT_ADAPTER = Path(
    "/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/checkpoints/"
    "gemma4_decision_nf4_qlora_v2_2_5_semantic_rebalance"
)
SCENARIO_ORDER = (
    "A-1", "A-2", "A-3",
    "B-1", "B-2", "B-3",
    "C-1", "C-2", "C-3",
    "D-1", "D-2", "D-3",
)


def base_robot(section="noodle", active_task=None):
    return {
        "section": section,
        "task_queue": [],
        "active_task": copy.deepcopy(active_task),
        "completed_tasks": [],
        "robot_started": active_task is not None,
    }


def action_event(**order_changes):
    return {
        "type": "turn_applied",
        "order_changes": order_changes,
        "restriction_added": [],
        "restriction_removed": [],
        "preference_added": [],
        "preference_removed": [],
    }


def build_case(case_id):
    session = new_session_state()
    robot = base_robot()

    if case_id == "A-1":
        session["order"]["toppings"] = {"소시지": "normal", "게살": "low"}
        session["pending_confirmation"] = {
            "type": "preselected_section",
            "section": "meat",
            "items": {"소시지": "normal", "게살": "low"},
        }
        robot = base_robot("meat")
        utterance = "아니, 그런데 소시지는 많이로 바꿔줘"
    elif case_id == "A-2":
        session["order"]["toppings"] = {"소시지": "normal"}
        robot = base_robot("meat")
        utterance = "소시지 알레르기 있으니까 소시지도 빼줘"
    elif case_id == "A-3":
        session["order"].update({"sauce": "오일", "noodle_type": "얇은면", "noodle_portion": "normal"})
        utterance = "토마토로 바꾸고 지금 면 종류가 뭐야?"
    elif case_id == "B-1":
        session["order"]["toppings"] = {"치즈": "high", "페퍼론치노": "low"}
        session["pending_confirmation"] = {
            "type": "preselected_section",
            "section": "extra",
            "items": {"치즈": "high", "페퍼론치노": "low"},
        }
        robot = base_robot("extra")
        utterance = "아니요 치즈만 그대로 담아줘"
    elif case_id == "B-2":
        session["order"]["toppings"] = {"게살": "normal"}
        session["pending_confirmation"] = {
            "type": "preselected_section",
            "section": "meat",
            "items": {"게살": "normal"},
        }
        robot = base_robot("meat")
        utterance = "응, 그리고 소시지는 많이"
    elif case_id == "B-3":
        proposal = {
            "sauce": None,
            "noodle_type": None,
            "noodle_portion": None,
            "toppings": {"소시지": "normal"},
        }
        record = {
            "action": "request",
            "scope": "current",
            "criteria": "무난한 육류",
            "proposal": proposal,
            "reason_tags": ["무난함"],
        }
        session["recommendation"] = {
            "phase": "proposed",
            "last_proposal": copy.deepcopy(record),
            "history": [copy.deepcopy(record)],
        }
        robot = base_robot("meat")
        utterance = "그 추천으로 바로 진행해"
    elif case_id == "C-1":
        session["order"]["toppings"] = {"치즈": "high", "페퍼론치노": "normal"}
        session["history"] = [
            {"role": "user", "content": "치즈 많이 넣고 페퍼론치노는 보통으로 해줘"},
            {"role": "assistant", "content": "치즈와 페퍼론치노 선택을 반영했습니다."},
        ]
        session["action_history"] = [action_event(**{"toppings.치즈": "high", "toppings.페퍼론치노": "normal"})]
        robot = base_robot("extra")
        utterance = "아까 거 적게로 해"
    elif case_id == "C-2":
        robot = base_robot("extra")
        utterance = "페퍼로니 조금 넣어줘"
    elif case_id == "C-3":
        session["order"]["toppings"] = {"치즈": "high", "페퍼론치노": "low"}
        session["history"] = [
            {"role": "user", "content": "치즈 많이 넣어줘"},
            {"role": "assistant", "content": "치즈를 많이로 반영했습니다."},
            {"role": "user", "content": "페퍼론치노 적게 넣어줘"},
            {"role": "assistant", "content": "페퍼론치노를 적게로 반영했습니다."},
        ]
        session["action_history"] = [
            action_event(**{"toppings.치즈": "high"}),
            action_event(**{"toppings.페퍼론치노": "low"}),
        ]
        robot = base_robot("extra")
        utterance = "그거 취소해"
    elif case_id == "D-1":
        robot = base_robot("veggie")
        utterance = "버섯은 진짜 실어"
    elif case_id == "D-2":
        session["order"].update({"noodle_type": "넓은면", "noodle_portion": "normal"})
        session["history"] = [
            {"role": "user", "content": "넓은면 보통으로 해줘"},
            {"role": "assistant", "content": "넓은면 보통으로 반영했습니다."},
        ]
        utterance = "우리 아까 대화한 것중에 면 종료가 있지 않았어?"
    elif case_id == "D-3":
        session["order"].update({"noodle_type": "얇은면", "noodle_portion": "normal", "sauce": "토마토"})
        robot = base_robot("noodle", {"class": "얇은면", "repeat_count": 2})
        utterance = "지금 로봇 뭐 하고 있어?"
    else:
        raise ValueError(f"unknown scenario: {case_id}")

    return session, robot, utterance


def has_query(decision, query_type, target=None):
    return any(
        query.get("type") == query_type
        and (target is None or query.get("target") == target)
        for query in decision["queries"]
    )


def validate(case_id, before, result):
    decision = result["decision"]
    session = result["session"]
    order = session["order"]
    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    if case_id == "A-1":
        check(decision["confirmation"] == "reject", "confirmation != reject")
        check(decision["order_patch"]["toppings"].get("소시지") == "high", "sausage patch != high")
        check(decision["commit"] is False, "unexpected commit=true")
        check(session["pending_confirmation"] is None, "pending confirmation not cleared")
        check(order["toppings"] == {"소시지": "high"}, "final toppings are not sausage=high only")
    elif case_id == "A-2":
        check({"target": "소시지", "reason": "allergy"} in order["restrictions"], "allergy restriction missing")
        check("소시지" not in order["toppings"], "sausage was not removed")
    elif case_id == "A-3":
        check(decision["order_patch"]["sauce"] == "토마토", "sauce patch != tomato")
        check(has_query(decision, "order_field", "noodle_type"), "noodle_type query missing")
        check(order["sauce"] == "토마토", "final sauce != tomato")
    elif case_id == "B-1":
        check(decision["confirmation"] == "reject", "confirmation != reject")
        check(decision["commit"] is True, "commit != true")
        check(order["toppings"] == {"치즈": "high"}, "final toppings are not cheese=high only")
        check(result["policy"]["execute"] is True, "policy did not allow execution")
    elif case_id == "B-2":
        check(decision["confirmation"] == "accept", "confirmation != accept")
        check(decision["order_patch"]["toppings"].get("소시지") == "high", "sausage patch != high")
        check(order["toppings"] == {"게살": "normal", "소시지": "high"}, "pending and new items were not both kept")
    elif case_id == "B-3":
        check(decision["recommendation"]["action"] == "select", "recommendation action != select")
        check(decision["commit"] is True, "commit != true")
        check(order["toppings"].get("소시지") == "normal", "proposal did not reach final order")
        check(result["policy"]["execute"] is True, "policy did not allow execution")
    elif case_id in ("C-1", "C-2", "C-3"):
        check(decision["understanding"] == "clarify", "understanding != clarify")
        check(order == before["order"], "order mutated before clarification")
    elif case_id == "D-1":
        check({"target": "버섯", "reason": "dislike"} in order["restrictions"], "mushroom dislike restriction missing")
        check("버섯" not in decision["order_patch"]["toppings"], "misheard as mushroom amount")
    elif case_id == "D-2":
        check(has_query(decision, "order_field", "noodle_type"), "noodle_type query missing")
        check(not has_query(decision, "robot_completed"), "misclassified as robot_completed")
    elif case_id == "D-3":
        check(has_query(decision, "robot_status"), "robot_status query missing")
        check(order == before["order"], "order changed on status query")

    return failures


def parse_args():
    parser = argparse.ArgumentParser(description="Run production LLM graph scenarios without ROS publishers.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--scenario", choices=SCENARIO_ORDER, default="A-1")
    group.add_argument("--all", action="store_true", help="Run A-1 through D-3 in order.")
    parser.add_argument("--adapter-path", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--output", type=Path, default=Path("evaluation_any_order_true_20260930/llm_full_cycle_results.jsonl"))
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.adapter_path.is_dir():
        raise SystemExit(f"adapter not found: {args.adapter_path}")

    model_runtime.MODEL_QUANTIZATION = "nf4"
    print(f"LOAD> base={model_runtime.MODEL_PATH}")
    print(f"LOAD> adapter={args.adapter_path}")
    model, processor = model_runtime.load_model(str(args.adapter_path))
    call_decision = make_call_decision(model, processor)
    call_recommendation = make_call_recommendation(model, processor)
    call_response = make_call_response(model, processor)
    graph = build_graph(call_decision, call_recommendation, call_response)

    selected = SCENARIO_ORDER if args.all else (args.scenario,)
    records = []
    for index, case_id in enumerate(selected, 1):
        session, robot, utterance = build_case(case_id)
        before = copy.deepcopy(session)
        trace_start = len(call_decision.trace_events)
        print(f"\n[{index}/{len(selected)}] {case_id} USER> {utterance}")
        try:
            result = graph.invoke(new_turn_state(session, utterance, robot))
            traces = call_decision.trace_events[trace_start:]
            failures = validate(case_id, before, result)
            status = "PASS" if not failures else "FAIL"
            raw_outputs = [event.get("raw") for event in traces if event.get("raw") is not None]
            record = {
                "scenario": case_id,
                "status": status,
                "utterance": utterance,
                "decision_raw": raw_outputs,
                "decision_final": result["decision"],
                "policy": result["policy"],
                "reply": result["reply"],
                "final_session": result["session"],
                "failures": failures,
            }
        except Exception as error:
            status = "ERROR"
            record = {
                "scenario": case_id,
                "status": status,
                "utterance": utterance,
                "error": {"type": type(error).__name__, "message": str(error)},
            }

        records.append(record)
        print(f"{status}> {case_id}")
        if record.get("decision_raw"):
            print(f"RAW> {record['decision_raw'][-1]}")
        if record.get("failures"):
            for failure in record["failures"]:
                print(f"  - {failure}")
        if record.get("error"):
            print(f"  - {record['error']['type']}: {record['error']['message']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    passed = sum(record["status"] == "PASS" for record in records)
    print(f"\nSUMMARY> {passed}/{len(records)} passed")
    print(f"RESULTS> {args.output.resolve()}")
    raise SystemExit(0 if passed == len(records) else 1)


if __name__ == "__main__":
    main()
