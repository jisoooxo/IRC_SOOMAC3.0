"""Bounded existing-adapter prompt A/B; no ROS, training, or runtime path changes.

Exact logged inputs and explicitly labelled controls are independent probes, not a
closed-loop scenario. Response meanings require review of the recorded raw text.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[3]
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(session, turn):
    path = ROOT / "logs/llm/2026-10-31" / ("session_" + session) / "agent_turns.jsonl"
    row = next(row for row in map(json.loads, path.read_text().splitlines()) if row["turn_id"] == turn)
    return row, str(path)


def prepare_cases():
    cream, cream_path = source("20261031_223257_597216_pid259195", 2)
    sausage, sausage_path = source("20261031_223257_597216_pid259195", 8)
    mushroom, mushroom_path = source("20261031_213138_630159_pid218963", 4)
    success, success_path = source("20261031_225558_592562_pid284444", 3)
    cases = []
    for name, row, path, execute, changes in (
        ("D_cream_accept_execute", cream, cream_path, True, {"sauce": "크림"}),
        ("D_sausage_select_execute", sausage, sausage_path, True, {"소시지": "high"}),
        ("D_mushroom_change_execute", mushroom, mushroom_path, True, {"버섯": "normal"}),
        ("D_pending_execute_control", success, success_path, True, {}),
    ):
        expected = copy.deepcopy(row["session_before"]["order"])
        for key, value in changes.items():
            if key == "sauce":
                expected[key] = value
            else:
                expected["toppings"][key] = value
        cases.append({"id": name, "stage": "decision", "source": path, "turn": row["turn_id"],
                      "authored_control": False, "input": row["llm_trace"]["decision"][0]["model_input"],
                      "expected_order": expected, "expected_execute": execute})
    for name, message in (
        ("D_negated_execute_control", "소시지만 많이 골라줘. 바로 담지는 마."),
        ("D_quoted_execute_control", "'바로 담아줘'라고 말하면 실행되는지 물어보는 거야. 지금 담지는 마."),
        ("D_existing_low_control", "소시지만 골라줘."),
    ):
        data = copy.deepcopy(sausage["llm_trace"]["decision"][0]["model_input"])
        data["message"] = message
        expected = copy.deepcopy(data["order"])
        if name == "D_negated_execute_control":
            expected["toppings"]["소시지"] = "high"
        if name == "D_existing_low_control":
            data["order"]["toppings"]["소시지"] = "low"
            expected["toppings"]["소시지"] = "low"
        cases.append({"id": name, "stage": "decision", "authored_control": True, "input": data,
                      "expected_order": expected, "expected_execute": False})
    for name, row in (("P_new_high", sausage), ("P_changed_normal", mushroom)):
        cases.append({"id": name, "stage": "response", "authored_control": False,
                      "input": copy.deepcopy(row["llm_trace"]["response"][0]["model_input"])})
    kept = copy.deepcopy(sausage["llm_trace"]["response"][0]["model_input"])
    kept["user_text"] = "소시지는 많은 양으로 선택돼 있는 거 맞지?"
    kept["applied_this_turn"]["order_changes"] = {}
    kept["recent_history"] = (kept["recent_history"] + [
        {"role": "user", "content": sausage["user_text"]},
        {"role": "assistant", "content": sausage["published_reply"]},
    ])[-20:]
    cases.append({"id": "P_existing_high_control", "stage": "response", "authored_control": True, "input": kept})
    systems = {}
    for stage, row in (("decision", cream), ("response", sausage)):
        systems[stage] = next(m["content"] for m in row["llm_trace"][stage][0]["messages"] if m["role"] == "system")
    assert len(cases) == 10
    return cases, systems


def arguments(case):
    from soomac_irc.agent_contract import new_session_state
    data = case["input"]
    session = new_session_state()
    session.update(order=copy.deepcopy(data.get("order", data.get("confirmed_order"))),
                   preferences=copy.deepcopy(data["preferences"]), pending=copy.deepcopy(data["pending"]),
                   history=copy.deepcopy(data["recent_history"]),
                   action_history=copy.deepcopy(data.get("recent_action_history", [])))
    if case["stage"] == "decision":
        return session, data["message"], copy.deepcopy(data["robot_state"])
    return (data["user_text"], session, data["policy"], data["applied_this_turn"], data["future_changes"],
            data["recommendation_result"], data["next_prompt"], data["robot_state"], "task")


def assess(case, result):
    if case["stage"] == "response":
        return {"semantic_verdict": "REQUIRES_RAW_REVIEW", "reply": result}
    from soomac_irc.agent_contract import new_turn_state
    from soomac_irc.llm_langgraph import build_graph
    unexpected_q = []
    def unused_recommendation(*_args):
        unexpected_q.append(True)
        return {"proposal": {"sauce": None, "noodle_type": None, "noodle_portion": None, "toppings": {}}, "reason_tags": []}
    graph = build_graph(lambda *_: copy.deepcopy(result), unused_recommendation,
                        lambda *_: "[state replay only; not model Response]")
    outcome = graph.invoke(new_turn_state(*arguments(case)))
    checks = {"order_matches": outcome["session"]["order"] == case["expected_order"],
              "execute_matches": outcome["policy"]["execute"] == case["expected_execute"],
              "preferences_unchanged": outcome["session"]["preferences"] == case["input"]["preferences"],
              "no_unrequested_recommendation": not unexpected_q}
    return {"checks": checks, "passed": all(checks.values()), "order": outcome["session"]["order"],
            "pending": outcome["session"]["pending"], "policy": outcome["policy"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    from soomac_irc import agent_prompts, response_model
    from soomac_irc.decision_model import build_decision_inputs, make_call_decision
    from soomac_irc.model_runtime import (MODEL_PATH, DECISION_ADAPTER_PATH, RESPONSE_ADAPTER_PATH,
                                       RECOMMENDATION_ADAPTER_PATH, load_model)
    from transformers import AutoProcessor
    cases, baseline = prepare_cases()
    current = {"decision": agent_prompts.DECISION_SYSTEM, "response": agent_prompts.TASK_RESPONSE_SYSTEM}
    processor = AutoProcessor.from_pretrained(MODEL_PATH, local_files_only=True)
    pins = {name: digest(ROOT / "src/soomac_irc/soomac_irc" / name) for name in (
        "agent_prompts.py", "prompt_examples.py", "prompt_contrast_examples.py", "llm_langgraph.py", "model_runtime.py")}
    token_counts = []
    for case in cases:
        for mode, systems in (("baseline", baseline), ("candidate", current)):
            if case["stage"] == "decision":
                _, inputs, rebuilt = build_decision_inputs(*arguments(case), processor, system_prompt=systems["decision"])
                assert rebuilt == case["input"], case["id"] + " history/input truncation"
            else:
                messages = [{"role": "system", "content": systems["response"]},
                            {"role": "user", "content": json.dumps(case["input"], ensure_ascii=False, separators=(",", ":"))}]
                inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                      return_dict=True, return_tensors="pt", enable_thinking=False)
            count = inputs["input_ids"].shape[1]
            assert count < 16384, (case["id"], count)
            token_counts.append({"id": case["id"], "mode": mode, "tokens": count})
    plan = {"cases": cases, "calls": 20, "source_sha256": pins, "token_counts": token_counts,
            "base": MODEL_PATH, "adapters": {"decision": DECISION_ADAPTER_PATH, "response": RESPONSE_ADAPTER_PATH,
                                              "recommendation": RECOMMENDATION_ADAPTER_PATH},
            "prompt_sha256": {mode: {k: hashlib.sha256(v.encode()).hexdigest() for k,v in systems.items()}
                              for mode,systems in (("baseline",baseline),("candidate",current))}}
    if args.prepare:
        args.out.mkdir(parents=True, exist_ok=False)
        dump(args.out / "PLAN.json", plan)
        print(json.dumps({"prepared": True, "cases": len(cases), "calls": 20, "max_tokens": max(r["tokens"] for r in token_counts)}), flush=True)
        return
    assert json.loads((args.out / "PLAN.json").read_text()) == plan, "plan/source drift"
    assert not (args.out / "results.jsonl").exists(), "no automatic replay of an existing run"
    import subprocess
    import torch
    assert not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True).strip(), "GPU occupied"
    for path in plan["adapters"].values():
        assert (Path(path) / "adapter_model.safetensors").is_file(), path
    identity = {name: {"path": path, "weight_sha256": digest(Path(path) / "adapter_model.safetensors"),
                       "config_sha256": digest(Path(path) / "adapter_config.json")} for name,path in plan["adapters"].items()}
    dump(args.out / "IDENTITY.json", {"adapters": identity, "pid": os.getpid(), "started": time.time()})
    try:
        model, processor = load_model(DECISION_ADAPTER_PATH, RESPONSE_ADAPTER_PATH, RECOMMENDATION_ADAPTER_PATH)
        assert all(parameter.device.type == "cuda" for parameter in model.parameters())
        calls = {mode: make_call_decision(model, processor, adapter_name="decision", system_prompt=systems["decision"])
                 for mode,systems in (("baseline",baseline),("candidate",current))}
        response = response_model.make_call_response(model, processor, adapter_name="response")
        done = 0
        with (args.out / "results.jsonl").open("x", encoding="utf-8") as stream:
            for case in cases:
                for mode,systems in (("baseline",baseline),("candidate",current)):
                    if case["stage"] == "decision":
                        call = calls[mode]
                    else:
                        response_model.RESPONSE_SYSTEM_BY_ROUTE["task"] = systems["response"]
                        call = response
                    result = call(*arguments(case))
                    trace = copy.deepcopy(call.trace_events[-1])
                    assert trace["model_input"] == case["input"], "actual input drift"
                    row = {"id": case["id"], "mode": mode, "result": result, "trace": trace, "assessment": assess(case, result)}
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    done += 1
                    print(json.dumps({"done": done, "total": 20, "id": case["id"], "mode": mode,
                                      "raw": trace["raw"], "assessment": row["assessment"]}, ensure_ascii=False), flush=True)
        assert all(digest(ROOT / "src/soomac_irc/soomac_irc" / name)==value for name,value in pins.items()), "source drift"
        dump(args.out / "EXIT.json", {"exit_code": 0, "calls": done, "gpu": torch.cuda.get_device_name(0),
                                     "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "closed_loop": False,
                                     "training": False, "actual_ros": False, "automatic_retry": False})
    except BaseException as error:
        dump(args.out / "EXIT.json", {"exit_code": 1, "error": repr(error), "automatic_retry": False})
        raise
    finally:
        response_model.RESPONSE_SYSTEM_BY_ROUTE["task"] = current["response"]


if __name__ == "__main__":
    main()
