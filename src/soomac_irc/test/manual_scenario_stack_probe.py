"""Real D/Q/P closed-loop dialogue; physical completion is an offline fixture.

No gold decisions or order repairs are injected. Strict mode stops on a failed
gate; measure-all mode feeds every utterance while preserving the actual robot
stage, including stage drift after a missed commit. No ROS/VLM calls are imported.
"""
import argparse
import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import time

from manual_prompt_alignment_probe import ROOT, digest, dump

os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")


def fixture_text():
    tree = ast.parse((ROOT / "src/soomac_irc/soomac_irc/llm_langgraph_node.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LLMLangGraphNode")
    start = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "_start_order")
    greeting = next(ast.literal_eval(n.value) for n in start.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "reply" for t in n.targets))
    method = next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "_section_prompt")
    announcements = {}
    for branch in method.body:
        if isinstance(branch, ast.If) and isinstance(branch.test, ast.Compare):
            section = ast.literal_eval(branch.test.comparators[0])
            value = next(n.value for n in branch.body if isinstance(n, ast.Return))
            announcements[section] = ast.literal_eval(value)
    return greeting, announcements


def fixture_constants():
    tree = ast.parse((ROOT / "src/soomac_irc/soomac_irc/llm_langgraph_node.py").read_text())
    return {target.id: ast.literal_eval(node.value)
            for node in tree.body if isinstance(node, ast.Assign)
            for target in node.targets if isinstance(target, ast.Name)
            and target.id in {"AMOUNT_TO_COUNT", "TOPIC_CLASS_NAMES"}}


def scenario(compound):
    initial = "면은 얇은면으로 보통 양으로 주고, 소스는 느끼한 걸로 추천해줘."
    if compound:
        return [
            ("noodle", "initial", initial, False),
            ("noodle", "cream_execute", "크림으로 선택할게. 바로 담아.", True),
            ("veggie", "vegetables", "양파랑 버섯 둘 다 적게 담아줘.", False),
            ("veggie", "mushroom_execute", "버섯만 보통 양으로 바꿔서 바로 담아.", True),
            ("meat", "allergy_sausage_execute", "게살 알레르기가 있어서 소시지만 많이 바로 담아줘.", True),
            ("extra", "cheese_execute", "페퍼론치노는 빼고 치즈만 적게 골라서 바로 담아.", True),
        ]
    return [
        ("noodle", "initial", initial, False),
        ("noodle", "cream_select", "크림으로 선택할게.", False),
        ("noodle", "cream_execute", "바로 담아.", True),
        ("veggie", "vegetables", "양파랑 버섯 둘 다 적게 담아줘.", False),
        ("veggie", "mushroom_change", "버섯만 보통 양으로 바꿔줘.", False),
        ("veggie", "mushroom_execute", "바로 담아.", True),
        ("meat", "allergy", "게살 알레르기가 있어요.", False),
        ("meat", "sausage_select", "소시지만 많이 골라줘.", False),
        ("meat", "sausage_execute", "바로 담아.", True),
        ("extra", "cheese_select", "페퍼론치노는 빼고 치즈만 적게 골라줘.", False),
        ("extra", "cheese_execute", "바로 담아.", True),
    ]


def expected_after(order, tag):
    expected = copy.deepcopy(order)
    if tag == "initial":
        expected.update(noodle_type="얇은면", noodle_portion="normal")
    if tag in {"cream_select", "cream_execute"}:
        expected["sauce"] = "크림"
    if tag == "vegetables":
        expected["toppings"].update({"양파": "low", "버섯": "low"})
    if tag in {"mushroom_change", "mushroom_execute"}:
        expected["toppings"]["버섯"] = "normal"
    if tag in {"allergy", "allergy_sausage_execute"}:
        expected["restrictions"] = [{"target": "게살", "reason": "allergy"}]
    if tag in {"sausage_select", "sausage_execute", "allergy_sausage_execute"}:
        expected["toppings"]["소시지"] = "high"
    if tag in {"cheese_select", "cheese_execute"}:
        expected["toppings"].pop("페퍼론치노", None)
        expected["toppings"]["치즈"] = "low"
    return expected


def response_evidence(events, reply, episode, tag):
    """Separate Response model output from runtime-owned fixed replies."""
    attempts = [event for event in events if event.get("stage") == "response"]
    if not attempts:
        return {"source": "runtime_fixed", "automatic_pass": None,
                "text_rule_flags": [], "semantic_review": "REQUIRES_INDEPENDENT_REVIEW"}
    last = attempts[-1]
    if last.get("error"):
        return {"source": "response_error_fallback", "automatic_pass": False,
                "text_rule_flags": ["response_model_error"],
                "semantic_review": "REQUIRES_INDEPENDENT_REVIEW"}
    from soomac_irc.evaluation_v43 import grade_response_prediction
    grade = grade_response_prediction(
        {"id": f"{episode}-{tag}", "route": last.get("route", "task"),
         "input": last["model_input"]}, reply)
    return {"source": "response_model", "automatic_pass": grade.passed,
            "text_rule_flags": list(grade.text_rule_flags),
            "all_violation_codes": [violation.code for violation in grade.violations],
            "semantic_review": "REQUIRES_INDEPENDENT_REVIEW"}


def run_episode(compound, model, processor, stream, *, measure_all=False, execution_reply_mode="fixed"):
    from soomac_irc.agent_contract import new_session_state, new_turn_state
    from soomac_irc.decision_model import make_call_decision
    from soomac_irc.recommendation_model import make_call_recommendation
    from soomac_irc.response_model import make_call_response
    from soomac_irc.llm_langgraph import build_graph
    from soomac_irc.llm_policy import is_future_recommendation_pending, section_execution_items
    constants = fixture_constants()
    amount_to_count, topic_classes = constants["AMOUNT_TO_COUNT"], constants["TOPIC_CLASS_NAMES"]
    decision = make_call_decision(model, processor, adapter_name="decision")
    recommendation = make_call_recommendation(model, processor, adapter_name="recommendation")
    response = make_call_response(model, processor, adapter_name="response")
    graph = build_graph(decision, recommendation, response, execution_reply_mode=execution_reply_mode)
    greeting, announcements = fixture_text()
    session = new_session_state()
    session["history"].append({"role": "assistant", "content": greeting})
    robot = {"section": "noodle", "active_task": None, "task_queue": [], "completed_tasks": [],
             "robot_started": False, "verification_history": [], "vlm_enabled": False, "vlm_confirmed": False}
    name = "compound" if compound else "split_control"
    steps = scenario(compound)
    oracle_order = new_session_state()["order"]
    result_summary = {"episode": name, "planned_turns": len(steps), "observed_turns": 0,
                      "execution_reply_mode": execution_reply_mode,
                      "passed": False, "stopped_at": None, "plans": [], "unreached": [],
                      "physical_simulation_only": True, "actual_ros": False, "failed_turns": [],
                      "measure_all": measure_all, "automatic_adoption": False,
                      "reply_sources": {"response_model": 0, "runtime_fixed": 0,
                                        "response_error_fallback": 0},
                      "response_auto_flagged_turns": []}
    for index, (section, tag, text, execute) in enumerate(steps):
        if not measure_all:
            assert section == robot["section"], "fixture tried advancing without authorized execution"
        before, robot_before = copy.deepcopy(session), copy.deepcopy(robot)
        actual_section = robot["section"]
        oracle_order = expected_after(oracle_order, tag)
        for call in (decision, recommendation, response):
            call.trace_events.clear()
        started = time.perf_counter()
        result = graph.invoke(new_turn_state(session, text, robot))
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        session = copy.deepcopy(result["session"])
        traces = {stage: copy.deepcopy(call.trace_events) for stage,call in (
            ("decision",decision), ("recommendation",recommendation), ("response",response))}
        reply_evidence = response_evidence(traces["response"], result["reply"], name, tag)
        result_summary["reply_sources"][reply_evidence["source"]] += 1
        if reply_evidence["automatic_pass"] is False:
            result_summary["response_auto_flagged_turns"].append(
                {"tag": tag, "source": reply_evidence["source"],
                 "violation_codes": reply_evidence.get("all_violation_codes",
                                                       reply_evidence["text_rule_flags"])})
        checks = {"order_matches": session["order"] == oracle_order,
                  "execute_matches": result["policy"]["execute"] is execute,
                  "section_matches": actual_section == section,
                  "history_prefix_preserved": session["history"][:len(before["history"])] == before["history"],
                  "history_pair_matches": session["history"][-2:] == [
                      {"role":"user","content":text},{"role":"assistant","content":result["reply"]}],
                  "no_model_errors": not any(t.get("error") for group in traces.values() for t in group)}
        if tag == "initial":
            pending = session.get("pending") or {}
            checks["cream_is_unconfirmed_candidate"] = (
                pending.get("type") == "recommendation" and (pending.get("candidate") or {}).get("sauce") == "크림")
        expected_plans = {
            "cream_execute":[{"class":"noodle_thin","repeat_count":2}],
            "mushroom_execute":[{"class":"onion","repeat_count":1},{"class":"mushroom","repeat_count":2}],
            "sausage_execute":[{"class":"sausage","repeat_count":3}],
            "allergy_sausage_execute":[{"class":"sausage","repeat_count":3}],
            "cheese_execute":[{"class":"cheese","repeat_count":1}],
        }.get(tag, [])
        actual_items = (section_execution_items(session["order"],actual_section)
                        if result["policy"]["execute"] else [])
        actual_plans = [{"class":topic_classes[entry["item"]],
                         "repeat_count":amount_to_count[entry["amount"]]
                         if actual_section not in {"sauce","lid"} else 1} for entry in actual_items]
        checks["plans_match"] = actual_plans == expected_plans
        row = {"episode": name, "turn": index+1, "tag": tag, "user_text": text,
               "before": before, "after": copy.deepcopy(session), "robot_before": robot_before,
               "decision_final": result.get("decision"), "policy": result["policy"], "reply": result["reply"],
               "traces": traces, "reply_evidence": reply_evidence,
               "checks": checks, "passed": all(checks.values()),
               "expected_section":section,"expected_order":copy.deepcopy(oracle_order),
               "latency_ms":elapsed_ms,"expected_plans":expected_plans,"actual_plans":actual_plans,
               "physical_simulation_only": True, "actual_ros": False}
        stream.write(json.dumps(row,ensure_ascii=False)+"\n"); stream.flush()
        print(json.dumps({k:row[k] for k in ("episode","turn","tag","user_text","decision_final","policy","reply","checks")},ensure_ascii=False),flush=True)
        result_summary["observed_turns"] += 1
        if not row["passed"]:
            result_summary["failed_turns"].append({"tag":tag,"checks":[key for key,value in checks.items() if not value]})
            if not measure_all:
                result_summary["stopped_at"] = tag
                result_summary["failed_checks"] = [key for key,value in checks.items() if not value]
                result_summary["unreached"] = [entry[1] for entry in steps[index+1:]]
                return result_summary
        if result["policy"]["execute"] and actual_items:
            for entry,plan in zip(actual_items,actual_plans):
                task = {"class":entry["item"], "repeat_count":plan["repeat_count"]}
                result_summary["plans"].append({"section":actual_section,"plan":plan,"actual_ros":False})
                robot["completed_tasks"].append(copy.deepcopy(task))
                session["action_history"].append({"type":"task_completed","section":actual_section,"task":task})
            robot["robot_started"] = True
            if not is_future_recommendation_pending(session.get("pending"),actual_section):
                session["pending"] = None
            if actual_section in {"noodle","veggie","meat","extra"}:
                next_section = {"noodle":"veggie","veggie":"meat","meat":"extra","extra":"lid"}[actual_section]
                change = {"type":"section_transition","section":actual_section,"next_section":next_section,"outcome":"completed"}
                session["action_history"].append(change)
                robot["section"] = next_section
                if next_section in announcements:
                    session["history"].append({"role":"assistant","content":announcements[next_section]})
    result_summary["passed"] = not result_summary["failed_turns"]
    return result_summary


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--measure-all",action="store_true")
    for agent in ("decision", "response", "recommendation"):
        parser.add_argument(f"--{agent}-adapter", type=Path,
                            help="Original PEFT export, not the vLLM-converted directory; evaluation only")
    parser.add_argument("--vllm-base-url", help="Evaluation server URL; only used by the vLLM wrapper")
    return parser.parse_args(argv)


def evaluation_paths(args, defaults):
    paths = {}
    for agent, default in defaults.items():
        override = getattr(args, f"{agent}_adapter")
        path = Path(override if override is not None else default).resolve()
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            if not (path / name).is_file():
                raise FileNotFoundError(path / name)
        if override is not None and (path / "CONVERSION.json").exists():
            raise ValueError("Use original PEFT export; converted directory belongs to the server")
        paths[agent] = str(path)
    return paths


def main():
    args = parse_arguments()
    from soomac_irc.model_runtime import load_model, MODEL_PATH, DECISION_ADAPTER_PATH, RESPONSE_ADAPTER_PATH, RECOMMENDATION_ADAPTER_PATH
    import torch
    paths = evaluation_paths(args, {"decision": DECISION_ADAPTER_PATH,
                                   "response": RESPONSE_ADAPTER_PATH,
                                   "recommendation": RECOMMENDATION_ADAPTER_PATH})
    args.out.mkdir(parents=True,exist_ok=False)
    pins={name:digest(ROOT/"src/soomac_irc/soomac_irc"/name) for name in (
        "agent_prompts.py","prompt_examples.py","prompt_contrast_examples.py","llm_langgraph.py","model_runtime.py")}
    dump(args.out/"PLAN.json",{"base":MODEL_PATH,"adapters":paths,"source_sha256":pins,
         "episodes":{"compound":scenario(True),"split_control":scenario(False)},
         "actual_ros":False,"training":False,"gold_order_repair":False,"physical_simulation_only":True,
         "measure_all":args.measure_all,"acceptance_owner":"user",
         "vllm_base_url":args.vllm_base_url})
    assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True).strip(),"GPU occupied"
    dump(args.out/"IDENTITY.json",{"pid":os.getpid(),"started":time.time(),"adapters":{
        key:{"path":path,"weight_sha256":digest(Path(path)/"adapter_model.safetensors"),
             "config_sha256":digest(Path(path)/"adapter_config.json")} for key,path in paths.items()}})
    try:
        model,processor=load_model(paths["decision"],paths["response"],paths["recommendation"])
        assert all(p.device.type=="cuda" for p in model.parameters())
        with (args.out/"turns.jsonl").open("x",encoding="utf-8") as stream:
            outcomes=[run_episode(compound,model,processor,stream,measure_all=args.measure_all) for compound in (True,False)]
        assert all(digest(ROOT/"src/soomac_irc/soomac_irc"/key)==value for key,value in pins.items()),"source drift"
        dump(args.out/"RESULT.json",{"episodes":outcomes,"field_execution_passed":all(r["passed"] for r in outcomes),
             "response_semantics":"REQUIRES_INDEPENDENT_REVIEW","actual_ros":False,"physical_simulation_only":True})
        dump(args.out/"EXIT.json",{"exit_code":0,"gpu":torch.cuda.get_device_name(0),
             "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"actual_ros":False,"training":False})
    except BaseException as error:
        dump(args.out/"EXIT.json",{"exit_code":1,"error":repr(error),"automatic_retry":False})
        raise


if __name__=="__main__":
    main()
