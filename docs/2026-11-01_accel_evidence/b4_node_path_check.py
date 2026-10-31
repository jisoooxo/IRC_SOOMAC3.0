import json, sys
sys.path.insert(0, "/home/roma/IRC_SOOMAC3.0/src/soomac_irc")
from soomac_irc import vllm_client
from soomac_irc.model_runtime import MODEL_PATH, DECISION_ADAPTER_PATH, RECOMMENDATION_ADAPTER_PATH, RESPONSE_ADAPTER_PATH
from soomac_irc.decision_model import make_call_decision
from soomac_irc.agent_contract import new_session_state
model, processor = vllm_client.load_vllm_backend(MODEL_PATH, {"decision": DECISION_ADAPTER_PATH, "recommendation": RECOMMENDATION_ADAPTER_PATH, "response": RESPONSE_ADAPTER_PATH})
print("served roots:", vllm_client.served_adapter_roots())
call = make_call_decision(model, processor, adapter_name="decision")
session = new_session_state()
robot = {"section": "noodle", "active_task": None, "task_queue": [], "completed_tasks": [], "robot_started": False, "verification_history": [], "vlm_enabled": False, "vlm_confirmed": False}
print("decision:", call(session, "면은 얇은면으로 보통 양으로 주고, 소스는 느끼한 걸로 추천해줘.", robot)["order_patch"])
t = call.trace_events[-1]; print("trace:", {k: t.get(k) for k in ("prompt_tokens", "output_tokens", "latency_ms", "raw")}, "usage:", model.last_usage)
try:
    vllm_client.check_served_adapters({"decision": "/tmp/some_other_adapter"}, vllm_client.served_adapter_roots())
except vllm_client.VllmError as e:
    print("mismatch detected OK:", str(e)[:80])
