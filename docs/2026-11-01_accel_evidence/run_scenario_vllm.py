"""Run the existing 17-turn harness unchanged, but with the vLLM server backend (VllmModel)."""
import subprocess, sys
ROOT = "/home/roma/IRC_SOOMAC3.0"
sys.path[:0] = [f"{ROOT}/src/soomac_irc", f"{ROOT}/src/soomac_irc/test"]
import torch
import soomac_irc.model_runtime as mr
from soomac_irc import vllm_client
import manual_scenario_stack_probe as probe

class HarnessVllmModel(vllm_client.VllmModel):
    def parameters(self):  # harness asserts weights are on CUDA; the server holds them
        return iter([torch.empty(1, device="cuda")])

def load_model(decision_path, response_path, recommendation_path):
    _, processor = vllm_client.load_vllm_backend(mr.MODEL_PATH, {
        "decision": decision_path, "recommendation": recommendation_path, "response": response_path})
    return HarnessVllmModel(), processor

_check_output = subprocess.check_output
def check_output(args, *a, **k):  # harness requires an idle GPU; the vLLM server is expected to occupy it
    if args and args[0] == "nvidia-smi":
        return ""
    return _check_output(args, *a, **k)

mr.load_model = load_model
probe.subprocess.check_output = check_output
probe.main()
