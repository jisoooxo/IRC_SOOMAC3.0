import os, sys, time
R = "/home/roma/CosyVoice_new"; sys.path.insert(0, R); sys.path.insert(0, os.path.join(R, "third_party/Matcha-TTS"))
import torch
from cosyvoice.cli.cosyvoice import AutoModel
tts = AutoModel(model_dir="/home/roma/models/audio/tts/Fun-CosyVoice3-0.5B-2512", fp16=False, load_trt=False)
tts.model.token_hop_len = 25
PT = "You are a helpful assistant.<|endofprompt|>반갑습니다. 산사의 아침 공기가 참으로 맑고 고요하지요?"
PW = "/home/roma/IRC_SOOMAC3.0/src/soomac_irc/soomac_irc/tts_prompts/sansa_natural.wav"
for _ in tts.inference_zero_shot("워밍업입니다.", PT, PW, stream=True, text_frontend=False): pass
torch.cuda.synchronize(); print("LOADED", flush=True); time.sleep(3)
text = "게살 알레르기 제한을 추가했어요. 소시지는 많이 선택되어 있고 양파는 적은 양 그대로예요. 이대로 담을까요?"
for i in range(1):
    t = time.perf_counter()
    for _ in tts.inference_zero_shot(text, PT, PW, stream=True, text_frontend=False): pass
    torch.cuda.synchronize(); print(f"TTS {i} {time.perf_counter()-t:.2f}s", flush=True)
print("HOLD", flush=True)
while True: time.sleep(5)
