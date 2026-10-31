import time, wave, numpy as np, torch
from transformers import AutoModelForMultimodalLM, AutoProcessor
P = "/home/roma/models/audio/stt/Qwen3-ASR-1.7B-hf"
proc = AutoProcessor.from_pretrained(P, local_files_only=True)
model = AutoModelForMultimodalLM.from_pretrained(P, local_files_only=True, dtype=torch.bfloat16, device_map="cuda:0").eval()
w = wave.open("/home/roma/IRC_SOOMAC3.0/src/soomac_irc/soomac_irc/tts_prompts/sansa_natural.wav"); pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
sr = proc.feature_extractor.sampling_rate
audio = np.interp(np.arange(0, len(pcm), 24000 / sr), np.arange(len(pcm)), pcm).astype(np.float32)
audio = np.concatenate([audio, audio])  # ~10 s utterance
torch.cuda.synchronize(); print("LOADED", flush=True); time.sleep(3)
for i in range(5):
    t = time.perf_counter()
    inputs = proc.apply_transcription_request(audio=audio, language="ko").to(model.device, model.dtype)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=256)
    torch.cuda.synchronize(); print(f"STT {i} {time.perf_counter()-t:.2f}s", proc.decode(out[:, inputs['input_ids'].shape[1]:], return_format='transcription_only'), flush=True)
print("torch peak GiB", round(torch.cuda.max_memory_allocated() / 2**30, 2)); time.sleep(2)
