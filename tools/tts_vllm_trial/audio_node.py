"""Isolated audio trial: production callbacks, no production topic publication."""
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ['SOOMAC_TEST_RUN'])
ROLE = sys.argv[1]
MODE = os.environ.get('TTS_TRIAL_MODE', 'baseline')
sys.path[:0] = [str(ROOT/'src/soomac_irc'), '/home/roma/CosyVoice_new', '/home/roma/CosyVoice_new/third_party/Matcha-TTS']


def save(path, value):
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)); temp.replace(path)


class Capture:
    last = None
    def publish(self, msg): self.last = msg.data


def install_vllm_factory(mod):
    from vllm import EngineArgs, LLMEngine, ModelRegistry
    from cosyvoice.cli.model import CosyVoice2Model
    from cosyvoice.utils.file_utils import export_cosyvoice2_vllm
    import torch
    import threading
    ModelRegistry.register_model('CosyVoice2ForCausalLM', 'cosyvoice_vllm_compat:CosyVoice2ForCausalLM')
    def load_vllm(self, model_dir):
        export_cosyvoice2_vllm(self.llm, model_dir, self.device)
        args = dict(model=model_dir, skip_tokenizer_init=True, enable_prompt_embeds=True,
                    max_model_len=4096, max_num_seqs=1, max_num_batched_tokens=4096,
                    kv_cache_memory_bytes=int(os.environ.get('TTS_KV_MIB', '512'))*1024**2,
                    gpu_memory_utilization=float(os.environ.get('TTS_GPU_MEM', '.12')),
                    dtype='bfloat16', seed=1234)
        save(OUT/'TTS_ENGINE_ARGS.json', args)
        self.llm.vllm = LLMEngine.from_engine_args(EngineArgs(**args))
        self.llm.lock = threading.Lock()
        del self.llm.llm.model.model.layers
    CosyVoice2Model.load_vllm = load_vllm  # This trial process only.
    original = mod.AutoModel
    def factory(**kwargs):
        kwargs.update(load_vllm=True, load_trt=False, fp16=False)
        return original(**kwargs)
    mod.AutoModel = factory
    mod.MODEL_DIR = str(ROOT/'tools/tts_vllm_trial/model')


def main():
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly
    import rclpy
    import torch
    from std_msgs.msg import String
    torch.set_num_threads(2)
    rclpy.init(args=['--ros-args', '-r', '__node:=soomac_'+ROLE+'_'+MODE+'_trial'])
    node = None
    try:
        if ROLE == 'stt':
            from soomac_irc.stt_qwen3_asr_node import Qwen3AsrSttNode
            class StoredInput(Qwen3AsrSttNode):
                def _initialize_runtime(self):
                    self._validate_constants(); self._load_silero_vad(); self._load_model()
                    self.question_pub = Capture(); self.capture_source = None
            node = StoredInput()
        else:
            import soomac_irc.tts_node as mod
            mod.LAST_WAV = str(OUT/'audio/startup.wav')
            mod.LAST_PACED_WAV = str(OUT/'audio/startup_paced.wav')
            if MODE == 'vllm': install_vllm_factory(mod)
            class TrialTTSNode(mod.TTSNode):
                pass
            node = TrialTTSNode()
            assert hasattr(node.cosyvoice_tts.model.llm,'vllm') == (MODE == 'vllm')
            class Speaker:
                def __init__(self, real): self.real=real; self.first_write=None
                def write(self, data):
                    if self.first_write is None: self.first_write=time.perf_counter()
                    return self.real.write(data)
                def __getattr__(self, key): return getattr(self.real,key)
            node.speaker = Speaker(node.speaker)
            node.stt_stop_pub = Capture(); node.stt_retrigger_pub = Capture()
            original_infer = node.cosyvoice_tts.inference_zero_shot
            synthesis = {}
            def timed_infer(*args, **kwargs):
                synthesis.clear(); synthesis['start']=time.perf_counter()
                try:
                    yield from original_infer(*args, **kwargs)
                except BaseException as error:
                    synthesis['error']=str(error)
                    raise
                finally:
                    synthesis['end']=time.perf_counter()
            node.cosyvoice_tts.inference_zero_shot = timed_infer
        save(OUT/(ROLE+'_READY.json'), dict(pid=os.getpid(),time=time.time(),mode=MODE,
            model='Qwen3-ASR-1.7B-hf' if ROLE=='stt' else 'Fun-CosyVoice3-0.5B-2512',
            cuda_initialized=torch.cuda.is_initialized(),live_microphone=False))
        seen=set()
        while not (OUT/'STOP').exists():
            for path in sorted((OUT/'commands').glob(ROLE+'_*.json')):
                if path.name in seen: continue
                seen.add(path.name); command=json.loads(path.read_text()); result=dict(command=command,pid=os.getpid())
                start=time.perf_counter()
                try:
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                    if ROLE == 'stt':
                        file=Path(command.get('file',ROOT/'src/soomac_irc/soomac_irc/tts_prompts/sansa_natural.wav'))
                        audio,rate=sf.read(str(file),dtype='float32')
                        if audio.ndim>1: audio=audio.mean(axis=1)
                        gcd=math.gcd(16000,rate); audio=resample_poly(audio,16000//gcd,rate//gcd)
                        if 'file' not in command:
                            n=int(command.get('seconds',10)*16000); audio=np.tile(audio,math.ceil(n/len(audio)))[:n]
                        pcm=np.clip(audio*32768,-32768,32767).astype('<i2').tobytes()
                        with node.state_lock: node.listening_enabled=True; node.gate_open=True
                        node.question_pub.last=None; infer_start=time.perf_counter()
                        node._recognize_utterance(pcm,node.session,infer_start,len(audio)/16000)
                        result.update(text=node.question_pub.last,inference_s=time.perf_counter()-infer_start,input_audio_s=len(audio)/16000)
                        assert result['text'], 'No STT output'
                        with node.state_lock: node.listening_enabled=False; node.gate_open=False
                    else:
                        from cosyvoice.utils.common import set_all_random_seed
                        set_all_random_seed(command.get('seed',1234))
                        wav=OUT/'audio'/(path.stem+'.wav'); paced=OUT/'audio'/(path.stem+'_paced.wav')
                        mod.LAST_WAV=str(wav); mod.LAST_PACED_WAV=str(paced)
                        node.speaker.first_write=None; node.stt_retrigger_pub.last=None
                        call_start=time.perf_counter(); node.llm_callback(String(data=command['text'])); call_end=time.perf_counter()
                        assert node.stt_retrigger_pub.last == 'finished', 'TTS incomplete'
                        assert not synthesis.get('error'), synthesis
                        audio,rate=sf.read(str(wav),dtype='float32')
                        assert rate==24000 and len(audio)>0 and np.isfinite(audio).all()
                        assert len(audio)/rate>=.3, 'Degenerate audio under 0.3s for full trial sentence'
                        assert float(np.max(np.abs(audio)))>0, 'Silent WAV'
                        generated=len(audio)/rate; gen_s=synthesis['end']-synthesis['start']
                        info=sf.info(str(paced if paced.exists() else wav))
                        result.update(callback_start=call_start,callback_end=call_end,first_write=node.speaker.first_write,
                            first_write_delay_s=node.speaker.first_write-call_start, audio_duration_s=info.duration,
                            raw_audio_duration_s=generated,generation_s=gen_s,rtf=gen_s/generated,
                            wav=str(wav),paced=str(paced) if paced.exists() else None,speaker_latency_s=node.speaker.latency,
                            playback_measured='software writes and callback, not acoustic onset/end',synthesis=synthesis.copy())
                    torch.cuda.synchronize()
                    result.update(ok=True,torch_peak_allocated=torch.cuda.max_memory_allocated(),torch_peak_reserved=torch.cuda.max_memory_reserved())
                except BaseException as error:
                    result.update(ok=False,error=str(error),traceback=traceback.format_exc())
                result.update(start=start,end=time.perf_counter()); result['seconds']=result['end']-start
                save(OUT/'results'/path.name,result)
            time.sleep(.02)
    finally:
        if node is not None: node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()


if __name__=='__main__':
    try: main()
    except BaseException as error:
        save(OUT/(ROLE+'_EXIT.json'),dict(error=str(error),traceback=traceback.format_exc())); raise
