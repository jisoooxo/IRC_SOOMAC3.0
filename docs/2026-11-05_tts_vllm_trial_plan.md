# TTS vLLM 별도 노드 동시 상주 실측 계획

사용자 승인: 기존 노드를 바꾸지 않는 별도 TTS 노드, STT/LLM/TTS 및 D/Q/P 세 adapter 동시 적재·실측. 필요하면 시험용 제한 변경 후 종료·원복한다. 예상 30~50분이며 의존성/컴파일에 따라 연장 가능하다.

1. [x] 사전 봉인·환경: 현재 source/launcher/adapter/환경 metadata를 기록한다. cosyvoice3의 Torch2.10/torchaudio2.10과 ABI가 맞는 vLLM0.17.0 후보를 **작업 폴더의 별도 venv**에 설치한다. 기존 환경은 상속 읽기만 한다. 모델 원본은 읽기 전용 참조하고 vLLM export만 시험 폴더에 저장한다.
2. [x] 별도 노드: 원본 TTS 클래스의 실제 전처리·문장 pause·출력 callback을 재사용하되 시험 프로세스 안에서만 모델 factory를 교체한다. 출력/파형은 시험 폴더, ROS domain217와 localhost만 사용한다. 마이크는 없는 것으로 보고 저장 음성으로 Qwen 실제 추론, 시나리오는 정확한 텍스트 입력을 사용한다. 실물 로봇/VLM/운영 topic은 호출하지 않는다.
3. [x] 사전 합성·비교: 같은 한국어 문장/화자/참조음성/stream=True 조건으로 현재 HF, 시험환경 HF(의존성 차이 대조), vLLM을 비교한다. 필요하면 후보 전용 API 호환 shim을 만들고 범위/차이를 기록한다. 원본 CosyVoice 소스는 수정하지 않는다.
4. [x] 동시 실측: 현재 LLM16K/.58·batch invariant·D6/Q6/Pbest150와 Qwen STT를 상주시키고 TTS 후보를 시험한다. TTS 자체는 max_seqs1 및 작은 KV 예산으로 시작한다. 각 조건의 최종5대본36턴과 고정 텍스트 합성 지연을 기록한다. 동시 STT/LLM/TTS 추론도 짧게 측정해 상주와 연산 중 peak를 구분한다. 필요할 때 시험 프로세스 인자만 조정하며 실제값을 남긴다.
5. [x] 종료·판정: 첫 오디오 요청 지연, 합성 RTF(재생 제외), 대본 총시간, 장치 전체 NVML peak, 출력 WAV/텍스트와 상태·수량·계약을 비교한다. 생성 음성의 자연스러움/발음은 텍스트 state 검사와 별개이며 실제 청취 여부를 명시한다. 소유 scope·port·GPU 반환 및 원본 해시 대조 뒤 결과/worklog를 기록한다.

## 실행 경계

- 기존 `tts_node.py`, CosyVoice 설치/소스, aliases, server launcher, adapter, checkpoint, VLM 정책을 수정하지 않는다.
- 새 환경의 pip 설치가 상속 환경을 uninstall/upgrade하지 않도록 venv 경로/설치 계획을 검증한다. 모델 export는 새 디렉터리다.
- 원래 TTS FP32 대비 vLLM export BF16 및 transformers 버전 변화가 생기면 순수 엔진 효과로 단정하지 않는다. 별도 HF 대조를 둔다.
- 현행 TTS+STT+LLM baseline과 후보를 새 프로세스에서 순차 시험한다. 설치/단독 합성 실패는 보존하고 동시 실측 성공으로 쓰지 않는다.
- 시스템 OOM/멈춤 방지를 위해 CPU thread 수·scope RAM 제한, GPU 여유/시스템 RAM 감시를 유지한다. 사용자 허용은 제한 완화 가능성이며 물리 VRAM을 넘겨 실행하라는 뜻이 아니다. 다른 소유 프로세스는 중단하지 않는다.
- 결과는 텍스트→TTS 요청/재생 callback 기준이다. 마이크/VAD·물리 동작·VLM·음향 센서 지연을 포함한 완전 시연 시간은 아니다.

공식 근거: CosyVoice `vllm_example.py`의 CosyVoice3 load_vllm 경로, vLLM v0.17.0 requirements의 torch/torchaudio2.10.0. 실제 설치 wheel metadata/import/추론으로 추가 확인한다. 기존 LLM 서버의 vLLM0.32와 TTS 엔진 버전이 같아야 할 이유는 없으며 별도 프로세스로 격리한다.

## 실행 전 비교·중단 기준 확정

- 모델은 현재 Fun-CosyVoice3-0.5B-2512와 같은 참조 음성이다. 후보는 공식 `CosyVoice3 → load_vllm → speech-token LM → 기존 flow/HiFT` 경로이며 TensorRT는 끈다. `fp16=False` 유지, export LM의 BF16 변화는 별도 표시한다.
- 세 조건 모두 동일 초기 TTS 워밍업·LLM 공통 접두 워밍업, 고정 문장 3회 반복, 같은5대본36턴을 사용한다. 독립 노드는 실제 source callback을 실행하며 원본 노드를 바꾸지 않는다.
- 첫 음성 지연은 TTS callback 입력→첫 speaker.write, 합성 시간은 generator 첫 next 직전→StopIteration(소비/재생과 분리된 producer), RTF는 이 시간/원본 음성 길이이다. 전체 대본은 첫 graph 입력→마지막 TTS callback 종료다. 실제 음향 센서 측정은 아니다.
- 기능 통과: HTTP 정상 종료, 음성 nonempty/finite/24kHz·잘린 오류 없음, 36/36 상태·수량·계약, 알려진 주문/수량 문장 ASR 재인식은 보조 검사. 자연스러움/화자 품질은 주관 청취 미실시 시 미검증으로 남긴다.
- 성능 채택 후보: 같은 환경 HF 대비 첫 음성 지연 중앙값10% 이상 개선·P95가10% 넘게 악화하지 않음·generation RTF<1·동시 부하 최소 VRAM 여유1.5GiB 이상을 잠정 기준으로 삼는다. 단회 대본은 기술 통계이며 통계적 유의성이나 SLA를 주장하지 않는다.
- CPU thread2, 각scope RAM38GiB/Swap0, GPU/RAM50ms 목표 샘플. GPU free<1GiB 또는 MemAvailable<6GiB가4샘플 지속되면 소유scope 중단한다. 명시적 오류/OOM은 즉시 실패 기록. 필요 시 TTS만 max_model_len4096/max_seqs1/KV512MiB부터 조정한다. LLM16K/.58 유지가 우선이며 변경이 필요하면 별도 condition으로 보존한다.
- Codex 일반 설계 교차검수는 조건부 찬성이며 위 비교/중단/정밀도 기준을 반영했다. 로컬 코드·실측 수치의 외부 독립 감사와 구분한다.

## 실측 중 확인된 실패와 진단 유지

- baseline 36턴은 끝났으나 34/36 상태·계약 통과다. `accepted_short_then_explicit_execution` turn2의 짧은 수락에 commit=true가 추가되어 면을 먼저 실행했고, turn3은 이미 야채 단계라 기대와 달랐다. TTS 교체 전 기준에서도 발생했으며 대본 정답을 변경하지 않는다. 전체 성공/안전한 채택으로 보고하지 않는다.
- 이후 진단은 계약 실패를 `CONTRACT_STATUS.json`에 보존하면서 음성·동시 메모리 검사까지 마친다. 실물 실행은 없다. 최초 baseline 전체 측정을 반복하지 않고 누락된 stress/ASR만 별도 fresh stack으로 보충한다. 그 stress LLM cache는 기존 full-scenario 이후와 다르다.
- HF 대조군 시작 전 포트 bind가 EADDRINUSE로 실패했지만 listener/GPU 잔류는 없었다. 시험 preflight socket에 SO_REUSEADDR를 적용했다. TIME_WAIT 원인은 추정이며 당시 소켓 상태를 캡처하지 않아 확정하지 않는다.
- 새 환경 HF는 fixed sentence 9개 모두 0.04초 음성으로 붕괴했다. nonempty/finite만으로 걸러지지 않았으며 수동 중단·실패 보존했다. 이제 전체 문장에 0.3초 미만이면 실패 처리한다. 이 조건의 낮은 지연은 성능 자료로 쓰지 않는다. 원래 HF와 vLLM 비교는 dependency/dtype/sampling이 함께 달라지는 제품 후보 비교이며 순수 엔진 비교로 주장하지 않는다.
- 추가 교란: 원본 YAML은 RAS(top_p=.8/top_k25)이고 공식 vLLM wrapper는 SamplingParams(top_k25, stop/min/max)다. 같은 seed라고 같은 오디오/길이를 보장하지 않는다.

## 종료

실측 및 정리 완료. [최종 보고](2026-11-05_tts_vllm_trial_review.md). 기존 HF 유지 권고, 운영 미교체·16K/.58 유지. 원본40파일/3환경 inventory/3adapter 동일. 시험scope·포트·GPU 프로세스 반환. 별도 RustDesk311MiB는 종료하지 않았다. HF 대조 실패와 baseline 계약34/36은 보존하며 전체 무결점 성공으로 쓰지 않는다.
