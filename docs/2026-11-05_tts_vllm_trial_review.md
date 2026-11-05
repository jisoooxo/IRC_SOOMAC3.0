# 별도 TTS vLLM 동시 상주 실측 — 2026-11-05

[실행 계획](2026-11-05_tts_vllm_trial_plan.md) → 운영 노드를 바꾸지 않고 후보의 속도·음성·GPU 여유를 실제 측정한다.

**실측 완료. vLLM TTS의 별도 노드 실행은 가능하지만, 현재 HF TTS 유지 권고이다.** 고정 문장 첫 출력 중앙값은 약43ms 개선됐지만 P95가 악화했고, 대조 환경 HF 실패와 품질 미검증이 남는다. 운영 노드는 교체하지 않았다.

## 1. 시험 범위와 실제 경로

기존 `tts_node.py`는 수정하지 않았다. 새 `tools/tts_vllm_trial/audio_node.py`가 원본 클래스를 상속하고 해당 프로세스 안에서만 모델 생성 함수를 교체한다. 원본 전처리, 동일 화자와 참조 음성, stream=True, 0.60초 문장 pause, speaker.write를 사용했다. ROS domain217/localhost이며 생산 토픽 대신 캡처 객체를 사용한다. 실제 로봇 명령은 발행하지 않는다.

| 구성 | 이번 실측 |
|---|---|
| LLM | 현행 Gemma4-12B NF4/BF16, vLLM0.32.0, 16,384토큰/.58/max_seqs2, batch invariant ON |
| LoRA | Decision epoch6, Recommendation epoch6, Response best150를 분리 적재; 병합 없음 |
| STT | 기존 gemma4_env의 Qwen3-ASR-1.7B-hf, 저장 음성10초/30초 실제 인식 |
| 기존 TTS | Fun-CosyVoice3-0.5B-2512, cosyvoice3 Python3.10/Torch2.10, HF FP32, TensorRT OFF |
| 후보 TTS | 같은 모델의 공식 speech-token LM vLLM 경로, 별도 venv/vLLM0.17.0/BF16; flow/HiFT FP32 유지 |

Gemma 서버와 TTS vLLM은 **서로 다른 프로세스/엔진**이다. D/Q/P는 Gemma 서버에만 적재된다. TTS는 별도512MiB KV cache, max_model_len4096/max_seqs1이다. `gpu_memory_utilization=.12`도 기록돼 있지만 명시적 `kv_cache_memory_bytes`가 우선하므로 실제 메모리 상한12%라고 해석하면 안 된다.

최종5대본36턴은 정확한 텍스트를 실제 D/Q/P graph에 입력하고 매 답변을 실제 합성/스피커 출력했다. STT 모델은 상주하며 별도 저장 음성 추론 및 동시 부하에 참여한다. **마이크→STT→36턴 전체 음성 파이프라인, 실제 음향 시작/끝, 로봇 동작, VLM 이미지 추론은 이 시험에 포함하지 않았다.** 로봇 완료는 평가 fixture이다.

## 2. 속도 비교의 정의와 제한

고정 문장3개×3회=9회는 같은 텍스트·참조 음성으로 측정한다. 첫 음성은 TTS callback 입력→첫 speaker.write 요청이다. 합성 실시간 계수(RTF)는 generator 완료 시간/원본 음성 길이이며, 재생은 별도 소비 스레드이므로 오디오 재생 대기를 포함하지 않는다. 실제 음향 측정이나 첫 소리가 귀에 도달한 시간은 아니다.

**이번 결과는 실행 환경 전체의 비교이며, 순수 vLLM 엔진만의 성능 비교가 아니다.** 새 HF 대조군은 0.04초 음성으로 붕괴해 무효 처리했다. 따라서 기존 HF 대 후보 vLLM은 transformers4.51.3→4.57.1, FP32→BF16 speech LM, RAS(top_p=.8/top_k25)→공식 vLLM top-k25 sampling 변경이 함께 존재한다. 원본 HF RAS는 `cosyvoice/utils/common.py:138`, 분기별 생성은 `cosyvoice/llm/llm.py:505`에서 확인했다. command의 seed를 같게 설정해도 별도 vLLM 엔진의 RNG가 매번 리셋되는 것은 아니다.

출력 문장·음성 길이가 달라지므로 전체 대본 시간 차이를 모두 연산 가속으로 해석하지 않는다. 9회 probe/조건별 대본1회는 기술 통계이며 통계적 유의성·장시간 안정성·SLA를 보장하지 않는다. NVML은50ms 목표로 장치 전체를 샘플링하며 관측 사이의 순간 peak는 놓칠 수 있다.

## 3. 실패를 보존한 내역

1. 첫 import는 upstream 커스텀 vLLM 모델의 `Union` wildcard 의존 때문에 실패했다. 새 시험 파일에만 typing import를 추가했다. 실제 export architecture는 Qwen2ForCausalLM으로 커스텀 클래스가 실제 inference를 맡지는 않는다. 원본 CosyVoice 코드는 그대로다.
2. 첫 GPU smoke는 CXXABI_1.3.15/SQLite 관련 libstdc++ 충돌로 실패했다. TTS 자식 프로세스에만 cosyvoice3의 libstdc++를 preload하여 재실행한 단독3문장 smoke는 통과했다. 시스템 라이브러리나 원본 환경은 수정하지 않았다.
3. baseline 대본은36턴까지 완료했으나34/36 계약 통과로 마지막 assertion이 발생했다. 후속 진단은 계약 실패를 그대로 저장하면서 음성·메모리 진단을 끝내도록 바꿨다. baseline 누락 stress/ASR만 별도 fresh stack으로 보충하며 full scenario는 반복하지 않는다.
4. 첫 HF 대조군 시작은18001 bind EADDRINUSE로 실패했다. 후속 listener/GPU 잔류는 없었고 trial preflight에 SO_REUSEADDR를 적용했다. TIME_WAIT는 추정이며 당시 해당 소켓 상태를 캡처하지 않아 원인 확정은 하지 않는다.
5. 새 환경 HF 대조군 재시도는9개 probe가 모두0.04초 음성이다. nonempty/finite 검사만으로 성공으로 잘못 표시된 원본 JSON을 남기고 `AUDIO_REJECTION.json`으로 전부 무효 처리했다. 중단 후 전체 문장0.3초 미만 실패 검사를 보강했다. 낮은 지연값을 성능 개선으로 계산하지 않는다. HF 붕괴의 내부 원인은 미확정이다.

`pip check`는 깨끗하지 않다. baseline의 whisper/triton 제약과 trial의 lightning/packaging, grpcio-tools/protobuf, gradio/numpy 및 gradio-client/websockets 제약이 남는다. 성공한 추론 경로가 그 UI/codegen 경로를 검증하지는 않는다. `BASELINE_PIP_CHECK.log`, `PIP_CHECK.log`에 보존했다. 새 venv를 운영용 범용 환경으로 채택해서는 안 된다.

## 4. 대화 계약 차이

기준 HF 실행의 `accepted_short_then_explicit_execution`에서 turn2 “그걸로 담아줘”에 Decision이 `confirmation=accept, commit=true`를 출력했다. 크림 선택 뒤 얇은면 보통2회 실행계획이 만들어졌고, turn3 명시적 면 실행 시 이미 야채 단계였다. 대본 기대는 turn2 소스 선택만, turn3 면 실행이므로2턴 실패이다. 실제 로봇 동작은 없다.

후보 실행은 동일한 요청에서 `confirmation=accept`만 출력했다. 직전 warmup 평가, 이번 HF 기준, 후보의 해당 요청 인자가 완전히 같은 것을 `DECISION_VARIATION.json`에 보존했다. batch invariant ON이어도 이번 서버 실행 간 출력 차이가 존재한다. **내부 원인은 UNKNOWN이며 TTS 교체가 Decision을 개선했다거나 워밍업이 원인이라고 단정하지 않는다.** 평가 정답·프롬프트·운영 정책을 이 시험에서 변경하지 않았다.

## 5. 최종 측정·종료 검증

### 동일 문장9회 — 실행 환경 전체 비교

| 지표 | 기존 HF | 후보 vLLM |
|---|---:|---:|
| 첫 speaker.write 중앙값 | 0.511초 | 0.468초 |
| 첫 speaker.write P95 | 0.523초 | 0.590초 |
| 합성 RTF 중앙값 | 0.359 | 0.280 |
| 정상 길이/finite/nonempty 음성 | 9/9 | 9/9 |

중앙값은0.043초(8.35%) 개선했으나 P95는12.81% 악화했다. 사전 잠정 성능 기준인 중앙값10% 이상 개선·P95 악화10% 이내를 모두 만족하지 못했다. RTF는약22% 낮아졌지만 이미 두 조건 모두 재생보다 빠르게 합성한다. 새 환경 HF 대조군은9/9 무효이며 표에서 제외했다.

### 최종 시나리오 — LLM 입력부터 마지막 TTS callback 종료

| 대본 | 턴 | 기존 HF | 후보 vLLM |
|---|---:|---:|---:|
| 복합 주문·추천·단계 실행 | 6 | 37.27초 | 36.14초 |
| 명시적 선택+면 실행 | 2 | 15.96초 | 16.13초 |
| 짧은 수락→명시 실행 | 11 | 78.23초 | 70.11초 |
| 수량 정정→지시어 실행 | 11 | 75.32초 | 77.97초 |
| 기분→음식→실행 거절 | 6 | 36.41초 | 37.64초 |

**전체36턴은 기존4분3.56초, 후보3분58.37초**이다. 핵심 복합 주문6턴은37.27초→36.14초였다. 사용자 말하기/생각하기, STT endpoint, 실제 로봇 대기, VLM 시간은 제외한다. 첫 대본 전 worker 준비/공통 접두 워밍업 시간도 표에서 제외된다.

두 조건은36답변 중4개가 다르고 원본 음성 합계174.12→169.08초, pause 포함177.92→173.77초이다. 전체약5.19초 단축은 음성 길이·대화 분기 차이의 영향을 크게 받으므로 TTS 가속만의 효과로 주장하지 않는다. 상태·계획·계약은 기존34/36, 후보36/36이며 정확도 향상을 TTS로 귀속하지 않는다. 두 조건 모두 scenario HTTP62/62 정상 stop이고, 워밍업3요청씩은 max_tokens1로 의도한 length 종료이다.

### 동시 부하와 음성 검사

후보는 실제 STT30초 저장 음성·LLM 요청·TTS 합성을 동시에 실행했다. 요청/추론 interval이3자간2.096초 겹쳤다. 이는 GPU kernel-level 동시 실행 시간을 직접 측정한 값은 아니다. STT3.232초, LLM2.096초, 동시 부하 TTS 첫 출력1.418초/RTF0.594이다. 직렬 첫 출력0.468초와 달리 동시 연산 시 지연이 늘었다.

| 후보 phase | 장치 전체 관측 peak | 최소 여유 |
|---|---:|---:|
| 로딩 포함 전체 | 29.29GiB | 2.57GiB |
| 36턴 대화 | 28.58GiB | 3.28GiB |
| 세 모델 동시 추론 | 28.82GiB | 3.04GiB |

NVML 측정 간격 최대80.6ms이다. 현재16K/.58을 그대로 사용했고 용량 제한 해제는 필요하지 않았다. 이번 부하에는 VLM 이미지 입력이 없으며 이전 더 높은 peak를 무효화하거나 임의 입력에서도 충분한 여유가 있음을 보장하지 않는다.

후보 고정 문장3개를 Qwen STT로 재인식한 결과는 다음과 같다. 수량·제외 의미는3/3 보존됐다. **자연스러움·화자 유사도·발음 품질의 인간 청취 검증은 미실시**이며 ASR 통과를 청취 통과로 쓰지 않는다.

| 합성 요청 | 재인식 |
|---|---|
| 얇은면을 보통 양으로 담을게요. | 얇은 면을 보통 양으로 담을게요. |
| 양파는 적게, 버섯은 많이 담을게요. | 양파는 적게, 버섯은 많이 담을게요. |
| 게살 알레르기가 있으시군요. 게살은 제외하고 소시지만 많이 담을게요. | 요청과 동일 |

baseline 보충 시험도 완료했다. 동시 부하 peak28.66GiB/여유3.20GiB, 3자 interval 겹침3.079초, STT3.089초/LLM5.217초/TTS 첫 출력2.812초였다. **이 baseline은 fresh stack의 cold LLM cache이고 후보는 대본 이후 warm cache라 두 동시 지연을 직접적인 속도 향상 비율로 계산하지 않는다.** baseline 원래 WAV3개 재인식도 의미3/3 보존됐다. 별도 baseline 보충 중 RustDesk가 GPU311MiB를 사용하기 시작했으며 장치 전체 수치에 포함된다.

최종 원본40파일 SHA256, 기존3환경 package name/version 전체 inventory, D/Q/P symlink target 및 adapter weight/config SHA는 모두 BEFORE와 같았다. 이는 전체 site-packages 설치파일 해시 검증은 아니다. 운영16K/.58, 기존 node/alias/adapter/VLM은 변경하지 않았으므로 설정 롤백은 필요 없었다.

소유 scope는 모두 inactive/반환, 시험18001과 운영8000 listener는 없다. 종료 검사가 GPU PID243672의311MiB 때문에 일단 실패했으나 host cgroup을 확인하니 `/system.slice/rustdesk.service`의 RustDesk였다. 그 프로세스는 종료하지 않았다. 최초 `RESOURCES.json`/`AFTER_INITIAL.json` 실패를 보존하고 최종 `AFTER.json`에서 **시험 소유 GPU 프로세스0, 별도 RustDesk311MiB 잔류**를 확인했다. 모든 GPU compute가0이라고 보고하지 않는다.

계획5단계와 실측을 끝냈다. 정밀도·sampling·의존성 차이와 불완전한 음질 검증, 작고 불균일한 첫 응답 이득을 고려해 **기존 HF TTS 유지, 후보는 별도 시험 노드로 보존**한다. 판단 자료/음성/실패 로그는 남겼으며 `FINAL_STATUS.json`은 실험 완료와 운영 미채택을 구분한다.

Cause: 후보는 합성 RTF를 줄였지만 첫 출력 중앙값의 개선이 작고 P95가 악화했으며, 동일 환경 HF 대조가 정상 음성을 만들지 못했다.
Effect: 운영을 바꾸지 않은 상태에서 후보의 실행 가능성·속도·동시 VRAM 범위를 확보했고, 현 단계에서는 교체 이득을 입증하지 못했다.


원본 자료: `training/learning_first_v43/final/goals/GL10/tts_vllm_trial_20261105/`. `PLAN/BEFORE`, 각 조건의 `REQUESTS/TURNS/VRAM`, 음성 WAV, 중단 로그를 보존한다. Codex 일반 설계 및 해석 원칙 교차검수는 별도로 완료했으며 내부 코드/수치 전체의 독립 감사라고 주장하지 않는다.
