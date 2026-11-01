# IRC LLM 추론 가속 — 작업 기록 (Worklog)

[결과·근거 본문](2026-11-01_llm_inference_acceleration.md) · [근거 파일](2026-11-01_accel_evidence/) · [서버 실행](../tools/vllm_server/start_vllm_server.sh)
→ **같은 모델·어댑터·프롬프트로 응답 시간을 줄인다. 의미 정확도(데이터·프롬프트·학습)는 범위 밖이다.**

- 기간: 2026-11-01 KST, 브랜치 `llm_final_2`(커밋하지 않음)
- 담당: Claude = **속도 최적화**. 데이터·프롬프트·재학습은 사용자와 Codex가 따로 진행한다(본문 §0).
- 모든 기술 판단은 Codex(`gpt-5.6-sol`)로 교차검증했다. 원문은 `training/learning_first_v43/final/goals/GL10/accel_prefix_cache_20261101/codex_*.txt`.

## 1. 한눈에 보는 결과

| 단계 | Decision 1회(캐시 적중) | 17턴 시나리오 | 최대 GPU | 상태 |
| --- | ---: | ---: | ---: | --- |
| 시작 전: HF transformers 5.13.1, 캐시 없음 | 4.64초 | 210.8초(당시 프롬프트 기준) | 13.5GB | 기준 |
| (중간) HF 5.13.1 + 접두 캐시 | 1.43초 | 99.5~103.0초 | 21.8GB | 환경 교체 전 |
| A: HF + 접두 캐시 + 새 환경 `soomac_llm` | 1.04초 | 71.9초 | 14.1GB | **운영 기본값(`LLM_BACKEND="hf"`)** |
| B: vLLM 서버 + NF4 + 어댑터 변환 | 0.28~0.54초 | 32.8~37.2초 | 서버 15.6GB(0.45 설정) | 선택 백엔드(`"vllm"`), 실전 로봇 시험 대기 |

VLM 판정 1회: HF 2.53초 → vLLM 1.56초(실제 판정 프롬프트 8종, 판정 8/8 동일).

## 2. 시간순 작업

| # | 작업 | 결과 | 본문 |
| --- | --- | --- | --- |
| 1 | 병목 조사: 로그 지연 분해, 토크나이저로 고정 앞부분 측정, SDPA·Linear 마이크로벤치 | 입력의 85~97%가 매번 같은 system prompt. FlashAttention은 head 512 때문에 SM120에서 불가, 효과 상한 약 20% | §3~4 |
| 2 | 웹 조사 3갈래(vLLM·SGLang / 어텐션 커널 / 양자화) | vLLM 후보, k_eq_v LoRA 불일치 발견, FP8은 사용자 결정으로 제외 | §4 |
| 3 | **접두 KV 캐시 구현**(`prefix_cache.py`), D/Q/P 호출부 연결, 기본 켜짐(사용자 결정) | 로그 재생 4.42 → 2.05초 | §5~6 |
| 4 | 출력 차이 진단(logits 비교) | 버그가 아니라 bf16/NF4 계산 순서 차이. 거의 동점인 지점에서 뒤집힘 | §7 |
| 5 | 17턴 시나리오 OFF/ON(기존 하네스 그대로) | OFF = 기존 결과 17턴 동일. 필수 시나리오 실패는 캐시와 무관(데이터 담당에 전달). ON 고유 차이 1건(분리 T8) | §8 |
| 6 | transformers 버그 발견: 5.13.1 `layer_types[:-0]` → 슬라이딩 층 KV가 잘리지 않음 | 5.14.0에서 수정됨 확인(PR #47118). 하지만 업그레이드만으로는 메모리가 안 줄어듦(view 문제) | §9, §14 |
| 7 | 외부 분석(vLLM, transformers 업그레이드) 팩트체크 | "업그레이드만 하면 됨"은 틀림 → 업그레이드 + 저장 직후 clone이 필요(1.97 → 0.44GiB) | §14 |
| 8 | PyTorch 2.13~2.15 팩트체크, 환경 충돌 조사 | torch 업그레이드의 자동 가속 근거 없음. ROS Humble 때문에 LLM 노드는 Python 3.10 고정 → vLLM은 별도 서버 필수 | §15 |
| 9 | 디스크 정리(사용자 지시) | 휴지통, `~/sapiens2_host`, `Desktop/Qwen3-14B`, `Desktop/qwen3_backup`, openvla `.git`의 Codex 스냅샷 삭제 → 여유 37 → 198GB | 아래 §5 |
| 10 | **A: 새 환경 `soomac_llm` 신규 설치**(복제본 `gemma4_env_v2`는 사용자 지시로 삭제) | 패키지 150개, `pip check` 0건, 테스트 통과 | §16 |
| 11 | clone 1차 실패 → 원인: `inference_mode`에서 view의 `_base`가 기록되지 않음 → "저장소 > 내용"으로 판정 변경 | Decision 캐시 1.97 → 0.445GiB | §16.2 |
| 12 | 같은 코드로 기존 환경 vs `soomac_llm` 비교 | 17턴 103.0 → 71.9초, 최대 VRAM 21.8 → 14.1GB, 새 의미 회귀 없음 | §16.3 |
| 13 | `.bashrc` `soomac_llm()` 함수를 새 환경으로 전환(사용자 승인) | 백업 `~/.bashrc.bak_20261101_llm_env` | §16.4 |
| 14 | VLM 지연 재현 측정 | 판정 1회 2.90 → 2.42초, 시간 대부분이 판정 문장 생성 | §17 |
| 15 | **B1~B2: vLLM 환경 `soomac_vllm_server` 설치, 어댑터 변환** | k_proj LoRA를 v_proj로 복제한 변환본(원본 불변) | §18 |
| 16 | B3: BF16으로 HF vs vLLM | vLLM LoRA 정상(11/14). **진짜 원인은 NF4 → BF16**(HF끼리도 2/14) | §18.3 |
| 17 | B3b: vLLM + NF4(bnb 플러그인) | HF-NF4와 12/14(Decision·Recommendation 전부 일치), 실제 조건 Decision 0.28~0.54초 | §18.4 |
| 18 | **B4: 서버 스크립트, `vllm_client.py`, 노드 `LLM_BACKEND` 스위치** | 오프라인(네트워크 격리) 동작 확인. 서버 기본 조각 계산 2,496 → 12,288로 수정 | §19 |
| 19 | 17턴 2회 + VLM 실제 이미지 비교 | 32.8~33.5초. 1회차 T11 오류(토마토), 2회차 17/17 동일 → vLLM은 실행마다 완전히 같지 않음 | §19.2 |
| 20 | Codex 지적 코드 결함 2건 수정 | 서버 실제 어댑터 경로로 대조, `token_ids` 누락 시 오류 | §19.2 |
| 21 | `.bashrc`에 `soomac_vllm_server()` 추가(사용자 승인) | 백업 `~/.bashrc.bak_20261101_vllm_fn` | §19.3 |
| 22 | **GPU 예산(상한 27GiB, 사용자 지시)**: STT·TTS 실측, 서버 설정 6종 시험 | 기본값 0.45 / 동시 2 / 최대 10,240 → 약 26.0~26.5GiB | §20.1~20.2 |
| 23 | 출력 한도를 서버 남은 길이에 맞춰 자동 축소 | 입력이 서버 한도를 넘으면 명확한 오류 | §20.2 |
| 24 | NVFP4·MTP 조사 | MTP × LoRA는 vLLM 공식 미지원. NVFP4는 FP4 연산 지원되지만 어댑터 품질 위험 | §20.3 |
| 25 | GPT·외부 조사 팩트체크(LoRA+MTP 이슈 #62340, LMCache 레시피) | 충돌 수정 PR이 v0.32.0에 포함, `--enforce-eager` 원인도 0.32에서 수정됨 | §21.1 |
| 26 | MTP 보조 모델 다운로드(`/home/roma/models/gemma-4-12B-it-assistant`, 838MB) | 인터넷 1회 사용 | §21.1 |
| 27 | **MTP·결과 고정 실험 A~D**(메모리 비율 0.55 고정) | MTP 1차 실패(보조 모델이 bitsandbytes 로더 상속) → `draft_load_config auto`로 해결. MTP 이득 3~6%, 적중률 76%. 결과 고정은 반복 17/17 일치, +5~26% 느림 | §21.3 |
| 28 | 결론 | MTP 미채택, 결과 고정은 선택 옵션, 기본 설정 유지 | §21.4 |
| 29 | 결과 고정을 운영 설정(0.45 + STT·TTS 상주)에서 확인 | 기동 성공, 25.8GiB(+추론 시 약 26.4GiB), 17턴 반복 17/17, 36.0초 → 로봇 시험용 권장 | §22.2 |
| 30 | NVFP4 체크포인트 2종 다운로드 시작, 서버 스크립트 `VLLM_QUANT=auto` 옵션 추가 | xet 전송 멈춤 → 일반 HTTP로 재시작(약 3~4MB/s) | §22.4 |
| 31 | **결과 고정을 서버 기본값으로 켬**(사용자 결정), `.bashrc` 주석 갱신 | 끄기: `VLLM_BATCH_INVARIANT=0 soomac_vllm_server` | §22.2 |
| 32 | NVFP4 LoRA 재학습 경로·어댑터 켜고 끄기 정리, GPT 의견 2회 팩트체크 | ModelOpt NVFP4 QLoRA 공식 예제 확인(Qwen3-8B). 재학습은 품질이 떨어질 때만 | §23 |
| 33 | 시험 대상을 RedHatAI W4A4로 확정, coolthor 제외 | RedHatAI는 인증 조직이고 KV는 bf16. coolthor는 개인 배포라 다운로드를 중단하고 삭제 | §23.5 |
| 34 | **사고**: NVFP4 시험 자동 시작 → FlashInfer FP4 커널 JIT 컴파일(동시 34개)이 RAM 62GB를 고갈 → 08:23 OOM으로 로그인 세션 종료(Codex 작업 포함) | `.done`을 aborted로 기록. 재시도는 메모리 상한(systemd-run) + `MAX_JOBS=2` + 사용자 승인 후 | §23.6 |
| 35 | **방법 C**: 입력을 서빙 한도(10,240 − 출력)에 맞춤. 넘으면 오래된 대화부터 뺌, 시작 시 system prompt 검사, 서버 길이 fail-closed | 한도 이하 입력은 불변. Decision 프롬프트 여유 259토큰. Codex 2회 검토 반영. 테스트 929개 통과. GPU 17턴 동일성은 Codex 학습 종료 후 | §24 |
| 36 | 계획 변경(사용자): 다른 어댑터 학습 먼저. 학습 후 확인 도구 작성: 변환 도구 이동, 17턴 3방식 실행·비교, 한 번에 실행하는 스크립트(단계별 RAM 상한), 노드의 어댑터 해시 검사 | GPU 실행 없음. CPU 확인 통과(변환 바이트 동일, 비교 도구 거짓 통과·거짓 실패 없음). Codex 2회 반영. 테스트 930개 | §25 |
| 37 | 사용자 질문: 이전 vLLM 측정 때 RAM은? → **기록 없음**(GPU만 기록). 검사 도구에 RAM·OOM 안전선 추가: 묶음 전체 상한 44G + 단계별 상한, 1초 RAM 기록, vLLM 단계에 STT·TTS 상주, 보고서 판정 | GPU 실행 없음(Codex Decision profile 컨테이너 build04가 GPU 사용 중). CPU 시험·Codex 2회 반영 | §25.5 |
| 38 | 팀원용 정리(사용자 요청, NVFP4는 시간상 보류): 속도 변천사 문서 [md](2026-11-01_llm_inference_speed_history.md) + 원리 그림·최종 비교 차트 [html](2026-11-01_llm_inference_speed_history.html). 라이브러리 버전 팩트체크 절 포함 | Codex 검수 반대 8건(설정 혼합 수치, VLM 합성/실제 혼합, 80~97%, Python ≥3.11, 입력/출력 한도 표기, 실행 시간, 추정 표시) 전부 반영. worklog 결과 고정 기본값 표기 충돌 수정 | 이 표 |

## 3. 바꾼 파일

**저장소 코드 (미커밋)**

| 파일 | 변경 | 기존 동작 영향 |
| --- | --- | --- |
| `src/soomac_irc/soomac_irc/prefix_cache.py` (신규) | Agent별 접두 KV 캐시, 저장 직후 clone | `PREFIX_CACHE_ENABLED=False`면 종전과 동일 |
| `decision_model.py` · `recommendation_model.py` · `response_model.py` | `model.generate` → `prefix_cache.generate`, trace에 캐시 상태 기록 | 프롬프트·문법·출력 처리 불변 |
| `src/soomac_irc/soomac_irc/vllm_client.py` (신규) | vLLM 서버 클라이언트, `VllmModel` 대리 객체, 원격 VLM, 어댑터·길이 검사 | `LLM_BACKEND="vllm"`일 때만 사용 |
| `llm_langgraph_node.py` | `LLM_BACKEND` 상수, 모델 로드·VLM 연결 2곳 분기 | 기본 `"hf"` = 기존과 동일. 다른 세션의 미커밋 변경이 같은 파일에 함께 있음 |
| `src/soomac_irc/test/test_vllm_backend.py` (신규) | 서버 없이 도는 단위 테스트 6개 | 전체 919개 통과 |
| `src/soomac_irc/soomac_irc/input_budget.py` · `test_input_budget.py` (신규) | 입력 길이 한도·자르기·시작 검사(방법 C). Decision 상한 16,384 → 9,216 | 한도 이하면 입력 불변. 전체 929개 통과 |
| `tools/vllm_server/{check_after_training.sh,convert_adapters_for_vllm.py,run_scenario.py,compare_scenario_runs.py}` (신규) | 학습 후 확인 도구(§25). `vllm_client.py`에 어댑터 원본 해시 검사 추가 | GPU를 쓰는 실행은 사용자가 학습 후 직접 |
| `tools/vllm_server/start_vllm_server.sh` (신규) | 서버 실행(NF4, LoRA 3개, 오프라인, 0.45/2/10240). 옵션 `VLLM_SPEC_MODEL`(MTP, 기본 꺼짐), `VLLM_BATCH_INVARIANT`(결과 고정, **기본 켬**, 사용자 결정) | — |
| `docs/2026-11-01_llm_inference_acceleration.md`, 이 worklog, `docs/2026-11-01_accel_evidence/`(63개) | 결과·근거·스크립트 | — |

`model_runtime.py`, 프롬프트(`agent_prompts.py`·`prompt_examples.py`), 그래프·정책은 **수정하지 않았다**(데이터 담당 하네스가 해시로 고정 감시).

**저장소 밖 / git 제외**

| 대상 | 내용 |
| --- | --- |
| conda `soomac_llm`(5.5GB) | Python 3.10.20, torch 2.12.1+cu130, transformers 5.14.1, peft 0.21.0, bitsandbytes 0.50.2, xgrammar 0.2.8. LLM 노드용 |
| conda `soomac_vllm_server`(8.6GB) | Python 3.12, vllm 0.32.0, torch 2.13.0, transformers 5.19.0, vllm-bnb-plugin 0.0.3. 서버용 |
| conda `gemma4_env` | **변경 없음**(STT 노드가 계속 사용) |
| `models/vllm_adapters/{decision,recommendation,response}` | vLLM용 변환 어댑터 + `CONVERSION.json`(git 제외) |
| `~/.bashrc` | `soomac_llm()` → `soomac_llm` 환경, `soomac_vllm_server()` 추가. 백업 2개 |
| `training/.../GL10/accel_prefix_cache_20261101/` | 시나리오 원본 trace, Codex 원문(git 제외) |
| `~/.claude/.../memory/` | 역할 분담, Codex 모델 우회 메모 |

## 4. 사용자 결정 기록

1. FP8은 가속 후보에서 제외(이전 시험에서 더 느렸음)
2. 접두 캐시 운영 켜 둠
3. 복제 대신 새로 설치한 환경으로 가고, 복제본은 삭제
4. vLLM 환경 이름은 `soomac_vllm_server`(나중에 다른 LLM 서버로도 쓸 수 있게)
5. A·B 둘 다 진행. 어댑터 재학습은 데이터 검수 뒤 Agent별로 따로(원격 GPU 포함). NF4(QLoRA) 유지
6. GPU 전체 상한 27GiB
7. vLLM 서버는 결과 고정(Batch Invariant) 기본 켬

## 5. 디스크 정리 기록 (사용자 지시)

| 삭제 | 크기 | 백업·복구 |
| --- | ---: | --- |
| 휴지통 | 약 23GB(실제 확보는 미미, 하드링크로 추정) | — |
| `~/sapiens2_host` | 27GB | `Desktop/backup_tar`의 sapiens tar 4개(→ NAS) |
| `Desktop/Qwen3-14B` | 28GB | `Qwen3-14B.tar.gz`(→ NAS), HF 공개 모델 |
| `Desktop/qwen3_backup`(Qwen3-VL 4B·8B FP8) | 18GB | tar 없음. HF 공개 모델로 재다운로드 |
| openvla `.git`의 Codex 스냅샷 ref | 약 66GB | 데이터셋은 HF `romalab-cbf/two_block_pnp`와 1000에피소드·171파일 일치 확인, tar는 Docker 이미지 존재 |

`IRC_SOOMAC3.0/.git`(13GB)에도 Codex 스냅샷 ref 6개가 있다. 다른 세션 작업 중이라 손대지 않았다. 예전 `ros2_ws` 코드 일부가 삭제한 Qwen 경로를 참조한다.

## 6. 작업 중 실수와 정정

- 공유 작업트리에서 `git stash`/`pop` 1회 실행 → 다른 세션 변경 13개 파일이 충돌 없이 복원됨을 확인. 이후 사용하지 않는다.
- "qwen3_backup = Qwen3-14B 사본"이라고 잘못 안내 → 실제는 Qwen3-VL 2종. 삭제 전에 확인해 정정했다.
- "torchaudio 버전 충돌"이라고 잘못 안내 → torchaudio 2.12는 존재하지 않아 정상 조합.
- "vLLM bnb + LoRA 미지원"이라고 잘못 안내 → 공식 예제 있음(Codex 정정).
- 메모리 표를 "GB"로 표기했지만 실제는 nvidia-smi MiB(Codex 정정, 본문 §20.2에 GiB 병기).
- `pkill -f`가 자기 명령줄까지 잡아 시험 스크립트가 중단됨 → PID로 정리하고 스크립트를 새로 작성했다.
- `~/.codex/config.toml`의 `gpt-6.1-sol`은 계정이 거부해 모든 교차검증을 `-m gpt-5.6-sol`로 실행했다(CLAUDE.md 기재와 다름).

- **NVFP4 시험 자동 시작으로 FlashInfer JIT 컴파일이 RAM을 고갈시켜 로그인 세션 종료(08:23).** 원인: `MAX_JOBS`를 설정하지 않음 + 메모리 상한 없이 자동 시작. 이후 무거운 작업은 메모리 상한 안에서, 승인 후에만 실행한다(본문 §23.6).

## 7. 되돌리기

| 되돌릴 것 | 방법 |
| --- | --- |
| 접두 캐시 | `prefix_cache.py`의 `PREFIX_CACHE_ENABLED = False` |
| vLLM 백엔드 | `llm_langgraph_node.py`의 `LLM_BACKEND = "hf"`(현재 기본값) |
| LLM 노드 환경 | `cp ~/.bashrc.bak_20261101_llm_env ~/.bashrc` 또는 `soomac_llm()`의 경로를 `gemma4_env`로 |
| 서버 함수 | `cp ~/.bashrc.bak_20261101_vllm_fn ~/.bashrc` |

## 8. 남은 일

1. **로봇 스택 실전 시험**(사용자): `soomac_vllm_server` → `LLM_BACKEND="vllm"` → `soomac_llm`. nvtop 전체 메모리(카메라·비전 포함), 대화·VLM 동작 확인
2. vLLM 비결정성 관찰: 같은 시나리오 반복 시 갈리는 턴 기록
3. 입력 길이: 방법 C로 해결(§24). 남은 것: GPU가 비면 HF 17턴 동일성 확인. Decision 프롬프트 여유 259토큰(데이터 담당 전달)
4. 선택: NVFP4 + 기존 어댑터 14건 비교(약 30분). 크게 다르면 재학습 판단은 데이터 담당
4-2. 데이터 담당 전달: 소스가 빈 상태의 "바로 담아"에서 소스('토마토') 지어내기(§21.3)
5. 어댑터 재학습 후: `bash tools/vllm_server/check_after_training.sh --preflight` → 통과하면 같은 명령을 옵션 없이 실행(변환·HF/vLLM 17턴·C 동일성 비교를 한 번에, §25)
6. 커밋은 하지 않았다. 다른 세션 변경과 섞여 있어 사용자 판단으로 나눠서 커밋해야 한다
