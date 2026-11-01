# Claude NVFP4 재시험 → 로컬 두 Agent 학습 연결 — build03

> **현재 계획 변경:** 사용자가 NVFP4 완료 대기를 철회하고 Response 추가 의미검수 후 Response→Recommendation 순차 학습으로 변경했다. 옛 NVFP4 waiting PID679304는 정확한 명령과 자식0을 확인한 뒤 종료했다. 아래 명령은 과거 기록이며 다시 실행하지 않는다. build03 원격 실행의 모델 메타데이터 불일치를 수정한 [build04 제작 기록](2026-11-01_decision_build04_processor_fix.md)을 우선한다. 새 로컬 직접 승인 launcher는 별도 출력·명시적 승인 플래그를 쓰며 실제 GPU guard와 순차 실행·복원 검증은 유지한다. 현재 원격 이미지 검증이 GPU를 사용하므로 로컬 본 학습은 아직 미시작이다.

[제작·실제 검증](2026-11-01_decision_build03_assets_fix.md) · [시간순 worklog](2026-11-01_training_package_build_worklog.md)
→ 현재 사용할 버전은 build03이며, 새 NVFP4 재시험 종료·GPU 반환 전 본 학습은 금지다.

## 1. Decision TAR — 실제 완료

- 파일: `/home/roma/Downloads/soomac-decision-r5-compound-semantics-20261101-build03.tar`
- 크기: 6,991,625,216bytes. SHA256: `4ded2e17fd4064e1c84e003ae76c9320fc522ed352ddd0bb15a69268df929e05`.
- 이미지: `soomac/decision-r5:20261101-build03`, ID `sha256:b312009fef020df9390e44820457b2859180bc448dbdf5a4a98ebb42ff402647`.
- 실제 assets22파일 hash/일반파일/상위 링크 검사, UID1000 preflight, CPU81건, 새 GPU profile→save5/fail→resume5→10/fail→resume10/finalize, 52layer 해시와 실제 Docker load가 모두 통과했다. 운영 adapter나 source 데이터는 바꾸지 않았다.
- 확인된 서버 오류의 경로를 `/app/assets/dataset`으로 교정했다. **실제 서버 재업로드·접수 결과는 아직 미확인**이며 root가 원격 업로드/실행을 하지 않았다.

build02는 잘못된 `/app/dataset` 내장 선언으로 서버가 거절했다. 이전 Docker/복원 시험 통과를 서버 규격 통과로 잘못 확대했던 설명은 철회했다. build02 파일·영수증은 역사적 증거로 보존하되 사용하지 않는다. 옛 waiting PID582218/session13153은 root가 종료했고 실제 exit143을 확인했다. 아래는 모두 새 build03 경로다.

## 2. 불변 릴리즈·완료 영수증

```text
/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/releases/conversation_r5_20261101_build03
RELEASE SHA256: e213d6188fc71739051c4d04f8acc419bc5727fc326d12c0f46c2707aeb4284c

/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/runs/r5_package_test_20261101_build03/DECISION_READY.json
```

이 새 Decision 영수증은 실제 존재한다. 원본 dataset manifest SHA `7850948287237c5ace9d1b0f56931419760fbce3f0afa02cea40681acb6ceffa`와 model pins/common/local 코드는 유지됐다. 바뀐 봉인 실행 파일은 remote packaging5개뿐이다.

## 3. 로컬 실제 상태 — 2026-11-01 14:34 KST 관측

새 launcher session12238/PID679304는 실제 running으로 확인됐지만 **학습 중이 아니라 waiting**이다. `pipeline_state.json`은 P/Q 모두 pending, 자식 프로세스0, GPU compute 목록도 비어 있었다. 이 관측을 새 NVFP4 시험 종료 증거로 쓰지 않는다. Goal UI의 외부 NVFP4 의존성 blocked와 실제 대기 프로세스는 별개다.

새 gate는 아래 경로이며 현재 미존재다. 약20:34 KST까지21600초 대기하고, 신호가 없으면 종료하며 자동 재시도하지 않는다. PID·session은 관측 기록이므로 나중에 살아 있다고 가정하지 말고 실제로 확인한다.

```text
/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/runs/r5_package_test_20261101_build03/LOCAL_READY_AFTER_NVFP4_RETEST.json
```

현재 실행 중인 명령(중복 실행 금지):

```bash
/home/roma/miniconda3/envs/gemma4_env/bin/python -B \
  /home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/releases/conversation_r5_20261101_build03/local/launch.py \
  --bundle /home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/releases/conversation_r5_20261101_build03 \
  --model /home/roma/Desktop/sLLM/gemma-4-12B-it \
  --output /home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/runs/r5_local_after_nvfp4_20261101_build03 \
  --mode train \
  --gate-receipt /home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/runs/r5_package_test_20261101_build03/LOCAL_READY_AFTER_NVFP4_RETEST.json \
  --gate-timeout-seconds 21600 --gate-poll-seconds 10
```

이 명령은 선행 shell 변수를 요구하지 않는다. timeout/장애 후 재개할 때만 기존 프로세스 종료를 확인하고 같은 출력으로 실행한다. OOM/손상 시 다음 Agent를 실행하지 않고 정상 commit checkpoint를 보존한다. `--resume auto`는 동일 release/data/config만 허용하며 이전 데이터 optimizer나 불완전 checkpoint로 fresh fallback하지 않는다.

## 4. Claude가 새 재시험을 실제 끝냈을 때만

이전 `training/learning_first_v43/gpu_handoff/CLAUDE_NVFP4_TEST.done`은12:55 기록이지만 내용은08:23 `aborted_system_oom`/시험 미완료다. **이를 새 재시험 완료로 승격하지 않는다.** 마지막 확인에서 이 PC의 NVFP4/vLLM/컴파일 프로세스와 localhost8000 서비스는 없었다. 다른 창/외부 PC의 후속 시험은 새 증거로 확인해야 한다.

Claude 담당자가 실제 재시험 종료와 GPU 반환을 확인한 뒤에만 아래 helper를 명시적으로 호출한다. 성공/실패 결과와 시험 종료는 다르며, GPU가 잠깐 비었다고 종료를 창작하지 않는다. root는 NVFP4 재실행·가속·VLM 코드를 변경하지 않는다.

Helper: `releases/conversation_r5_20261101_build03/workflow_receipt.py` (아래는 호출 형식 템플릿이며 실제 보고서 경로가 없으므로 현재 실행 금지).

```text
python3 /home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/releases/conversation_r5_20261101_build03/workflow_receipt.py
  --decision-receipt /home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/runs/r5_package_test_20261101_build03/DECISION_READY.json
  --claude-report <이번에 실제 종료한 재시험 보고서의 절대 경로>
  --output /home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/runs/r5_package_test_20261101_build03/LOCAL_READY_AFTER_NVFP4_RETEST.json
  --nvfp4-test-ended-and-gpu-returned
```

이 helper 호출은 운영자의 명시적 attestation이다. 실제 TAR 해시·실재 보고서를 결속하고 신호를 덮어쓰지 않는다. 완료 파일에 과거 보고서·placeholder·원래 중단 기록을 넣지 않는다. 새 신호가 생겨도 launcher가 시각/identity/해시·실제 GPU free/compute 목록·GPU lock을 모두 확인해야 Response를 시작하고, 실제 train 완료 검증 뒤에만 Recommendation을 시작한다. 대기 중에는 GPU lock/모델을 점유하지 않는다.

NVFP4 결과를 NF4/BF16 학습 설정으로 자동 복사하지 않는다. 로컬 학습 환경은 검증된 gemma4_env이며 추론 환경 migration을 학습 환경 변경으로 혼동하지 않는다. 원격 실행/운영 adapter 교체/ROS/자동 commit·push는 승인 범위 밖이다.

Cause: 패키지 수정과 외부 GPU 시험·학습을 서로 다른 버전/신호로 연결하면 조기 실행이나 잘못된 자료 채택이 생긴다.

Effect: build03의 실제 TAR/릴리즈와 새 재시험 신호만 사용하고, 본 학습은 조건을 모두 만족할 때만 순차 실행한다.
