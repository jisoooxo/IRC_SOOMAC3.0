# 최신 데이터 전용 학습 패키지 제작·검수 계획

[작업 기록](2026-11-01_dataset_rebuild_worklog.md) · [설정 점검](2026-11-01_training_parameter_check.md)
→ Decision 원격용 TAR·최적화/복원 시험을 먼저 완료하고, Claude NVFP4 시험 종료와 GPU 여유가 확인된 뒤 로컬 Response→Recommendation 본 학습을 순차 자동 실행한다.

## 0. 최신 실행 순서·승인 갱신 (2026-11-01)

사용자 최신 지시가 아래의 과거 ‘본 학습 금지/제작만’ 범위보다 우선한다. 원격 대상은 사용자가 **Decision으로 확정**했다. 이제 세 Agent의 총시간 비교로 원격 대상을 다시 고르는 작업은 하지 않는다. 이는 원격 총시간 실측을 확보했다는 뜻은 아니다.

1. 최신 고정 데이터로 Decision 원격 학습 이미지와 TAR를 제작하고, 유한한 최적화 profile 및 승인된 짧은 GPU 저장→의도적 실패→재개 시험을 완료한다. 기본 CMD는 실학습용으로 하며 시험용 TAR와 혼동하지 않는다.
2. 결과와 검증 영수증을 남기고 GPU를 반환한다. 이후 **Claude가 NVFP4 시험을 담당**한다. 이 작업자는 NVFP4·추론 Runtime·캐시·VLM 코드에 관여하거나 그 프로세스를 중단하지 않는다.
3. Claude NVFP4 시험의 **명시적 완료 신호**와 GPU 여유를 모두 확인한다. GPU가 일시적으로 비었다는 이유로 시험 종료를 추측하지 않는다. 결과의 성공/실패와 ‘시험 종료·GPU 반환’은 구분한다. 완료 신호 인터페이스와 경로는 실제 구현 뒤 기록한다(현재 미구현).
4. 위 조건과 학습 preflight를 통과하면 로컬 **Response→Recommendation**을 별도 프로세스로 순차 학습한다. 최신 데이터/학습 환경·모델 경로를 검증하며, 추론용 NVFP4 결과를 학습 설정에 자동 적용하지 않는다.
5. 첫 Agent 검증 완료 후에만 다음 Agent를 시작한다. OOM/저장 손상/설정 불일치면 중단하고 마지막 정상 체크포인트를 보존한다. 사용자 재개 시 같은 릴리즈의 완료 체크포인트에서 복원한다. 무한 자동 재시도·자동 batch 변경·동시 학습은 하지 않는다.

**현재 상태:** Goal active. 순서와 조건은 기록 완료. common 런타임·완료 신호 연동·실제 GPU 시험·이미지/TAR 완성·자동 학습은 아직 미완료이다. ‘학습 자동 이어가기’ 승인은 위 순서 충족 후의 로컬 두 Agent에 한정된다. 원격 업로드/실행, 기존 adapter 교체/삭제, ROS, commit/push는 승인하지 않았다.

Cause: 사용자가 Decision 원격용과 Claude 시험 이후의 로컬 자동 학습 순서를 명확히 지정했다.

Effect: GPU 사용 순서를 고정하고, 시험 종료 확인 없는 조기 학습과 동일 GPU 동시 점유를 방지하는 조건을 구현·검수 범위에 추가한다.

## 1. 고정 입력과 승인 범위

사용자 지정 규격: `/home/roma/Downloads/TRAINING_IMAGE_SPEC.ko_v2_9 (2).md`, 1002행, SHA256 `3754ca51cf391bc7cde7684f4a992f57fdf1c87f558004745886165922a44af5`를 전체 읽었다. 이전 `(1)` 사본으로 대신하지 않는다.

**오직 다음 최신 데이터만 사용한다.**

```text
/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/candidates/compound_semantics_20261101_r2/dataset
MANIFEST SHA256: 7850948287237c5ace9d1b0f56931419760fbce3f0afa02cea40681acb6ceffa
```

기존 r3/r4/old TAR의 데이터는 학습 소스로 쓰지 않는다. 모든 실행/복원 영수증을 위 원본 hash와 선택한 Agent 파일 hash에 결속한다. 원 후보의 false/pending manifest는 보존한다. 사용자 승인과 의미 채택 기록을 별도 새 학습 릴리즈에 연결하며 옛 manifest를 덮어써 통과시키지 않는다.

생성 코드의 정본 위치는 `training/learning_first_v43/training_release_20261101_r5/`다. 병렬 수정 소유권은 `remote/`와 `local/`을 분리하고 root가 `common/`·실행 계약·최신 데이터 릴리즈·통합 테스트를 소유한다. 기존 Runtime/prompt/cache/VLM/ROS/운영 adapter와 다른 작업자 변경은 수정하지 않는다.

최초 승인에서는 본 학습을 금지하고 **짧은 GPU 저장→의도적 실패→재개 시험만 허용**했다. 최신 §0에 따라 로컬 두 Agent 본 학습은 Claude 시험 종료·GPU 반환·검증을 조건으로 추가 승인됐다. 그 전에는 폐기용 LoRA와 신규 시험 출력만 사용한다. 원격 접속/업로드/학습, 기존 adapter/checkpoint 삭제/교체, 자동 commit/push는 하지 않는다.

## 2. 병렬 작업과 실행 계약

원격 역할은 Decision 전용 Dockerfile/입력 선언/entrypoint/빌드·CPU 검사·TAR 생성 코드를 작성한다. 로컬 역할은 Response 성공 뒤에만 Recommendation을 별도 프로세스로 실행하는 순차 launcher와 tests를 작성한다. 둘 다 root가 제공하는 공통 CLI를 사용하고 공통 파일을 수정하지 않는다.

공통 CLI 예정 계약:

```text
python common/train_agent.py --mode preflight|profile|train|smoke \
  --agent decision|response|recommendation --dataset PATH --model PATH \
  --output PATH --release PATH --budget-bytes INTEGER \
  [--batch auto|1|2|4|8] [--resume auto|none|PATH]
```

`--mode preflight`는 모델/GPU/optimizer를 생성하지 않는다. `train`은 제작·검증·Claude NVFP4 시험 종료와 GPU 반환 전에는 호출하지 않는다. `smoke`는 별도 smoke 출력에서만 짧게 수행한다. `profile`은 저장하지 않을 probe LoRA로 유한한 batch 후보를 검사한다. 호출자의 환경 `MODEL_DIR/DATASET_DIR/CHECKPOINT_DIR`과 서버 `LAB_RESUME/RESUME_CHECKPOINT_DIR/RESUME_STEP`을 처리해야 한다. 최종 인터페이스 변경은 한 곳에 기록하고 병렬 작업자에게 전달한다.

원격 요청 예산은44000000000bytes, 로컬 기본 요청 예산은31000000000bytes이며 실제 GPU/서버 제한을 우회하지 않는다. batch 자동 선택은 긴 입력·실제 역전파/optimizer를 포함한 유한 프로파일 뒤에만 기록한다. microbatch×accumulation16을 비교 기준으로 쓰되 수치/손실 정규화의 완전 동등성은 별도 검증한다. profile 실패 시 더 큰 batch/OOM 무한 반복 금지.

## 3. 저장·OOM·재개 계약

완료 체크포인트에는 adapter, optimizer, scheduler, Trainer/global step, RNG, 학습 설정·data/model/source 지문을 보존한다. 저장 완료와 필수 파일·내용 검증 이후에만 `.lab-monitor/resume.json`을 원자적으로 게시한다. 중간 저장 실패는 직전 완료 marker와 체크포인트를 보존한다.

OOM에서는 미완료 step이나 깨진 optimizer 상태를 즉석 저장해 정상 checkpoint로 게시하지 않는다. 마지막 저장 완료 지점만 재개한다. 명시적인 재개 실행으로 model/optimizer/scheduler/진행 위치를 복원하며 파일 누락/손상·다른 Agent/데이터/설정·step 불일치면 오류 종료한다. 조용한 fresh fallback이나 무한 재시도는 금지한다. 저장 이전까지의 미저장 작업 손실과 최초 완료 checkpoint 이전 OOM은 복구를 보장하지 않는다.

로컬 launcher는 P의 검증된 완료 영수증이 있을 때만 Q로 넘어간다. P 실패/OOM이면 멈추고 재개 위치를 알린다. 재실행 시 완료한 P를 다시 학습하지 않으며 미완료 Agent의 저장 완료 checkpoint를 사용한다. 두 학습 프로세스를 동시에 띄우지 않는다.

## 4. 규격과 검수 완료 조건

linux/amd64, lab spec2.9/progress1/resume output-dir-v1/inputs 라벨, UID1000 읽기·실행/출력 쓰기, readonly model/data, 서버 env 경로, HF CPU import, 초기에 progress 생성/10초 heartbeat, 실제 optimizer/sample 계수, 실패 nonzero/exec 전달을 확인한다. 모델·데이터·환경 비밀과 토큰을 이미지에 넣지 않는다. dataset은 최신 데이터 실제 파일을 내장할 수 있으며 base model은 서버 입력으로 받는다.

CPU 테스트는 최신 데이터 hash 고정·부정확한 입력 거부·순차 실행/실패 차단·완료 건너뛰기·resume path traversal/symlink/손상/step/data mismatch·저장 중단 시 이전 marker 보존을 다룬다. mock/CPU 테스트를 실제 12B GPU 복원 성공으로 세지 않는다.

승인된 GPU smoke는 신규 출력에서 실제 모델로 저장→실패→재개, optimizer/scheduler/global step 복원·새 checkpoint 생성과 초기 출력 불변을 확인한다. 빌드한 이미지의 실제 UID1000·기본 entrypoint/CMD로도 확인해야 규격 검증 완료로 보고한다. Docker/GPU 접근 등 실행 제약이 남으면 실제 미완료 항목을 명시하며 TAR를 검증 완료라고 제출하지 않는다. 본 학습용 기본 CMD와 smoke 시험 전용 설정을 혼동하지 않는다.

독립 Codex sol/high 코드·실제 테스트 로그 대조 뒤 결과/명령/출력 경로/SHA와 남은 한계를 기록한다. 추가 모델/계정으로 사용량·인증 제한을 우회하지 않는다.
