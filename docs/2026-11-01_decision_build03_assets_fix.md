# Decision build03 — 서버 내장 데이터 경로 교정

## 최종 실제 결과 — 2026-11-01 14:34 KST

**수정 TAR 제작과 로컬 검증 완료. 실제 서버 재접수는 아직 미확인이다.**

```text
/home/roma/Downloads/soomac-decision-r5-compound-semantics-20261101-build03.tar
크기: 6,991,625,216 bytes
SHA256: 4ded2e17fd4064e1c84e003ae76c9320fc522ed352ddd0bb15a69268df929e05
```

실제 finalizer session17984 exit0, Docker load session45071 exit0이다. load 뒤 새 image ID b312009f…/linux-amd64/UID1000:1000/CMD train 불변도 확인했다. 새 `runs/r5_package_test_20261101_build03/DECISION_READY.json`은14:33:29 KST에 작성됐다. 이 영수증은 실제 새 GPU/릴리즈/TAR/config/manifest/layer 검증을 결속하며 옛 결과를 승격하지 않았다. 확인된 assets 경로 오류는 소스만 아니라 실제 새 이미지의 파일22개와 TAR config에서도 교정됐다.

검사 범위는 CPU81건, 실제 UID1000 preflight(D4360/309), assets 일반 파일/상위 링크/hash, 새 GPU3 stress probe+10 optimizer smoke/native4종 복원/원본 체크포인트 불변,52 layer hash 및 Docker load다. 실제 서버는 이 작업에서 업로드/실행하지 않았으므로 서버 접수 성공이나 모델 시나리오 품질을 합격으로 주장하지 않는다. 원본 데이터/모델/common/local/운영 adapter/VLM·가속 코드는 바꾸지 않았다.

새 로컬 대기는 실제 session12238/PID679304로 복구했다.14:34:45 KST status=waiting, P/Q pending, 자식0/GPU compute0으로 관측했다. 새 build03 NVFP4 재시험 gate는 미존재이며 본 학습은0회다. 약20:34 KST까지6시간 기다리고 자동 재시도하지 않는다. 명령과 완료 신호는 [새 버전으로 정리한 인계 문서](2026-11-01_nvfp4_to_training_handoff.md)를 따른다. 옛 build02 launcher는 종료했으며 새 프로세스와 혼동하지 않는다. 아래는 구현 과정의 시간순 기록이다.

→ build02의 서버 업로드 규격 위반을 수정하며 데이터·학습 의미론은 바꾸지 않는다.

## 승인과 보존

사용자 `제작 ㄱㄱ` 승인 뒤 진행한다. 원본 dataset manifest SHA는 `7850948287237c5ace9d1b0f56931419760fbce3f0afa02cea40681acb6ceffa` 그대로다. build02 release/image/TAR/checkpoint는 덮어쓰기·삭제하지 않는다. 원격 조작/본 학습/adapter 배포/가속·VLM/commit/push는 하지 않는다. 새 NVFP4 재시험 뒤 로컬 본 학습이라는 조건도 그대로다.

## 최소 수정

1. 실제 원격 입력은 `/app/assets/dataset`에 일반 파일로 COPY한다. label/ENV/entrypoint 기본값/profile 입력도 맞춘다. 공통 로컬/체크포인트 검증기의 canonical bundle 경로를 위해 동일한 `/app/dataset` 복사본도 유지한다. 링크나 학습자료 수정은 아니다.
2. builder는 옛 경로의 잘못된 선언도 release에 봉인돼 있더라도 거부한다. finalizer는 assets 하위 경로·상위 이동 금지·실제 DATASET_DIR 일치를 확인한다. 옛 잘못된 정상 기대값을 수정하고 부정 테스트를 보강한다.
3. build03 release→새 이미지→실제 내장 파일/UID1000 preflight→새 이미지 GPU 저장/실패/복원→새 TAR/config/layer 검증·Docker load 순으로 진행한다. 이전 GPU 시험을 새 image ID에 복사해 통과시키지 않는다.

RustDesk는 사용자의 연결 프로세스라 종료하지 않는다. 폐기용 GPU 시험의 외부 operator에만 명시한 단일 RustDesk PID(실제 exe 일치,1024MiB 이하)를 허용하는 옵션을 추가한다. 다른 compute/컴파일/NVFP4 작업은 여전히 차단한다. 이 PC 시험 예산은30GB로 줄여 남은 메모리를 확보하고 본 학습 guard/학습 batch는 변경하지 않는다. 이 옵션도 CPU 검사한다.

## 현재 상태

build03 불변 릴리즈가 생성됐다. 경로 `/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/releases/conversation_r5_20261101_build03`, RELEASE SHA256 `e213d6188fc71739051c4d04f8acc419bc5727fc326d12c0f46c2707aeb4284c`이다. source manifest는 그대로다. 관련 CPU39건이 통과했다. 기존 자체 waiting launcher PID582218을 실제 argv 확인 후 TERM으로 종료했고 session13153의 실제 exit143을 확인했다. GPU/본 학습 프로세스가 아니라 옛 gate 대기만 종료한 것이다. 새 이미지 빌드는 session에 시작했으며 완료/실제 GPU 검증/최종 TAR는 아직 미완료다.

### 실제 새 이미지 검사

이미지 빌드는 exit0이며 image ID는 `sha256:b312009fef020df9390e44820457b2859180bc448dbdf5a4a98ebb42ff402647`이다. 실제 label/ENV 모두 `/app/assets/dataset`이다. UID1000·읽기 전용 루트/기본모델로 실제 entrypoint preflight가 D train4360/val309에서 exit0이다. 별도 실제 이미지 검사도 일반 파일22개 전체 SHA가 원본과 일치하고 상위 폴더/파일에 symlink가 없음을 확인했다(`REAL_EMBEDDED_FILES_PASS`). 전체 CPU81건도 exit0이다(기존74+신규7, 중복 합산 아님).

새 GPU 시험은 별도 output `runs/r5_package_test_20261101_build03/gpu_verification`에 시작했다. 컨테이너 RAM40GiB/동일 swap 상한/pids256을 추가해 데스크톱을 보호한다. local 시험 VRAM30GB, 타 NVFP4/학습이 있으면 차단한다. 본 학습은 아니다. 아직 결과 전에는 저장/복원 합격·최종 TAR 완성을 주장하지 않는다.

초기 경로 교정 CPU35건 뒤 외부 operator4건을 추가해 관련39건을 통과했다. 위의 최신 실제 관측을 우선한다. 새 결과를 받은 뒤 실제 경로·SHA·관측을 추가한다. 수정 전의 ‘서버 준비 완료’ 표현은 유지하지 않는다.

### 독립 검수

Codex sol/high의 국소 검수는 조건부 찬성이었다(`/tmp/soomac_build03_review.out`). source의 정적 경로 계약은 일치한다고 확인했다. 지적은 외부 GPU operator의 미래 기본값32GB를30GB로 맞출 것,35/39건의 시점 구분을 명시할 것이었다. root가 해당 줄을 확인해 기본값30GB·시점 표기를 교정했다. **현재 실행 중 시험에는 이미 `--test-budget-bytes 30000000000`을 명시했으므로 실제32GB로 실행된 오류는 아니다.** 이미지 안의 동결 학습 코드는 바꾸지 않았다. reviewer가 미확인으로 남긴 실제 이미지 필수 파일·링크 검사도 root가 위22파일 검사를 통해 확인했다. 최종 TAR 실물/실제 새 GPU 결과와 서버 접수는 여전히 각각 별도 관문이다.

### 실제 TAR / profile 후속

임시 내보내기 session14813은 exit0이며 최종 게시 전 `.tar.partial`로 보존한다. 별도 실제 TAR 검증 session59472도 exit0이다.52개 ordered layer 참조/53개 member의 저장·해제 해시와 OCI chain을 통과했다. native manifest `83d56df7e8d52c60d4d2cf3b342182f14e0bfc8dddf24d436e15c7209188ed88`, config `ede63258f916704bc2281c6d9f60129fbaa5da4fe1ac018d3e85c96e2baa985c`이다. source manifest와 model pins는 build02와 같고 바뀐 봉인 code_file5개는 remote/Dockerfile/build_export.py/lab-inputs.json/run_training.sh/test_remote.py뿐이다. common/로컬 학습/모델/데이터는 바뀌지 않았다.

14:11 KST 실제 profile exit0이다. batch1의3가지 stress/reference optimizer 시험 모두 finite/nonzero gradient로 통과했고 reserved peak23,559,405,568bytes였다. batch2는 이 PC30GB 시험 예산의 allocator 한도 때문에 한 번 OOM으로 거부돼 탐색이 종료됐다. batch4/8 무한 재시도나 본 학습 실패가 아니다. remote44GB auto 측정 조건은 그대로다. 실제 새 smoke는 save5_fail 단계로 넘어갔으며, 입력 전체를 변환한 뒤 최장32/8개로 저장·재개를 검사하는 기존 코드 그대로라 CPU 준비 시간이 있다. 이것을 새 데이터 재감사나 중복 실행으로 세지 않는다.

후속 실제 관측: save5_fail은 완료된 step5를 저장한 뒤 예상한 `Injected resume test failure after global step 5`로 exit1이었다. `.lab-monitor/resume.json`의 checkpoint는 `attempt-f896605687b44ba7acaf4c77d97a70ab/decision/checkpoint-5`이다. driver가 이를 확인하고 `resume5_save10_fail`을 시작했다. 아직 최종10step/4종 native 복원 equality/TAR 완료를 앞당겨 주장하지 않는다. 새 로컬 P/Q preflight(session10253)는 모두 exit0이며 본 학습·새 대기 프로세스는 아직 없다.

### 14:31 KST — 실제 새 GPU 시험 최종 통과

driver session97525는 exit0이다. 실제 profile0→step5 저장/의도적 종료1→native5→10 재개/저장/의도적 종료1→최종10 native 복원/완료0을 확인했다. `GPU_VERIFICATION.json`은 새 image b312009f…/release e213d618…에 결속돼 있으며 `final_step_resume_verified=true`, `original_checkpoints_unchanged=true`, `full_training_started=false`다. 최종 저장본의 adapter/optimizer/scheduler/RNG 모두 실제 저장 상태와 equality=true다. 옛 build02 GPU 영수증을 새 ID로 바꿔 쓰지 않았다. 실제 optimizer 업데이트는 폐기 profile3+smoke10이고 본 학습은0회다. 최종 TAR 게시 검증을 시작했다.
