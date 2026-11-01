# 기존 build03 기반 프로세서 내장 수정 — build04

[실제 원인](2026-11-01_build03_model_metadata_failure.md) · [작업 기록](2026-11-01_training_package_build_worklog.md)
→ 동일 데이터·가중치·학습 설정을 유지하고, 원본 인코더 파일만 이미지에 봉인한다.

## 0. 최종 결과 — 2026-11-01 16:11 KST

**수정 TAR 제작·실제 검증 완료.** 파일은 `/home/roma/Downloads/soomac-decision-r5-compound-semantics-20261101-build04.tar`, 6,998,478,336bytes다.

```text
SHA256 def26c665ea75dc0342126950607eb5cca81a7e217f2b44a27db192942d69bac
IMAGE  sha256:bf4b17bea0c9b10d68894f7e05c39bde6f55f371247b155e6fa332b4e0831352
```

CPU90건과 공식 메타데이터 overlay를 둔 실제 GPU profile→save5/의도적 실패→resume5→save10/의도적 실패→native final resume10가 통과했다. 최종 optimizer/scheduler/adapter/RNG 네 digest 모두 같고, 원래 checkpoint 불변도 검증됐다. GPU operator session5281 exit0이다. 실제 새 GPU 결과만 검토한 Codex CLI도 범위 한정 찬성(`/tmp/soomac_build04_gpu_evidence_review.out`)이며, 이를 실제 서버 실행/모델 품질 합격으로 확대하지 않았다.

finalizer session86462 exit0, 완료 영수증 `runs/r5_package_test_20261101_build04/DECISION_READY.json`의 시각16:09:50 KST다. 실제57layer payload와 OCI native manifest/config의 연결·최종 TAR SHA를 검사했다. `.partial`은 최종 파일·영수증 게시 후 없어졌다. 실제 `docker image load` session14067 exit0이며 같은 ID/linux-amd64/UID1000/CMD train을 확인했다. 기존 build03 TAR 6,991,625,216bytes는 보존했다. 원격 업로드·서버 실행은 아직 root가 하지 않았다.

로컬 Response1,992행 의미 추가검수와 P/Q CPU preflight도 완료했다. 사용자 승인에 따라 외부 launcher session86950/PID764128로 새 출력 `runs/r5_local_response_reaudited_20261101_build04`를 시작했다. 실제 `pipeline_state.json`은 running, Response started, Recommendation pending이다. 현재 초기 입력/배치 검사이며 **본 학습 optimizer update 완료는 아직 확인하지 않았다.** GPU/model을 로드한 상태나 step1 성공으로 과장하지 않는다. Recommendation은 검증된 Response 완료 뒤에만 시작한다. 이 PID/session은 관측값이며 이후에는 실제 생존을 다시 확인해야 한다.

이하 제작 중/미완료 문구는 해당 시점 이력이다. 최신 실제 상태는 이 절과 외부 실행 JSON을 우선한다.

## 1. 사용자 승인과 불변 범위

2026-11-01 사용자가 「기존꺼에서 바꾸셈 기존 tar 파일에서」라고 승인했다. 기존 `soomac/decision-r5:20261101-build03` 이미지 ID `sha256:b312009fef020df9390e44820457b2859180bc448dbdf5a4a98ebb42ff402647`을 `FROM`으로 사용했다. 의존성을 새로 설치하거나 데이터·프롬프트·epoch·batch 정책·가중치를 교체하지 않는다. 원본 build03 TAR와 receipt를 보존하며 결과는 별도 build04로 저장한다.

최신 원본 dataset MANIFEST는 `7850948287237c5ace9d1b0f56931419760fbce3f0afa02cea40681acb6ceffa`다. 모든 dataset 파일은 build03과 byte 동일이고 `model_files` 6개 pin도 동일함을 직접 비교했다. `core.py`의 SFT encoder와 학습 본체는 수정하지 않았다.

## 2. 수정한 경로

1. 원본 config/tokenizer/tokenizer_config/processor_config/chat_template 5파일을 `/app/assets/model-processor`에 일반파일로 내장한다. release의 `processor_files`는 데이터 인코딩 원본 5개 SHA와 정확히 같아야 하며 각 파일을 `source_files`에도 봉인한다.
2. release verifier는 내장 파일 전부를 검사하고, 외부 `/model`의 `config.json` 및 가중치 검사를 유지한다. 내장으로 공급한 processor-only 파일에만 외부 사본을 요구하지 않는다. 옛 release는 기존 검사 그대로 유지한다.
3. `configure()`가 ENV와 실제 `core.PROCESSOR_DIR`를 같은 봉인 경로로 설정한다. profile/학습은 기존 processor 로더를 그대로 쓰며 completion 검증도 같은 release 검사에 연결된다. ENV만 바꾸고 실제 로더가 `/model`을 계속 읽는 우회는 없다.
4. Docker 입력 선언은 외부 모델의 실제 필수 파일을 config+weight로 한정한다. 실제 실행은 UID1000, CMD train, 원격 예산44GB, batch auto를 유지한다. root는 빌드 시 inherited assets 권한 정리에만 사용하고 실제 실행 전에 UID1000으로 복귀한다.

Cause: 원격 cache의 메타데이터와 source 인코딩용 파일을 동일한 것으로 가정한 포장 계약이었다.

Effect: 외부 모델 메타데이터를 덮어쓰거나 검사를 끄지 않고, 원래 인코딩을 쓴 채 동일 가중치로 학습할 수 있는 경로가 생긴다.

## 3. 제작 중 실제 결과 — 완료와 구분

최종 사용 릴리즈는 `/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/releases/conversation_r5_20261101_build04_r2`, RELEASE SHA `f709f5c69264b0030f4f039016b29bc3de46bcac0027a518b8a99bff45372def`다. 처음 두 릴리즈는 삭제/재봉인하지 않았다. 첫 초안은 빌드 전 삭제 경로를 보완했고, r1 빌드는 inherited root-owned assets의 chmod 권한 오류로 중단됐다. 원인을 실제 stderr로 확인하고 r2의 **빌드 단계만 USER0**으로 바꾼 뒤 실제 image build exit0을 확인했다. 실패를 통과로 바꾸지 않는다.

최종 CPU 검사: **88 tests, 44 subtests PASS**, `/tmp/soomac_build04_cpu_final.out`이다. ROS pytest autoload의 `lark` 누락으로 최초 collection이 실패해 이 LLM 패키지 검사는 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`로 실행했다. ROS 기능을 검증했다고 주장하지 않는다.

별도 native restore CPU2건도 통과해 이 패키지 검증의 고유 CPU 합계는90건이다(`/tmp/soomac_build04_restore_cpu.out`). SWIG 사용중단 경고3개가 있으며 실제 GPU restore 성공과는 구분한다. 이후 추가한 로컬 gate 변경28건은 외부 launcher의 별도 검증이며 봉인된 원격 이미지의90건에 섞어 더하지 않는다.

Codex CLI 읽기 전용 code review는 조건부 찬성, 확인된 코드 결함 없음이다(`/tmp/soomac_build04_code_review.out`). 당시 Dockerfile 변경보다 앞선 CPU 결과라는 지적을 받아 최종 소스로88건을 다시 실행했다. 마지막 빌드 권한 교정을 리뷰어가 다시 확인했다고 쓰지 않는다.

공식 고정 commit의 template/config를 readonly overlay로 넣은 실제 build03 preflight에서 사용자와 동일한 `base model drift: chat_template.jinja`를 재현했다(`/tmp/soomac_build03_metadata_repro.log`). 같은 overlay로 build04 preflight와 GPU 시험을 실행 중이며, 모델 원본은 수정하지 않았다. 새 이미지 저장/실패/native resume와 최종 TAR 해시는 아직 이 절에서 PASS로 표시하지 않는다.

실제 build04 이미지 ID는 `sha256:bf4b17bea0c9b10d68894f7e05c39bde6f55f371247b155e6fa332b4e0831352`, UID1000:1000, CMD `["train"]`이다. Docker RootFS를 직접 대조해 원래52layer가 새57layer의 동일 prefix임을 확인했다. build03을 버리고 다른 기본 이미지로 교체한 것이 아니다. 공식 메타데이터 overlay가 있는 실제 build04 preflight는 exit0, Decision train4360/validation309이며 `model_loaded=false,training_started=false`다. `/tmp/soomac_build04_metadata_preflight.log`에 저장했다. 실제 서버에 접속·재업로드한 것은 아니다.

GPU profile은 exit0으로 완료했다. 배치1의 mixed stress/max scored span/reference longest 세 실제 optimizer update가 finite nonzero gradient로 성공했고 시간20.35/18.35/20.29초, peak allocated22,523,206,656bytes/reserved23,559,405,568bytes다. 배치2는 의도한 로컬30GB 시험 예산에서 OOM 판정 후 더 큰 후보를 시도하지 않았다. 이는 폐기용 유한 profile이며 본 학습 실패/원격44GB 측정이 아니다. 원격 요청 예산44GB·batch auto는 그대로다. 저장/복원 smoke10step은 다음 진행 중이며, 완료 전에는 최종 PASS로 승격하지 않는다.

15:54 KST 첫 smoke는 실제5 optimizer step과 평가·checkpoint commit 후 예정된 `Injected resume test failure after global step 5`로 exit1이었다. `.lab-monitor/resume.json`이 step5를 가리킴을 확인했고 operator는 이를 성공적 중단 시험으로 확인한 뒤 같은 이미지/metadata overlay의 resume5→10을 시작했다. 이 exit1을 본 학습 실패나 완료로 오기록하지 않는다. 마지막 native restore·원래 checkpoint 불변·TAR layer/hash 검증은 여전히 대기다.

## 4. Response 병렬 의미검수 마감 — 데이터 변경 없음

두 Codex sol/high 검수자가 실제 Response1,992행을 분할했다. 전반996행: challenge29, holdout20, test127, train1–820. 후반996행: train821–1664의844개와 validation152개다. 두 보고서 모두 확증된 새 FIX0/HOLD0이며 이는 검수 범위에서 발견이 없었다는 뜻이지 의미 무오류 보증은 아니다.

「둘 다 조금으로 바꿨어요」에 실제 신규 변경이 양파뿐인 일부 행은 합계 상태를 말하는 것으로도 읽힌다. 최초 후반 FIX 제안을 root와 두 검수자가 대조해 선택적 명료화로 낮췄다. 기존 MANUAL_REVIEW는 모델 출력 관측이지 dataset KEEP 영수증이므로 전수 의미 합격 근거로 승격하지 않았다. 거짓 발화 뒤 현재 주문을 바로잡는 답변의 직접성도 선택적 문체 개선으로 구별했다. 정상 데이터를 문구 취향으로 삭제하지 않았다.

기본 시나리오의 P 예시·누적 수량 보존이 포함됨을 확인했지만, 고정 실행 안내로 P 미호출인 턴에 가짜 P 학습행을 만들지 않았다. 이 검수는 D/Q 전수 재감사나 새 adapter 실제 추론 합격이 아니다. source/원격 Decision 데이터/현재 adapter는 변경하지 않았고, 로컬 본 학습도 아직 시작하지 않았다.

## 5. 로컬 순차 학습 준비 — 기존 승인 변경 반영

사용자가 NVFP4 대기를 철회한 범위만 외부 source `local/launch.py`의 `--user-approved-without-nvfp4-gate`로 반영했다. train 전용·gate receipt와 상호 배타적이며, 기본값에서 gate 없이 시작하는 것은 여전히 거부한다. `identity.authorization_mode`를 기록해 옛 waiting 출력과 섞지 않는다. source launcher는 불변 build04_r2 bundle의 실제 common trainer/verifier를 호출하고, bundle 안의 봉인된 옛 launcher를 고쳐 쓰지 않는다.

GPU/output lock·실제 free/compute 검사·P 완료 검증 후에만 Q 시작·P 실패 시 Q 차단·동일 설정 full-state resume는 유지한다. root가 소스를 직접 확인하고 별도28개 CPU/mock 테스트를 실행했다. Codex CLI 읽기 전용 검수도 조건부 찬성(`/tmp/soomac_local_gate_change_review.out`), 실행 차단 결함은 없었고 gate 도움말 문구를 새 플래그에 맞게 고친 뒤28건을 다시 통과했다. 원격 GPU smoke가 끝나 잠금을 반환한 뒤에만 실제 로컬 실행할 수 있다.

옛 대기 PID679304는 실제 명령이 build03 NVFP4 waiter이고 자식0임을 확인한 뒤 root가 SIGTERM으로 종료했다. 모델/GPU/Claude 프로세스는 중단하지 않았으며 기존 출력은 보존했다. 새로운 로컬 출력은 별도 경로를 사용해야 한다. 아직 이 절에서 본 학습 시작을 주장하지 않는다.
