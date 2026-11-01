# 속도 우선 학습 — TAR 먼저

[진단 근거](2026-11-01_training_speed_tar_audit.md) · [작업 기록](2026-11-01_training_package_build_worklog.md)
→ 최신 데이터는 그대로 두고 실제 장치에서 측정한 처리량으로 학습 설정을 고른다.

## 1. 사용자 승인과 순서

2026-11-01. 기존 로컬 Response/순차 launcher는 사용자 요청으로17:06 KST에 종료했다.12/624step이며 저장 완료 marker가 없었으므로 그 진행의 native 복원은 불가능하다. 출력/로그를 삭제하지 않았고 Recommendation은 시작하지 않았다. 원격 중단은 사용자가 담당하며 이 작업자가 원격 중단 API를 호출하지 않는다.

최신 Goal 순서는 **Decision TAR 완성 → 로컬 Response → Recommendation 순차 본 학습**이다. 원격 TAR 업로드와 본 학습 시작은 사용자 담당이다. Goal이 완료되려면 TAR 검증과 **두 로컬 Agent 각각의 실제 GPU 설정 실측·본 학습 optimizer 진행·첫 full-state checkpoint 저장**을 확인한다. Response만 시작하고 Recommendation을 예약한 상태는 진행 성과이지만 두 Agent의 실측/실행 증거를 대신하지 않는다. 순차 조건상 Recommendation은 Response 완료 검증 뒤에만 시작하며, Recommendation 전체 학습 종료까지 확인했다는 뜻은 아니다.

## 2. 고정할 학습 내용

최신 `compound_semantics_20261101_r2/dataset`, MANIFEST SHA `7850948287237c5ace9d1b0f56931419760fbce3f0afa02cea40681acb6ceffa`를 유지한다. 데이터/정답/시스템 지침/few-shot/history/기반 가중치·processor·LoRA 구조·LR·6epoch·유효 배치16을 속도를 이유로 줄이거나 바꾸지 않는다. FP8/NVFP4·VLM·추론 가속 코드는 범위 밖이다. 기존 build04 bundle/TAR와 checkpoint는 보존한다.

## 3. 새 학습 후보 설정

1. `gradient-checkpointing=auto`는 OFF와 ON 각각의 배치1→2→4→8을 유한 비교한다. OOM이면 **그 모드에서만** 큰 배치를 멈춘다. 동일 실패 조합을 반복하지 않는다. OFF가 들어가지 않으면 ON을 별도 후보로 측정하며, ON이 항상 정답이라고 고정하지 않는다.
2. 모든 후보가 최장 입력/정답 범위 stress를 거친다. 같은 최장 reference의 두 microbatch warmup 후 **실제 학습과 동일하게16개 예시를 누적한 optimizer update 한 번**을 측정한다. 유한·비영 gradient 검사는 수행하되 계측 구간 밖에 둔다. **실측 sequences/s가 가장 높은 후보**를 고르고, 수치가 정확히 같은 때만 checkpointing OFF·큰 microbatch 순으로 선택한다. 단일 정상 update 표본이며 장시간 정상상태 최적성을 보증하지 않는다는 한계는 영수증에 남긴다.
3. 원격은 서버 `GPU_MEMORY_BUDGET_BYTES`를 사용하고 Budget에서 bytes/fraction/실제장치 최솟값을 재검사한다. 이미지의44GB 고정 ENV는 빈 값으로 바꾸며, allocator 외 여유는2GiB→1GiB 후보로 둔다. 명시적 환경 override가 있어도 서버 허용량을 넘지 못한다. RAM은 VRAM과 별개다.
4. 로컬은32,000,000,000bytes/allocator 외**512MiB(후속 CLI 검수한 최신 후보)**를 두되 actual GPU free·타 프로세스·최장 역전파를 확인한다. 최초1GiB 후보보다 allocator 한도만512MiB 늘리고 전체 process 상한은 그대로다. Decision의1GiB OOM 기록을 P/Q 통과 근거로 쓰지 않는다. 최종 mode/batch는 Agent별 실측으로 고른다. 두 학습은 절대 동시 실행하지 않는다. 원격 봉인 이미지의1GiB는 바꾸지 않는다.
5. 선택한 checkpointing·budget·reserve를 실행 identity에 기록하고 저장본 설정 불일치는 거부한다. 훈련 중 OOM에 자동으로 batch를 바꿔 계속하지 않는다. 이미 끝난 전수 데이터 감사를 반복하지 않는다.

첫 full-state 저장은 본 학습5step에서 추가로 수행하고, 이후50step/epoch 경계 저장은 유지한다. 이 첫 저장 때문에 validation 전체를 추가 실행하지 않는다. gradient checkpointing OFF와 학습 복구 checkpoint 저장은 별개다.

## 4. 완료 기준과 증거

CPU: 모드 전파(PEFT/Trainer/profile), 잘못된 예산 거부, 서버 clamp, OFF OOM 뒤 ON 유한 탐색, 불일치 resume 차단, 로컬 순차/실패 차단을 확인한다. 독립 Codex sol/high 검수 뒤 실제 근거와 반례를 다시 읽는다.

GPU: 로컬에 다른 compute가 없음을 확인하고 새 이미지 UID1000으로 폐기 profile 및10optimizer-step 저장5→의도적 실패→resume10→실패→최종 native 복원을 검증한다. 로컬32GB 시험을 원격46.8GiB 실측으로 과장하지 않는다. 원격에서는 첫 실행 시 실제 예산으로 profile이 다시 수행된다.

TAR: 최신 release/data/model/processor pin, 기본 CMD train, spec2.9, native full-state resume, 실제 layer/config 연결·전체 SHA와 Docker load를 확인한다. 새 build05 이름으로 내보내고 build04는 삭제하지 않는다. 완성 영수증이 게시되기 전 ready라고 하지 않는다.

로컬 본 학습: TAR 완성 뒤에만 새 release/output으로 P→Q를 시작한다. **P와 Q 각각** 실제 optimizer 증가와 첫 원자적 full-state checkpoint를 확인하고, P 완료 검증 전 Q 시작 금지를 유지한다. Q pending/미측정 상태에서는 전체 Goal 완료로 표시하지 않는다. 검증된 checkpoint만 재개하며 일시적인 스텝 로그를 복원 지점으로 세지 않는다.

## 5. 현재 상태

**18:05 KST: [build05 학습용 TAR 완성](2026-11-01_decision_build05_speed.md).** 실제 GPU 저장/실패/재개, 독립 로그 검수, TAR57layer/OCI/config/SHA와 Docker load까지 통과했다. 원격 업로드·본 학습은 사용자 담당이다. 아래 GPU/TAR 미완료 문구는 이전 상태 이력이며 최신 결과를 우선한다. 로컬 P→Q는 읽기 전용으로 보존한 `runs/r5_package_test_20261101_build05_speed/local_operator/launch.py`와 정책을 사용해 새 output으로 시작한다. 두 Agent 각각의 GPU 실측·optimizer 진행·첫 저장 확인은 아직 남아 있다.

기존 로컬 중단과 속도 설정/자동 비교 코드의 CPU 검증은 완료했다. 새 봉인 release `conversation_r5_20261101_build05_speed` SHA는 `ff3285e46bff4f56c18327adb3b038a1010ab5e7482ca59729ab86c1f3a9459e`다. 새 이미지 `soomac/decision-r5:20261101-build05-speed`를 빌드했고 실제 GPU profile→저장5→실패→재개10→최종 복원 검증을 진행 중이다. **GPU 검증·TAR 내보내기·로컬 본 학습 재시작은 아직 완료하지 않았다.** 이 문서의 후보 숫자는 측정 결과가 아니다.

사용자의 추가 지시에 따라 Recommendation·Response 설정을 서로 다른 Codex `gpt-6-sol/high` 작업자에게 분담했다. 각 작업자는 자신의 설정 JSON·CPU 테스트·근거 문서만 수정한다. 별도 통합 작업자가 `local/launch.py`에 `--agent-settings-dir`를 연결하여, JSON을 적어놓기만 하고 실제 명령은 옛 기본값으로 도는 불일치를 막는다. 이 통합은 원격 release를 재봉인하거나 GPU 검증을 다시 시작할 이유가 아니다. 로컬 launcher와 JSON의 실제 해시는 새 pipeline identity로 별도 고정한다.

후속 통합은 완료했다. [실행·재개 계약](2026-11-01_local_speed_integration.md)의 agent별 예산/인자/설정 변경 차단을 연결하고, 완료 검증기 pin을 import 전에 확인한다. 최종 관련 CPU41건 통과, 독립 CLI delta 검수는 새 설정 경로의 확인된 회귀 없음/legacy 가짜 release 범위 한정 조건부다. 실제 봉인 build05의 검증기 pin은 검사한다. 두 Agent의 실제 CPU preflight도 `runs/r5_local_speed_20261101_build05/.preflight`에서 exit0이다. 이 preflight는 GPU나 본 학습을 시작하지 않았다.

원인: P/Q의 입력 길이·점수 대상 범위가 다르고, 설정 파일만 작성하면 실제 실행 인자는 바뀌지 않는다.
결과: P/Q 설정 검수와 원격 GPU 시험은 병렬 진행하되, 실제 학습은 TAR 검증 뒤 P→Q 순차로만 실행한다.
