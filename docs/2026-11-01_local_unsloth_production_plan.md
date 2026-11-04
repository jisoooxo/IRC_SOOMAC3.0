# 로컬 Unsloth Response → Recommendation 본 학습 계획

[기존 작업 기록](2026-11-01_training_package_build_worklog.md) · [실행기](../tools/unsloth_training/production.py) · [순차 실행기](../tools/unsloth_training/launch.py) · [원격 후보 실패 기록](2026-11-01_unsloth_remote_v3_validation.md)
→ **목표는 기존 로컬 Unsloth 실행기로 Response와 Recommendation을 실제 학습하고, 저장된 adapter의 기능까지 확인하는 것이다.**

## 1. 최신 사용자 결정과 현재 상태

### 2026-11-03 14:52 KST — 두 본 학습 완료, 통합 품질 미합격

Response624step/6epoch와 Recommendation234step/6epoch 모두 정상 종료했다. 각 full-state·완료 영수증·best/epoch6 원본 PEFT와 실제 원본 PEFT 추론·격리 vLLM0.32 변환/로딩을 확인했다. 봉인24건에서 P best/epoch6 모두8/8, Q best7/8·epoch6 8/8이지만 **기존 D를 고정한 네 조합의17턴 누적 회귀는 모두 실패**했다. Q가 전 조합에서 한 번도 호출되지 않았고 P의 무근거 추천/허위 크림 선택도 관측됐다. 학습 성공과 기능 합격을 분리하고 운영 adapter를 교체하지 않는다. 상세 근거는 [2026-11-03 품질 결과](2026-11-03_unsloth_local_quality_result.md)다.

Cause: 기존 D의 추천·복합 실행 누락과 P의 state 밖 발화가 실사용 흐름을 끊었다.
Effect: 새 원격 D 도착 뒤 같은 회귀를 재평가하고 P 오류를 국소 보강할 근거가 확보됐다.

### 2026-11-03 09:09 KST — Response 정상 완료, Recommendation 사전 검증 진행

Response는6epoch/624step 정상 종료했고 `production.verify_response_completion()`이 full-state·두 원본 PEFT export·영수증을 통과했다. 최종 validation loss1.1994315385818481, best는step150의0.8207007646560669이다. **학습 실행 완료이지 기능 품질 합격은 아니다.** 기존 Q 대기 서비스는 자동으로 Q 대표 입력 목적함수 검증을 통과했고, 저장·실패·재개/재로드 사전 검증을 진행 중이다. Q 본 학습과 P/Q 후보 기능 평가는 미완료다. 상세 경로·SHA는 [worklog](2026-11-01_training_package_build_worklog.md)와 [STATE](training_acceleration_20261101/STATE.json)를 따른다.

Cause: Response 정상 종료와 완료 영수증이 Q 대기 실행기의 시작 조건을 충족했다.
Effect: 중복 학습 없이 Q GPU 사전 검증으로 넘어갔다. 운영 adapter는 교체하지 않았다.

### 2026-11-02 00:59 KST — 20/624step, warmup 완료

Response 동일 서비스/PID1405238은20/624step, loss1.25, grad_norm10.84, learning_rate5e-5, epoch0.1923이다. 설정된20step warmup이 완료됐고, Q CPU 대기 서비스 PID1416620도 active다. 표시ETA29:50:08은 잠정치이며 최초 validation/save50step 전에는 최종 성능을 판정하지 않는다. 현재 마지막 검증된 production checkpoint는 step5다. 학습 정책·데이터·원격 Decision·HF checkpoint와 운영 adapter를 변경하지 않았다.

Cause: 20번째 실제 optimizer update가 끝났다.
Effect: warmup 뒤 설정된 learning rate로 학습을 이어가며, 다음 검증 관문은50step validation이다.

### 2026-11-02 00:29 KST — 본 학습10step loss 확인

Response 동일PID1405238이 **10/624step**에 도달했다. stdout logging loss는5step2.696 →10step2.586, grad_norm은1887 →18.34, learning_rate2.368e-5, epoch0.09615다. 각 logging 구간의 train loss이며 고정 validation 입력의 성능 비교는 아니다. 최신 표시는 ETA29시간07분53초이며 초기10step에 근거한 잠정치다. 최초 validation50step과 최종 기능 평가 이전에는 학습 품질 합격으로 보지 않는다. 기존 checkpoint5가 마지막 저장본인 것은 정상적인 production 저장 간격이며 smoke의10step 저장 규칙과 혼동하지 않는다. P와 Q대기서비스는 active이고 중복 실행/재시작/학습 정책 변경은 없다. 이번 Goal turn은 동일 live PID를 기다려8→9→10step과 새loss를 확인한 verified wait다.

Cause: 실제 optimizer update10회와 다음 logging 구간이 끝났다.
Effect: 큰 gradient norm이 지속되는지는 다음 로그에서 재확인할 수 있고 현재10step 값은 유한하다. train loss 감소를 최종 기능 개선과 동일시하지 않는다.

### 2026-11-02 00:18 KST — 6/624step·격리 평가 경로 준비

동일 Response 서비스/PID에서6step을 확인했다. Q 서비스는 CPU 대기이며 GPU compute는 Response 한 개다. 이전 Goal turn은 checkpoint 무결성 확인·평가 입력 교정으로 progress이며, 이번 turn은 live service/step 전진 확인 및 아래 평가 실행 경로 준비다. 전체 목표는 완료하지 않는다.

기존 `manual_scenario_stack_probe.py`에 `--decision-adapter`, `--response-adapter`, `--recommendation-adapter`를 추가했다. 지정하지 않으면 기존 운영 경로를 읽되, 지정하면 평가 프로세스 내부에서만 원본 PEFT 경로를 사용한다. 운영 `model_runtime.py`를 바꾸지 않는다. 원본 두 파일이 없거나 vLLM 변환본을 원본 대신 지정하면 거부한다. PLAN/IDENTITY 및 실제 load_model에 같은 경로를 쓴다. 기존 `tools/vllm_server/run_scenario.py vllm`은 `--vllm-base-url http://127.0.0.1:18001`을 받아 process-local로 격리 서버를 선택하며 해당 override는 localhost18001만 허용한다. 서버의 기존 CONVERSION 원본 경로/SHA 검사는 유지한다.

새4건과 기존 measure-all2건을 pytest로 실행해 **6건PASS**였다. 최초 unittest 호출은 pytest 함수2건을 수집하지 않았으므로 그 결과는 합격 증거로 세지 않고 pytest로 다시 확인했다. 기존 실패 턴 뒤에 로봇 단계를 강제로 전진시키지 않는 동작과 실제17턴 시나리오·정답은 변경하지 않았다. GPU 모델/서버/ROS는 실행하지 않았다. 실제 endpoint 및 최종 adapter 검증은 학습 종료와 GPU 반환 뒤 수행한다. GPU 점유 검사는 기존 vLLM wrapper가 서버 프로세스를 예외 처리하므로 실행 전에 서비스/compute 목록을 별도로 확인한다.

Cause: 기존 시나리오 도구가 운영 adapter 경로에 고정돼 새 학습 결과를 격리 평가하기 어려웠다.
Effect: 기존 하네스를 재사용하면서 운영 경로·운영 포트를 바꾸지 않고 최종 산출물을 검사할 수 있다. 실제 모델 품질은 아직 미검증이다.

### 2026-11-02 00:15 KST — production checkpoint5 실파일 검증

Response 동일 PID1405238은 **5/624step, loss2.696, epoch0.04808**이며 Q CPU waiter PID1416620도 대기 중이다. `resume_marker()`와 `verify_checkpoint()`로 실제 production `attempt-63872c5d7d584aab81aa73bbb6cc0155/response/checkpoint-5`의 7개 파일 SHA 및 trainer/receipt step5 일치를 읽기 전용 확인했다. 설정은 mode=train,6epoch,target624,acc16,max9216이며 smoke 복원본이 아니다. Adapter SHA는 `35a35726e407a400e976a24092319ccfd5db5985fcb7fcd7c3ce420422b5549f`다. 저장 무결성과 실제 재시작 검증은 다르다. 본 학습을 중단해 재시작하지 않았으며 저장·재개 동작 근거는 앞서 통과한 폐기용 시험이다.

표시 ETA31시간54분은5step 시점 잠정치다. grad_norm1887은 유한하지만 크게 관측됐고 기존 max_grad_norm1.0 설정을 유지한다. 원인 미확정이며 한 번의 값으로 정상 품질을 선언하거나 학습 정책을 바꾸지 않는다. 아직 validation/기능 점수는 없다. 원격 Decision, HF checkpoint, 운영 서버, 학습 데이터와 production.py는 변경하지 않았다.

학습과 병행한 CPU 작업은 **평가 입력 교정**이다. 새 후보 `/tmp/soomac-training-accel-20261101/new_holdout_v4_runtime/NEW_FUNCTIONAL_CASES.json`(SHA `855a000c0fbb26546b83c5d2a22ffac3d2592f27161e9ba97fb4c2e125d8aa06`)은 D/Q/P 각8건이다. Q/P 입력은 실제 runtime builder로 scope/starting_now를 생성한다. 이전 대화가 필요한 수량·알레르기 사례에 근거 history를 넣었다. 독립 검수에서 발견한 Q03의 과도한 소스-only 채점은 교정해 안전한 추가 후보는 허용하되 소스 누락·페퍼론치노·근거 없는 high/low 추천량은 실패하도록 했다. D06에는 실제 이전 실행 질문을 보충했고 P03은 대기 상태를 나타내는 과거 발화로 모호함을 없앴다. v2/v3 원본과 학습 release는 보존했다.

CPU9건에는 D8건의 정답 주입 graph 전이, Q03 정상/소스누락/매운재료/수량오류 대조가 포함된다. 실제 모델 추론 성적이 아니다. Train+validation의 입력·발화·group 완전 일치0건이나 **알려진 시나리오의 새 표현이므로 독립 주제 일반화/의미 중복 없음은 주장하지 않는다.** Response 자동 채점은 보수적인 사실 오류 검사이며 불필요한 추천·실행 질문까지 완전 판정하지 않는다. 실제 답변의 state/history 대조와 누적 대화 평가가 남았다. 기존 검수 로그 `/tmp/soomac_eval_fixture_review.out`의 조건부 지적을 root가 확인·교정했으며 수정본의 별도 외부 승인이나 모델 품질 합격으로 표시하지 않는다.

Cause: 본 학습은 정상 진행하나 평가 입력의 수동 scope와 지나치게 좁은 추천 정답은 결과를 오판할 수 있었다.
Effect: 본 학습은 유지하고 평가 입력만 런타임 계약과 맞췄다. 다음 학습 관측은10step loss, 최초 validation은50step이며 최종 품질 판정은 별도다.

### 2026-11-02 00:02 KST — Response 본 학습 실제 1/624step

**본 학습이 실제 시작됐다.** 11월1일23:56:22 실행한 systemd user service `soomac-unsloth-response-r5-20261101.service`, PID1405238에서 1/624step, loss2.756, grad_norm88.86, epoch0.009615를 확인했다. 첫 update219.15초, 표시 ETA37:55:00은 1step 기준 잠정치이며 validation/저장 지연을 포함한 완료 보장이 아니다. 첫 learning_rate0은 기존20step warmup 정책을 유지한 결과다. 첫 production checkpoint는5step에서 저장하며 현재는 미생성이다.

새 durable 출력은 `/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/runs/r5_unsloth_local_authorized_20261101/response`, 로그는 같은 부모의 `response.training.log`이다. `production.py` SHA `86638bad3614a633b6fa023694f5c7b37ca840e9e37177be0eeff9d6a27e3208`는 변경하지 않았다. systemd RAM44G/Swap0 상한, 기존 GPU lock/32GB process budget을 유지한다. 학습1664행/검증152행,6epoch,624step,batch1×acc16,NF4/BF16,r16/alpha32/dropout.05,SmartGC,native CE,cosine/paged_adamw8bit,history불변이다. HF full-state나 smoke checkpoint를 본 학습에 이어받지 않았다.

P 사전 검증 완료: native 저장5→재개10과 네 상태 hash 일치; 기존 checkpoint5 보존;656개 LoRA tensor 변경/328 B nonzero/finite; long8430·short1507·multiturn7678 원본/labels/history/CE 검증; 표준 PEFT 새 프로세스656 tensor exact restore; 실제 vLLM0.32 staging 로드와3건 token-ID/grammar/JSON 추론. vLLM 결과는 각 원본 상태/정답과 대조해 수량·선택·완료 사실이 맞음을 확인했다. 이는 작은 호환성 시험이지 최종 누적 시나리오 합격이 아니다. 임시18001 서버는 종료했고 운영 adapter 경로는 그대로다.

Recommendation은 **학습 미시작·CPU 대기 등록 완료**다. service `soomac-unsloth-recommendation-queue-r5-20261101.service`, PID1416620이 기존 `launch.py`로 P 정상 종료를 기다린다. GPU는 P만 사용한다. P 서비스 정상 종료와 full-state 완료 영수증을 모두 확인한 뒤 Q 대표 목적함수→smoke 저장5/재개10→표준 PEFT 재로드/실제 parameter 변화→234step 순서로 진행한다. P 실패·서비스 소실·Q 검수 실패는 다음 단계로 넘기지 않는다. 새 대규모 pipeline 대신 기존 launch.py에 대기와 기존 검수 호출만 추가했으며 CPU22건과 Codex 델타 검수를 통과했다. 최종 기능 평가·전체 품질 합격은 미완료다. 아래 문서-only/미시작 문구는 이전 단계의 이력이다.

### 후속 실행 승인 — Autonomous Goal-Oriented 지시

사용자 첨부 `3d6b8acb-e140-47d0-a759-6f64ce436cf0/Pasted text.txt`로 읽기 전용 조사 한정이 해제됐다. Goal을 활성화했고 **필수 사전 검증 통과 후 Response → Recommendation 실제 본 학습·평가까지** 연속 수행한다. 아래 문서 작성만 한다는 문구는 이전 단계의 이력이다. 기존 HF/원격 Decision/운영 adapter 경로는 그대로 보존한다.

기존 환경 목록 37개 실행 가능·8개 백업, CPU 메타데이터 및 코드 감사 결과를 재사용한다. Codex CLI 검수 `/tmp/soomac_library_compat_review_20261101.out`는 버전 차이만으로 로컬 P/Q를 막을 확인된 근거가 없다는 조건부 판정이며, 새 adapter 실제 vLLM 로딩 합격을 뜻하지 않는다. xformers ABI 경고는 기존 PyTorch attention 대체 실행과 구별한다. 패키지를 재설치하지 않는다.

실행기 CPU 16건 통과. 기존 P/Q benchmark는 현재 `verify_benchmark_result`의 코드·패키지·입력 identity 관문을 통과했다. GPU compute 목록이 비었음을 확인하고 기존 `production.py --mode smoke`의 Response 5step 저장→의도적 실패→10step 재개 시험을 시작한다. 신규 출력 `/tmp/soomac-training-accel-20261101/authorized_response_smoke_v1`, 로그는 같은 이름 `.log`이다. 이 시험은 accumulation1/constant scheduler의 저장 인터페이스 시험이며 본 학습 624step/accumulation16과 구별한다.

Orca CLI `open` 뒤 `run-create`가 `runtime_unavailable: Could not connect to the running Orca app. Restart Orca and try again.`로 실패했다. 새 병렬 작업자를 실행했다고 보고하지 않는다. 현재는 주 담당자가 GPU를 단독 소유하고 순차 진행한다. 기존 CLI 읽기 전용 교차검수는 재사용한다.

현재 미완료: 대표 단문/장문/멀티턴 목적함수, native full-state 복원, 표준 PEFT 새 프로세스 로딩, 격리 vLLM staging 로딩, 본 학습, 기능 평가. 새 사전 GPU 결과는 실제 영수증 뒤에만 기록한다.

이번 지시는 짧은 사전 검증에 문제가 없으면 로컬 Response 본 학습을 시작하고 Recommendation을 순차 실행하는 조건부 승인을 포함한다. 이번 문서 작성 단계에서는 실행 명령을 발행하지 않는다. 먼저 계획과 기존 기록을 정리한다는 마지막 요청을 따른다. 기존 문서의 `production 미승인` 문구는 과거 상태이며, 앞으로의 로컬 P/Q 실행 권한은 이 결정으로 대체한다.

기존 HF checkpoint는 보존한다. HF full-state를 Unsloth로 이식하지 않고 신규 출력에서 시작한다. 원격 Decision 작업·실패 candidate v3·운영 adapter 경로는 변경하지 않는다. 원격 학습을 다시 시작하거나 중단할 권한으로 확대하지 않는다.

| 구분 | 확보한 증거 | 아직 남은 것 |
| --- | --- | --- |
| 로컬 Response | Unsloth warmup 2 + timed 3 optimizer update; 평균 232.230초/update | 대표 입력의 목적함수 점검, native full-state 저장/재개, 최종 PEFT 재로드, 본 학습 |
| 로컬 Recommendation | Unsloth cold 1 update, 131.381초 | 동일 사전 점검·저장/재개·본 학습·기능 평가 |
| 기존 HF Response | 과거 본 학습 step 6까지 로그, checkpoint-5 보존 | 복구·재학습은 이번 목표 아님 |
| 원격 candidate v3 Decision | HF/U forward·backward 비교 완료, 큰 차이 관측 | 원인 미확정; 제출용 TAR 없음, 기존 원격 작업 유지 |

측정 근거: `/tmp/soomac-training-accel-20261101/quality_warm_response_v2/RESULT.json`, `recommendation_unsloth_native1_smoke/RESULT.json`. `mode=full` benchmark는 전체 학습이 아니다. 위 실험의 `production_training=false`를 그대로 해석한다.

## 2. loss 차이를 어떻게 판단하는가

**HF와 loss가 다르다는 사실만으로 Unsloth 학습이 잘못됐다고 결론내리지 않는다. HF와의 비트 단위 일치를 본 학습의 필수 조건으로 삼지 않는다.** 동시에 finite loss/gradient 또는 한 행의 loss 일치만으로 올바른 학습을 보장하지 않는다.

원격 후보 이미지에서 HF loss 3.917093, Unsloth 4.086672와 큰 gradient 차이가 관측됐다. 두 Unsloth GC 모드의 forward가 같았고 초기 LoRA B=0이었으므로 이를 smart GC 또는 LoRA dropout 탓으로 단정하지 않는다. **근본 원인은 미확정이다.** 해당 이미지와 로컬은 torch/Transformers/Unsloth 버전이 다르므로 그 결과를 로컬 P/Q의 실패 판정으로 그대로 옮기지도 않는다.

로컬 승인 기준은 다음이다.

1. 실제 토큰·history·labels·target shift·선택 logits·mask 의미가 봉인 데이터와 일치한다. 짧은 입력, 장문, 멀티턴을 포함한다.
2. 동일 Unsloth 출력 logits에서 독립적으로 계산한 masked CE와 학습 loss의 정규화가 일치한다. 본 학습의 microbatch 평균/accumulation 16 정책을 유지한다.
3. 실제 train 모드에서 loss·LoRA gradient가 유한하고, optimizer update 후 학습 대상 값이 실제 변경된다. dropout 0.05와 text-only LoRA 대상이 유지된다.
4. full-state 저장→새 프로세스 재로드→이어지는 update와 최종 표준 PEFT adapter 재로드가 성공한다. 저장된 optimizer/scheduler/RNG와 step을 검증한다.
5. validation 및 사전에 고정한 작업별 평가를 기록한다. 사전 검증 통과는 본 학습 시작 가능 판정이며 최종 모델 품질 합격과 구분한다.

토큰/목적함수/가중치 로딩의 의미 오류, NaN/Inf, 실제 update 누락, 손상된 checkpoint는 차단한다. 미해결 수치 차이는 위험에 기록하되 무한 원인 추적을 본 학습의 선행조건으로 만들지 않는다. 의미 오류가 확인되면 위험 고지만으로 통과시키지 않는다.

## 3. 재사용할 실행 구성

| 항목 | Response | Recommendation |
| --- | --- | --- |
| 실행 파일 | `tools/unsloth_training/production.py` | 같은 파일 + 기존 `launch.py` |
| train / validation | 1664 / 152 | 623 / 30 |
| epochs / 예상 optimizer steps | 6 / 624 | 6 / 234 |
| max_length | 9216 | 5120 |
| learning rate | 5e-5 | 1e-4 |
| 공통 정책 | NF4 double quant, BF16, r16/alpha32/dropout0.05, Smart GC, batch1×acc16 | 동일 |
| optimizer / schedule | paged_adamw_8bit, weight decay0.01, cosine, warmup0.03 | 동일 |

Python: `/home/roma/IRC_SOOMAC3.0/training/learning_first_v43/training_release_20261101_r5/env/bin/python`.

봉인 release: `/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/releases/conversation_r5_20261101_build05_speed`.

데이터 manifest SHA256: `7850948287237c5ace9d1b0f56931419760fbce3f0afa02cea40681acb6ceffa`.

기존 runner의 smoke는 5/10step, accumulation1·constant scheduler 등 폐기용 설정이다. **저장/재개 인터페이스 검증용이며 production의 acc16/cosine과 동일한 학습 증거로 세지 않는다.** acc16 update 증거는 기존 benchmark 및 본 학습 초기 update에서 확인한다. 새 파이프라인을 만들지 않고 필요한 경우 기존 실행기/검수 도구만 국소 보완한다.

## 4. 실행 순서와 완료 기준

1. **계획·기록 정렬 — 완료.** 승인과 과거 실험/본 학습을 구분하고 문서·STATE의 낡은 미승인 표기를 정정한다.
2. **로컬 Response 사전 점검 — 완료.** 대표 입력의 목적함수, 저장·실패·재개, 표준 PEFT 및 격리 vLLM 로딩을 확인했다. 세부 증거는 이 문서의 최신 상태 절에 있다.
3. **Response 본 학습 — 6epoch/624step 정상 완료, 기능 품질 미판정.** 새 durable 출력에 `mode=train`으로 실행했고 완료 영수증·full-state·두 원본 PEFT export를 검증했다. 기존 HF checkpoint는 재개 원본으로 지정하지 않았다.
4. **Recommendation 준비와 순차 실행 — 완료.** Response 정상 종료 뒤 기존 대기 실행기가 Q 대표 목적함수·저장→실패→재개·PEFT 재로드를 통과하고 6epoch/234step 본 학습을 완료했다. 서비스 정상 종료·GPU 반환·최종 영수증을 확인했다.
5. **진행·결과 검증 — 저장/단일 턴/격리 vLLM 완료, 통합 품질 미합격.** 5step train loss, 50step 및 epoch 종료 validation/checkpoint를 기록했다. 공통 `checkpointing.verify_completion(..., "recommendation", ...)`이234step·full-state·export 해시를 확인했고 `trainer_state.json`은 epoch6이다. P/Q 두 후보를 원본 PEFT로 실제 로딩해 봉인24건을 네 조합에서 추론하고, 임시 vLLM0.32 포트18001에서 네 조합17턴을 완료했다. 이 평가에서 모두 통합 실패했으므로 기능 합격/운영 교체는 미완료다. 자세한 실패는 [품질 결과](2026-11-03_unsloth_local_quality_result.md)에 있다.

**P/Q 각각의 사전 합격 기록 완료 전에는 해당 `mode=train`을 시작하지 않는다.** 기존 runner의 한 행 CE 관문·학습 도중 gradient 검사만으로 이 사전 검증을 대신하지 않는다. 짧은/장문/멀티턴의 목적함수, train 모드 gradient와 실제 값 변경, 저장→새 프로세스 재개→update, 표준 PEFT adapter 새 프로세스 로딩을 각 agent에서 확인한다. adapter 파일 export/해시 확인은 실제 로딩 합격이 아니다. 기존 도구를 재사용하고 미비한 증거 연결만 국소 보완한다.

현재 기능 평가 후보는 `/tmp/soomac-training-accel-20261101/new_holdout_v4_runtime/NEW_FUNCTIONAL_CASES.json`이다. D/Q/P 각8건의 실제 runtime 입력·스키마·학습/검증 행과의 정확 일치 누출 검사는 통과했다. root의 24건 입력/정답 대조에서 새 명시적 모순은 발견하지 못했으나 독립 의미 승인과 실제 모델 예측은 미완료다. 이전 v2는 사용하지 않는다. 사용자가 학습에 넣은 시연 대화는 회귀용이며 미학습 일반화 성적으로 세지 않는다.

P 평가: 없는 주문/수량/실행 완료 사실을 말하지 않기, 변경량과 보존량 구분, 불필요한 추천·반복 금지. Q 평가: 요청 대상과 문맥에 맞는 추천, 매운맛/알레르기/느끼함 제한, 근거 없는 추천량 normal, 후보를 확정 주문으로 오인하지 않기. 통합 회귀는 크림 수락+즉시 실행, 버섯만 normal 변경+양파 low 보존, 게살 allergy+소시지 high, 치즈 low·페퍼론치노 제외를 포함한다. 기존 Decision을 사용한 통합 실패는 D/Q/P 원시 필드와 실제 state를 대조해 원인을 구분한다.

**최종 adapter 선택은 검증 손실만으로 끝내지 않는다.** `production.py`는 P/Q마다 `best_adapter`(최저 eval loss)와 `epoch_6_adapter`(마지막 full-state)의 원본 PEFT 두 개를 별도로 내보내고 `training_complete.json.exports`에 SHA를 남긴다. 완료 영수증과 두 export 해시·표준 PEFT/vLLM 로딩을 확인한 뒤, P 두 후보와 Q 두 후보를 동일한 봉인24건에 각각 적용하고, 기존 Decision을 고정한 2×2 조합의17턴 회귀를 비교한다. 턴별 D/Q/P 실제 호출·응답 출처·주문 필드·실행·허위 완료/수량 발화를 함께 기록한다. 점수 높은 조합도 사용자 승인 전 운영 경로에 교체하지 않으며, 최신 원격 Decision이 도착하지 않은 경우 그 조합의 결론은 기존 Decision 기준으로 한정한다.

**모델 출력을 보기 전 고정한 품질 판정 기준:** P와 Q의 두 후보 각각에 동일한 봉인 8건씩 적용하고, 스키마/필드 자동 검사 결과와 독립적인 문맥·사실성 검수를 별도로 기록한다. 운영 추천 후보의 합격은 해당 P 8/8, Q 8/8과 허위 주문·수량·실행 완료·근거 없는 추천 0건, 그리고 기존 D 고정 2×2의 두 에피소드(복합 6턴·분리 11턴)에서 `order_matches`·`execute_matches`·`plans_match`·추천 후보 근거·history 보존·model error 없음이 전 턴 통과해야 한다. `manual_scenario_stack_probe.py`의 `measure_all`은 실패 뒤 턴까지 진단하려는 모드이므로, 이 모드에서 끝까지 실행했다는 것만으로 합격 처리하지 않는다. Response가 호출되지 않은 고정 안내는 `runtime_fixed`로 표시해 모델 품질 근거에서 제외한다. 현재 봉인 24건 중 D 8건은 기존 Decision의 별도 진단이며 새 원격 D의 성적이 아니다. 자동 검사 통과는 독립 의미 검수나 실제 로봇 완료 증거가 아니다. 모든 후보가 미달이면 정상 학습 완료와 **품질 미합격**을 함께 보고하고, 운영 adapter를 교체하지 않는다.

봉인 영수증의 exact input/발화/group 누출 검사는 모두 음성이지만, 이 24건은 알려진 시나리오 계열의 재표현이다. 따라서 8/8 및 17턴 합격도 **이 계열의 회귀 통과**이지 새로운 주문 유형 전반으로의 일반화 보증은 아니다. 최종 보고에서 이 한계를 숨기지 않는다.

격리 vLLM 검증에는 기본 `convert_adapters_for_vllm.py`나 `check_after_training.sh`를 직접 실행하지 않는다. 이 둘의 기본 출력은 운영 `models/vllm_adapters`이고 경로 교체를 포함한다. 새 `tools/training_acceleration/stage_vllm_candidates.py`는 기존 K→V 변환 함수를 **새 평가 폴더 안에서만** 재사용하며 원본 해시·변환 해시·설정 동일성을 기록한다. CPU fixture 3건은 PASS이고 실제 완성 P/Q의 변환·포트18001 로딩은 학습 종료 후 별도로 확인한다. 현재 운영 adapter나 서버를 변경하지 않았다.

## 5. 시간·상태 보고

Response 232.230초×624 = **약40시간15분**, Recommendation cold 131.381초×234 = **약8시간32분**은 단순 산술 참고치다. 동일 최장 행 반복 benchmark를 실제 전체 데이터 ETA나 상한으로 확정하지 않는다. validation/저장/기능 평가·초기 compile 시간은 별도이다. 실제 학습의 서로 다른 입력 10~20update를 확보하면 남은 시간과 완료 시각을 갱신한다.

보고에는 `사전 검증 / 본 학습 시작 여부 / optimizer step·전체 step / loss·최신 정상 checkpoint / ETA / 남은 품질 위험`을 분리한다. 완료 step 수만으로 모델 품질 완료를 선언하지 않는다.

현재: **Response·Recommendation 본 학습 모두 정상 완료, 봉인 단일 턴 평가 완료, 기존 Decision 기준 통합 품질 미합격.** 과거의 “본 학습 미시작”과 “진행 중”은 작성 당시 기록일 뿐이다. 새 원격 D 재평가와 P의 무근거 발화 보강 전 운영 adapter는 교체하지 않는다. 최신 근거는 [품질 결과](2026-11-03_unsloth_local_quality_result.md)와 [STATE](training_acceleration_20261101/STATE.json)를 우선한다.

읽기 전용 Codex `gpt-6-sol/high` 교차검수는 조건부 찬성이다. 실제 production.py의 한 행 CE 관문, smoke 복원 검사, adapter export, logging_steps=5를 직접 대조하여 위의 agent별 사전 합격·실제 재로드 구분·loss 기록 간격을 명시했다. 검수 로그: `/tmp/soomac_local_unsloth_plan_review.out`. 문서/STATE JSON·diff 검사는 완료했으며 학습 코드나 GPU 검사를 이번 문서 작업에서 실행하지 않았다.

Cause: 속도 benchmark, 원격 후보의 수치 비교, 로컬 본 학습 승인·준비 상태가 서로 섞여 실제 학습 시작 조건이 불명확해졌다.
Effect: 기존 실행기에서 짧은 목적함수·저장 검증을 마치면 로컬 P→Q 본 학습으로 진행하며, 원격 후보 실패를 로컬 전체 학습 금지로 확대하지 않는다.
