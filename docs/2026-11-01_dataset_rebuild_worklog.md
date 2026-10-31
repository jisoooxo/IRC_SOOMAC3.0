# 새 학습본 조립·검수 작업 기록

[실행 계획](<IRC LLM Final dataset rebuild plan.md>) · [기존 감사](../training/learning_first_v43/final/goals/GL10/data_reaudit_20261031/RESULT.md)
→ 기존 판정 장부를 먼저 연결하고 최신 계약의 영향 행만 다시 확인한다.

**최신 후속:** 사용자가 교정·불량행 제외·학습 진행을 승인했다. 아래 「교정·신규 학습 승인 후」가 현재 상태다. 이전 미리보기와 영수증은 그대로 보존한다. 과거 절의 별도 승인 필요/미수정 표현은 당시 상태이며 현재 완료 증거가 아니다.

## 교정·신규 학습 승인 후 — 2026-11-01

### 최신 설정 점검 — 원격 VRAM 약44GB 사용자 지정 반영

[학습 파라미터 실제 점검](2026-11-01_training_parameter_check.md)을 작성했다. 사용자 「점검해보고, 원격은 약44GB VRAM으로 생각」을 현 작업 기준으로 반영한다. 로컬은 호스트 조회에서 RTX PRO4500 Blackwell 총32623MiB/여유30770MiB, compute 프로세스 목록 비어 있음을 확인했다. 샌드박스 내부 조회 실패는 호스트 재조회에서 해소됐으며 드라이버 고장으로 판단하거나 설치·재시작하지 않았다. 원격은 사용자 지정44GB 기준이지 이번에 원격 GPU를 조회한 것이 아니다.

정적 점검 결과 기존 `train_r3.py`는 batch1/acc16 및 Budget31GB를 고정하고, profile도 `measure_one(...,1,...)`만 호출한다. 기존1/2/4/8 profiler는 있으나 이 wrapper의 자동 탐색에 연결되지 않았다. 최신 D/P/Q의 실제 train 길이 중앙값은8069/7546/3981token이다. 따라서 짧은 답변만 보고 학습 비용을 추정하지 않으며, 더 큰 batch가 들어가는지는 역전파/optimizer 실측 전에는 미확정이다. NF4/BF16·SDPA·gradient checkpointing·scored-position loss는 이미 사용 중이다. 특정 최적화 부재나 batch1만을 과거 원격 지연의 유일 원인으로 단정하지 않았다.

원격 Decision44GB / 로컬 Response→Recommendation 순차를 유지하고, microbatch·accumulation은 effective batch16 기준 후보 비교로 기록했다. 최종 batch/LR/epoch·환경은 아직 채택하지 않았다. 새 후보는 실행 승인 결속 파일이 아직 없고 manifest에 조립 당시의 false/pending이 남아 있어 옛 r3 실행 파일을 바로 재개할 수 없음을 명시했다. 기존 사용자 승인을 철회한 것이 아니다. 두 conda 환경의 package metadata 차이도 기록했으며 추론 migration을 학습 검증으로 간주하지 않는다.

설정 점검의 독립 Codex sol/high는 핵심 진단에 조건부 찬성했다. 지적대로 「새 승인 영수증」뿐 아니라 manifest=true 요구/해시를 포함한 실행 검증 계약을 정해야 함을 추가했다. 원44GB도 실측 한도가 아닌 요청 예산이며, budget 모듈은 다른 작업 점유를 자동 차감하지 않음을 명시했다. 원후보의 불변 manifest나 검사를 우회해 통과시키지 않는다. CPU 예산/누적 산술 검사는 통과했고 GPU 시험은 미실행이다. 마지막 문서 보강 후 재교차검수는 하지 않았다.

**이번 실행:** GPU 상태 읽기/코드·과거 로그·token 영수증·metadata 확인과 MD 작성뿐이다. 모델 로드·GPU 역전파/프로파일·학습·설정 코드 변경·Docker 제작/업로드·원격 실행은 하지 않았다. 원본·기존 adapter·checkpoint·다른 작업자 변경은 보존했다. 다음 작업은 최신 manifest와 장소별 설정을 받는 학습 경로에 유한 batch 프로파일을 연결하는 것이다.

### 최상위 후속 지침 — 평가기 불신 검증·원격 분리·학습 설정 재점검 (2026-11-01)

사용자 최신 요청: 「검수기도 틀릴 수 있다. 가장 큰 데이터는 원격으로 따로 학습한다. 이전 학습이 느렸으므로 파라미터를 잘 확인한다. 지금까지 worklog에 기록한다」. **이번 턴은 문서 기록과 기존 자료/설정의 읽기 전용 확인만 수행한다. 학습·GPU 프로파일·설정 변경·Docker 제작/업로드·원격 작업 시작은 하지 않는다.** 아래 과거 P→Q→D 전부 로컬 순차 계획은 최신 실행 지침이 아니다.

#### 1. 현재까지 완료한 것과 미완료 경계

최신 데이터 후보는 외부 `candidates/compound_semantics_20261101_r2/dataset`, manifest SHA `7850948287237c5ace9d1b0f56931419760fbce3f0afa02cea40681acb6ceffa`다. A06 토마토 의미 교정, 조사 오류와 영향 history 교정, 기존 HOLD/정확 중복 제외 장부 보존, 복합 즉시 실행24대화/60턴 보강·Claude/Codex 국소 검수·root 채택, 최종 train 시나리오6발화/실행4곳과 누적 상태 대조까지 완료했다. 새109행 중81신규/28train alias, 전체8311/train6647이며 테스트51건 통과다. 자세한 행별 근거는 [최종 결과](2026-11-01_compound_semantic_result.md)와 [포함 재확인 JSON](2026-11-01_scenario_inclusion_recheck.json)에 있다.

**완료하지 않은 것:** 최신 후보의 실행 승인 영수증/학습 runner 재결속·새 CPU preflight, 로컬/원격별 학습 속도·메모리 프로파일과 최종 파라미터, 최신 데이터용 원격 패키지 검증, 실제 새 학습과 adapter GPU 회귀. 기존 r3 run STATE는 실제 재조회 시 `STOPPED_BEFORE_FULL_TRAINING_DATA_CORRECTION_REQUIRED`, `full_training_started=false`, `automatic_retry=false`다. 과거 데이터로 묶인 큐/학습 승인 파일을 그대로 재개하지 않는다. 이 상태는 기존 사용자 학습 승인을 철회했다는 뜻이 아니다.

#### 2. 목표는 오답 복구가 아니라 첫 응답부터 사실에 맞는 대화

사용자 「잘못된 말 자체가 안 나오게 해야 한다」를 최상위 품질 기준으로 기록한다. 정상 첫 응답에서 실제 반영한 주문·수량만 설명하고, 선택/실행 시작/실제 완료를 구분하며, 없는 추천·실행 질문·단계 이동을 발명하지 않아야 한다. 수량 변경이 실제로 있으면 「이미 그 양」이라고 하지 않는다. 이전 assistant 오답은 사용자 확정 사실로 승격하지 않는다. **history 삭제·Python 키워드 강제·새 고정 문구로 이 문제를 가리는 것은 하지 않는다.**

오염 복구 대조도 실제 train에 남아 있다. `NS01-C05-T01-P/T06-D`는 이전 assistant의 면 normal 오답을 유지한 history에서도 실제 low를 보존한다. `NS01-D03-T01-P`는 잘못된 「이미 high」를 따라가지 않고 실제 변경을 설명한다. `NS01-D04-T01-P`는 소시지 거짓 완료, `NS01-D05-T01-P`는 가짜 소스 단계 이동, `NS01-D06-T01-P`는 가짜 양파 선택을 바로잡는다. 의도적으로 틀린 상류 D01/D02 T01 Decision은 정상 D 정답에 포함되지 않았음을 실제 train에서 확인했다. 해당 오염 질문의 Codex sol/high 대조도 조건부 찬성이다(`/tmp/soomac_history_pollution_question_review.out`). 이는 정상 경로를 대신하는 목표가 아닌 사고 대비 자료이며, 실제 모델의 오염 내성은 아직 미검증이다. canonical state 자체까지 틀리거나 수음 의도가 모호한 경우의 자동 복구를 보장하지 않는다.

#### 3. 평가기 자체도 검증 대상이다

| 검사 축 | 후속 적용 기준 |
| --- | --- |
| 평가기의 오채점 | 이미 확인한 무승인 시작·알레르기 충돌 소스·틀린 재료 완료·관용 표현의 수량 오인 같은 알려진 반례로 거짓 통과/거짓 탈락을 확인한다. 기존 완료 검사는 재사용하고 변경 영향분만 다시 확인한다 |
| 독립 정답 | 평가기 출력이나 Runtime 현재 동작을 그대로 정답으로 삼지 않는다. 사용자 요구에서 동결한 기대값과 실제 before/after 주문·제약·pending·물리 증거를 따로 대조한다. 기대값 변경은 근거를 남기며 통과시키려고 고치지 않는다 |
| 문장 의미 | 문자열 규칙·schema 성공·동일 계열 모델 간 동의만으로 의미 합격을 선언하지 않는다. 원문+state+모델 원출력+repair/최종 응답을 보존하고 정상 첫 응답과 오류 복구를 분리해서 검토한다. LLM judge를 쓰면 사람 정답 대조 세트로 먼저 확인한다 |
| 최종 시연 합격 | 필수 시나리오 및 변형에서 수량 날조·거짓 시작/완료·없는 실행 질문·제약 누락·명시 즉시 실행 누락이 있으면 해당 조합을 합격시키지 않는다. 유한 시험의 오류0은 모든 자연어에 대한 무오류 보장이 아니다 |
| 실패 귀속 | 모델 의미 오류, Python 계약/안전 차단, 입력 오염/수음 모호함, 평가기 오류를 구분한다. 올바른 안전 차단을 모델 실패로 세거나 실패한 단계를 강제로 넘긴 결과를 정상 전체 성공으로 세지 않는다 |

#### 4. 학습 장소 분리 — Decision 원격, Response/Recommendation 로컬

최신 영수증의 train 행 수·실제 encoded token 합을 읽어 학습 분량을 확인했다. 파일 크기만으로 대상을 정하지 않는다. 아래 token은 system/history/input/target을 포함한 저장 인코딩 길이의 1회 합이며, 6epoch 실측 시간이나 정답 target token만의 합이 아니다.

| Agent | train 행 | 인코딩 token 합(1회) | train 최장 token | 학습 배치 위치 |
| --- | ---: | ---: | ---: | --- |
| Decision | 4,360 | 35,348,919 | 8,608 | **원격 별도 학습 대상** |
| Response | 1,664 | 11,535,187 | 8,430 | 로컬 순차 첫 대상 |
| Recommendation | 623 | 2,485,522 | 4,201 | 로컬에서 Response 다음 |

Decision이 행 수와 encoded token 합 기준 가장 크므로 사용자 원격 분리 지시를 Decision에 연결한다. 원격 GPU 총용량/현재 실제 할당량/속도는 이 조회로 확인하지 않았다. 과거 원격 조건 「48GB까지 사용, 44GB 예산」은 새 작업에서 재확인해야 할 기준이며 96GB 장치라는 이유로 전부 쓸 수 있다고 가정하지 않는다. 로컬 P/Q는 OOM 방지를 위해 동시 실행하지 않는다. 원격 D와 로컬 P/Q도 **동일 data manifest·schema/system·base revision·tokenizer·EOS/mask**를 사용해 D→Q→P 의미 정렬을 유지한다. 원격 업로드/실행은 이번 기록 작업에 포함하지 않는다.

#### 5. 학습 설정 — 기존 값은 비교 기준, 새 최종값은 미확정

실제로 읽은 기존 `train_r3.py`는 batch1·gradient accumulation16, int4 설정, manifest의 agent별 cap(D9216/Q5120/P9216), `Budget(31_000_000_000)`을 사용한다. 불러오는 `trainer_runtime_v5.py`는 세 Agent 모두6epoch, LR D/Q1e-4·P5e-5다. budget 모듈은 총 프로세스 예산에서 별도2GiB를 빼 allocator 상한을 정하므로 free VRAM만으로 OOM 가능성을 판단하지 않는다. **이 값들은 과거 로컬 runner 설정 확인이지 이번 학습의 최종 채택값이 아니다.**

후속에서는 실제 로컬/원격의 GPU 사용 가능량·다른 프로세스·allocator 상한, 실제 train+validation의 긴 입력 및 대표 길이에서 전진/역전파·optimizer·저장 비용을 확인한다. microbatch·accumulation·effective batch를 함께 비교하고, `step/s`만이 아닌 `sample/s`·실제 처리 `token/s`·메모리 peak·검증/저장 시간을 구분한다. 배치 증가를 무조건 빠르다고 하거나 VRAM이 크다는 이유로 학습시간을 단정하지 않는다. 데이터/history를 잘라 속도를 맞추지 않는다. 학습률·effective batch·epoch·정밀도·attention/checkpointing·loss 처리·길이별 padding/배치 구성의 적용 여부를 runner 실제 코드로 확인한 뒤 바꾼다. 이번에는 특정 microbatch나 새 epoch/LR을 미리 확정하지 않았다.

추론 가속은 별도 Claude 담당이며 prefix cache/vLLM 환경 이관이 학습 속도까지 개선했다고 가정하지 않는다. FP8 재도입은 하지 않는다. 기존 adapter·checkpoint·로그를 보존하고 새 데이터는 과거 optimizer의 정확 resume으로 취급하지 않는다. 파라미터 교정의 세부 계획/실측 결과는 실제 실행할 때 별도 영수증에 남긴다.

본 문서 변경은 독립 Codex sol/high가 **찬성**으로 교차확인했다(`/tmp/soomac_worklog_remote_split_review.out`). 지적한 wrapper92~111행 및 기반 trainer55~79행은 root가 직접 읽은 과거 설정과 일치한다. 이는 기록의 범위·수치·시제 대조이며 학습 속도·파라미터 성능 검증은 아니다. 작업 트리의 다른 사용자/가속 담당 변경은 수정·되돌리기·stage하지 않았고 `git diff --check`는 통과했다.

Cause: 평가기 통과와 실제 대화 품질, 파일 크기와 학습 계산량, GPU 총용량과 실제 학습 가능 예산이 각각 다른 정보다.

Effect: 첫 응답 사실성·평가기 역검증을 품질 기준으로 고정하고, D 원격/P→Q 로컬로 분리하며, 과거 느린 설정을 자동 재사용하지 않도록 다음 작업 경계를 명시했다.

### 사용자 후속 — KEEP 뜻 기록 및 최종 시나리오 포함 재확인

사용자가 KEEP 설명을 MD에 남기고 본인 시나리오가 최신 저장 후보에 실제 포함됐는지 재확인하도록 요청했다. KEEP=검수 범위의 gold 유지/채택, FIX=국소 수정 후 영향 검증, HOLD=근거 부족·모호한 행의 학습 투입 보류로 [최종 결과 §1](2026-11-01_compound_semantic_result.md)에 기록했다. 모델 성공이나 이미 학습 완료한 것으로 해석하지 않는다. 이번 확인은 기존 감사의 재시작이 아니라 최종 train의 기본 시연 ID·변형 ID·정답·history 연결을 확인하는 국소 점검이다.

[실제 시나리오 대조 §6](2026-11-01_compound_semantic_result.md#6-사용자-시나리오의-최종-train-포함-재확인)에 사용자5단계/실제6발화와 ID를 기록했다. 최신 r2의 `DA31-CORE` train D26/Q1/P10 보존을 확인했고, 공통 첫 턴 T01-D/Q/P→compound T02/T04/T05/T07/T09의 저장된 gold를 실제 Graph에 주입하여 수량·제약·실행과 인접 저장행 order/history 연속성을 대조했다. 6턴/실행4턴 모두 기대값과 일치, 마지막 턴은 앞선 사용자5턴을 포함한다. 단계 전환은 기존 snapshot의 명시적 오프라인 bridge 근거이며 새 물리 완료나 닫힌 루프 모델 성공으로 세지 않는다. 새109행도 train 정답/정확 alias로 모두 존재했다. 테스트51건 재통과, exit0. manifest SHA 불변. 검사 코드와 결과 JSON을 보존했으며 Runtime·학습 자료·adapter 변경, 학습/GPU 추론/ROS/commit/push는 하지 않았다.

이번 국소 대조는 독립 Codex sol/high도 **찬성**으로 확인했다. 지목한 실제 train 행 번호와 ID를 root가 재확인했고, alias의 동일성은 metadata가 아닌 input/target/route 기준임을 명시했다. 본 결과는 데이터 포함·작성 정답의 CPU 정책 동작 근거이지 모델 성공이 아니다. 추가 데이터 생성·제외·전면 감사는 하지 않았다.

### 최신 저장 후보 — 복합 실행 보강 r2 (검수·국소 교정 완료, 본 학습 미시작)

[대화별 정답·검증 상세](2026-11-01_compound_semantic_result.md). 최신 후보는 외부 `candidates/compound_semantics_20261101_r2`, manifest SHA `7850948287237c5ace9d1b0f56931419760fbce3f0afa02cea40681acb6ceffa`이다. 원본 r4_a2 8230행을 보존하고 24대화/60턴 초안109행 중 정확 중복28행을 train alias로 재사용, 새81행(D47/Q6/P28) 추가했다. 전체8311/train6647(D4360/Q623/P1664)이다. 기존 nontrain 네 split은 byte 동일이며 기존 train도 원본 byte prefix를 보존했다. 109행 모두 실제 train 또는 동일 train 정답에 연결됨을 저장 후 재확인했다.

원 초안과 a2는 비교용이다. 최종 literal 원본·행·before/after trace는 이 후보 `authored/`에 보존했다. 최초 별도 r1 조립은 G07~09의 같은 P 입력에 표현만 다른 정답이 있어 상충 검사에서 중지했다. 해당3개 P를 기존 `NS01-C01-T01-P` 정답으로 통일한 뒤 다음 history까지 재생성했고, 검사 완화 없이 새 r2에 조립했다. 독립 Codex sol/high는 원24대화에 KEEP21/FIX3을 제시했다. 게살을 육류로 오분류할 수 있는 G10~12의 안내만 구분해 교정하고 다음 history에도 연결했다. Claude는 실제24대화/60턴을 읽고 이 국소 교정까지 재대조하여 최종24 KEEP으로 보고서를 저장했다. root도 채택했다. 직접 고친 P6행과 영향 history를 포함하면 실제 변경18행/동일91행이며, 보고서의 「103행 재사용」은 직접 변경 개수와 혼동한 표현으로 root 상세 문서 §5에 정정했다.

현재 실제 Graph gold 재생60턴 오류0/예외0, commit20턴/실행19턴(소스 누락 G22만 차단), 신규81행 실제 trainer 인코딩 통과, 기존8230행 full-record/system 영수증 재결속이다. 실제 저장 후보의 manifest/alias/원본 보존까지 포함한 **51건 통과**, exit0이다. cap D9216/Q5120/P9216 유지, history 절삭0이다. Claude 보고서 SHA는 `af57e9daf939c5871dd2c563b9222d9266e4897d5d31a79f69babba8cc97f8de`. 이 작업에서 본 학습·GPU 모델·ROS·운영 adapter 교체·Runtime/prompt/cache/VLM 변경·commit/push는 하지 않았다. 실제 run STATE도 본 학습 미시작·중지 상태로 재확인했다. 기존 사용자 승인 보존을 유지하며 의미 검수 완료를 모델 성능 보장으로 해석하지 않는다. 다음 한 작업은 새 manifest를 결속한 순차 학습 preflight이며 이전 r3 큐를 그대로 재개하지 않는다.

Cause: 주문 변경·추천 수락·제약·실행을 복합 발화에서 동시에 맞추고 다음 턴까지 같은 정답 흐름으로 학습해야 한다.

Effect: 「바로 해줘/넣어줘」 긍정뿐 아니라 부정·인용·미래조건·모호한 지시와 실제 소스 누락 차단을 함께 저장하고 추적 가능하게 만들었다.

### 당시 착수 기록 — Claude 독립 의미 검수와 복합 실행 보강

사용자가 Claude 교차검수와 「바로 해줘/바로 넣어줘」 복합 발화 보강을 승인했다. [독립 검수 지시](2026-11-01_claude_compound_review_brief.md)를 먼저 작성하고 새 Orca Claude 세션 `term_18e9fe4f-4d4b-4f1d-8667-63e0301c9c7a`에 전달했다. 실제 화면에서 지시문·관련 코드 읽기와 검수 중 응답을 확인했다. 이는 최종 검수 완료 증거는 아니다.

Orca 감독 Run은 root의 `ORCA_TERMINAL_HANDLE` 부재로 `no_active_sender_terminal` 거부였다. Run/Task/Dispatch는 생성되지 않았다. 다른 패널 handle을 빌리지 않고 Orca `agent launch` 단발 검수로 전환했다. 모델은 사용자 기본 설정을 상속했으며 기존 속도 담당 Claude를 재사용하지 않았다. 추가 하위 CLI/worker·GPU·학습·Runtime/VLM 변경은 금지하고 보고서 한 개만 작성하도록 범위를 고정했다.

root는 `training/learning_first_v43/compound_semantics_20261101/author_compound.py`로 24대화/60사용자턴을 literal 작성하고 실제 Graph로 재생했다. 현재 D60/Q8/P41, 109행 초안, 필드·실행·호출/질문계약 불일치0, 예외0이다. 기본 mood+비매운전체추천, 추천수락+실행, 채소번복+실행, allergy+새주문+실행, 제외+치즈감량+실행, 기존 pending/없는 pending, 부정·인용·미래조건·모호한 대상·필수소스누락, 8턴 복합 누적을 포함한다. 초안 `/tmp/soomac_compound_r5/`는 아직 학습본 병합·의미 최종채택·실제 모델 성공이 아니다. 학습 중지는 유지한다.

### 최신 — 사용자 메뉴 기준 반영·추천 의미/범위 국소 감사

사용자가 A06을 토마토로 바꾸도록 명시했다. 아래 과거 「맛 근거 부족/오일 채택 보류」는 이 지시 전 기록이다. 현재 교정본은 외부 `candidates/conversation_supplement_20261101_r4_a2/dataset`, manifest SHA `474a93e8008b02b28cf6f99a459f369dbdf3561956210228e3e6c4845f26069e`이다. **학습은 중지 유지**이며 운영 adapter/Runtime/prompt/cache/VLM은 변경하지 않았다. 기존 사용자 학습 승인을 철회한 것이 아니라 새 선택의 의미 검수/보강이 끝나지 않아 training release를 보류한 상태다.

| 감사/교정 항목 | 실제 결과와 한계 |
| --- | --- |
| A06 토마토 | Q 후보를 토마토로 변경하고 수락 후 T02~T06의 order·P·실제 후속 history를 일치시켰다. Q 이유에는 제작 메타 문구를 빼고 「느끼한 맛 비선호에 맞춘 토마토 소스 제안」만 남겼다. 이 선택은 프로젝트 사용자 메뉴 기준이며 모든 오일 소스의 보편적 맛 단정이 아니다 |
| F08 조사 | 「게살는/게살를」을 「게살은/게살을」로 교정하고 다음 D history까지 재생성. A06과 합쳐 변경14행만 재인코딩, 불변8216행 영수증 재결속. 전체8230/train6566과 분할 유지, 원본과 r3/r4 첫 시도 보존 |
| 초기 소스 누락 검사 | Q 전체706행(train617 포함)을 scope/주문/후보와 대조. 초기 noodle·소스미정·소스선택가능인데 proposal.sauce=null 후보는 `QF28-Q-QF28-T03-narrow-00` 1행이며 실제 요청은 「다른 건 말고 채소 한 가지만」이라 정상. Q train의 면만 제안한1행 `NS01-E05-T02-Q`도 오일확정 뒤 「면 추천만」이므로 정상. 자동 검색을 전역 의미 합격으로 쓰지 않는다 |
| 넓은/좁은 요청 의미 대조 | `QF28-Q-QF28-T03-broad-00`은 면+토마토+버섯, narrow는 버섯만. `NS01-A04-T03-Q`는 현재noodle인데 미래 소시지를 추천. `NS01-A02-T01-Q/P`는 cannot_eat 유지+크림후보/응답, `NS01-E05-T01-Q/P`는 비매운 요청에 크림+얇은면. 실제 추천 pending/소스 proposal이 있는 P에서 소스명 미언급 표면 후보0이며 모든 표현의 의미 무오류 인증은 아니다 |
| 분포·보강 공백 | Q train617 중 초기 noodle27. 다른 단계 후보도 있는데 출력이 현재 단계뿐인 진단 후보525행/고유 발화64종을 확인했다. 대부분 「현재 단계 하나/야채만/고기만/두번째 수정」 등 정당한 좁은 요청이므로 일괄 삭제하지 않는다. 이는 모델 실패 원인 증명이 아니며 수치 편향 합격/탈락도 아니다. Q 현재 사용자 발화의 「기분/스트레스/우울」 표면 변형0; 「편안하게」 등 의미 유사 표현이 없다는 뜻은 아니다 |

독립 Codex sol/high는 토마토·조사 교정과 위 대표 대조에 조건부 찬성했다. 이유의 제작 메타 문구 지적은 채택해 r4_a2에서 교정했다. 반영 뒤 실제 저장행을 읽는 회귀6건 통과, 39대화201턴 Graph 재생 오류0, 변경14행 schema/builder/EOS/mask/무절삭 인코딩 통과 및 전체파일 hash/정확 중복·상충 검사를 수행했다. 이 검사를 새 adapter의 자연어 추론 성공으로 표현하지 않는다. 코드 빈칸 검사는 통과했다. 원시 CLI 로그 `/tmp/soomac_recommendation_r4_review.log`, Q 전체 문맥 목록/후보·분포 `/tmp/soomac_recommendation_meaning_r4/`.

**후속 보강 우선순위:** 「기분+매운맛 cannot_eat+전체 식사 추천」과 말투/수음 변형을 실제 누적 D→Q→P에 연결하고, 「면만/채소만/소스는 이미확정/유제품allergy/이전제약기억」 대조를 같은 기준으로 구성한다. 초기 넓은 요청에는 선택 가능한 안전 소스를 반드시 포함하고 면만 바꿔 매운맛을 해결했다고 하지 않는다. 넓은 요청의 후보 범위를 현재부터 미래 전체로 두되 좁은 요청에 모든 코스를 억지로 끼워 넣거나 이전 확정량을 덮지 않는다. 이 추가 보강은 아직 생성/채택/학습하지 않았으며 기존525행을 keyword로 버리거나 새 Python 자연어 판단을 넣지 않는다.

Cause: 이전 감사에서 schema/state 연결을 통과한 것과 맛·요청 폭·추천 근거의 적절성을 충분히 구분하지 않았다.

Effect: 사용자 지정 맛 교정은 실제 후속 state까지 반영했고, 과거 오류 정답의 잔존 여부와 관련 학습 분포/의미 변형 부족을 별도 결과로 남겼다. 학습 재개 전에 이 우선 범위를 보강해야 한다.

**최신 상태 정정 — 시나리오 포함 대조 중 새 조사 오류 발견:** `NS03-F08-T01-P`의 「게살를」, 연결된 입력/다음 턴 history의 「게살는」을 확인했다. 이번 작업의 큐292073이 GPU 프로파일296700 단계에 도달했음을 명령/PID로 확인한 뒤 **이 두 프로세스만 종료했다. 본 학습 미시작**이며 자동 재시도하지 않는다. 실제 run의 `STATE.json`/`DATA_QUALITY_STOP.json`은 `STOPPED_BEFORE_FULL_TRAINING_DATA_CORRECTION_REQUIRED`다. 기존 adapter·다른 Claude GPU 작업·원본·동결 dataset은 변경하지 않았다. 「의미·문장 품질 전부 이상 없음」으로 판정하지 않는다. 확인된 해당 대화와 후속 history의 국소 교정/재결속 뒤에만 새 선택으로 재개해야 한다.

**추가 의미 지적 — A05/A06:** 사용자 지적 후 실제 발화·Q/P 정답·domain 정의를 다시 읽었다. A05는 「지금 추천하지 마」 뒤 명시적으로 「이제 고소한 소스를 추천해줘」가 있어 최신 요청에 따라 추천하는 정상 시간적 전환이다. A06은 「느끼함 비선호 → 오일이면 피할 수 있음」이라는 정답 근거가 실제 제공한 메뉴 특성에 결속돼 있지 않다. `domain.py`에는 소스 이름과 제한 분류만 있고 덜 느끼함/레시피 근거가 없으며 Q의 reason_tags도 추천 자체가 적합함을 증명하지 않는다. 오일 추천이 언제나 틀리다고 단정하거나 토마토를 무조건 정답으로 바꾸지 않는다. **A06의 맛-메뉴 적합성은 미해결/채택 재검토 필요**로 기록한다. Q train에서 오일 proposal+느끼함 reason 명시 조합은 해당1행이며, 이는 전체 다른 표현의 오류 부재 인증이 아니다. 지금은 데이터·prompt·학습을 변경하지 않았고 본 학습 중지 상태를 유지한다.

사용자 시나리오 포함 여부를 실제 train에서 재조회했다. `DA31-CORE` D26/Q1/P10(공통 첫 턴은 중복 제거 후 공유), NS03-F01~F09, AMOUNT/QMEM의 low/normal/high 대조가 존재한다. 전체8230행 target schema를 다시 검사해 오류0, manifest 모든 파일 hash 일치를 확인했다. 관련 D/P history는 최대20 message/그중 선행 user 최대8턴으로, 20사용자턴이라는 뜻이 아니다. 알려진 메뉴+조사 표면 검사에서는 위 P1행이 잡혔으며 전역 한국어 무오류 인증은 아니다. 독립 Codex sol/high도 시나리오/변형/필드 구조 포함에는 조건부 찬성하고 동일 F08 오류와 연결 D history를 확인했다. CLI 로그 `/tmp/soomac_scenario_inclusion_review.log`; token/MCP 시작 경고와 실제 최종 판정은 구분한다. 이번 조회로 데이터·schema·prompt를 수정하지 않았다.

진행 **필드·인코딩 조립 3/4 완료, 문장 교정 후 4/4 재개 필요**. ① 확정 P 오류와 영향 history 교정, ② 부족한 정상 복합 실행 분기 보강 및 실제 Graph/필드·인코딩 검증, ③ 제외 장부와 결속한 신규 학습본 조립을 마쳤으나 이후 문장 검수에서 위 오류가 발견됐다. 기존 hash/통과 영수증은 당시 검사 범위의 사실로 보존하며 유리하게 덮어쓰지 않는다. 현재 모델이 못 맞힌다는 이유로 정상적인 어려운 데이터를 삭제하지 않는다. 모호한 것은 HOLD로 격리하고 원본은 보존한다.

- 학습 승인: 최신 사용자 「교정 ㄱㄱ … 전체 데이터에서 뺄건 빼고 학습시키자」. 새 데이터는 기존 optimizer/state의 정확한 resume이 아니므로 기본 모델에서 신규 adapter 학습한다. 운영 adapter 경로 교체·실물 ROS·원격 변경·commit/push는 범위 밖이다.
- 확정 FIX10과 일반 첫 안내 17개를 구분하여 교정한다. 상태 질문/명시적 유예/이미 안내한 대상에는 실행 질문을 일괄 추가하지 않는다. P 정답을 바꾸면 다음 턴 D/Q/P 입력도 실제 Graph로 재생성한다.
- 기존 HOLD13과 감사 단독 4행 제외는 이미 적용된 `base_filtered`를 재사용한다. 중복 제외/중복 검수로 세지 않는다. 추가 제외는 ID·원문 hash·사유를 새 장부에 남긴다.
- 로컬 학습은 Response → Recommendation → Decision 순차, 기존 체크포인트 보존. 05:52 KST 실제 GPU에는 별도 Claude의 STT/TTS/vLLM 시험이 실행 중이었다. 이 작업을 강제 종료하거나 동시에 학습하지 않는다. GPU 프로파일/학습 시작 여부는 별도 실제 영수증으로 기록한다.

Cause: 검수된 초안에는 필요한 첫 실행 질문 누락과 정상 복합 commit 분기 부족이 남아 있었다.

Effect: 이전 검수의 확정 지적만 국소 교정하고, 데이터 동결과 실제 학습 시작을 구분하여 새 학습으로 연결한다.

### r3 최종 교정·검증·큐 시작 영수증

[최종 채택과 학습 지침](2026-11-01_r3_training_selection.md). 최종은 외부 `candidates/conversation_supplement_20261101_r3_a3/dataset`이며 manifest SHA는 `487679ad4b65a48bcbaeb6d4fb88bab803666febda4986bdbc531a9057986c8d`이다. 초기 r3/a2는 비교·실패 증거로만 보존하고 학습하지 않는다.

| 항목 | 실제 결과 |
| --- | --- |
| 교정·보강 | 필요한 첫 실행 안내 P26행 교정. 기존30대화에 복합 실행/지시 대상/조건·부정 대조9대화 추가. 총39대화/201턴 실제 Graph 재생 불일치0. 다음 턴 입력도 변경된 실제 history로 재생성 |
| 제외·조립 | 기존 HOLD13 제외와 감사 단독4행 제외 유지. 신규 질문 시점 HOLD2행 격리, 정확 중복6행 alias 제외. 원본/history는 보존. 새370행(D197/Q15/P158) 채택, 전체8230/train6566(D4313/Q617/P1636) |
| 필드·인코딩 | 전체 builder/schema·정확 누수/상충 검사. 기반7860행의 동일 full-record/system 영수증 재결속. 새370행 실제 trainer AST/EOS/정답 mask 재인코딩과 저장 후 hash 대조. history 절삭0. 최대 D8616/Q4332/P8430, cap D9216/Q5120/P9216 |
| 검수·테스트 | Codex sol/high 1차 조건부 지적 F07/F08 참조 모호함을 교정. 새 packager/trainer/유한 큐 최종 코드 검수는 찬성. 관련100건+신규복합6건 통과, diff 공백 검사 통과. 새 테스트의 잘못된 pending=null 단언은 실제 Runtime 분기를 읽고 실행 비허가·정확한 대상·량 보존 검사로 교정. CPU 3-agent preflight 모두 PASS |
| 학습 실행 상태 | 큐 PID292073. Run: `/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/runs/conversation_r3_20261101_dec29eead775`. 최초 `STATE.json=WAITING_GPU_NOT_TRAINING`. Response preflight PASS. STT/TTS/vLLM 외부 compute PID287199/287200/288863/290694 점유를 확인했으며 종료하지 않음 |

실제 순서는 P→Q→D, 각 Agent의 최장 train+validation 입력으로 b1 역전파/optimizer 프로파일 PASS 후 별도 fresh full 학습이다. NF4/batch1/acc16/6epoch, D/Q LR1e-4·P LR5e-5, text-decoder LoRA r16/alpha32/dropout0.05, 기존 EOS/mask/저장 방식이다. 이전 optimizer를 복구하지 않는다. 출력은 새 run과 후보 내부 `gpu_profiles`로 한정한다. GPU 대기는 각 단계45분/30초 간격이며 오류·OOM·기한 초과는 자동 재시도 없이 `STOPPED_NOT_COMPLETE`다. 대기 큐 실행은 본 학습 성공 증거가 아니다.

학습 승인과 source SHA는 후보 `TRAINING_APPROVAL.json`, 제외는 `HOLDS.json`/`validation/ALIASES.json`, 수정은 `CORRECTIONS.json`, 전후 필드는 `TRACES.jsonl`에 있다. 실행 상태는 위 run의 `STATE.json`, 각 `*-profile.log`/`*-full.log`, 실제 `training_complete.json`을 읽는다. 정상임을 보장하려고 옛 영수증·미학습 모델 결과를 고쳐 쓰지 않는다. 전역 의역 누수 미입증·수치 편향 사용자 보류는 유지한다.

Cause: 교정 전 데이터의 첫 질문 누락을 해결하면서도 상류 오류 복구·명시적 보류를 섞어 자동 실행으로 오학습시키지 않아야 했다.

Effect: 원본을 보존한 최종 학습본과 순차 실행 큐를 결속했고, 다른 GPU 작업 종료 뒤에만 실제 학습을 시작하도록 했다. 실제 adapter 성능 개선은 학습 후 누적 모델 회귀에서 별도 판정한다.

## 최종 통합 판정 — 2026-11-01 05:48 KST

**진행3/3: 이번새자료의지시동결·독립검수·결과대조/기록을마감했다.** Claude의30대화/180턴/342고유ID를실제자료와대조해누락/추가/중복0을확인했다. 원문·기반·생성물의기존64개핀도전부일치한다. 검수완료는자료최종채택이나학습완료가아니다.

| 구분 | 최종 처리 |
| --- | --- |
| 실제개별정답대조 | 새D178/Q14/P150의state사실성·수량·제약·완료근거/P-only분기에서두검수범위내명백한추가사실오류미발견. 무오류증명/모델성공으로쓰지않음 |
| P계약교정 | rootFIX10(추천수락뒤필요한첫실행질문누락), HOLD1(미래소시지수락/현재면질문시점). 질문을일괄추가하지않으며, 일반첫안내17행은별도국소검토목록 |
| 새보강공백 | 정상추천수락+즉시commit, 초회allergy+신규주문+commit, 실행위치다양화, 선행대상/조건·과거회상대조. 기존DA31 gold는재사용하며전체재생성하지않음 |
| 오탐·과잉판정 배제 | C05/D03초기history오탐제외. B05-T02에인용뜻질문비commit실제있으므로“인용0”주장기각. next_prompt104/미질문103은103개오류가아님. Claude와root의세부대상목록이완전일치했다고쓰지않음 |
| 작업정산 | Claude공식worker_done성공수신, `ctx_6f220268bafb`만release, 해당Run미정리worker0. 중복하위Codex실패/TERM은검수완료로세지않고이전기록보존. Runtime/prompt/cache/VLM/adapter/원본미수정·학습/GPU/ROS/commit/push없음 |

Claude실제보고서는외부 `candidates/conversation_supplement_20261101_r1/reviews/CLAUDE_SEMANTIC_REVIEW.md`와coverageJSON이다. 완료payload의repo상대경로와실제외부위치가다르므로실제경로를기록했다. coverage의축약source pin에오타가있어그값은재현지문으로쓰지않고root의full SHA/기존64핀불변으로결속했다. 보고서SHA와지적별채택/기각은위root리뷰§6에있다. 원보고서나영수증을유리하게덮어쓰지않았다.

**다음한작업:** 별도r3에서필수P정답과그다음턴history영향범위를교정하고, HOLD/일반첫안내를누적문맥으로정렬한다. 이후누락정상branch를추가·검수하고학습길이/배치/재개설정을확정한다. 실제재학습과새모델GPU누적회귀는그뒤이며별도승인없이시작하지않는다. numeric bias수치합격은사용자보류, 전역의역누수는미입증상태다.

Cause: 반복질문금지를강조한새정답일부가필요한첫확인까지생략했고, 정상수락+즉시실행보강도부족했다.

Effect: 기존자료를전부버리거나Python키워드로강제하지않고교정/보류/보강을구분한후속범위를고정했다.

## 후속 의미 검수 착수 — 사용자 Claude/Codex 교차검수 승인

현재 단계는 **1/3 지시·입력 동결 → 2/3 독립 의미 검수 → 3/3 지적 대조·결과 기록**이다. [Claude에 전달할 전체 지시](2026-11-01_supplement_semantic_review_brief.md)를 작성했다. 대상은 F절의 새30대화/180턴/342행과 실제 전후 상태다. `AUTHORED_DIALOGUES_r2.json` 및 `replay_r2`가 최신이며 r1과 기존 영수증은 보존한다. 시작 기록은 Claude 실행/완료 증거가 아니다. 실제 Run/Dispatch/검수 결과가 생기면 아래에 별도로 기록한다.

CPU 재생·builder/인코딩 성공과 모델/의미 합격을 명시적으로 분리해 전달한다. D01/D02의 P-only 상류 오류 분기, 감사4행 실제 제외, 기존 HOLD13 재확인, 새 미리보기 비활성, 정상/실패 복구/오프라인 물리 fixture, 수량 보존·복합commit·불필요 질문/추천·멀티턴을 빠짐없이 지시에 포함했다. 다른 Claude의 속도/환경/vLLM 작업과 이 검수를 분리하며 원본·Runtime·cache·VLM을 건드리지 않는다.

Cause: 새 자료는 자동 필드 검사를 통과했지만 독립 의미 최종판정은 아직 없었다.

Effect: 검수 범위·한계·사용자 목표를 고정한 동일 자료로 Claude와 root가 각각 판정하고, 과거 전수검수를 반복하지 않는다.

### Claude 실제 시작 영수증

Orca `run_5dcc02059ec1` / `task_3c4656128c0f` / `ctx_6f220268bafb`, 새 Claude 전용 터미널 `term_2e9ba6c7-8abf-401c-a69b-4cc371005508`로 시작했다. 시작 응답은 `state=ready`, `turnStart=observed`, `input_accepted`와 `turn_started`다. Claude 모델은 사용자 기본 설정을 상속했으며 특정 모델명을 임의로 주장하지 않는다. 이는 **실제 턴 시작 증거**이며 지침 전체를 읽었거나 검수가 완료됐다는 증거는 아니다. 첫 입력은 위 지시 MD 전체를 먼저 읽도록 전달했다. 기존 최적화 터미널은 재사용하지 않았다. IPC 조회는 sandbox에서 연결 불가였으나 승인된 동일 CLI의 escalated 조회로 정상 연결됐으며 Orca 재시작/설정 변경은 하지 않았다.

### 실행 중 감독 기록 — 완료 증거와 구분

Claude는새30대화180턴을읽고실제342행의gold/export·오류주입/물리차단입력대조를보고했다. C05/D03초기history오탐은직접원문을확인해철회한것도전사에서확인했다. 이중자동정합검사를의미합격으로세지않는다.

추가작업자금지를명시했음에도Claude가자신의기존교차검수지침을따라별도`codex exec`를실행했다. 최초시도는모델거부/MCP403으로실패했으며모델명변경재시도도관측했다. root는중복실행종료·인증오류후재시도금지·inbox후속지침처리·Claude본인리뷰마감만하도록명시했다. 이추가Codex시도는완료검수로집계하지않으며,기존root독립판정을대체하지않는다. 데이터/Runtime/가속/VLM에는변경하지않도록소유권을계속제한한다. 최종정산때실제중단/완료상태를별도로기록한다.

05:38 KST: 미처리중복시도가계속대기해고유prompt stdin경로와부모관계(이번Claude PID248017→bash258867→node258870→codex258877)를읽기전용으로확인했다. 해당중복Codex258877에만TERM을보냈고후속ps에서258877/258870종료·본Claude248017유지를확인했다. 원본/검수로그/다른사용자작업은삭제하지않았다. 이는Orca주검수worker취소가아니라잘못추가한별도읽기전용자식만범위에서제외한것이다. 종료143/모델거부/403을데이터나모델정확도실패로집계하지않으며추가재시도를금지했다.

### root 의미·계약 1차 판정 — 05:32 KST

[root 독립 리뷰](../training/learning_first_v43/conversation_supplement_20261101_r1/reviews/CODEX_SEMANTIC_REVIEW.md)에30대화/180턴/342행의 대조를 저장했다. state사실성만본첫판정과실제system까지확대한추가판정을구분한다. 현재root잠정은 **KEEP331/FIX10/HOLD1**이다. FIX10은추천수락뒤유효한첫실행확인을빠뜨린P행이다. 명시적미실행2행은그대로유지하며상태질문에일괄“담을까요”를붙이지않는다. HOLD1은미래소시지후보수락에현재면실행next_prompt가붙는질문시점계약이다. Claude최종판정대기이며데이터/Runtime/prompt는아직수정하지않았다.

추가로새정상D의recommendation pending수락+commit0, 초회강한제한추가+commit0, 정상commit30행전부T06인보강공백을확인했다. 기존DA31 gold부재나342행전부폐기라는뜻은아니다. 개별오류교정과필요한소수정상branch추가를구분한다. 검수도중Claude의중간덤프가초기history를놓칠수있는점을발견해원문위치를전달했다. 실제C05초기history의assistant수량오답은존재하며, 덤프누락을데이터결함으로세면안된다.

교정시에는P정답10개뿐아니라그답변을history로읽는다음턴D/Q/P입력의영향범위도재생성해야한다. 변경된target이아니더라도history가바뀌면새SFT검증대상이다. 영향없는기반/나머지대화는재감사하지않는다. A04질문시점HOLD는T02첫면확인을보완한뒤T04가불필요반복을피하는상황인지누적문맥으로다시판정한다.

## 최신 통합 작업 기록 — 2026-11-01 04:58 KST

**이 절부터 읽는다.** 아래의 시간순 R1~R5 기록과 후속 기록은 보존한다. 당시 “다음 작업”, “미수정”, “완료”는 그 시점의 한정 범위이며 전체 프로젝트의 현재 상태를 뜻하지 않는다. 현재 브랜치는 `llm_final_2`, HEAD는 `3798a0e012c1e9165c9c78694b9bfcf98794469b`이다. 이 통합 마감에서는 문서만 갱신했으며 테스트·GPU·학습을 새로 실행하지 않았다. 아래 숫자는 각 실제 실행/검수 시점의 증거다.

### A. 지금까지 수행한 작업과 근거

| 작업 | 실제 수행한 일 | 결과와 한계 / 읽을 문서 |
| --- | --- | --- |
| Python pending·독립commit 수정 | 실행 유예 때 확정 주문을 지우던 분기 제거. 추천/제한철회 거절과 함께 온 별도commit을 최종 안전 검사까지 전달 | 코드 수정 완료. 당시812건/하위6건 통과. 모델이commit 자체를 누락하는 문제까지 해결한 것은 아님. [수정 기록](2026-11-01_llm_pending_runtime_fix.md) |
| 실제 로그·prompt 원인 대조 | 10/31 크림·버섯·소시지 복합실행 성공/실패의 raw, 실제system, history, pending, 변경 사실을 비교 | 규칙/예시 미전달과 해당입력history잘림을 근거에서 제외. 모델의 내부 실패 인과는 미확정. [prompt Worklog](2026-11-01_llm_prompt_alignment_worklog.md) |
| D/P prompt·few-shot 보완 | D C01~C07로 선택/추천수락/수량변경/알레르기+실행과 부정을 대조. P S10~S12로 신규/변경/기존유지 구별 | prompt 적용 및 당시834건/하위6건 통과. 예시는 실제history가 아니라 독립system 시연. Python 자연어 keyword 해석 추가 없음 |
| 기존best GPU prompt A/B | 독립입력10개에 old/new prompt로총20회 실제추론(D14/P6, Q로드만) | 일부 크림·버섯commit과 P시점표현 개선. 소시지commit/기존low보존/인용route 회귀는 남음. 추가D7회는 예고만 했고 미실행. Base-only나전체닫힌루프와 구분 |
| 누적시나리오 GPU 계측 | 최초strict2행을 보존하고 사용자 지시로복합6+분리11턴을 실제after/history로 끝까지 측정 | D17/Q0/P17, 실제소스null/실행0/plan0/전행noodle. 후반오류는선행실패전파와분리해야 함. [17턴 결과](2026-11-01_llm_scenario_regression.md) · [계측 JSON](2026-11-01_scenario_measurements.json) |
| 대화·필드 시각화와 독립검수 | 실제17턴과미학습보강13턴/22target을HTML에분리하고내장JSON/원본을대조. 당시Codex3작업을Orca로병렬진행해마감 | 실제17개user/reply일치, 보강전후13표, 표시수정·브라우저확인 완료. 모델실패수정이나재학습이 아님. [시각화 Worklog](2026-11-01_conversation_visual_worklog.md) · [HTML](2026-11-01_conversation_view.html) |
| 원본·장부·기존의미판정 결속 | 원본v5 7784행/39파일과기존FIX108/HOLD12/보강93을먼저연결. 불변판정재사용, 새계약영향D65/P139만독립검수 | D FIX52/KEEP12/HOLD1, P FIX16/KEEP123. P16은validation이다. 감사결과재사용을새성과로중복집계하지 않음. [R2최종판정](../training/learning_first_v43/rebuild_20261101_r1/reviews/ROOT_ADOPTION.md) |
| 승인된Q 규칙·예시 교정 | 근거없는새추천량만normal 규칙1개와정적시연4값을교정. 명시량·기존확정량·유효후보량은유지 | Q104기존교체 재사용. Runtime/새학습계약 정렬, Python으로추천량덮기없음. Q-only71건 및 당시확장836건/하위6건 통과 |
| 새비활성후보 조립·실제trainer CPU검사 | 7864행조립, 전행실제인코딩/EOS/정답mask·schema·builder·정확누수/분포검사. P명확화입력64행만국소이관 | 최신 `dataset_aligned_r2` 7864행/train6200행. 모델성공·전역의역누수·수치편향합격은아님. [재조립 결과](2026-11-01_dataset_rebuild_result.md) |
| 조립영수증 오류 국소교정 | 저장전dict순서와저장후JSON순서때문에893행byte영수증이달라진원인을확인. 최초실패보존, 도구/저장후영수증교정·64행재인코딩 | 의미오류893건이아님. 불변SFT7800행증거재사용·완전metadata115행갱신. 원본/옛결과를통과로덮지않음 |
| 현재D/Q/P adapter 출처·실패·새정답 대조 | 실파일weight/config3개와17턴계측identity일치, v5→R2필드delta, 대표31행, 실제로그6턴을결속 | D원격attempt의dataset hash직접영수증은부족. Q/P의v5manifest결속확인. 현재모델은새후보를학습한모델이아님. [대조 보고서](2026-11-01_adapter_training_failure_alignment.md) |
| 추가대화·정답 명세 | 기본시연재사용, 복합출력·다중선행참조·수량/제약보존·잘못된assistant복구·불필요추천/반복질문대조를구체화 | **SPEC_ONLY**. 이번문서추가분의정식JSONL생성/병합/학습은미실행. [대화 명세](2026-11-01_conversation_supplement_spec.md) |
| 단계전환 토글·로그·회귀 | `hard_cording_response`를상단에추가해주석처리된기존Q/P event경로를조건부복원. 새Response trace만선별하고로그에모드/next_prompt추가 | 기본True. 신규17건, 관련85건/하위6건, 확장LLM15파일853건/하위6건통과. 모델발화품질이나실물시험은아님. [토글 기록](2026-11-01_section_response_toggle.md) |
| 보강93행현계약검사·감사예시제외목록 | D59/Q4/P30을현재builder/schema/session불변으로대조. 사용자최신지시에따라감사단독D2/P2를다음조립제외대상으로결속 | 93행검사통과는새전수의미검수아님. **기존후보4행실제삭제전**, 원본/history불변. [검사와제외영수증](../training/learning_first_v43/section_response_toggle_20261101/SUPPLEMENT_CHECK.json) |

812/834/836/853 등의 테스트 수는 단계별로 겹치는 suite 결과이다. 합산해 독립 사례 수나 모델 성능으로 보고하지 않는다. 당시 독립검수·작업자 완료 기록은 해당 문서에 보존하며 이번 문서 갱신을 새멀티에이전트 검수라고 표현하지 않는다.

### B. 현재 코드·모델·데이터의 실제 선택 상태

| 대상 | 현재 확인 상태 | 주의 |
| --- | --- | --- |
| 단계전환 안내 | `llm_langgraph_node.py:54`의 `hard_cording_response=True`를 AST로 재확인 | False로 바꾸고 노드를 재시작하면 야채/육류/추가재료의 기존모델event경로. lid/sauce자동실행·시작안내·안전fallback은 별개 |
| Decision best | checkpoint750, epoch2.9444, weight SHA앞12자리 `c2cf15bb3a35` | 원격결과의실제가중치는계측과일치. 직접학습source영수증부족은그대로유지 |
| Recommendation best | checkpoint100, epoch2.6421, weight `cda007d8e7e9` | 전체6epoch완료와선택checkpoint를혼동하지않음 |
| Response best | checkpoint50, epoch0.5517, weight `7f528012569d` | 재개최종run이끝나도선택best는초반checkpoint계승. 낮은epoch가실패원인이라고단정하지않음 |
| 새후보 | 외부 `candidates/natural_order_rebuild_20261101_r1/dataset_aligned_r2` | 7864행은감사4행제외/새문서보강을아직적용하지않은현버전. 비활성/loader미연결/미학습 |

새후보manifest SHA는 `2a2c1687bdc390e05c9f4f254f84068f20f6fa9ceeb76d8d10ee119f6b58f230`이다. 외부 `FINAL_VERIFICATION.json`의상태는 `POSTWRITE_R2_RECEIPTS_AND_ENCODINGS_BOUND`이다. 당시전행CPU검증기록을이번에읽었으며전행인코딩을다시돌리지는않았다. D/Q/P현재학습계약5system은앞선대조에서Runtime와문자열전체일치했다. 코드토글을추가했다고과거runtime전체hash가불변이라고고쳐쓰지않는다.

### C. 최신 사용자 결정과 변경 소유권

1. 목표는 **이전대화+state를바탕으로자연스럽게선택·추천·변경·실행하는직원형대화**이다. 필수시연의복합commit·수량보존·제약저장·사실에맞는응답을함께검수한다.
2. 불필요한동일추천/확인질문반복은금지한다. 단어만바꾸는반복도문제다. 주문변경/실제질문상태에필요한새확인·사용자요청재설명·안전고지는허용한다. 반복을막으려고사실·수량·메뉴를바꾸지않는다.
3. 감사표현변형추가는철회됐다. 이번새보강명세에서감사단독사례를제외한다. 감사표현을종료·실행동의로하드코딩하지않으며, 기존history의해당표현을keyword삭제하지않는다.
4. 이작업의소유범위는LLM이다. VLM프롬프트/판정/카메라·ROS topic·busy STT·기존history·사용자미커밋변경은보존한다. 자동commit/push·실물실행·adapter교체·추가학습은하지않았다.
5. 속도·prefix cache·새환경migration은Claude/다른작업자소유다. [가속문서 §16](2026-11-01_llm_inference_acceleration.md)의 `soomac_llm`, 103.0→71.9초/21.8→14.1GB 결과는**그문서담당자의별도실험기록**으로연결한다. 이번worklog갱신으로재현했다고하지않는다. 실제ROS/카메라smoke와운영실행명령고정도그담당범위의후속조건이다.

Cause: 서로다른실행시점·작업자·데이터버전의완료표현이한대화에섞여있었다.

Effect: 실제구현/계측/데이터후보/명세/외부담당결과를분리해다음작업자가중복감사나미완료작업의완료오인을하지않게한다.

### D. 아직 하지 않은 작업 — 완료로 처리하지 않는다

1. **신규명세의학습행생성·의미검수·조립:** 복합선택+추천/commit·오염된앞assistant를보존한P복구·반복질문대조를실제builder/Graph로만든다. 감사4행제외는새버전에적용하고원본/history를보존한다.
2. **학습설정과학습:** 현후보train595행이8192초과(D590/P5)이다. D/P9216은길이상제안일뿐GPU검증완료값이아니다. 새row/split/prompt버전에기존optimizer resume을자동사용하지않으며학습은별도승인범위다.
3. **새모델의누적회귀·채택:** 실제D→조건부Q→P와선택/실행/완료를보정없이끝까지검사한다. 모델모드(False)단계안내의자연스러움·반복·상태일치도실제추론이필요하다. CPU정답주입통과를대신쓰지않는다.
4. **미확정품질범위:** 수치편향은사용자보류, 전역의역누수는미입증, 더깊은복합참조와원격D attempt source직접결속도미확정이다. HOLD12위임실행계약/HOLD1모호한취소를임의해제하지않는다.
5. **기존계획의잔여항목:** 불완전일반주문의next_prompt보완·별도추천범위토글 등은이번단계전환토글과다르며일괄구현완료로처리하지않는다. 새Docker/tar·원격재학습·실물시연완료도이번작업의산출물이아니다.

### E. 파일 찾는 순서와 다음 한 작업

현재상태는이통합절 → [추가대화명세](2026-11-01_conversation_supplement_spec.md) → [adapter대조결과](2026-11-01_adapter_training_failure_alignment.md) → [토글구현/테스트](2026-11-01_section_response_toggle.md) → [후보/길이검증결과](2026-11-01_dataset_rebuild_result.md) 순서로읽는다. 원문세부근거는A표의문서와아래기존시간순기록에서찾는다.

증거추출/검사도구는 `training/learning_first_v43/adapter_failure_alignment_20261101/`와 `training/learning_first_v43/section_response_toggle_20261101/`에있다. training경로는기존ignore정책으로Git상태에보이지않을수있다. 이번에stage/강제추가하지않았다. `.claude/settings.json`, cache·모델래퍼 등다른작업자의변경은그대로두었다.

**다음한작업은추가대화명세A/B/D의입력·정답행을생성하고기존장부와연결하는것**이다. 이번사용자요청은여기까지의기록저장이므로그작업이나추가검증을자동시작하지않는다.

### F. 후속 생성 승인 뒤 — 30대화 초안과 실제 제외 반영

사용자 “30대화·약300~450행계획을MD에쓰고진행” 승인뒤명세§5에계획을저장했다. 기존후보를직접덮지않고새외부경로 `/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/candidates/conversation_supplement_20261101_r1/`를만들었다. 운영코드·prompt·cache/환경·VLM·adapter경로는이번생성작업에서바꾸지않았다.

| 구분 | 실제 결과 |
| --- | --- |
| 이전폐기/교정대조 | HOLD13은기존후보에실제없음. 기존교체108과추가정답교정68의최종target일치. 의미키정확중복0/persisted none0. 원본/기존판정은재작성하지않음 |
| 감사단독4행 | 새 `base_filtered/`에서실제제외(D2/P2). 기반7860행/train6196. 이전후보/원본/history불변으로복구가능. 더이상“목록에만기록”상태가아님 |
| 생성 | 5묶음×6대화, 사용자6턴씩총180턴. D178/Q14/P150의고유342행. 잘못된D2개는P-only오류주입분기이며D학습정답으로내보내지않음 |
| CPU연결/인코딩 | 최신r2의180턴실제Graph검사오류0. 저장후새342행actual builder/schema/trainer AST·EOS·정답mask일치. history삭제0. 동일입력상충0/중복alias0 |
| 미리보기/한계 | `dataset_preview_r1/` 8202행/train6538. 비활성·loader미연결·의미검수대기. 새행최대D8308/Q4090/P8069토큰, D81행8192초과. 합친학습설정/새모델성공은미확정 |

최초r1초안/재생은보존했다. Graph통과후뒤단계초기fixture의면완료/전환근거를더명시하도록r2를만들었고다시180턴통과했다. 이근거는오프라인모의이며실제ROS완료가아니다. 신규자료불변조건5건도통과했다. 최초pytest는ROS자동플러그인의lark누락으로수집전실패했으며, 환경변경없이순수CPU테스트에한해자동플러그인로드를꺼서5건통과했다. 모델/데이터의의미실패로집계하지않는다.

`GENERATION_RECEIPT.json`으로기반불변·HOLD/감사제외·342행·파일지문을결속했다. `DIALOGUES_FOR_REVIEW.ko.md`는30대화의발화·정답·실제전후state를읽는검토본이다. 로컬도구와외부 `tools/`에재현스크립트를보존했다. 이전7864행을새로전수감사·전수인코딩하지않았다.

현재남은한작업은**새342행의문맥의미·복합참조·반복·실패복구를최종검수하고영향행만교정**하는것이다. 기존전체시연은기존gold와연결해재사용한다. 모든새대화가초기면부터최종제조까지진행하는것은아니며취약구간별6턴대화다. 독립검수완료·학습채택·실제adapter성공으로표시하지않는다.

Cause: 제외대상장부와현재파일의실제내용을먼저확인해야잘못된원본이새자료와함께들어가는것을막을수있었다.

Effect: 검증한기반에서만새342행초안을생성했으며, 감사단독행은새버전에서제외하고기존증거와history는보존했다.

---

2026-11-01 KST. 사용자 “진행 ㄱㄱ”로 R1~R5 데이터 조립·검수와 학습 준비 판정 실행 승인. 학습·추가 GPU·adapter 경로 교체·ROS·VLM·캐시/속도 코드·commit/push는 제외한다. **진행 5/5: 한정 재조립·검수·학습 준비 판정 마감. 최신본은dataset_aligned_r2이며 학습 길이/배치 설정은 별도 미확정이다.** 원본을 덮거나 stash/reset하지 않는다. [최종 결과와 남은 조건](2026-11-01_dataset_rebuild_result.md)을 먼저 읽는다.

## 저장 위치와 현재 상태

학습 산출물은 `/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/candidates/natural_order_rebuild_20261101_r1/`에 생성했다. R1 동결 자료·204행 영향 packet과 R3 `dataset/`·정제/제외/직렬화 장부를 저장했다. 비활성·loader 미연결이며 실제 학습하지 않았다.

현재 원본 v5 7,784행, 확정 delta 새93/교체108/HOLD12와 Agent별 기존 판정 장부를 읽었다. 중복 재검수 방지용 GL07의 `ledger_current.jsonl`·`REVIEW_COVERAGE.json`·`ORCHESTRATION.json`은 입력 근거로만 사용하며 재작성하지 않는다. 옛 중간 상태 문자열만으로 해당 판정을 다시 열지 않는다.

## 완료 기준

R1 입력·계약·행 판정 근거 결속 → R2 영향 의미 검수/정제 → R3 후보 조립 → R4 구조·의미·누수·분포·trainer encode/mask → R5 실제 학습 준비 판정이다. 원본 및 기존 수정 장부는 보존한다. 실제 모델이 시연을 성공했다고 보고하지 않는다.

Cause: 과거 KEEP의 근거와 현재 계약이 일치하는 행은 재사용할 수 있지만 새 계약 영향 행은 별도로 확인해야 한다.

Effect: 전수 재감사나 근거 없는 전부 통과 대신 행별 재사용·재검수·교체·제외를 구분해 새 후보에 반영한다.

## 실행 근거와 승인 Q 변경

1. R1: 보존 원본 39파일 SHA 및 7,784행 source·기존 판정·index 연결을 확인했다. 기존 의미 판정 7,460행 재사용, 기존 교체108행 적용 예정, 기존 HOLD12행 제외, 변경 계약204행(D65/P139)만 검수했다. 동일 행을 신규 데이터로 중복 집계하지 않는다.
2. R2: Decision 65행은 FIX52/KEEP13의 독립 의견·CPU 전이65건을 받았다. KEEP2행의 추가 문맥 대조 후 주담당자가 `DV3-L01460` 1행만 HOLD로 조정했다. “그 조합은 아니고”가 재료/수량 어느 조합인지 확정할 근거가 없어 원 의견과 경쟁 해석을 보존했다. 최종 **D FIX52/KEEP12/HOLD1**이다. Response139행은 **KEEP123/FIX16**이며 FIX는 전부 validation이다. 기존 HOLD12와 합쳐 총13행을 제외할 예정이다.
3. Q 승인: 사용자 “해당 Q 규칙·예시만 수정”에 따라 `agent_prompts.py`의 수량 근거 규칙1개와 정적 `prompt_examples.py` R01 양파·소시지, R02 양파, R06 치즈의 양4개만 normal로 교정했다. 정적 시연 교정이며 live 사용자 입력을 Python으로 재판단하는 규칙이 아니다. 기존 면 low·명시 수량 R04·다른 Agent·추천 범위는 보존했다.
4. Q 검증: 신규2건 및 기존 GL06 69건, 총 **71건 통과**했다. 기존 frozen R1은 덮지 않고 `CONTRACT_REVISION_01.json`과 `contracts/prompts_aligned.json`에 승인된 Q-only 차이만 기록해 새 학습 계약과 연결한다. 아직 모델 성공을 뜻하지 않는다.
5. 도구 독립 검수: Codex 작업자가 새 조립/검증 스크립트의 identity/hash 결속, FIX 입력 불변, 누락 판정 차단, token/mask와 Q 개정을 검사한 fixture7개를 통과했다. 실제 데이터 조립은 주담당자가1회 수행해 exit0을 확인했다. 이번 병렬 작업자는 모두 완료·release됐으며 reclaimable0이다. 캐시·가속 작업 파일은 수정하지 않는다.

추가 보호 회귀: 관련 LLM14개 파일은 **836건 통과·하위 검사6건 통과**, exit0이다. SWIG 사용 중단 경고와 sandbox NVML 초기화 경고가 있었고 GPU 모델은 실행하지 않았다. 이 수치를 모델 자연어 이해 합격으로 보고하지 않는다.

## R3 실제 결과

| Agent | train | validation | test | challenge | holdout | 합계 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Decision | 4118 | 309 | 235 | 417 | 286 | 5365 |
| Recommendation | 602 | 30 | 31 | 15 | 13 | 691 |
| Response | 1480 | 152 | 127 | 29 | 20 | 1808 |

합계7,864행, train6,200행이다. `7784−기존HOLD12−새HOLD1+고유보강93`과 일치한다. 기존교체108과 추가교체68은 append가 아닌 제자리 대체다. schema 오류0이다. 기존 metadata 직렬화 영수증 완전4키115행은 새 계약에 맞게 갱신했고, 부분97행은 역사적 근거로 보존해 sidecar에 최신 직렬화를 따로 썼다.

유효 계약 SHA `3c64d8afdc1bb264023b20c3ba32fbe0c8be80bbaf2539e436a9decdf0183ff5`, 개정 파일 SHA `8872a9b102d3c6d00e8a7e2d46dd080a72ba120f1f77acd36fd122bd5959ab42`다. 원본·이전동결계약·test/validation 봉인 파일은 바꾸지 않았다. 새 평가본의 수정은 이 버전 장부의 errata로만 적용된다.

## 시연 대화와 Agent 연결

기존에 독립 의미 판정이 끝난 10대화·88기록의 literal-gold trace를 새 후보에 결속했다. 논리 Agent행104개(D64/Q7/P33)는 고유93개(D59/Q4/P30)와 alias11개로 모두 연결됐다. 실제 call의 input·target·route와 후보 대표행이 일치하고, 물리 bridge가 없는 연속 턴의 전체 session 전후가 같다. 이 확인은 옛 대화를 모델로 다시 돌린 것이 아니라 새 조립본의 연결 추적 검사다.

`SCENARIO_FLOW_REPORT.json`·`SCENARIO_FLOW_LINKS.jsonl`에 저장했다. 24개 물리 bridge는 당시 명시적 오프라인 모의 완료/단계 전환이며 실제 로봇 성공이 아니다. Q는 조건부 호출이고 기존 고정 안내 경로를 P 데이터인 것처럼 만들지 않았다. 사용자 핵심 복합 주문·분리 확인·존댓말·명령형·양 low/normal/high의 변경/유지/완료 오인 방지·과거 치즈 양 취향 회상 사례가 포함된다. 전체 원본의 모든 episode 연결이나 unseen 일반화까지 검증했다고 확대하지 않는다.

## R4 첫 전행 검사 결과 — 지우지 않는 최초 증거

CPU387.068초, 전체7,864행의 실제 builder 필드·값, schema, trainer template·EOS·target mask, 16,384 cap 인코딩 오류0이다. history 삭제0, 모델/GPU/학습0이다. 최대8,616토큰이며8,192를 넘는615행(D610/P5)이 있다. train만 보면D590/P5이다. Agent별 최대는D8616/Q4332/P8430이다. 실제 인코딩으로 통과한 cap은16,384이며9,216은 수치상 제안값이지 실제 GPU 메모리·학습 성공값이 아니다.

정확 입력 충돌·정확 교차split·문장부호/공백 정규화 교차split·명시episode 교차split은 각0이다. 전역 의역 누수0이라는 뜻은 아니다. 수치 편향은 DEFERRED를 유지한다. 실제 선행 사용자 발화는0~10개로 분포하며, 복합 의미의 충분성을 history 길이만으로 판정하지 않는다.

옛 quality screen 플래그170개를 원인별로 분리했다. 실제 단계 이벤트의null 사용자 입력(Q3/P85)·현재 읽기 전용 로봇 검증 상태9개는 현 builder/노드가 허용한다. 한국어9개는 모두 과거 사용자 구어체라 보존한다. 반면 P64행(train60/val2/test2)은 실제 응답 노드가understanding에서clarification을 전달하는데 데이터next_prompt가null이라 실제 호출 계약과 다르다. 정답64개는 모두 명확화 질문으로 적절하다. schema/builder 형태 통과를 최종 계약 합격으로 덮어쓰지 않았다.

Codex 추가 독립 검수자는 전체64입력·답변과 실제 호출 경로를 읽고 input.next_prompt만명확화로 바꾸는 데 동의했다. 최초dataset/validation은 보존하고 별도dataset_aligned_r2/validation_aligned_r2에 반영한다. 수정64행만 실제 재인코딩하고, 나머지7,800행은 전체record와현재system/messages/input/target hash가 부모직렬화sidecar와 같을 때만 증거를 재사용하도록 보완했다. 이행 전후 검수와 실행은 다음 결과에 기록한다.

## R4 국소 교정 실행·최초 영수증 실패·R5 마감

첫64행 교정 시도는 부모target SHA assert에서exit1로 멈췄고 출력 생성 전이라 최초dataset/validation을 바꾸지 않았다. 원인은 조립 도구가 저장 전dict 순서로 직렬화 SHA를 계산한 뒤 canonical key order로 저장한 것이다.893행(input/messages867·target778 중복)의 저장 전 원형을source/replacement/review에서 재구성해 옛해시와 일치함·저장 후canonical값과 같음을 확인했다. 의미 데이터893오류로 세지 않는다. 미래assembler는 저장 순서 정규화 후 영수증을 계산하도록 고쳤다. 이전 독립 검수의 승인을 마지막root 패치까지 확대한 것으로 쓰지 않는다.

그 뒤64행 국소교정은exit0이다. 실제 변경64행 재인코딩, 같은model system/input/target7,800행 증거 재사용,metadata 영수증115행 교정이 완료됐다. 재사용 단위는 모델에 영향을 주는SFT다. metadata만 고친 행을 full record가 불변인 것처럼 보고하지 않는다. 최초영수증과실패는별도보존했다. 마지막기존fixture7건 통과·git diff check통과다.

최종저장본7,864행 전체의actual messages/input/target/system SHA와record SHA·token mask증거를 다시결속했고, 완전한metadata영수증도전부일치했다. 원본39파일·기존장부·Q승인외의semantic pin은불변이다. 외부 `FINAL_VERIFICATION.json`은 `POSTWRITE_R2_RECEIPTS_AND_ENCODINGS_BOUND`이다. 최신dataset manifest SHA는 `2a2c1687bdc390e05c9f4f254f84068f20f6fa9ceeb76d8d10ee119f6b58f230`, 최종encode summary SHA는 `2479f95e060abb67418efb7855d637f642d70b15ecaad63fa50b51317e0ad284`다.

R5판정은 **비활성정제후보완료·학습설정미확정**이다. 최대8,616토큰,8,192초과615개(train595)라기존길이그대로학습하면안된다. D/P9216은수치상제안이며실제GPU profile은아직이다. 전역의역누수·수치편향·더깊은복합대화·새모델성능은별도한계다. Runtime와학습JSON은필드/값이같지만키순서까지byte-identical하다고주장하지않는다. 다음한작업은새후보와prompt를동결한길이·배치·직렬화설정검토이며자동학습하지않는다. 병렬작업자최종마감은Orca공식완료와release로확인한다.

Cause: 현재 Runtime은 실행 거절로 확정 주문을 지우지 않는데 일부 옛 정답은 그 삭제를 암묵적으로 기대했고, Q 정적 시연은 무근거 low/high를 남겼다.

Effect: 실제 선택 취소만 명시적 none patch로 분리하고, 수량 근거를 보존하는 새 Q 계약을 불변 원본과 분리해 조립할 수 있다.

## 후속 — 현재 세 adapter의 실제 교재와 실패 대조

사용자의 “지금 adapter 3개 다” 승인에 따라 [별도 대조 보고서](2026-11-01_adapter_training_failure_alignment.md)를 작성했다. 완료 범위는 현재 선택 weight/config와 기존 계측 identity3개 일치, Q/P v5 manifest 결속, 과거7784행과 새7864행의 필드 delta, 대표 원본/후보31행 추출, 실제 로봇 연계6턴과 기존17턴 결과 대조다. D 원격 추출본은 attempt dataset identity 영수증이 없어 직접 source hash 결속 한계를 남겼다. source/모델/로그/prompt58파일은 마감 시 hash 불변이었다. 새 전수 의미 합격·GPU 회귀·학습은 하지 않았다.

새 후보5system은 현재 Runtime과 문자열 전체가 같다. 과거 학습 system은 general 외 D/Q/task/mixed가 다르다. D train의 order patch+추천request 조합은 과거1/새2행이며 전체 복합 의미 수나 편향 합격 점수로 해석하지 않는다. P 기존train 정답 변경0·입력60교정·새30보강을 구별했다. 정상gold 자료에 더해 실제 소스확정 누락·pending 없는 질문·앞assistant 오류를 보존한 복구 자료가 후속 우선 항목이다. 이는 이번에 새 자료를 만들었다는 뜻이 아니다.

Cause: 새 비활성 후보와 현재 모델이 배운 과거 교재를 같은 버전처럼 취급하면 데이터 교정과 모델 실패를 혼동한다.

Effect: 다음 순서는 무조건 길이/배치 확정으로 직행하기보다, 보고서 §5의 **D 복합 출력·Q 조건부 추천·P 오류 복구 연결 보강 범위를 먼저 확정**하는 것이다. 기존 전수감사 재시작·자동 재학습·운영 adapter 교체는 하지 않는다.

## 후속 — 보강할 대화와 정답 명세

사용자의 “지금 넣을 대화랑 정답” 요청에 따라 [대화·정답 명세 r1](2026-11-01_conversation_supplement_spec.md)을 작성했다. 기존 핵심 D/Q/P와 복합실행5역할을 실제 후보의 ID·target·입력pending에 연결해 재사용 대상으로 분리했다. 추가 대화는 주문+추천, pending별 실행 의미, 복합 멀티턴/수량보존, P오류복구, 불필요추천/정보부활 대조의 다섯 묶음이다. 정상흐름과 상류실패주입 branch를 혼합하지 않고, 현 고정실행안내에 가짜 P행을 만들지 않도록 명시했다.

상태는 SPEC_ONLY이다. 새 JSONL 생성·Graph 재생·의미판정·후보병합·학습·adapter교체·GPU/ROS·가속/VLM 수정은 이번에 하지 않았다. 의미가 같은 답변의 표현 다양성은 허용하며 P예시를 Python 고정문구로 추가하지 않는다.

Cause: 이미 들어간 기본 시연을 복사하면 보강 수만 늘고, 실패 복구·문맥 대조의 공백은 남는다.

Effect: 기본 시연은 재사용하고 부족한 대화의 입력조건·정답·금지주장을 먼저 명시해 다음 정식 학습행 생성 범위를 한정했다.

## 후속 — 불필요한 동일 응답·질문 반복 금지

사용자의 “동일하게 반복해서 응답하면 안 됨”을 대화·정답 명세 §2 D와 §3.4에 반영했다. 상태 질문 뒤 같은 실행 질문 재부착, 실행 수락 후 재확인 루프, 기다리겠다는 사용자에게 같은 추천을 반복하는 경우를 실패로 명시했다. 반면 주문 변경에 따른 새 확인·사용자 요청 재설명·관련 단계의 유효 제약 안내는 허용한다. 표현만 바꾸는 반복도 문제이며 재료명/짧은 문구가 같다는 이유만으로 정상 응답을 버리지는 않는다.

Cause: 단일 턴의 정답 문장만 맞추면 직전 답변을 반복하며 실제 대화를 진행하지 못할 수 있다.

Effect: 실제 assistant history와 이번 변경·pending 목적을 함께 보고 반복/필요한 재확인을 대조하도록 명세를 보완했다. 아직 데이터 행 생성·학습·모델 성능 교정이 완료된 것은 아니며 Runtime/가속 코드는 변경하지 않았다.

## 후속 — 단계 전환 토글 구현과 감사 표현 제외(최신 지시)

사용자 승인으로 운영 노드 상단에 `hard_cording_response=True`를 추가했다. True는 기존 고정 안내, False는 야채/육류/추가재료 단계의 기존 Q/P event 경로이다. 기존 시작안내·lid/sauce 자동작업·안전fallback·ROS topic·busy STT·VLM 정책은 유지한다. 과거 Response trace를 이번 event 응답으로 오기록하지 않도록 호출 전후 개수로 구분한다. 노드 재시작이 필요하며 현재 기본값은True이다. [구현/검증 기록](2026-11-01_section_response_toggle.md)에 범위와 명령을 남겼다.

LLM15파일853건·하위6건 통과(exit0), 그중 새 토글 회귀17건이다. 운영 메서드의 CPU/mock 테스트이며 실제 모델·ROS 실행은 아니다. 감사 표현 변형 추가 제안은 사용자 최신 “아예 빼는 게 낫지”로 철회됐다. 명세의 감사 단독 예시를 제거했고 대화 종료/실행 수락 규칙으로 삼지 않는다.

기존 DA31 보강93행(D59/Q4/P30)은 현재 실제builder·target schema·session불변 검사를 통과했다. 이전 의미 판정은 재사용하며 fresh semantic93건으로 세지 않는다. 감사 단독4행(D2/P2)은 정확source/line/recordSHA로 다음조립 제외대상에 결속했다. 후보 파일/원본/history는 아직 바꾸지 않았다. 이번 새 문서 추가분은 SPEC_ONLY이며 정식 생성·의미검수·조립·학습·모델모드 실제GPU회귀가 남아 있다. 캐시/가속 코드와 VLM 파일은 수정하지 않았다.

Cause: 단계 전환 모델 호출은 주석 처리돼 있었고, 감사 단독 보강은 최신 사용자 범위에서 제외됐다.

Effect: 기본 고정 안내를 보존하면서 모델 모드를 선택할 수 있고, 신규 주문 중심 보강과 다음 후보 조립의 제외 대상을 분리해 기록했다.
# 최신 승인 — Decision TAR → Claude NVFP4 → 로컬 순차 학습 (2026-11-01)

[현재 제작 계획](2026-11-01_training_package_build_plan.md#0-최신-실행-순서승인-갱신-2026-11-01)을 실행 정본으로 사용한다. 사용자 원격 대상은 Decision으로 확정됐다. 최신 데이터는 `compound_semantics_20261101_r2/dataset`, 원본 MANIFEST SHA256 `7850948287237c5ace9d1b0f56931419760fbce3f0afa02cea40681acb6ceffa`이다. 기존 r3 실행 폴더의 데이터를 재사용하지 않는다.

사용자는 **Decision 이미지/TAR·최적화/복원 시험 완료 → Claude NVFP4 시험 → GPU 반환 후 Response→Recommendation 본 학습 자동 순차 실행**을 승인했다. 이전 ‘본 학습 금지’는 이 조건부 로컬 실행에 한해서 갱신한다. Claude 시험이 끝나기 전에는 실제 본 학습을 하지 않는다. GPU 유휴만으로 시험 종료를 추정하지 않으며 명시적 종료 신호와 여유를 함께 확인한다. OOM이면 다음 Agent를 시작하거나 무한 재시도하지 않고 정상 저장 checkpoint를 보존한다.

Goal은 active로 새 생성했다. 기존 원격/로컬 Codex 편집 세션은 실제 Orca 일반 터미널이며, root sender identity 부족으로 Run/Task/Dispatch를 생성하지 못했던 상태다. 감독 Run이 존재한다고 주장하지 않는다. 원격 handle `term_30171300-13fc-4f24-80c6-edfba4c84692`, 로컬 handle `term_5de9cde5-9da2-4f90-a216-92cd0936e7c7`를 재확인해 재사용하며 중복 편집 세션을 만들지 않는다.

현재 완료는 계획/승인 기록·Goal 활성화이며, common 런타임 통합·Claude 종료 신호 연동·실제 GPU 시험·TAR 완성·학습 시작은 미완료이다. 이 기록에서 테스트가 통과하거나 자동 대기가 이미 실행 중이라고 주장하지 않는다. 기존 사용자 변경·가속/캐시/VLM 작업을 보존한다.

Cause: 사용자가 다른 GPU 작업 이후 두 로컬 학습을 자동으로 잇도록 실행 승인을 확장했다.

Effect: 최신 순서와 승인 경계를 하나의 문서에 고정했으며, 제작 전/Claude 시험 중에는 로컬 본 학습을 차단한다.

