# 현재 D/Q/P adapter 학습 출처·실패·새 정답 대조

→ **현재 운영 선택 모델이 실제 배운 자료와 새 비활성 후보를 구분한다. 의미 검수의 재시작이 아니라 실패 조건에 대한 출처 대조이다.**

2026-11-01. `llm_final_2`. **상태: 한정 대조 4/4 완료. 전체 데이터 의미 무결성 인증은 아님.** 사용자 승인 범위는 현재 세 adapter의 데이터 의미 대조이다. 학습·GPU 추론·adapter 교체·Runtime/가속/VLM 수정·ROS·원격·commit/push는 하지 않았다.

## 1. 확인 순서와 종료 조건

1. 실제 loader의 D/Q/P 경로·weight/config SHA와 선택 checkpoint를 고정한다.
2. 실제 학습 source manifest·train 행·당시 prompt를 확인하고 새 후보와 연결한다.
3. 실제 실패 로그의 입력·출력과 원본/교정/보강 예시를 사람이 대조한다. 기존 확정 판정은 재사용하며 중복 성과로 세지 않는다.
4. 각 Agent의 확인된 결함·모델 실패·상류 전파·미확인을 분리해 이 문서를 마감한다.

현재 최신 후보는 외부 `candidates/natural_order_rebuild_20261101_r1/dataset_aligned_r2/`이다. 현재 adapter가 이 자료를 배웠다고 간주하지 않는다. 전역 의역 누수·수치 편향·새 모델 성공의 합격 인증은 이번 범위가 아니다.

## 2. 현재 adapter

| Agent | 실제 선택 best | 전체 학습 완료 | 현재 weight SHA 앞12자리 |
| --- | --- | --- | --- |
| Decision | checkpoint750, epoch2.9444 | step1530, epoch6 | `c2cf15bb3a35` |
| Recommendation | checkpoint100, epoch2.6421 | step228, epoch6 | `cda007d8e7e9` |
| Response | checkpoint50, epoch0.5517 | step546, epoch6 | `7f528012569d` |

`model_runtime.py`를 AST로 읽어 선택 경로를 얻었다. 실제 세 weight/config 파일을 SHA256으로 읽어 11/01의17턴 측정 identity와 모두 일치함을 확인했다. 모델을 GPU에 올리지 않았다. 전체 경로·hash·best/완료 기록은 [ADAPTERS.json](../training/learning_first_v43/adapter_failure_alignment_20261101/ADAPTERS.json)에 남겼다.

Q의 profile fingerprint와 P의 재개 MIGRATION은 과거 v5 manifest `dcf2a6861631ec90516ead6b29c54c12f540099882717ef97f90a0581ce4d37c`에 결속된다. P의 checkpoint50 weight는 이전 run의 SOURCE_SNAPSHOT 기록과도 같다. 즉 최종 재개 run이 끝났어도 선택 best는 초반 checkpoint를 이어받은 것이다. **이 사실만으로 P 실패의 원인을 조기 checkpoint라고 단정하지 않는다.** loss 최저 선택과 대화 시나리오 최고 선택은 다르다.

D의 원격 추출본에는 run별 dataset manifest 영수증이 없다. 기존 v5 runner/package 기록·train4065/총1530step은 v5 계열과 일치하지만, **원격 작업이 실제 읽은 데이터 hash의 직접 결속은 Q/P보다 약하다.** 아래 D 원본 대조는 이 한계를 가진 v5 source 대조이다. 원격 작업의 `.lab-monitor/resume.json` 또는 동일 역할의 attempt identity를 확보해야 직접 확정할 수 있다. 세 모델 모두 새 `dataset_aligned_r2`를 학습했다는 근거는 없다.

Cause: `best` 폴더는 마지막 epoch가 아니라 validation loss로 선택된 checkpoint를 가리킨다.

Effect: 마지막6epoch 결과라고 잘못 생각하고 모델 품질을 판단하거나, 새 교정 자료가 이미 학습됐다고 간주하지 않는다.

## 3. Agent별 의미 대조

### 3.1 Decision — 복합 의도의 연결이 핵심이다

| 대조 | 확인된 사실 | 판정 |
| --- | --- | --- |
| 기존 v5 `DV3-L02640/02641/02644/02645` | “치즈 많이 넣고 바로 진행해” 등의 원본 정답은 주문 patch만 있고 commit이 없다. 새 후보는 commit=true를 넣었다 | 기존 정답 오류의 기확정 교정 재사용. 이번 새 발견4건으로 세지 않음 |
| 기존 선택 축소 | 예: `DV3-L01548`의 소시지만 유지+실행에 게살none이 없었다. 새 후보는 게살만 제거한다 | 실행 유예와 선택 취소를 분리한 Runtime 계약 변경 반영 |
| 실제11/01 C1 | 면 normal은 맞지만 recommendation request 누락 → Q0회 | D의 명시 추천 요청 누락. Q 자체 실패가 아님 |
| 실제11/01 C2 | commit=true지만 크림 patch 누락, 소스null 유지 | 실행만 켜는 것으로는 부족. 선택과 실행을 함께 출력해야 함 |
| 실제11/01 C4/C5/C6 | C4 버섯normal·양파low·commit은 맞음. C5/C6은 선택/제약을 반영하나 commit 누락 | C4는 수량 변경 실패가 아니라 선행 소스 누락의 전파. C5/C6은 실제noodle 상태에서 관측한 복합commit 누락 |

새 train에 `DA31-CORE-*`의 면 선택+추천, 크림 선택+commit, 버섯normal 변경+commit, allergy+소시지high+commit, 페퍼론치노none+치즈low+commit이 실제 존재한다. “아직 바로 담지 마”는 실행 거절이며 주문을 취소하지 않는 대조도 들어 있다. 양파low를 유지하고 현재 버섯만 바꾸는 문맥과 누적history를 대표 행에서 직접 확인했다.

**중요한 분포:** 구조상 `target.order`가 있고 동시에 `recommendation.action=request`인 train 행은 v5에서 `DV3-L01855` **1개**, 새 후보에서는 이에 `DA31-CORE-separate-T01-D`를 더한 **2개**이다. 전체 train은4065→4118행이다. 조건부 추천을 order patch 없이 표현하는 유효한 자료까지 이 수치에 넣지 않았다. 단어만으로 전체 복합 의미 수를 센 것도 아니다. 하지만 사용자가 첫 턴에 요구한 바로 그 출력 조합의 저장 정답이 적다는 것은 확인됐다.

Cause: 선택·추천·제약·실행을 각각 아는 것과 한 발화에서 모두 출력하는 것은 다른 학습 조건이다. 현재 실패의 내부 인과 기여도는 미확정이다.

Effect: “멀티턴 자료가 많다”만으로 복합 시나리오 충분성을 통과시키지 않는다. 재학습 전에는 이 조합의 다양한 의미 대조를 우선 보강해야 한다.

### 3.2 Recommendation — 정답 교정은 있지만 실제 호출 여부부터 나눠야 한다

| 대조 | 확인된 사실 | 판정 |
| --- | --- | --- |
| `REC2-T0070` | 수량 근거 없는 새 버섯low → 새 후보normal | 최신 사용자 수량 정책과 옛 정답의 불일치 |
| `REC2-T0414` | 무관한 산책/음악 등의history 뒤 새 페퍼론치노high → 새 후보normal | 긴history 존재가 추천량의 근거는 아님 |
| `LF43-R-T-005` | 게살allergy·육류 많이 요청 → 소시지high | 올바른 명시 수량. normal로 바꾸면 안 되는 정상 자료 |
| 실제10/31 22:55 T1 | Q는 크림을 proposal로 냈고 이유에 느끼한 소스 요청을 반영했다 | Q가 크림 추천을 할 수 있었던 실제 호출. 그 run의 가중치hash는 없으므로 현재best 통제 성능시험은 아님 |
| 실제11/01 17턴 | Q 호출0 | 이17턴으로 Q 이해 실패나 면만 추천하는 모델 오류를 단정할 수 없음 |

Q 전체 기존104행 교체의 실제 split별 차이는 train 정답93, test정답2, challenge정답8이며 scope 입력3행 중1은 train/2는 test이다. 입력과정답 변경이 겹치므로 숫자를 단순 합산해 고유 오류 수로 쓰지 않는다. 승인된 Q 규칙1개와 few-shot4수량 교정은 현재prompt와 새후보 모두 반영됐다.

새 Q4행은 느끼한 소스→크림1행과, 앞서 치즈의 적게/보통/많이 선호를 기억하는3행이다. **전체 단계 추천의 다양성·임의 제약 조합까지 새 Q4행으로 충분하다는 판정은 아니다.** 반대로 현재·미래 scope가 넓다고 매번 모든 재료를 나열해야 하는 것도 아니다. 사용자 요구와 근거에 맞는 후보를 내야 한다.

Cause: 수량 근거 정책이 바뀌었고 옛 모델은 그 교정을 학습하지 않았다. Q미호출은 추천 모델의 선택 문제가 아니라 상류 D의 라우팅 문제이다.

Effect: Q교재를 고치는 작업과 D가 Q를 실제로 부르게 하는 작업을 분리한다. 한쪽을 고친 것을 다른 쪽 성공으로 보고하지 않는다.

### 3.3 Response — 새 선택·기존 선택·실제 담기·실패 복구를 구분해야 한다

| 대조 | 확인된 사실 | 판정 |
| --- | --- | --- |
| 실제11/01 C2/S2 | confirmed_order.sauce=null인데 “크림으로 선택했어요” | 실제 state와 다른 확정 주장. 상류가 실패했더라도 P의 사실성 문제는 별도 존재 |
| 실제11/01 C3 등 | pending/next_prompt가 없는데 “이대로 담을까요?” | 질문에 대응하는 확인 대상이 없는 계약 불일치 |
| 실제10/31 22:32 T8 | 신규 소시지high 적용 뒤 “이미 많은 양으로 선택되어 있어요” | 수량값은 맞지만 신규 변경을 기존 유지처럼 설명한 문제 |
| 기존 `RV1-S1081` | policy=understanding/clarify인데 next_prompt=null | 답변은 올바른 명확화 질문. 새 후보는 next_prompt만clarification으로 맞춤 |
| 기존 `RV1-S1321` | 현재 게살 실행 질문에 다른 단계의 미리 고른 재료도 묶음 | validation답변 교정. train 답변 교정으로 계산하면 안 됨 |

**“이미 담겼다”라는 문구가 있다고 전부 오염은 아니다.** `RV1-R007`은 completed_tasks에 치즈2회가 있고, `RV1-S0721`은 양파 완료와 task_completed 이력이 있다. 이 행들의 완료 설명은 근거가 있으므로 문구만 보고 삭제할 수 없다. 다만 이 근거로 해당 답변의 모든 안내 문장까지 완벽하다고 인증하지 않는다. 아직 제조되지 않은 재료를 담겼다고 하는 실패와 구별해야 한다.

새 P30행에는 `버섯 low→normal=늘렸어요`, `normal→normal=그대로예요`, `high→normal=줄였어요`가 각각 들어간다. “벌써 담겨 있어?”에는 실제 완료가 없으면 “아직 담기 전”이라고 답하고, 실행 유예 뒤 선택은 유지한다. 취향을 앞 대화에서 꺼내 권유하는 QMEM3양 대조도 있다. 실제pending이 있는 상태에서만 담기 질문을 하는 대표 입력을 확인했다.

**남은 공백:** 새 DA31 P30행은 주로 upstream이 정답대로 진행한gold 상태다. 그30행 안에는 실제 C2처럼 소스null·missing_order(sauce)인데 직전 assistant가 크림을 제안/확정했다고 잘못 말한 실패 복구 입력이 없다. 기존 전체 P train에는 일반missing_order 자료42행이 존재하므로 “누락 질문 학습자료가 전혀 없다”고 하지 않는다. 다만 새gold흐름을 넣은 것만으로 잘못된 앞 응답을 바로잡는 능력까지 보강했다고 할 수 없다.

Cause: 정상 정답이 이어지는 대화와 실제 상류 오류·잘못된 assistant 발화가 섞인 대화는 입력 분포가 다르다.

Effect: history를 삭제하지 않고 canonical state·이번 변경·확인 대상·실제 완료 증거를 근거로 복구하는 P 자료를 다음 국소 보강 대상으로 잡는다.

### 3.4 이전 학습본과 새 후보의 실제 차이

| Agent | train 수 | 기존 train의 변경 | 새 train 보강 | 평가용 정답 교정 |
| --- | --- | --- | --- | --- |
| D | 4065→4118 | 정답36행, 제외6행 | 59행 | validation2/test1/holdout17 |
| Q | 598→602 | 정답93행, 입력1행(겹침 가능) | 4행 | test정답2/challenge8; test입력2 |
| P | 1450→1480 | 입력60행, **기존 train 답변 변경0행** | 30행 | validation답변16행; validation/test입력 각2 |

이는 새 오류 검수 건수가 아니라 과거 v5 원본과 현재 새 후보의 **실제 필드 delta**다. 기존 FIX/HOLD를 그대로 재사용했다. 전split을 합치면 D정답56(기존4+추가52), Q정답103·scope3의 고유replacement104, P답변16 및명확화 입력64이다. D제외13행도 기존검수결과이며 train제외는그중6행이다.

train 선행 사용자 발화2개 이상은 D1422→1466, Q537→540, P756→781이다. **사용자 발화가 누적됐다는 구조적 수치이지 여러 선행 발화가 각각 정답 결정에 필수라는 의미 합격 수치가 아니다.** 새10대화/88trace의 D/Q/P 연결·중복alias·모의물리bridge는 이전 `SCENARIO_FLOW_REPORT.json` 결과를 재사용했다. Q는조건부이며 고정실행안내에는P호출이 없어 D/Q/P 개수가 같을 필요가 없다.

## 4. Prompt·few-shot은 실제로 들어가 있는가

현재 Runtime의 렌더링된 system을 직접 import해 과거 학습 계약·새 학습 계약의 문자열 전체와 비교했다. 모델 runtime은 import하지 않았다.

| 계약 | 과거 학습source ↔ 현재 Runtime | 새 후보 ↔ 현재 Runtime |
| --- | --- | --- |
| Decision | 다름 | 동일 |
| Recommendation | 다름 | 동일 |
| Response task/mixed | 다름 | 동일 |
| Response general | 동일 | 동일 |

**새 후보의 D/Q/P prompt가 현재 Runtime과 일치한다.** D의C01~C07 복합commit 대조, P의S10~S12 신규/변경/유지 대조는 실제 렌더링에 포함된다. Q의 최신수량 규칙·시연도 일치한다. 과거adapter는 이 최신 system으로 학습한 것이 아니므로 지금prompt를 넣었다는 사실만으로 새학습과 같아지지 않는다. 문자열 차이 자체는 버그가 아니며 정책 교정에 따른 의도된 변경도 포함한다.

캐시bench와 누적시나리오도 구분한다. `bench_prefix_cache.py`는 로그에 저장된 `messages`를 그대로 재생하고, `run_scenario_cache.py`는 현재모듈에서 prompt를 조립하는harness를 부른다. 따라서 벤치의 과거입력6209토큰에서 추천request가 나왔다는 사실을 현재시나리오8092토큰의 통제성공으로 합칠 수 없다. 두환경을 같은bench입력으로 비교한 속도실험 자체를 부정하는 말은 아니다. 최적화 담당 코드는 바꾸지 않았다.

Cause: prompt 존재·전달과 모델 준수는 별개이며, 서로 다른system/history의 성공과실패는 통제 A/B가 아니다.

Effect: 이미있는 규칙을 “없었다”고 다시 추가하거나 토큰을 계속 늘리는 대신, 실제 실패 입력과대조 입력을 고정한 후 바뀐한요소만 비교해야 한다.

## 5. 다음 보강은 세 Agent를 연결한 한정 묶음으로 한다

1. **D 복합 출력:** 선택+추천, 수락+선택+commit, 변경+기존량보존+commit, allergy+선택+commit의 의미변형과 반대/부정/유예 대조. 같은 문장을 복사해 양만 늘리는 방식은 쓰지 않는다.
2. **Q 맥락 추천:** 매운맛제한→소스/추가재료 대안, 게살allergy→소시지, 사용자 취향에 대한 이유, 근거있는양vs새normal, 현재·미래추천과현재실행의 분리. D가Q를부른 뒤의 state로 연결한다.
3. **P 정상·복구 양쪽:** 신규/변경/유지/미완료, 소스확정누락, pending없음, 앞assistant오발화가있는 실제입력을 포함한다. 실제state와 다르면 확정/완료를 주장하지 않고 빠진정보나현재선택을 자연스럽게 안내한다. history는 보존한다.
4. **연결 검수:** 각 대화의 D정답→실제Graph변경→조건부Q→P입력을 연결한다. 턴전후 주문·선호·제약·pending·수량·commit·plan·응답을 함께 대조한다. 정답주입CPU성공과실제adapter성공은 별도 기록한다.
5. **채택 경계:** 신규데이터의split·중복·의미판정·현재prompt를 고정한 뒤 학습길이·배치검토로 넘어간다. 기존 전체전수감사를 반복하지 않고 영향행만 추가한다. GPU/profile·학습·adapter교체는 이 문서로 실행 승인된 것이 아니다.

위5항은 **권장 후속 작업이며 이번에 실행하지 않았다.** 아직 새복구자료를 만들거나 새adapter가시나리오를성공했다고 말하지 않는다. 최신후보는 여전히 비활성이며 D/P의8192초과길이·전역의역누수·수치편향 미확정 상태도 유지한다.

## 6. 증거와 재현

[실행 스크립트](../training/learning_first_v43/adapter_failure_alignment_20261101/audit.py) · [행별 delta](../training/learning_first_v43/adapter_failure_alignment_20261101/DATA_DELTAS.json) · [31행 원본/후보 전문](../training/learning_first_v43/adapter_failure_alignment_20261101/SELECTED_FULL_EXAMPLES.json) · [prompt비교](../training/learning_first_v43/adapter_failure_alignment_20261101/PROMPT_COMPARISON.json) · [구조분포](../training/learning_first_v43/adapter_failure_alignment_20261101/STRUCTURAL_COVERAGE_NOT_SEMANTIC_SCORE.json)

실제10/31 세세션의6턴은 `REAL_LOG_SELECTED_TURNS.json`에 원형으로결속했다. 11/01 17턴은 기존계측JSON을재사용했다. source/adapter/manifest/prompt/log SHA는 `EVIDENCE_PINS.json`에 기록했다. 31행추출은전체semantic재인증31건이나신규보강31건을뜻하지않는다. 자동delta·hash검사와위대표맥락대조를분리한다.

```bash
/home/roma/miniconda3/envs/gemma4_env/bin/python training/learning_first_v43/adapter_failure_alignment_20261101/audit.py
```

실행exit0. weight/config3조합계측identity일치·원본/후보전split ID중복없음·전체delta생성·현재/후보5system문자열일치를확인했다. 이번과무관한Runtime테스트를다시돌려의미검수합격수로부풀리지않았다. 수정한파일은이보고서와감사스크립트/생성증거뿐이다. 캐시·환경migration·VLM·원본/새후보데이터·운영경로는수정하지않았다.

## 7. 다음 한 작업

현재소스에합칠전면재생성대신 **§5의 D복합출력과 P실패복구를 연결한 보강 묶음의 행목록·정답기준부터 확정**한다. Q는같은흐름의조건부호출로함께검수한다.

Cause: 과거 adapter·현재 prompt·새 교재의 버전이 서로 다르다.

Effect: 잘못 배운 정답과 올바른 정답을 배워도 실패한 경우를 나눌 수 있다. 아직 재학습 효과를 증명한 것은 아니다.
