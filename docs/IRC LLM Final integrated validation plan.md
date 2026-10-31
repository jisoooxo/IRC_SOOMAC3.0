# IRC LLM Final — 통합 테스트·보강·검수 계획

**2026-10-31 현재 실행 상태:** [잔여 작업·D→P 보강 계획](<IRC LLM Final remaining work.md>)을 먼저 읽는다. 아래 테스트/보강/학습 요구는 각각 완료 증거가 있는 범위만 완료이며 새 D→P 집중 검수·토글 구현·새 대본 누적 회귀·신규 데이터 보강은 미실행이다. 17:39/18:07 로그 분석 및 선별 데이터 교차검수는 끝났지만 전체 모델 합격은 아니다. event scope 확장·Q 소스 prompt는 후속 승인으로 일부 적용됐으므로 아래 과거 종료선의 “Runtime/prompt 변경 없음”을 현재 전체 상태로 해석하지 않는다.

[최종 원칙](<IRC LLM Final plan.md>) · [기존 인계](<IRC LLM Final handoff.md>) · [최신 best 실제 회귀](../training/learning_first_v43/final/goals/GL10/docker_v5_20261028/best_stack_regression_20261031/RESULT.md) · [이번 조합 탐색](../training/learning_first_v43/final/goals/GL10/docker_v5_20261028/scenario_combo_search_20261031/STATE.md)
→ **2026-10-31 사용자 지시: 조합 탐색을 멈추고, 계획 MD를 Claude/Codex로 교차검수하고, 실물 테스트 준비용 최신 best 3개 경로까지만 지정한다.** 본 문서는 기존 최종 plan의 목차를 유지하는 후속 실행 명세이다.

## 0. 최신 요구 — 변경 불가 기준

**후속 승인·실행(2026-10-31):** 사용자가 실패 조건을 좁히도록 요청하여 [15조건·Base/best30회 진단](../training/learning_first_v43/final/goals/GL10/docker_v5_20261028/failure_conditions_20261031/RESULT.md)을 추가 완료했다. 아래 종료선은 직전 경로 설정 작업의 범위다. 이번 후속 진단은 전체 조합 탐색 재개가 아니며 Runtime/prompt/학습/ROS 변경도 없다. next_prompt 누락 필드 안내 계약과 D 지시어·복합 의도/direct commit, Q 추천 범위가 후속 국소 개선 후보이다.

**이번 종료선은 계획·교차검수·LLM 모델 경로 설정이다.** 추가 GPU 탐색, 프롬프트 수정, 데이터 생성·정제, 재학습, Docker, ROS 기동, 로봇 동작은 이번에 하지 않는다. 아래 후속 작업은 계획이며 실행 완료가 아니다.

사용자는 `model_runtime.py`에 최신 best 경로를 직접 지정하라고 추가 지시했다. 별도 테스트 선택 분기를 추가하지 않는다. `all_gates_pass=true`로 미통과를 합격 처리하지 않는다. 기존 모델/복구 경로는 보존한다. 기존 manifest 자동 채택 경로와 현재 직접 지정 경로의 관계를 문서·테스트로 명확히 한다.

Python 자연어 키워드 판단·추가 고정 멘트·history 삭제를 도입하지 않는다. ROS topic, busy STT, 물리 실행 검증, 강한 제한 철회 확인은 유지한다. VLM 프롬프트·판정·재시도·카메라와 사용자의 `.claude/settings.json` 변경은 수정하지 않는다.

원인: 세 Agent가 같은 모델의 별도 adapter이면서 대화와 state를 공유하여, 한 Agent의 출력이 다른 Agent의 입력과 다음 턴까지 바꾼다.
결과: 개별 loss나 짧은 정답률만으로 실사용 성능을 확정하지 않고, 단독 원인 분리와 실제 연결 대화를 모두 검증한다.

## 1. 다음 담당자가 시작하는 순서

1. 이 문서와 이번 `REVIEW.md`를 읽고 승인된 범위·현재 모델·미해결 실패를 확인한다. 이전 전수 감사나 끝난 학습을 다시 시작하지 않는다.
2. 실물 테스트 로그의 실제 import 파일·base·3개 adapter 경로/해시·prompt/serializer·세션 ID를 먼저 고정한다. 소스 수정과 이미 떠 있는 ROS 프로세스의 모델 변경을 혼동하지 않는다.
3. 사용자 실물 로그를 같은 턴·같은 state로 재생해 최초로 틀어진 출력부터 분리한다. STT 원문과 사용자 의도는 별도 필드로 기록한다.
4. 후속 순서는 **Response 사실성/질문 목적 → Decision 복합 의도·실행 → Recommendation 적합성/scope → 실제 누적 통합 검수**이다. 이 우선순위는 다른 Agent의 문제를 덮어쓰라는 뜻이 아니다.
5. 정확한 실패 범위를 확정한 뒤 필요한 prompt/데이터만 보강한다. 재학습·실물 재실행은 별도 승인 후 수행한다.

## 2. 현재 확인된 기준점과 증거

### 2.1 코드·모델·데이터를 혼동하지 않는다

- 최신 완료 학습 계보는 10월28일 v5 묶음이다. Decision은 받은 `a519cf2ab8704abb9d1b70ef03c2f7b1-result.tar`, Recommendation/Response는 로컬 완료 학습이다. **10월31일 alignment 보강 후보로 다시 학습한 모델은 아니다.**
- `best`는 `minimum_eval_loss_provisional`이다. 자연스러운 대화·안전 의미·전체 시나리오의 최고 모델을 뜻하지 않는다.
- 종료 전 완성된 조합은 **11개**, 통과 후보는 **0개**였다. 범위는 Decision 1종(best)·Recommendation 2종·Response 8종이며 전 조합/모든 Decision의 검증 완료가 아니다.
- 원본 실제 흐름과 `forced_gold` 진단은 별개이다. 후자의 assistant history는 저작된 올바른 답변이며 실제 연결 대화의 성공 증거가 아니다.
- 사용자 요청으로 PID802808을 SIGINT로 종료했고, PID/GPU 해제를 확인했다. 12번째 부분 결과와 KeyboardInterrupt 영수증은 사용자 중단이며 모델 추론 오류·OOM으로 집계하지 않는다. native Goal 도구는 현재 등록 Goal 없음으로 반환하여 pause 전환이 불가능했다. 완료로 표시하지 않았다.

### 2.2 실물 테스트 준비에 사용할 최신 best

| Agent | 학습 완료 | best 실제 위치 | 가중치 SHA-256 |
|---|---|---|---|
| Decision | 1530step / 6epoch | checkpoint750 / epoch2.944417 | `c2cf15bb3a351c016974b69e5126b427373c9f06a9dbe367b402daf038c8a3d4` |
| Recommendation | 228step / 6epoch | checkpoint100 / epoch2.642140 | `cda007d8e7e985c483a96821cf32e9b6e0fcb8b713e3305c5c0d905e45c7e9de` |
| Response | 546step / 6epoch | checkpoint50 / epoch0.551724 | `7f528012569df694ecefa0bcb2b816c26dec0875995fdbd300e6fb32dc9a5dff` |

다음은 경로 값이며 실행 명령이 아니다. 기존 adapter 파일은 이동·덮어쓰지 않는다.

```text
Decision:
/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/runs/remote_all_a519cf_20261031_r2/results/checkpoints/attempt-d8a4ee465c95443daa47e2caf7fa1ee4/decision/best_adapter

Recommendation:
/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/runs/local_v6_batch_20261028/recommendation-c492d6c7538b/attempt-8f34316e864e429ba8ebdbd8b60f8a8a/recommendation/best_adapter

Response:
/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/runs/local_v6_batch_20261028/response-b1acc16-resume-bed9d88cbc5a/full/attempt-2b74a8919fce4e5690767295942c27d5/response/best_adapter
```

완료 표시 파일·best 선정 기록·실제 weight/config SHA를 함께 대조한다. Response의 best50은 최종 완료 attempt에도 보존된 초기 저장본이다. 마지막546과 같은 가중치가 아니며, 재개 전 디렉터리라는 이유만으로 구형 별도 학습으로 간주하지 않는다.

### 2.3 실제 실패와 성공을 분리한다

확인된 예시는 다음과 같다. 왜 학습된 모델이 그 출력을 냈는지의 근본 원인은 아직 unknown이다.

| 관측 | 확인된 직접 원인 | 분리할 책임 |
|---|---|---|
| 얇은면+조금 요청 뒤 면 종류가 null | Decision raw부터 양 low만 있고 면 종류가 없다 | D 의미 누락. Python이 정상 주문을 지웠다는 증거는 없다 |
| 실제 면 미선택인데 “얇은면을 선택했다” 응답 | Response가 confirmed state보다 많은 사실을 말한다 | P 사실성; D 오류와 별도로 잡는다 |
| 채소 low/low·execution pending에서 명시 실행 후에도 정지 | D raw가 commit 없이 양파 low 수정만 반복한다 | 실행 의도 판단 누락. Python이 임의 실행해서 보완하면 안 된다 |
| 초기 추천에 소스가 없어 추가 소스 발화가 필요 | Q best raw의 sauce가 null이었다 | 넓은면 자체의 오류와 소스 선택 흐름 부족을 구분한다 |
| 게살 알레르기+나머지 선택에서 제한·소시지 확정 누락 | 기존 best 실제 raw가 게살 none patch만 출력했다 | D 복합 의도·지시 대상; 이후 응답만 고쳐 성공 처리하지 않는다 |

## 3. 최종 동작 계약

### 3.1 세 Agent와 Python의 책임

```text
이전 user/assistant history + confirmed state + pending + robot evidence
    → Decision → 상태 반영/안전 검증
                  → 필요할 때만 Recommendation → 후보 검증
                  → Response 또는 기존 제어 발화 경로
                  → 실제 최종 발행문 → 다음 턴 history → 다시 Decision
```

Q 호출 여부는 조건부이다. 정상 호출되지 않은 Q/P를 모델 실패로 세지 않는다. Python이 학습할 자연어 의미를 다시 키워드로 판정하지 않는다. schema·지원 메뉴/양·물리 가능성·강한 제한 충돌·중복 실행·ROS 발행 검증은 보존한다.

### 3.2 추천·확정·실행 전이

추천 후보와 확정 주문, 추천 선택 수락과 실행 수락을 구분한다. 같은 “응, 그걸로 해”도 직전 질문과 실제 pending으로 해석한다. 새 주문+“바로 담아줘”는 order와 commit을 함께 판단하지만, 일반 “넣어줘”만으로 임의 실행하지 않는다. 새 항목 수량 미지정만 normal이며 기존 확정량은 유지한다.

### 3.3 직원형 대화와 남은 코스 추천

“매운 걸 못 드신다고 하셨으니 크림 소스는 어떠세요?”, 단계가 왔을 때 “앞서 말씀하신 취향을 고려하면 페퍼론치노 대신 치즈는 어떠세요?”처럼 이유와 자연스러운 권유를 연결한다. 실제 주문이 확정돼 있으면 재주문 질문 대신 적절한 실행 확인을 한다. 특정 문장 복제를 정답으로 강제하지 않는다.

사용자가 전체 구성을 요청하면 현재 및 앞으로 변경 가능한 단계 전체를 추천할 수 있다. 단계 이벤트의 현재 단계 안내와 사용자 broad 추천을 혼동하지 않는다. “나머지만”은 직전 후보·질문·state의 실제 지시 대상을 해석한다. 명시적 미래 지시도 허용하고, 언제나 현재 section으로 제한하거나 history 전체의 재료를 합치는 규칙을 만들지 않는다.

### 3.4 물리 상태와 완료 표현

선택/실행 승인/실행 시작/실제 완료를 구분한다. 선택만 했는데 “담았어요”, 완료 증거 없이 “다음 단계로 넘어갔어요”라고 말하지 않는다. 이미 실제로 담은 재료를 빼 달라는 요청은 물리적으로 가능한 범위와 불가능한 범위를 설명한다. VLM/완료 증거는 읽기만 하며 이번에 판정 정책을 바꾸지 않는다.

## 4. Runtime 변경 명세

이번 실제 변경은 `model_runtime.py`의 LLM 3개 직접 경로, 선택 상태 문서/manifest의 미검증 표시, 그 경로에 대한 CPU 검사뿐이다. **기존 모듈 전역 `_resolve_runtime_adapter_paths()` 호출을 제거하고 세 경로를 직접 문자열 상수로 정의했다**(`model_runtime.py:66/71/76`). legacy manifest 해석 함수·기존 rollback 상수는 보존하되 활성 경로 선정에는 호출하지 않는다. 정보용 `config/llm_active_stack.json`은 `active=false`, `all_gates_pass=false`이고 동일 경로를 기록한다. 따라서 manifest가 best 상수를 덮거나 미검증 모델을 합격이라고 표시하지 않는다.

실제 소비 지점은 `llm_langgraph_node.py:99`의 `load_model(DECISION_ADAPTER_PATH, RESPONSE_ADAPTER_PATH, RECOMMENDATION_ADAPTER_PATH)`이다. 함수의 인수 순서는 D/P/Q이며 문서상의 D/Q/P 표 순서와 혼동하지 않는다. AST/CPU routing 검사로 상수→Node 인수→이름별 PEFT 등록을 확인한다. 직접 지정 파일이 없으면 Peft 로드가 실패하므로 조용히 구모델을 사용했다고 착각하지 않도록 실제 시작 경로/오류를 확인한다.

복구할 때는 직접 경로 상수 3개를 각각 보존된 `ROLLBACK_*_ADAPTER_PATH`로 바꾼 뒤 사용자가 재기동한다. 정보용 manifest만 바꿔서는 선택이 바뀌지 않는다. 현재는 외부 학습 작업장 export를 직접 참조하므로 파일 이동/삭제와 이 PC 이외의 경로 이식은 별도 작업이다.

Decision adapter config의 `/model`은 원격 학습 경로이고 현재 base 경로는 `/home/roma/Desktop/sLLM/gemma-4-12B-it`이다. 이 문자열 차이만으로 불일치/동일성을 단정하지 않는다. 기존 학습 base weight/processor 증거와 실제 GPU 로드 이력, 이번 weight/config hash·형식 검사를 구분해서 기록한다. CPU 파일 검사는 새 GPU/실물 호환성 전체 검증이 아니다.

기존 실행 검증/Graph/추천 범위/응답 소유권/프롬프트/token/물리 상태 로직은 이번에 수정하지 않는다. 경로 설정 검증에는 ROS를 기동하지 않고 Python 상수·PEFT config·weight SHA·모의 adapter routing 테스트를 사용한다. VLM 영역은 diff가 없어야 한다.

모델은 시작 때 로드되므로 이미 떠 있는 프로세스에는 즉시 반영되지 않는다. 사용자가 수동 기동하기 전 실제 import가 source인지 오래된 install 복사본인지 확인하고, 필요하면 별도 빌드 절차를 따른다. 이 문서만으로 새 ROS 실행을 승인한 것은 아니다.

## 5. 엄격한 프롬프트 작성·검증 기준

### 5.1 공통 규칙

Prompt 후보와 adapter를 동시에 바꾸지 않는다. 실제 system 문자열·schema·serializer·생성 설정·역할·history와 state를 입력 계약으로 동결한다. 과거 assistant의 추측/잘못된 완료를 사용자 사실이나 실제 완료로 승격하지 않는다는 규칙을 명시하되 history는 삭제하지 않는다. 우선 정본 state를 사용하고 모호한 지시는 질문한다.

### 5.2 Decision 필수 대조

복합 발화의 주문+양+제약+추천+수락+즉시 실행이 함께 빠짐없이 반영되는지 검사한다. cannot_eat/allergy와 단순 비선호, 제약 철회와 주문 취소, 일반 선택과 실행을 구분한다. 지시어 예시는 현재 후보/확정량/미래 대상/명확화 대조쌍으로 만든다. Python으로 문장을 다시 해석하지 않는다.

### 5.3 Recommendation 필수 대조

현재/미래 유효 scope, 강한 제한, 이미 확정된 선택과 양, 새 선택의 기본량을 유지한다. broad/narrow 추천 대조를 넣는다. “매운맛→항상 크림·치즈”라는 전역 데이터 편향을 만들지 않는다. 이번 사용자 기준 시나리오의 크림 기대와 별도 일반화 평가를 분리한다.

### 5.4 Response 필수 대조 — 후속 작업 최우선

| 입력 상태 | 허용되는 발화 | 금지할 의미 |
|---|---|---|
| 추천 pending | 후보와 이유·선택 권유 | 후보를 이미 주문/실행했다고 주장 |
| 실제 execution pending + 유효 대상 | 최신 대상과 양에 맞는 실행 확인 | 과거 양·다른 대상으로 확인 |
| pending/next_prompt 없음 | 실제 적용 내용·필요한 정보 요청 | 새 실행 확인을 만들어 루프 유발 |
| 실행 승인 없음 | 주문 선택·변경 사실 | “담기 시작할게요” 뒤 질문을 붙여 시작 주장을 숨김 |
| 실제 완료 증거 없음 | 선택/대기/시작 단계의 정직한 설명 | 완료·이동 사실을 창작 |

Response의 실행 질문과 Runtime의 pending 생성/전달 경로를 함께 검수한다. 발화에 “담을까요?”가 있다는 이유로 pending을 Python이 새로 만들지 않는다. 반대로 모델이 실행 질문을 하지 않았으면 “응”을 무조건 실행 승인으로 간주하지 않는다.

### 5.5 입력 동일성 검사

**후속 보강·미실행:** 추천 없는 직접 주문은 D→검증/최종 state→P를 독립 필수 경로로 검사한다. 주문 누락, 기존량의 normal 덮어쓰기, 없는 면 선택 주장, 주문을 완료로 말하는 문제를 D raw/적용 후/P input/P raw/발행까지 분리한다. 상세 oracle·검사표는 [잔여 계획 §2~3](<IRC LLM Final remaining work.md>)이며, 이 문단 추가는 검사 완료가 아니다.

D/Q/P에 동일한 JSON을 넣는 것이 정렬이 아니다. 같은 dialogue_id/turn_id에서 **D는 해당 발화 전 state, Q는 D 적용 뒤 state, P는 검증된 최종 state/질문 목적/실행 증거**를 받는다. 실제 런타임의 호출 순서·조건·source·역할·누적 history로 직렬화한다. 이전 user를 새 user처럼 복제하거나 event에 가짜 user를 넣지 않는다.

현재 task Graph의 source 근거는 `llm_langgraph.py:646` 제한 반영·`:649` 선호 반영·`:653` 주문 patch 적용·`:661` 제한 검증 주문 반영, `:818` 조건 분기, `:825`~`:832` 갱신된 session을 Q에 전달하는 경로이다. 이는 **Q를 실제 호출하는 정상 task 경로**의 순서이며, 조기 clarify/blocked/general 또는 Q 비호출까지 모두 호출된다는 주장이 아니다. 단계 event는 별도 입력 계약을 따른다. `test_graph_role_refactor.py`의 전후 필드/조건부 방문 검수로 대조한다.

### 5.6 Session·멀티턴 few-shot — 필수 산출물

단일 턴 외 2/3/6/8/12턴 길이를 나눠, 취향→추천→변경→번복→알레르기→지시어→선택/실행의 누적 의미를 검수한다. 같은 마지막 문장에 서로 다른 선행 history/state/pending을 주는 대조쌍을 반드시 포함한다. 예시는 테스트 원문을 복제하지 않고 표현·메뉴·대상·양·실행 질문 위치를 바꾼다.

### 5.7 토큰 예산 증액 — history를 희생하지 않는다

현재 코드 기준 D 입력 상한16384/output1024, Q output1024, P output512, 최근 history20message 창이다. 이 값은 이번 변경 대상이 아니다. 긴 예시 추가 시 실제 tokenizer의 prompt/output 토큰·D history trimming·VRAM·지연·중단 여부를 측정한다. 숫자만 늘리거나 중요한 앞선 제약이 잘린 채 token PASS라고 하지 않는다.

## 6. 데이터 정제·재구성 명세

### 6.1 행별 판정과 삭제 절차

기존 ledger의 hash/semantic_key/KEEP·FIX·DROP·HOLD를 먼저 연결한다. raw 원본을 덮지 않고 새 후보/수정 provenance를 남긴다. 이미 고친 행을 원본의 옛 target만 보고 다시 작업하지 않는다. 재훈련이 필요해도 “세 모델 전부 초기부터”를 먼저 결정하지 않는다.

### 6.2 중복·누수

엄격 JSON 중복, 의미 중복, 대화 prefix/변형 family, template 골격, prompt few-shot과 평가 split 누수를 각각 검사한다. 동일 대화의 턴·표현 변형이 train/test로 갈라지지 않게 dialogue/family 단위로 나눈다. `false/0`, null/누락은 구분한다. 테스트 시나리오 원문과 근접 변형을 보강 train으로 누출시키지 않는다.

### 6.3 확인된 우선 검수 대상

P: 없는 pending의 실행 질문, false start/false completion, 양/재료의 과거값 회상, assistant 추측의 사용자 사실 승격을 먼저 본다. D: 면 종류/제약/선택/commit 중 일부만 빠지는 복합 의도를 본다. Q: 전체 추천 요청이 현재 면으로 축소되는 원인을 입력 scope·raw 제안·후처리로 분리한다. 기존 GL07~10 검수 완료/보류 행과 신규 실패를 중복 집계하지 않는다.

### 6.4 새 데이터 생성 방법

1. 실제 실패 로그에서 익명화한 상황과 독립 의미 oracle를 먼저 작성한다. 현재 모델/Runtime의 출력을 무조건 gold로 쓰지 않는다.
2. canonical session trace를 저작하고 pre-state→D target→적용/검증→Q target(호출 시만)→P input/target→post-state/history를 연결한다.
3. 자연스러운 정답 assistant history를 가진 학습 흐름과, 오염된 history를 포함하되 state로 교정하는 강건성 흐름을 구분한다. 후자를 정상 정답 주장으로 누적하지 않는다.
4. clean prefix·부분 모호성·ASR 유사 오류·불필요한 general/mixed 끼움 턴·사용자 번복을 대조 생성한다. 불확실한 수음은 억지로 확정하지 않는 예시도 포함한다.
5. 구조/상태/독립 의미/언어 검수와 누수 검사가 끝난 후보만 채택한다. 원래 wrong prefix를 gold로 덮어 원래 대화가 성공한 것처럼 만들지 않는다.

### 6.5 언어·편향 검사표

조사/받침/수량 연결, 존댓말 일관성, 긍정/부정의 범위, 지시어와 대명사, 무의미한 반복·장황함을 실제 문맥으로 본다. 한 줄/긴 문장 어느 쪽도 획일 강제하지 않는다. 취향 설명+짧은 제안/질문이 충분한 경우 불필요하게 장문을 학습시키지 않는다.

### 6.6 엄격한 편향 감사 — 데이터·few-shot·평가기 공통

section·메뉴·양·취향/제약 강도·추가/삭제/철회·일반/즉시 실행·history 길이·질문 목적을 조건별 교차 집계한다. 특정 단어 하나와 정답 action을 연결하는 shortcut, 크림/치즈 쏠림, commit 과소/과다, clarification 남발, 현재 section 쏠림을 확인한다. 사용자 보류 상태인 과거60/70/10 등의 수치를 임의로 합격 기준으로 다시 쓰지 않는다. 분류표와 충분한 표본·판정 기준을 확정하기 전에는 수치 편향 감사를 완료했다고 쓰지 않는다.

## 7. 평가기 교정과 합격 기준

### 7.1 먼저 고칠 오채점

다음 반례는 기존 검수 증거를 재사용하고 관련 계약 변경 시만 다시 검사한다: 무승인 시작 주장 뒤 질문 추가, 유제품 제한과 실제 크림 값 충돌, 면만 완료됐는데 크림 완료 주장, “보통 이 정도면”을 양normal로 오인하는 문구 검사. 정상 표현 대조도 포함한다. 정규식으로 문장 의미 전체를 확정하지 않는다.

### 7.2 두 종류의 평가를 분리한다

**구조 평가:** schema/필수 필드/strict boolean/지원 값, 전후 전체 필드diff, 주문·선호·제약·pending/targets/양·scope·robot facts·허용된 plan 횟수를 본다. Q 실패 안내나 정상 skip을 모두 ok/실행으로 강제하지 않는다. 각 기대값을 실제 코드 계약과 oracle로 먼저 확인한다.

**의미 평가:** 사용자 의도 보존·복합 지시·실제 답변 사실성·질문 목적·자연스러움·누적 맥락을 사람이 기준화한 세트로 검증한다. `text_rule_flags`는 검토 표시일 뿐 단독 DROP/합격 근거가 아니다. LLM judge는 인간 adjudication 기준으로 교정하고 judge와 생성/모델 출력의 의견이 갈리면 근거를 남겨 HOLD로 둔다.

### 7.3 Release gate

최소 수집 필드: dialogue/turn/variant ID, adapter/prompt/schema/serializer hash, STT 원문/실제 입력, before/after 전필드, raw/외부JSON/정규화/repair와 원인, Q raw/허용·거부 필드, P raw/실제 발행문, pending/next_prompt/전달 여부, robot/action evidence, plan(offline/실제 구분), token/latency/error, oracle/adjudication.

단독 oracle state 통과, gold 강제 진단, 실제 생성 history 누적, 실물 결과를 별도 판정한다. 로봇이 중간 멈췄으면 그 뒤는 NOT_REACHED이다. 실제 완료·이동 증거가 없으면 생성 문구만으로 완료시키지 않는다. 안전 관련 오실행/금지 재료/false completion은 평균 점수로 상쇄하지 않는다.

## 8. 필수 회귀 시나리오

### 8.0 1차 필수 시나리오 P1 — 사용자 지정

| 구간 | 사용자/상황 | 반드시 볼 것 |
|---|---|---|
| 비매운 추천→면 | “매운 걸 못 먹어. 추천해줘” → “좋아, 얇은면 조금만 담아줘” | cannot_eat 유지, 이번 기준의 크림 제안/수락·면low, 선택과 실행 분리·유효 확인 질문 |
| 채소 번복 | 버섯low·양파high → “아 그냥 둘 다 조금” | 양파만high→low, 버섯low 유지, 일반 변경으로 임의 실행하지 않음 |
| 육류 복합 의도 | 소시지/게살 후보 뒤 “게살 알러지라 나머지만/그거 빼고 담아줘” | 게살 제한 저장+제외, 문맥상 소시지 확정·기존 후보량 유지, 선택/실행 수락 구분 |
| 토핑 회상 | 이후 extra 단계, 매운맛 못 먹음 재확인 | 앞선 제한을 근거로 페퍼론치노 제외·치즈 자연스러운 제안, 철회된/추측된 사실 회상 금지 |
| 명시 실행 | 각 실제 execution pending에서 “바로 담아줘/지금 진행해” | 최신 주문·대상·실행 허용, 중복plan 금지, 실제 완료 전 완료 발화 금지 |

### 8.1 실제 로그와 이상적 흐름 대조

원래 사용자의 발화와 실제 assistant 응답을 그대로 보존한 흐름을 먼저 본다. 중간 실패를 고쳐 진행한 diagnostic branch에는 바꾼 필드/대상/이유/oracle를 기록한다. 새로운 보완 user 발화가 필요했던 흐름을 원래 발화만으로 완주했다고 쓰지 않는다. 실패 이후 능력 관측은 허용하되 실제 누적 성공과 분모를 합치지 않는다.

### 8.2 추가 필수 대조군

| 묶음 | 대조 사례 |
|---|---|
| 실행 의도 | 새 주문+즉시 실행 / 이전 execution 수락 / 일반 주문 / 추천 수락 / 명시적 skip / 대상 없는 “바로 해줘” |
| 주문·제약 | 기존량 유지 / 선호 철회 / 확정 주문 취소 / 강한 제한 철회 확인·거절 / 이미 담은 재료 변경 / 미래 단계 사전 주문 |
| 질문 목적 | recommendation·execution·clarification·restriction 철회 확인 / pending 없음 / stale pending·미전달 질문 / 동일 “응” |
| 대화 다양성 | general·mixed 끼움 / 부정·조건부 수락 / 조사·받침 / 정중·구어·오타·ASR / 관련 없는 오래된 과거 |
| grounding | 선택만 됨 / 승인됐으나 미시작 / 시작 기록만 있음 / 일부 재료만 완료 / 다음 단계 전환 증거 없음 |

### 8.3 복합 멀티턴과 모델 판단 자유도 — 필수 검수

같은 마지막 발화의 의미가 2/3/6/8/12턴의 다른 선행 맥락에서 어떻게 달라지는지 본다. history 길이만 세고 복합 의미 검수를 끝내지 않는다. 전후 diff에는 바뀌어야 하는 필드, 보존해야 하는 필드, 바뀌면 안 되는 필드를 미리 명시한다. 정답 모델 출력을 강제로 맞추는 Python 규칙을 새로 추가하지 않는다.

## 9. 저장소·학습 디렉토리 정리

기존 ledger/원본 데이터·학습 run·checkpoint·tar는 보존한다. 이 MD는 docs에 두고 실제 검수 영수증/실험 로그는 `training/learning_first_v43/final/goals/GL10/docker_v5_20261028/real_robot_test_plan_20261031/`에 모은다. 이번 작업에서 대량 파일 이동/삭제를 하지 않는다. source별 날짜→입력 모델/설정→실행/미실행→결과→다음 한 단계를 기록한다.

## 10. Agent별 비교·재학습·Docker

우선순위는 P 사실성/질문 목적이지만 D/Q 상태 의미도 동시에 독립 원인으로 보존한다. 단독 비교는 oracle 입력, 실제 연결 비교는 실제 생성 history를 사용한다. **한 번에 하나의 adapter 또는 하나의 prompt만 변경**한다. 기본 Q best 고정·D epoch1/2/3 우선·P 후보 비교는 후속 탐색안이며 이번에64/320조합 탐색을 재시작하지 않는다.

재학습 필요성은 기존 가중치/현 prompt → prompt-only → 최소 계약 수정이 실제 필요한 경우 → 데이터 보강 후 adapter 재학습 순으로 Agent별 판단한다. prompt 실패를 덮으려고 Python 문구를 늘리거나 단독 loss가 낮다는 이유로 전부 재학습하지 않는다. 잘못된 학습 예시가 확인되면 기존 자료 KEEP/FIX/DROP/HOLD와 신규 aligned trace를 먼저 확정한다. 원격 기존 작업 취소·checkpoint 교체·학습/Docker 제작은 새 승인 없이는 하지 않는다.

## 11. 작업 패키지 — 다른 Agent가 그대로 실행할 목록

| 단계 | 담당·허용 산출물 | 통과 조건/경계 |
|---|---|---|
| 현재 A | Codex: 계획 MD·최신 best 경로·파일/해시/routing CPU 검사 | source/모델 선택 일치, 미검증 표시, 복구 경로 보존, ROS/학습 실행0 |
| 현재 B | Claude: 이 MD와 LLM 경로 변경의 읽기 전용 교차검수 | 누락·모순·과도한 방어·정답 오염·잘못된 합격/선택 주장 지적, 수정 반영 장부 |
| 후속 C | 사용자 실물 테스트 → 새 로그 수집·재생/원인 분리 | 실제 로드 identity와 STT/발행/물리 증거 분리, 없는 완료 창작 금지 |
| 후속 D | 승인된 prompt/정렬 데이터 보강·독립 의미/편향/누수 검수 | family split, 필드/전후/복합 의미, 모호한 행HOLD, 기존검수중복0 |
| 후속 E | 별도 승인된 필요한 Agent 재학습·새 회귀·최종 채택 | 실제 user-history 기반 흐름과 일반화/안전 대조 통과 후 채택 |

후속 검수자는 미확인 원인을 추측으로 단정하지 않는다. Claude 의견과 Codex 의견이 다르면 실제 코드/입력/전후 state를 우선 대조하고 상충 이유를 남긴다. Claude 검수 보고는 실물 테스트 또는 데이터 원본 전수검수 완료가 아니다.

## 12. 최종 완료 정의

이번 작업은 MD 초안→Claude/Codex 교차검수→조건 반영, 최신 best의 코드 경로/실제 artifact/SHA 일치, 제한된 CPU 회귀와 VLM/ROS 무변경 확인으로 끝낸다. 사용자 실물 테스트의 성공이나 재학습 완료라고 쓰지 않는다. 기존 조합 탐색은 중단 상태로 보존하며 자동 재개하지 않는다.

장기 최종 채택에는 실제 생성 history 기반 전체 P1 및 변형·추가 대조, 안전/실행/pending/응답 일치가 필요하다. **미검증 best 준비와 검증 완료 채택은 서로 다른 상태이다.**

## 13. 이번 계획 작성의 진행 기록

- 2026-10-31: 사용자 요청으로 조합 탐색 종료. 완성11/320·후보0, 다음 부분 실행과 원본 cache/trace 보존. 모델 ranking 또는 전체 Decision 검증 완료로 확대하지 않는다.
- 사용자 추가 지시: 코드에 최신 best 경로를 직접 지정한다. 테스트 선택용 새 분기는 추가하지 않는다. 경로 변경을 의미 gate 통과로 표시하지 않는다.
- Claude 계획 검수: 조건부, 반대 없음. 직접 경로와 manifest 우선권 명세·Node 소비 지점·폴더 날짜/SHA 구분·D 적용 뒤 Q 전달 근거의4조건을 주담당자가 source/검사와 대조해 이 문서에 반영했다. Claude의 최초 의견은 적용 전 코드를 기준으로 한 것이며 최종 코드 재검수와 구분한다.
- Codex 계획/경로 변경안 검수: 조건부 찬성. 미활성 정보 manifest, 직접 경로 복구 절차, 실제 weight 동일성/adapter 형식/원격 base 경로 차이의 해석을 반영했다. 적용 후 최종 Codex 한정 검수는 찬성이다.
- 최종 CPU79건 PASS/3.32초/exit0. 실제 best3개 weight/config SHA와 runtime/eval processor5파일 동일, 공유 VLM/loader 함수 AST 및 나머지 동결 LLM source 불변, develop 설치가 수정 source를 참조함을 확인했다. 이는 새 GPU/실물 성능 합격이 아니다. 상세 [REVIEW.md](../training/learning_first_v43/final/goals/GL10/docker_v5_20261028/real_robot_test_plan_20261031/REVIEW.md).
- 이번 종료선에 도달했다. 사용자 실물 테스트를 기다리며 자동 탐색·새학습·프롬프트 변경을 시작하지 않는다.

## 용어 정리

| 용어 | 뜻 |
|---|---|
| 정렬(alignment) | 같은 실제 턴의 역할별 pre/post state와 의도·응답·실행 증거가 일치하는 것 |
| 연결 흐름(closed loop) | 모델의 실제 발행 답변이 다음 턴 history로 들어가는 누적 대화 |
| 강제 진단(gold-forwarded diagnosis) | 올바른 저작 맥락으로 뒤 단계 능력을 따로 관찰하는 시험; 원래 흐름 성공이 아님 |
| provisional best | validation loss 기준의 임시 저장본; 대화 최고 성능 확정 아님 |
| HOLD | 의미가 모호하거나 근거가 부족해 활성 정답으로 채택하지 않은 검수 결과 |

(내가 의역해보자면, Response의 말 한마디가 다음 Decision의 판단 조건이 되므로, 세 모델을 따로 잘 학습시키는 것과 함께 같은 대화의 흐름을 서로 맞춰야 한다.)

## 14. 2026-10-31 최신 추천·학습 데이터 정책

[별도 보강 계획](../training/learning_first_v43/final/goals/GL10/dataset_supplement_plan_20261031/PLAN.ko.md)이 이번 후속 범위이다. 전체 주문과 남은 변경 가능 코스를 추천 기준으로 고정하고 상황에 맞게 제안한다. 현 단계만 보는 추천과 소스 없는 면 안내를 보강한다. 사용자 원문 시나리오도 학습 후보에 포함하며 이전 보호 회귀 F01은 새 버전에서 seen 회귀로 분류하고 별도 unseen 평가를 확보한다. 기존 봉인본/split은 이번에 변경하지 않았다. 이 정책을 Runtime/prompt/data에 이미 적용했다고 보고하지 않는다. 데이터 검수·생성은 계획 다음 별도 작업이다.
