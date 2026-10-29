# IRC LLM Final — 그래프 역할 분리·평평한 Python 리팩토링 계획

[상위 계획](<IRC LLM Final plan.md>) · [인계](<IRC LLM Final handoff.md>) · [작업 범위](../AGENTS.md) · [현재 그래프](../src/soomac_irc/soomac_irc/llm_langgraph.py) · [ROS 연동](../src/soomac_irc/soomac_irc/llm_langgraph_node.py)
→ **Recommendation을 독립 노드로 드러내고 Python은 처리 순서가 읽히게 정리한다. 기능·학습 계약은 바꾸지 않는다.**

작성: 2026-10-29 KST. **계획 작성 후 사용자 `ㄱㄱ`로 구현 승인, R1~R5 완료. 최종 778건·2subtests 통과, 독립 Codex 검수 찬성.** [구현 결과](../training/learning_first_v43/final/audits/2026-10-29_graph_refactor/RESULT.md). 기존 계획 작성만으로 자동 구현하지 않았으며 후속 승인을 받아 진행했다. 별도 GL01~15 재개·학습 재시작·원격 변경 승인은 아니다.

## 0. 기준점과 읽는 순서

1. 이 문서에서 범위·불변 조건·실행 순서를 읽는다. 기존 프로젝트 전체 감사나 데이터 검수를 처음부터 반복하지 않는다.
2. `AGENTS.md`와 상위 계획의 사용자 고정 조건을 확인한다. 과거의 “학습 미실행” 같은 문구를 현재 상태로 복사하지 않는다. 학습의 실제 상태는 해당 학습 실행 기록으로 따로 확인한다.
3. `llm_langgraph.py`의 `build_graph`, `process_task`, `generate_response`, `generate_section_event_response`를 읽는다.
4. 모델 입력 생성기·기존 테스트를 확인하고 변경 전후 비교의 기준점을 고정한다.
5. 코드 변경은 사용자가 구현을 승인한 다음 아래 R1부터 순차 진행한다.

계획 조사 시 branch는 `llm_final`, HEAD는 `4f084f686f050322611c4a080393fa964d7c53d0`이고 tracked 변경은 없었다. 이 문서와 연결 안내를 추가한 뒤에는 문서 변경만 생긴다. 실제 구현 직전 Git 상태를 다시 확인해 사용자 변경을 보존한다. 원본 `flexible_llm` 수정·자동 commit/push는 금지한다.

### 0.1 과거 코드에서 가져오는 것과 가져오지 않는 것

2026-10-06 `6d927dc`의 부모 코드에는 `run_recommendation` 독립 노드가 있었다. 해당 리팩토링에서 그래프가 11개 노드에서 4개로 줄고 추천 호출이 `process_task` 안으로 들어갔다. 현재 그래프 주석은 “Graph를 평평하게 둔다”고 설명한다.

**과거 커밋은 스타일 참고일 뿐 동작 복원 기준이 아니다.** 초기 반환, 위에서 아래로 읽히는 구성, 의미 있는 단계 구분과 한국어 주석을 참고한다. 삭제됐던 자연어 키워드 guard, 과거 state field, 옛 pending 정책, 옛 추천 scope·고정 멘트를 다시 가져오지 않는다. 10월 11일 고정 발화 기준을 포함한 기존 최신 계약을 보존한다.

원인: 노드 개수를 줄이는 것과 함수 내부를 단순하게 만드는 것은 서로 다른 설계 선택이다.
효과: 추천 노드는 독립시키면서 각 함수는 적은 중첩으로 읽을 수 있게 한다.

## 1. 목표와 작업 경계

### 1.1 이번에 바꾸려는 것

| 대상 | 허용하는 구조 변경 | 반드시 유지할 것 |
|---|---|---|
| 그래프 | Recommendation 호출을 실제 독립 노드로 이동 | 기존 호출 조건·횟수·입력 시점·예외 처리·후속 정책 |
| 태스크 처리 | 추천 전 상태 반영과 추천 후 실행 정책을 의미 있는 단계로 분리 | 기존 우선순위와 조기 반환 결과 |
| Python 구성 | 불필요한 중첩·중복 전달을 줄이고 명시적인 초기 반환 사용 | 검사 조건·타입·반환값·부작용·예외 경계 |
| 주석 | 목적·상태 변경·안전 경계를 짧은 한국어로 설명 | 실제 코드와 일치하는 설명, 가짜 완료 주장 금지 |
| 테스트·인계 | 구조와 동등성 검사를 추가하고 변경 범위를 기록 | 기존 oracle·성공 회귀·실패 원기록 |

### 1.2 파일별 수정 상한

| 파일 묶음 | 계획상 역할 | 제한 |
|---|---|---|
| `llm_langgraph.py` | 우선 구현 대상. 그래프 결선·태스크 전후 단계·추천 노드·주석 | 실제로 호출되는 노드여야 한다. 장식용 노드 추가나 기존 내부 호출 중복 유지 금지 |
| `llm_langgraph_node.py` | 다음 순서의 LLM 연동 가독성 정리 | `_process_turn`의 LLM 상태/발행 결과 기록, 단계 이벤트의 LLM 호출 부분 등 사전 지정한 영역만. 카메라·VLM·ROS 제어·worker/thread 정책은 수정하지 않는다 |
| 세 Agent 모델 래퍼·`response_contract.py`·`llm_policy.py` | 입출력 계약의 비교 기준. 필요한 설명 주석만 후보로 검토 | prompt/schema/직렬화/생성 설정/validator 의미 변경은 이번 범위 밖. 본문 변경이 필요하면 먼저 이유·영향 범위를 분리 보고한다 |
| `agent_contract.py` | 공개 계약의 불변 기준 | SessionState/Decision/RobotState/new_turn_state 공개 key·enum·기본값은 변경하지 않는다. private 그래프 전달용 타입은 우선 그래프 모듈 내부에서 해결한다 |
| LLM 관련 테스트·이 문서·변경 기록 | 동등성·구조 검사 및 인계 | VLM 테스트를 고치거나 기존 기대값을 새 동작에 맞춰 낮추지 않는다 |

### 1.3 금지하는 변경

1. 데이터셋·prompt·few-shot·모델 입력 창·토큰 한도·생성 옵션·adapter 로딩·학습 설정을 변경하거나 진행 중 학습을 중단·재시작하지 않는다. GPU 추론 시험도 이번 구조 검증의 기본 절차가 아니다.
2. history를 삭제·요약·마스킹하거나 새로운 Python 자연어 규칙, 키워드 gate, 자동 수락, 자동 추천 확정, 새로운 고정 대화 멘트를 추가하지 않는다.
3. VLM prompt/판정/재시도/이미지·카메라 경로, 물리 완료 사실, ROS 토픽·메시지·lid/sauce 자동 제어, busy STT·TTS 정책을 변경하지 않는다.
4. 새 성능 향상·말귀 개선·학습 완료를 구조 리팩토링의 성과로 주장하지 않는다. 모의 모델 검사와 실제 모델 의미 성능은 구분한다.
5. 전체 파일 롤백·저장소 전체 포맷·대량 파일 이동·원격 작업 변경·자동 commit/push를 하지 않는다. 발견한 기존 버그를 별도 승인 없이 함께 고치지 않는다.

## 2. 그래프 역할 분리 설계

### 2.1 현재 구조에서 주의할 사실

현재 추천은 `process_task` 중간에서 실행된다. 그 앞에는 pending 해소·제한/취향 적용·주문 검증·강한 제한 철회 확인이 있고, 뒤에는 거절 처리·skip·commit·새 execution pending 생성이 있다.

**단순히 `process_task → Recommendation → Response`를 덧붙이면 안 된다.** 기존 내부 호출을 남기면 이중 추천이 되고, 호출을 제거한 채 실행 정책을 먼저 끝내면 최신 추천 pending과 실행 차단의 순서가 달라질 수 있다.

### 2.2 목표 결선 — 설계안

```text
START → interpret_decision
           ├─ general → process_general ───────────────────────┐
           └─ task/mixed → prepare_task                        │
                              ├─ 이미 판정 종료 ──────────────┤
                              ├─ 추천 불필요 → finalize_task ─┤
                              └─ 추천 필요 → run_recommendation
                                                  ├─ 판정 종료 ┤
                                                  └─ 계속 → finalize_task
                                                               │
                                                generate_response → END
```

노드명은 역할을 설명하기 위한 후보이며 구현 시 기존 caller·검사기 영향과 함께 확정한다. 과거 11개 노드를 그대로 복원하지 않는다. Decision·Recommendation·Response 세 모델 역할은 구분하되 Python의 상태/안전 처리도 명시한다. `generate_response` 노드가 지나간다는 것과 Response 모델을 반드시 호출한다는 것은 다르며 기존 우회/fallback 조건을 그대로 보존한다.

### 2.3 각 노드의 책임

| 단계 | 책임 | 하지 않는 일 |
|---|---|---|
| `interpret_decision` / `process_general` | 기존 해석·최대 1회 구조 repair·general/clarify 경로 유지 | Python으로 사용자 의도 재해석, task 변경을 general로 누락 |
| `prepare_task` | 기존 pending 해소, 제한/취향/주문 반영, 오류/강한 제한 철회 등 선행 종료 판정 | 최신 제약 반영 전 추천, 아직 남은 commit 검사를 미리 확정 |
| `run_recommendation` | 기존 조건으로 모델 1회 호출, 기존 결과 검증, 후보/pending/policy 갱신 | 모든 턴 무조건 추천, 후보 자동 주문 확정, 새 자연어 검사, 중복 호출 |
| `finalize_task` | 기존 추천 후 거절·대체 주문·skip·commit·execution pending·최종 reason 처리 | 모델 재호출, 이미 적용한 주문/제약 중복 적용 |
| `generate_response` | 기존 최종 state·추천 결과·next_prompt·실행 권한 전달 및 기존 history 기록 | 새로운 ROS 명령·실행/완료 사실·확인 질문 권한 생성 |

추천 노드 조건은 기존의 검증된 Decision `recommendation.action`과 선행 종료 상태로 결정한다. 자연어 문자열을 새로 검색하지 않는다. 추천 수락·거절만 하는 턴을 새로운 추천 요청으로 바꾸지 않는다.

### 2.4 추천 뒤에도 이어지는 현재 주문 commit을 보존한다

현재는 추천 후에도 명시적 commit이 있고 pending이 현재 실행을 막지 않으면 후속 실행 검사로 이어질 수 있다. 따라서 **추천은 무조건 Response로 끝난다는 새 규칙을 만들지 않는다.** 특히 미래 치즈 후보와 현재 확정된 소시지 실행, 추천 생성 실패와 독립된 현재 주문 commit을 원래 코드·독립 oracle 양쪽으로 대조한다. 반대로 새 미수락 후보가 현재 실행을 막아야 하는 경우는 그대로 차단한다.

### 2.5 단계 이벤트 경로를 사용자 턴과 혼동하지 않는다

`generate_section_event_response`에도 기존 추천 호출이 있다. 이 경로를 빠뜨리거나 사용자 턴 그래프로 무조건 우회시키지 않는다. 첫 노드 분리 단계에서는 이벤트 경로를 그대로 보존하고 사용자 턴의 추천 호출만 이동하는 것을 기본으로 한다. 이후 공통화가 필요하면 별도 동등성 검사 뒤 결정한다. 이벤트에 가짜 user message나 새로운 Decision 추론을 추가하지 않는다.

사용자 추천은 기존 현재+미래 가능 범위를 유지한다. 이벤트 추천의 기존 현재 단계 범위와 기존 후보 재검증·확정 주문 확인·fallback 조건도 유지한다. 두 scope를 편의상 하나로 합치지 않는다.

## 3. 평평한 Python과 주석 기준

### 3.1 코드 구성

1. 한 함수에서 정상 경로가 위에서 아래로 읽히게 한다. 기존 우선순위를 확인한 뒤 종료 분기는 초기 반환으로 둔다. 조건을 앞당겨 조회·반영 순서가 바뀌면 단순 스타일 수정이 아니므로 채택하지 않는다.
2. 함수는 모델 호출·추천 검증·상태 반영·실행 정책처럼 의미 있는 단계로만 나눈다. 한 줄 전달 함수·범용 dispatcher·strategy/plugin·새 decorator 체계·generator continuation을 만들지 않는다.
3. 기존 validator·입력 생성기·모델 래퍼를 그대로 사용한다. 같은 분기/검사를 여러 노드에 복사하거나 state를 다시 재생해 임시 값을 복원하지 않는다.
4. 현재 코드의 타입·변수명·import 스타일을 최대한 유지한다. 삼항식·`getattr`·동적 호출·압축된 한 줄 조건으로 분기를 숨기지 않는다. 단순한 dict/조건문/명시적 반환을 우선한다.
5. 독립 노드는 직렬 모델 호출이다. 노드 분리가 모델 호출 병렬화·재시도 증가·thread/lock 변경을 뜻하지 않는다.

### 3.2 주석 기준

함수 앞에는 필요한 경우 목적과 주요 부작용을 1~3줄로 적는다. 함수 안에는 실제 안전 경계·선행 상태·조기 종료 이유만 설명한다. 모든 줄을 한국어로 번역하거나 형식적인 설명란을 반복하지 않는다.

설명용 주석 예시이며 붙여넣기용 구현 코드는 아니다.

```text
# 이번 턴의 제한·취향·주문 변경을 먼저 반영한 상태로 추천한다.
# 추천은 검증된 후보를 pending에만 보관하며, 여기서 주문을 확정하지 않는다.
# 미래 추천 때문에 독립된 현재 주문의 명시적 실행 요청을 버리지 않는다.
# Response에는 이번 턴을 아직 history에 추가하지 않은 상태를 전달한다.
```

오래된 주석이 현재 코드와 다르면 근거를 확인한 뒤 설명만 교정한다. 주석을 맞추려고 실제 동작을 바꾸거나 원래 없던 성공·완료 사실을 작성하지 않는다. 수정 스니펫을 제시할 때는 import와 각 변수의 정의·분기별 생존·pop/del·도달 가능성을 실제 파일에서 확인한다.

## 4. 노드 사이 상태 전달과 공개 계약

### 4.1 분리 때문에 잃으면 안 되는 임시 값

현재 함수의 `pending_result`, `applied`, `accepted_patch`, `restriction_validation`, `restriction_enforcement`는 추천 후 분기에서도 쓰인다. 구현 전 실제 정의·사용 줄과 모든 조기 반환을 표로 연결한다. `previous_session`은 턴 시작 스냅샷으로 유지하고, 이를 추천 후 상태로 덮어쓰지 않는다.

우선안은 그래프 모듈 내부의 private 작업 문맥 하나로 필요한 임시 값만 전달하는 것이다. 전역변수·클로저의 가변 턴 상태·노드 밖 side channel로 보관하지 않는다. private 문맥을 SessionState, Decision schema, prompt payload, pending, action_history 또는 ROS 메시지에 섞지 않는다. public key에 임시 제어 플래그를 숨겨 넣지 않는다.

### 4.2 private 문맥 격리의 통과 조건

현재 설치 환경의 `StateGraph` 생성자 시그니처에 `input_schema`와 `output_schema`가 있음을 읽기 전용으로 확인했다. **실제 필드 격리가 검증된 것은 아니므로** R1에서 작은 CPU 대조 테스트로 동작을 확인한다. 공개 TurnState는 그대로 두고 모듈 내부 그래프용 타입/입출력 투영으로 임시 문맥을 격리하는 방식을 우선 검토한다.

`new_turn_state()`가 반환하는 현재 8개 공개 key와 SessionState 5개 key는 변경하지 않는다. `graph.invoke()`의 공개 반환 key·타입·값도 baseline과 같아야 한다. 내부 상태를 `None`으로 지우는 것만으로 반환 key가 추가되지 않았다고 간주하지 않는다. `stream/get_state` 등 실제 소비자가 있으면 별도 목록화하고 노드 구조 관측의 의도된 변경과 업무 데이터 누출을 구분한다.

분리 때문에 영속 상태나 모델 입력 계약을 바꿔야만 한다면 이 계획 안에서 임의 확대하지 않는다. 사유와 대안을 보고하고 사용자 결정을 받는다.

## 5. 기능 보존 불변 조건

| 묶음 | 변경 전후 비교 대상 |
|---|---|
| 상태 | session의 order/preferences/pending/history/action_history 전필드, previous_session, robot_state, Decision, policy의 bool/null/enum·nested값, recommendation_result, reply |
| 모델 계약 | 세 Agent의 실제 전달 인자·직렬화 입력·history 역할/순서/창·prompt/schema·tokenizer 설정·호출 순서·횟수·오류 경계 |
| 실행 권한 | 일반 주문/추천 수락/실행 수락/직접 commit/skip/강한 제한 철회·물리 불가·busy·중복 실행 검증, no-op과 정상 실행 양쪽 |
| 응답·질문 | next_prompt와실제pending대상·수량, 모델 소유 응답과 기존 고정/fallback 분기, 실제 발행문과history 일치, 가짜 완료/진행·권한 없는 실행 질문 금지 |
| 연동 | 사용자/이벤트 구분, 입력 복사본 보존, ROS 발행 mock payload/횟수/순서, STT 상태, 자동 진행, VLM/실제완료 증거 읽기 전용 전달 |

모의 모델로 비교할 때 응답 문자열까지 정확히 같아야 한다. 시간·지연·원래부터 비결정적인 로그 식별값은 정확한 제외 목록을 먼저 고정한다. 정책·주문·질문·plan 값의 차이를 비결정성이라는 이유로 제외하지 않는다. 그래프 node 이름/edge/내부 stream 단계의 의도된 차이는 업무 결과와 분리해 기록한다.

원인: 노드 이동만으로도 state 읽기 시점과 조기 반환이 바뀔 수 있다.
효과: 최종 답변만 비교하지 않고 입력→중간 상태→모델 호출→최종 상태→발행 결과를 함께 비교한다.

## 6. 검증 시나리오

### 6.1 추천과 실행의 결합

| 사례군 | 필수 대조 |
|---|---|
| 제약+추천 | “매운 걸 못 먹어/게살 알레르기 있어+추천”에서 최신 제약을 추천 입력에 먼저 반영. 위험 후보 전체/부분 거절, 모델 이유 원문 보존 |
| 추천 생명주기 | request/revise/수락/거절/부분 수락/후보 수정, 독립된 현재 주문 변경 시 미래 후보 보존, 오래된 후보 무효화, 호출0/1회 |
| 즉시 실행 | 새 주문+commit, 유효 execution pending 수락, 추천 수락+commit, 현재 주문+미래 추천+commit, 빈 주문·현재/미래 pending 대조 |
| 오류·선행 종료 | Decision 구조 repair·duplicate key·clarify, unsupported/invalid/protected/제약 충돌, 추천 예외/빈 후보/잘못된 schema, Response 예외/빈 문자열 |
| 제어·상태 질문 | general/mixed/no-op, 거절+대체 주문, optional skip, busy/완료항목/오래된 실행 확인, 강한 제한 철회 수락·거절·deferred |

### 6.2 복합 멀티턴과 이벤트

| 사례군 | 필수 대조 |
|---|---|
| 전체 사용자 흐름 | 매운맛 제한+기분 추천→크림 선택→얇은면 low→버섯 low/양파 high→둘 다 low 번복→게살 알레르기+나머지 선택→토핑 제한 반영→현재 단계 실행 |
| 복합 참조 | 2/3/6/8/12턴 등 기존 episode를 재사용하고 단일턴 내 복합의도도 포함. “나머지/그거 빼고”의 같은 문장·다른 history/pending 대조 |
| 수량·철회 | 기존 high/low 유지, 새 명시 수량 우선, 선호 철회≠주문 취소, 제한 철회≠이미 담은 재료 회수, 관련 없는 필드 불변 |
| 대화 삽입 | general·상태 질문·혼합 발화·수음 모호성·잘못된 과거 assistant 질문을 끼워도 필요한 state/history/질문 권한 보존 |
| 단계 이벤트 | 선호-only 추천/이미 확정된 주문 확인/기존 후보 재검증/후보없음·실패 fallback/연속 이벤트·lid/sauce, 새 user 없이 event 유지 및 중복 실행 없음 |

복합 멀티턴은 턴마다 기존/신규 세션을 각각 이어받아 전체 episode를 끝까지 대조한다. 각 턴을 독립된 정답 state로 리셋해 비교한 것을 누적 멀티턴 성공이라고 보고하지 않는다. fixture의 Decision/추천/Response는 고정 oracle/모의 출력이며 실제 LLM의 말귀·자연스러움 합격 근거가 아니다.

### 6.3 우선 재사용할 실제 테스트

`test_natural_multiturn_contract.py`, `test_gl04_state_contract.py`, `test_gl05_response_contract.py`, `test_gl05_fast_path.py`, `test_section_question_events.py`를 핵심 묶음으로 사용한다. prompt·호출 계약 확인에는 `test_gl06_prompt_contract.py`, adapter 경로에는 `test_adapter_routing.py`, 평가기 호환에는 기존 evaluator 테스트를 참조한다. VLM 테스트는 읽기/수정/재생성 대상으로 삼지 않는다. 이번 계획 작성에서는 이 테스트들을 실행하지 않았으며 과거 테스트 수를 새 실행 결과로 복사하지 않는다.

추가할 검사는 실제 독립 노드 결선/조건부 방문/중복 호출 부재, private 문맥 비누출, baseline differential, 복합 episode 전후 필드 비교이다. 기존 테스트를 동일한 이름으로 대량 복제하지 않는다.

## 7. 구현 승인 후의 순서

| 단계 | 한정 작업 | 통과해야 다음 단계로 이동 |
|---|---|---|
| R1 기준 고정 | Git/허용 파일 hash·기존 테스트 결과, 모델 입력/호출 캡처, private 문맥 입출력 격리 시험, 현재 코드 baseline fixture 고정 | 원래 실패와 새 실패 구분, 동등성 관측점·모호한 사례 목록, 임시 값 전달 설계 확정 |
| R2 그래프 분리 | `llm_langgraph.py`에서 추천 전/추천/후속 정책 분리 및 실제 조건부 결선 | 기존 추천 호출 제거·중복0, 선행 종료/후속commit 보존, 공개 state 불변·전체 differential 일치 |
| R3 가독성·주석 | 그래프 파일 마감 후 `llm_langgraph_node.py`의 사전 지정 LLM 구간 정리, 필요 주석 보완 | ROS/VLM/worker/STT 정책 불변, 실제 변수 생존/예외 경계·복사 시점 보존, 파일별 국소 회귀 |
| R4 통합 회귀·검수 | 핵심 CPU/mock 테스트, 누적 episode, payload/발행 대조, 독립 Codex sol/high diff 검수 | 신규 실패0, 독립 지적 국소 교정 후 영향 테스트 재실행, 미실행 검증 명시 |
| R5 인계 | 변경 전후 노드 지도·파일/함수 목록·불변 근거·실제 테스트 명령/결과·남은 위험 기록 | 사용자 보고 후 중지. 학습/모델교체/실물ROS/원격/commit/push로 자동 확대하지 않음 |

구현은 R1→R2→R3→R4→R5 순차가 기본이다. 같은 파일·정책을 여러 작업자가 동시에 수정하지 않는다. 별도 병렬 작업은 필요성과 독립 범위가 확인될 때만 승인 범위에서 수행한다. 이번 계획 작성 자체는 멀티에이전트 구현을 시작하지 않는다.

테스트는 설치된 환경과 실제 의존성을 R1에서 확인하고 CPU/mock 범위로 실행한다. 학습 중인 GPU에 추가 모델을 로드하거나 live ROS 노드를 생성하지 않는다. 기존 AST 기반 Node 테스트의 제한을 기록하며 mock 통과를 실제 ROS 전달 성공이라고 표현하지 않는다.

`_process_turn`의 중첩 `write_turn`을 정리한다면 `turn_id`, 시작 시간, 변경 전 session/robot_state, 현재 result, 실제 발행 reply의 캡처·정의·갱신 시점을 먼저 연결한다. `result=None`인 오류 경로와 finally/초기 반환에서도 로그가 달라지지 않아야 한다. 새 의미 있는 메서드가 필요하면 `test_section_question_events.py`의 AST 메서드 선택 목록도 함께 검토한다. fixture가 새 helper를 로드하지 않아 생긴 실패를 모델 실패로 보고하거나 테스트를 건너뛰지 않는다. harness 수정은 실제 helper 로드만 보강하고 원래 상태·발행 assertion은 유지한다.

## 8. 실패·중복 작업 방지

1. baseline과 oracle이 충돌하면 충돌을 별도 기록한다. oracle이나 기존 기대값을 새 결과에 맞춰 고쳐 통과시키지 않는다. 기존 버그 수정은 이 구조 리팩토링에 몰래 섞지 않는다.
2. 실패는 영향을 받은 분기·입력·변경 파일로 좁힌다. 완료 데이터 재감사·전체학습 재시작·과거커밋 전체복원으로 대응하지 않는다. 동일한 가정으로 세 번 실패하면 중단하고 의심되는 가정을 보고한다.
3. 한 번 통과한 검사/검수는 입력·소스 hash가 그대로면 근거를 재사용한다. 해당 변경이 영향을 주는 검사만 추가 실행한다. 검수자를 다시 나누거나 같은 작업을 무기한 반복하지 않는다.
4. 사용자/병렬 변경이 있으면 해당 diff를 보존하고 충돌 범위만 확인한다. VLM 변경 때문에 hash가 달라져도 되돌리지 않으며 원본 검사 결과와 LLM 범위 검사를 구분한다.
5. 코드가 단순해 보인다는 이유만으로 완료하지 않는다. node 구조는 바뀌어도 공개 계약·모델 입력·질문 권한·실행 결과가 같다는 실제 증거가 있어야 한다.

## 9. 완료 기준과 인계 산출물

1. Recommendation이 실제 조건부 독립 노드이며 일반 턴/명확화/선행 실패에서 불필요하게 호출되지 않는다. 현재+미래 추천 및 별도 이벤트 경로가 보존된다.
2. 공개 함수 서명·입출력 schema·state key·prompt/data/token 계약과 현행 안전/실행/발화 정책을 보존했다. private 문맥이 모델·세션·공개 반환에 새지 않는다.
3. 고정 모델 출력 기준으로 before/after의 전필드·호출·payload·응답·발행 mock가 같고, 누적 복합 episode 및 정상/거절 대조를 통과했다.
4. 한국어 주석이 실제 코드의 역할·부작용·안전 이유를 설명하고 불필요한 중첩/추상화/단순 전달 helper를 늘리지 않았다. 최종 독립 검수의 미해결 차단점이 없다.
5. `RESULT`에는 실제 변경 파일·baseline hash·테스트 환경/명령/종료코드·독립 검수 결과·미실행 GPU/실물 검증·원래 존재한 미해결 이슈를 남긴다. 새 모델 성능이나 학습 완료를 이번 구조 검증으로 주장하지 않는다.

계획 검수/구현 산출물은 이후 이 계획에서 직접 연결한다. 대량 학습 자료를 `docs/`로 복사하지 않는다. 작업 완료 후 현행 상위 계획·인계의 읽는 순서만 국소 갱신하고 과거 이력을 지우지 않는다.

후속 증거 경로는 `training/learning_first_v43/final/audits/2026-10-29_graph_refactor/`를 기본으로 한다. R1 baseline·함수/지역변수 지도·차등 fixture, 이후 실제 테스트/검수/RESULT를 구분해 둔다. 과거 GL04~06의 source hash가 이번 의도된 리팩토링으로 달라지면 그 사실을 기록하되 옛 manifest/봉인/합격 영수증을 고쳐 새 코드를 통과시킨 것처럼 만들지 않는다.

## 10. 현재 상태와 다음 한 단계

계획 검수의 **조건부 찬성·실제 계획 차단점 0건** [원기록](../training/learning_first_v43/final/audits/2026-10-29_graph_refactor/PLAN_REVIEW.md)은 보존한다. 이후 사용자 승인에 따라 구현했다.

R1~R5 완료: private 문맥 실제 입출력 투영·사칭 입력 차단, 추천 전/추천/후속 정책 노드 분리, 로그 클로저의 명시 인자 메서드 이동. R4 CPU/mock 최종 통합 **778 passed·2 subtests, exit 0, 10.21초**. 모델 입력·누적 episode·plan/STT/발행문·실패 로그를 변경 전 실제 파일과 대조했다. 기존 모델/프롬프트·정책·공개 계약 파일과 VLM 보호 파일 hash는 동일하다.

Codex CLI sol/high 읽기 전용 [코드 검수](../training/learning_first_v43/final/audits/2026-10-29_graph_refactor/CODE_REVIEW.md)는 **찬성·신규 회귀/차단점 0건**이다. 검수자의 독립 pytest는 read-only 임시 폴더 제한으로 미실행이며 주 작업자의 실제 실행과 구분했다. 검수 후 runtime 변경 없이 이중 오류 누락 테스트 1개만 추가했다. [실행 영수증](../training/learning_first_v43/final/audits/2026-10-29_graph_refactor/TEST_RECEIPT.md) · [작업 장부](../training/learning_first_v43/final/audits/2026-10-29_graph_refactor/WORKLOG.md).

기존 한계인 graph 오류+fallback publish 오류 시 로그 누락, Python3.10 TypedDict/Pydantic schema export 오류는 [RESULT §4](../training/learning_first_v43/final/audits/2026-10-29_graph_refactor/RESULT.md#4-독립-검수와-기존-한계)에 별도 기록했다. 동작 보존 범위에 몰래 수정하지 않았다. 과거 source manifest도 변경하지 않았다.

다음 한 단계: 사용자가 이번 diff를 확인한다. 학습·원격·adapter·실물 ROS·자동 commit/push는 진행하지 않는다.
