# IRC SOOMAC3 Base Routing Runtime Hardening Log

참고 자료:

- [`IRC_SOOMAC3_DECISION_ROUTING_FOCUS_PLAN.md`](IRC_SOOMAC3_DECISION_ROUTING_FOCUS_PLAN.md)
- [`base model routing test.jsonl`](<base model routing test.jsonl>)

→ **Decision LLM이 semantic mistake를 내더라도 canonical session, dialogue focus, restriction, robot execution을 Python runtime이 바로 오염시키지 못하게 하는 작업이다.**

작성일: 2026-10-02  
대상 branch: `flexible_llm`  
현재 상태: 계획 작성 완료, 코드 수정 전

---

## 1. 최초 문제

현재 XGrammar와 JSON Schema는 출력 형식은 제한하지만 의미가 사용자 발화와 맞는지는 보장하지 않는다.

실제 Base Model 출력에서는 다음 문제가 확인됐다.

```text
사용자: 햄 많이 넣어줘
Decision: 양파 high
결과: canonical order에 양파 high 반영
```

```text
사용자: 페퍼로니 조금 넣어줘
Decision: 페퍼론치노 low
결과: canonical order에 페퍼론치노 low 반영
```

**JSON이 schema-valid하다는 사실과 semantic grounding이 맞다는 사실은 다르다.**

Cause: Python은 지원 domain에 포함된 Decision target을 사용자 발화와 대조하지 않고 적용한다.

Effect: Base Gemma가 unsupported 명사를 지원 메뉴로 치환하면 state mutation까지 이어질 수 있다.

---

## 2. Base test 조건

| 항목 | 값 |
| --- | --- |
| Repository | `jisoooxo/IRC_SOOMAC3.0` |
| Branch | `flexible_llm` |
| Base model | `/home/roma/Desktop/sLLM/gemma-4-12B-it` |
| Quantization | NF4 4-bit |
| Decision Adapter | 없음 |
| Response Adapter | 없음 |
| 실행 범위 | LLM-only |
| 제외 | STT, TTS, 실제 VLM 호출, UI, robot node |

실제 테스트에는 다음 구성이 모두 적용됐다.

```text
DECISION_SYSTEM
DECISION_SCHEMA
XGrammar
any_order=True
TASK_RESPONSE_SYSTEM
GENERAL_RESPONSE_SYSTEM
MIXED_RESPONSE_SYSTEM
RESPONSE_SCHEMA
```

따라서 실패를 prompt 또는 schema 미적용 문제로 해석하지 않는다.

---

## 3. Base 133 test 결과

| 지표 | 결과 |
| --- | ---: |
| 총 테스트 | 133 |
| PASS / FAIL | 36 / 97 |
| Route 정확도 | 81 / 131 |
| Mentions 정확도 | 17 / 81 |
| Mutation safety | 96 / 99 |
| Reference 요청 | 25 |
| Reference 독립 평가 가능 | 0 |
| Mixed PASS | 2 / 10 |
| Direct unsupported safety PASS | 3 / 5 |
| 기존 task regression PASS | 15 / 31 |
| Decision 평균 | 약 2,885 ms |
| Response 평균 | 약 841 ms |
| Total 평균 / P95 | 약 3,728 / 6,547 ms |

Reference 25건은 Python resolver가 25번 틀린 것이 아니다.

```text
setup turn의 mentions 누락
→ dialogue_focus.current=null
→ reference resolver에 후보 없음
→ 독립 평가 불가능
```

(내가 의역해보자면, 주소 찾기 로직을 평가하려 했지만 주소록에 이름이 한 번도 저장되지 않은 상태이다.)

---

## 4. 실제 failure example

### 4.1 General/task와 mentions

```text
사용자: 치즈랑 버섯은 뭐가 달라?
기대: route=general, mentions=["치즈", "버섯"]
실제: route=task, mentions=[], understanding=clarify
```

이 실패로 `dialogue_focus`가 만들어지지 않아 뒤의 `그거`, `둘 다`, `첫 번째 거`가 연쇄 실패했다.

### 4.2 Duplicate JSON key

Base raw output에 다음 패턴이 여러 번 존재한다.

```json
{
  "type": "order_item",
  "target": "치즈",
  "target": "버섯"
}
```

현재 `json.loads()`는 앞의 `target`을 조용히 덮어쓴다.

### 4.3 Mixed grounding

```text
test_id: 122
Python reference_targets: []
Response: 치즈와 버섯 중 어떤 것을 제외하고 싶으신지 ...
```

Task mutation은 rollback됐지만 Response가 history에서 task 후보를 다시 만들었다.

### 4.4 VLM-like 질문

```text
사용자: 지금 카메라에 뭐가 보여?
실제 route: task
실제 reply: 현재 로봇은 면 섹션에 있으며 ...
```

이번 작업에서는 `route=vlm`을 추가하지 않는다.

---

## 5. Cause / Effect

| Cause | Effect |
| --- | --- |
| Base Gemma가 새 semantic contract를 학습하지 않음 | route, mentions, semantic extraction 실패 |
| `parse_decision()`이 일반 `json.loads()` 사용 | duplicate key가 silent overwrite됨 |
| reference 계산이 Decision inference 뒤에 있음 | LLM이 target과 operation을 모두 추론함 |
| `clarify`여도 `run_recommendation()`까지 진행 | 불필요한 inference와 오염 가능성 발생 |
| repair 최종 실패가 `new_decision()`으로 초기화 | route와 mentions까지 유실됨 |
| Response에 recent history가 함께 전달됨 | Python에 없는 task 후보를 history에서 재생성 가능 |

---

## 6. 기존 구조

### 6.1 현재 graph 순서

```text
interpret_decision
→ validate_and_repair
→ resolve_reference
→ general no-op 또는 task pipeline
→ resolve_confirmation
→ apply_workers
→ run_recommendation
→ check_policy
→ generate_response
```

### 6.2 현재 reference 역할

```text
LLM이 dialogue_focus/history를 보고 target 해석
→ LLM이 canonical semantic 생성
→ Python resolve_focus_reference()가 target 재계산
→ reference_targets_match_decision()이 일치 여부 검사
```

따라서 현재 구조는 **LLM resolver + Python verifier**이다. Python 단독 resolver라고 표현하지 않는다.

### 6.3 현재 robot execution 경계

`llm_langgraph_node.py::_process_turn()`은 `policy.execute=True`일 때만 `_start_current_section()`을 호출한다. 로봇 작업 중에는 새 STT turn도 무시한다.

이 경계는 유지하며 이번 P0에서 ROS node를 수정하지 않는다.

---

## 7. 현재 코드와 작업 지침의 차이

### 7.1 Duplicate-key 오류는 기존 semantic repair에 들어갈 수 없다

현재 semantic repair는 `call_decision()`이 정상적인 Decision을 반환한 뒤 실행된다. `parse_decision()`에서 ValueError가 발생하면 `validate_and_repair()`까지 도달하지 않는다.

최소 안전안:

```text
duplicate key 발견
→ parser reject
→ graph error path
→ graph_state 반영 없음
→ 사용자 재입력 안내
```

Duplicate parse 오류까지 자동 repair하려면 별도의 parse-retry 구조가 필요하다. P0 첫 변경에서는 state 무오염을 우선하고 자동 retry는 추가하지 않는다.

### 7.2 `reference_context`는 존재하지만 Decision 입력에는 없다

현재 `TurnState.reference_context`는 Decision 뒤의 resolver가 Response용 이유와 후보를 보관하는 값이다. `build_decision_inputs()`에는 전달되지 않는다.

### 7.3 Spacing variation은 현재 resolver가 모두 지원하지 않는다

현재 지원:

```text
둘 다
첫 번째 거
두 번째 거
아까 그거
전에 말한 거
```

현재 누락:

```text
둘다
첫번째거
첫 번째거
두번째 거
아까그거
전에말한거
```

### 7.4 Direct unsupported grounding은 P0/P1 목록만으로 완전히 해결되지 않는다

`햄→양파`, `페퍼로니→페퍼론치노`는 reference가 아닌 직접 발화이다. Reference pre-resolution만으로는 막히지 않는다.

따라서 P0 이후 별도의 **direct task target grounding guard**가 필요하다. 새 schema field 없이 user text, grounded mentions, resolved reference, canonical target을 대조하는 보수적 helper로 설계한다.

### 7.5 Prompt hardening만으로 Response grounding을 deterministic하게 보장할 수 없다

Prompt를 강화할 수는 있지만 Base Gemma가 항상 지킨다는 보장은 없다. Python fact가 없는 후보명을 절대 말하지 않는 보장이 필요하면 task clarification을 Python에서 만들거나 task/general 응답 생성을 분리해야 한다.

이번에는 prompt hardening까지 진행하고 동일 replay에서 재발하면 구조 분리안을 사용자에게 먼저 제안한다.

---

## 8. P0 최소 수정 계획

### P0-1. Duplicate JSON key reject

수정 대상:

| 파일 | 함수 | 변경 |
| --- | --- | --- |
| `decision_model.py` | `parse_decision()` | `object_pairs_hook`로 동일 object 내부 중복 key 탐지 |
| `decision_model.py` | 신규 작은 helper | 중복 key면 key 이름을 포함한 `ValueError` 발생 |

예상 동작:

```text
정상 JSON
→ 기존 normalize_decision 그대로

duplicate target
→ parser reject
→ graph state 반영 없음
```

하지 않을 것:

```text
Decision schema 변경
duplicate 중 마지막 값 선택
중복값 임의 병합
```

검증:

```text
동일 nested object의 duplicate target → reject
서로 다른 object에 같은 key 사용 → 허용
정상 sparse Decision → 기존과 동일
```

### P0-2. Clarify Recommendation inference 차단

수정 대상:

| 파일 | 함수 | 변경 |
| --- | --- | --- |
| `llm_langgraph.py` | `run_recommendation()` | 시작부에서 `understanding=clarify`면 모델 호출 없이 반환 |

반환 원칙:

```text
candidate_session은 그대로 전달
recommendation_result=None
call_recommendation 호출 횟수=0
최종 check_policy의 기존 rollback 유지
```

검증:

```text
clarify + recommendation.request → call count 0
정상 recommendation.request → call count 1
clarify 턴 canonical session → 기존 working_session으로 rollback
```

### P0-3. Reference pre-resolution

수정 대상:

| 파일 | 함수 | 변경 |
| --- | --- | --- |
| `dialogue_focus.py` | 신규 `build_reference_context()` | inference 전에 none/resolved/ambiguous/missing/stale와 targets 계산 |
| `dialogue_focus.py` | reference 표현 탐지 | spacing variation을 같은 의미로 정규화 |
| `decision_model.py` | `build_decision_inputs()` | Python-generated `reference_context`를 model input에 추가 |
| `agent_prompts.py` | `DECISION_SYSTEM` | target은 internal context를 사용하고 operation/amount를 해석하도록 역할 수정 |
| `llm_langgraph.py` | `resolve_reference()` | 동일 helper 결과로 final verifier와 policy context 구성 |
| `llm_policy.py` | `reference_targets_match_decision()` | reference target과 명시적 새 target을 구분해 검증 |

내부 입력 예:

```json
{
  "reference_context": {
    "status": "resolved",
    "targets": ["치즈"]
  }
}
```

이 값은 Decision OUTPUT이 아니며 `DECISION_SCHEMA`에 추가하지 않는다.

명시적 새 target 혼합 예:

```text
focus=["치즈"]
사용자: 그거 빼고 버섯 많이 넣어줘

reference target: 치즈
explicit mention: 버섯
기대 semantic:
  치즈=none
  버섯=high
```

현재 verifier처럼 전체 Decision target을 reference target 하나와 완전히 동일하다고 요구하면 이 정상 발화를 false ambiguous로 처리한다. 새 verifier는 reference target이 Decision에 포함되는지 확인하고, 추가 target은 현재 발화에 직접 grounded됐는지 별도로 확인한다.

### P0-4. Repair fallback amplification 방지

수정 대상:

| 파일 | 함수 | 변경 |
| --- | --- | --- |
| `llm_langgraph.py` | `validate_and_repair()` | 최종 실패 시 전체 `new_decision()` 초기화 대신 fail-closed fallback 생성 |
| `dialogue_focus.py` 또는 작은 grounding helper | surface mention 검사 | user text에 실제 존재하는 mention만 보존 |

최소 fallback 원칙:

```text
첫 Decision의 route 보존
user_text에 실제 존재하는 grounded mentions만 보존
order/restriction/preference/recommendation/commit/confirmation/queries 폐기
understanding=clarify
```

예:

```text
사용자: 치즈랑 버섯은 뭐가 달라?
첫 Decision: route=general, mentions=[치즈, 버섯], invalid query 포함
repair도 실패

fallback:
route=general
mentions=[치즈, 버섯]
understanding=clarify
mutation semantic 없음
```

Route까지 Python이 새로 추측하지 않는다. 첫 출력의 schema-valid route를 보존하되 위험 semantic을 모두 제거하는 방향이다.

---

## 9. P1 판단

P1은 P0 뒤로 미룬다. 이유는 reference context와 공통 grounding helper의 최종 형태가 먼저 정해져야 같은 검사를 중복 구현하지 않기 때문이다.

P0 완료 직후 다음 순서로 진행한다.

| 순서 | 항목 | 이유 |
| ---: | --- | --- |
| 1 | Direct task target grounding | test 93·94의 실제 canonical mutation 차단 |
| 2 | Commit conservative guard | false-positive robot execution 차단 |
| 3 | Restriction remove guard | hallucinated safety restriction 삭제 차단 |
| 4 | Focus mention grounding | hallucinated mention의 persistent contamination 차단 |
| 5 | Mixed Response prompt hardening | Python에 없는 task fact 생성 억제 |

VLM-like 질문은 이 P1까지 끝난 뒤 별도 안전 경계로 검토한다. 이번 단계에서 route enum은 늘리지 않는다.

---

## 10. 테스트 계획

기존 `test_runtime.py`에 모든 새 테스트를 섞지 않는다. Runtime hardening의 실패 원인을 작게 분리하기 위해 다음 파일을 제안한다.

| 새 파일 | 범위 |
| --- | --- |
| `src/soomac_irc/test/test_decision_runtime_hardening.py` | duplicate key, repair fallback, grounded mentions |
| `src/soomac_irc/test/test_reference_runtime_hardening.py` | pre-resolution, TTL, spacing, explicit target 혼합 |
| `src/soomac_irc/test/test_graph_runtime_hardening.py` | clarify recommendation call count, rollback, Response input facts |

### 10.1 P0 deterministic gate

```text
1. duplicate target reject
2. clarify recommendation call count=0
3. reference current/plural/ordinal/temporal/spacing
4. reference + explicit target가 false ambiguous가 아님
5. repair 실패 시 위험 semantic 폐기
6. repair 실패 시 grounded mentions 보존
7. 기존 canonical session 불변성
```

### 10.2 P1 deterministic gate

```text
1. 햄→양파 mutation 차단
2. 페퍼로니→페퍼론치노 mutation 차단
3. 명시적 진행 표현 없는 commit 차단
4. 명시적 철회 없는 restriction remove 차단
5. user text에 없는 mention focus 저장 차단
```

### 10.3 Response grounding gate

Prompt-only 단계에서 deterministic하게 확인할 수 있는 것은 Response 입력의 Python fact가 정확한지까지이다.

```text
policy.reference_targets=[]
→ Response model_input에도 []
→ applied_changes=[]
```

Base Model이 최종 문장에서 후보를 생성하지 않는지는 model replay에서 확인한다. 재발하면 Python task reply와 Base Gemma general reply 분리안을 승인받는다.

### 10.4 Replay

기존 [`base model routing test.jsonl`](<base model routing test.jsonl>)은 수정하지 않는다.

```text
P0 + P1 deterministic PASS
→ 동일 133 test 재실행
→ base_model_routing_test_runtime_hardened.jsonl 저장
→ Decision failure와 runtime failure 재분류
```

---

## 11. 예상 변경량

| 파일 | 예상 변경 | 판단 |
| --- | ---: | --- |
| `agent_contract.py` | 0줄 | Decision schema freeze |
| `agent_prompts.py` | 15~30줄 | input-only reference context와 Mixed grounding 문구 |
| `decision_model.py` | 25~45줄 | duplicate parser와 reference context 입력 |
| `decision_overrides.py` | 0~20줄 | 공통 conservative override가 필요할 때만 사용 |
| `dialogue_focus.py` | 50~90줄 | pre-resolution context, spacing, surface grounding helper |
| `llm_policy.py` | 20~50줄 | final verifier와 direct target grounding |
| `llm_langgraph.py` | 40~80줄 | recommendation early exit, repair fallback, reference verifier 연결 |
| `response_model.py` | 0줄 예상 | 우선 prompt만 강화 |
| `recommendation_model.py` | 0줄 | 제거·교체하지 않음 |
| `llm_langgraph_node.py` | 0줄 | robot execution 경계 유지 |
| `domain.py` | 0줄 예상 | 기존 canonical menu 사용 |
| 신규 test 3개 | 220~360줄 | P0/P1 deterministic regression |

보호 대상인 STT, TTS, VLM, UI, HTML, robot control 파일은 수정하지 않는다.

---

## 12. 단계별 실행 순서

한 번에 전체를 수정하지 않는다.

```text
Step 1: P0-1 duplicate key reject
→ diff
→ parser test
→ 문서 갱신

Step 2: P0-2 clarify recommendation skip
→ diff
→ call-count/rollback test
→ 문서 갱신

Step 3: P0-3 reference pre-resolution
→ diff
→ resolver/spacing/explicit target test
→ 문서 갱신

Step 4: P0-4 safe repair fallback
→ diff
→ route/mentions/semantic-drop test
→ 문서 갱신

Step 5: P1 high-risk grounding
→ 항목별 승인
→ deterministic safety test

Step 6: Base 133 replay
→ 새 JSONL
→ baseline 비교
```

각 단계는 사용자가 실제 출력 결과를 확인한 뒤 다음 단계로 넘어간다. 자동 commit과 push는 하지 않는다.

---

## 13. 수정한 코드

현재 없음. 계획 승인 전에는 코드에 손대지 않는다.

---

## 14. 수정 전 / 수정 후 목표 동작

| 사례 | 수정 전 | 목표 |
| --- | --- | --- |
| Duplicate target | 마지막 값으로 silent overwrite | parser reject, state 불변 |
| Clarify + recommendation | 추천 LLM 호출 후 rollback | 추천 호출 0 |
| `그거 많이` | LLM이 target부터 추론 | Python target 제공, LLM은 operation 해석 |
| Repair 재실패 | `task + mentions=[]` 초기화 | 위험 semantic 폐기, safe route/mentions 보존 |
| `햄 많이` | 지원 메뉴로 잘못 변환 후 mutation 가능 | 직접 target grounding 실패로 mutation 차단 |
| Empty reference target | Mixed Response가 history에서 후보 생성 가능 | Python fact 없으면 후보명 생성 금지 |

---

## 15. 남은 문제

P0 이후에도 Base Gemma의 route와 mentions 정확도 자체는 낮을 수 있다. Runtime hardening은 잘못된 출력을 안전하게 막는 작업이지 Decision 성능을 대신하는 작업이 아니다.

남는 문제는 다음 단계로 분리한다.

| 대상 | 다음 단계 |
| --- | --- |
| route / mentions / mixed extraction | Decision dataset과 Adapter 학습 |
| 최신 사실 | 별도 web/RAG 설계, 이번 범위 제외 |
| VLM orchestration | LLM-only safety 이후 별도 설계 |
| Response deterministic grounding | Prompt replay 실패 시 구조 분리 검토 |

---

## 16. Decision Adapter로 넘길 문제

```text
general/task lexical trap
general mention 추출
mixed task semantic 추출
capability question과 mutation 명령 구분
unsupported 주문의 clarify semantic
```

Python은 명백한 오염을 막지만 자연어 의미 전체를 regex로 다시 구현하지 않는다.

---

## 17. Recommendation test에서 볼 문제

Runtime hardening 후 별도 test를 수행한다.

```text
담백하게 / 꾸덕하고 매콤하게 추천
restriction과 충돌하는 추천
기존 선택을 보존한 remaining 추천
추천 revise / select / commit
unsupported 음식이 포함된 추천 공격 케이스
```

Recommendation LLM 유지 또는 Python recommender 전환은 이 결과를 본 뒤 결정한다.

---

## 용어 정리

- **런타임 강화(runtime hardening)**: 모델이 틀려도 상태 변경과 실행 권한을 Python이 보수적으로 제한하는 작업이다.
- **정본 상태(canonical state)**: 실제 주문과 실행 판단에 사용하는 유일한 확정 SessionState이다.
- **그라운딩(grounding)**: 모델이 출력한 대상이 실제 사용자 발화나 Python이 계산한 reference와 연결되는지 확인하는 과정이다.
- **자동복구(repair)**: invalid Decision을 같은 발화로 한 번 더 생성해 고치는 과정이다.
- **실패 폐쇄(fail-closed)**: 확신할 수 없을 때 실행하거나 mutation하지 않고 clarify 또는 오류로 종료하는 방식이다.
- **내부 문맥(internal context)**: Decision output schema가 아니라 Python이 모델 입력과 graph 한 턴에서만 사용하는 정보이다.

