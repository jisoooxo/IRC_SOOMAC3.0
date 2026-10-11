# IRC_SOOMAC 3.0 Natural-language-first 리팩토링 작업 기록

참고 코드: [agent_contract.py](../src/soomac_irc/soomac_irc/agent_contract.py), [agent_prompts.py](../src/soomac_irc/soomac_irc/agent_prompts.py), [decision_model.py](../src/soomac_irc/soomac_irc/decision_model.py), [llm_langgraph.py](../src/soomac_irc/soomac_irc/llm_langgraph.py), [llm_policy.py](../src/soomac_irc/soomac_irc/llm_policy.py), [llm_langgraph_node.py](../src/soomac_irc/soomac_irc/llm_langgraph_node.py)

→ 2026년 10월 6일 `flexible_llm` 브랜치에서 Decision 의미 해석을 모델로 모으고, Python은 최종 domain·safety·physical 검증만 담당하도록 정리한 기록이다.

참고 테스트: [test_natural_multiturn_contract.py](../src/soomac_irc/test/test_natural_multiturn_contract.py), [test_section_question_events.py](../src/soomac_irc/test/test_section_question_events.py), [test_vlm_ui.py](../src/soomac_irc/test/test_vlm_ui.py)

→ 최종 동작은 문서 설명이 아니라 위 회귀 테스트가 실제로 고정한다.

---

## 작업 기준

- 작업 브랜치: `flexible_llm`
- 시작 기준: `19b7947`
- 문서 작성 직전 HEAD: `f558965`
- 오늘 커밋 수: 10개
- 최종 작업 트리 상태: 문서 작성 전 clean
- 전체 변화량: 241 files changed, 2,618 insertions, 6,908 deletions

**오늘 작업의 핵심은 자연어 의미를 Python 규칙으로 재해석하지 않고 Decision Agent가 맡게 한 것이다.**

```text
Language meaning             = Decision Model
Structural contradiction     = one-shot repair
Domain / safety / physical   = final validator
Response                     = confirmed structured facts only
```

(내가 의역해보자면 모델은 “사용자가 무엇을 원했는가”를 결정하고, 코드는 “그 요구를 지금 로봇이 실제로 수행해도 되는가”만 결정한다.)

---

## 커밋 순서

| 시각 | 커밋 | 작업 |
|---|---|---|
| 11:26 | `6921b11` | 전날 runtime log 임시 백업 |
| 11:45 | `6d927dc` | Natural-language-first 1차 리팩토링 |
| 12:47 | `942879d` | Decision `cancel` field 제거 및 삭제 의미 통합 |
| 14:27 | `df9ea20` | one-shot structural repair 복구 |
| 14:59 | `b3f48f4` | semantic guard 제거, final validator 수렴 |
| 15:04 | `3626aa7` | VLM UI 발행 안정화 |
| 15:10 | `44d90e1` | 사용하지 않는 legacy runtime 제거 |
| 15:11 | `09a93e6` | recommendation annotation 정리 |
| 15:24 | `7e752b3` | contract 정본·로그 version 정리 |
| 15:42 | `f558965` | TypedDict 축소 및 `vlm_ui.py` 병합 |

`6921b11`에서 임시로 저장한 로그를 포함한 기존 runtime log는 `7e752b3`에서 모두 제거했다. 따라서 커밋 순서는 “백업했다가 source repository에서는 제거한 과정”까지 포함한다.

---

## 1. Natural-language-first 1차 리팩토링

### 1.1 이전 구조의 문제

이전 runtime에는 자연어 의미를 Python이 다시 판단하는 코드가 많았다.

```text
사용자 발화
→ keyword/phrase/mention/reference 해석
→ 여러 중간 guard
→ Decision 보정
→ policy
```

이 구조에서는 다음 문제가 발생했다.

- 모델이 올바르게 여러 field를 출력해도 Python의 좁은 분기에서 일부 의미가 사라졌다.
- `mentions`, `queries`, `pending_question`, `pending_confirmation`처럼 이전 계약의 state가 계속 남았다.
- 같은 의미를 prompt, parser, override, policy가 각각 조금씩 다르게 해석했다.
- 코드를 추가할수록 예외가 늘고 자연스러운 멀티턴 성능이 오히려 낮아졌다.

**1차 리팩토링은 Python semantic parser를 줄이고 sparse Decision 하나가 여러 의미를 동시에 표현하도록 바꿨다.**

### 1.2 외부 sparse Decision

현재 Decision Agent가 출력할 수 있는 top-level field는 다음과 같다.

```text
route
order
restrictions
preferences
recommendation
commit
confirmation
clarify
```

`route`만 필수이며 나머지는 해당 턴에 실제 의미가 있을 때만 출력한다.

예를 들어 한 발화에서 다음을 동시에 출력할 수 있다.

```json
{
  "route": "task",
  "order": {
    "noodle_portion": "high",
    "toppings": {
      "페퍼론치노": "low"
    }
  },
  "preferences": [
    {"value": "매콤하게", "action": "add"},
    {"value": "푸짐하게", "action": "add"}
  ],
  "commit": true
}
```

이제 `order + preferences + commit`, `restrictions + order`, `commit + preferences + order`처럼 복합 의미를 하나의 Decision으로 보존한다.

### 1.3 내부 normalized Decision

외부 sparse JSON은 `normalize_decision()`이 Graph 전용 고정 구조로 바꾼다.

```python
def normalize_decision(sparse_decision: dict) -> NormalizedDecision
```

파라미터 동작:

| 파라미터 | 동작 |
|---|---|
| `sparse_decision` | 모델이 실제 출력한 외부 JSON이다. 없는 field는 내부 기본값으로 채운다. |

반환 key:

```text
route
understanding
order_patch
restriction_options
preference_options
recommendation
commit
confirmation
```

외부와 내부의 역할은 다음처럼 분리했다.

```text
External Decision      = JSON Schema가 정본
NormalizedDecision     = TypedDict가 정본
```

(여기서 sparse와 normalized가 분리되는구나. 모델에게는 필요한 것만 말하게 하고 Graph에는 매번 같은 모양을 주는 구조이다.)

### 1.4 SessionState와 pending

멀티턴 동안 유지되는 state는 다음 다섯 key만 가진다.

```text
order
preferences
pending
history
action_history
```

pending은 별도 state를 여러 개 두지 않고 discriminator로 구분한다.

```json
{"type": "execution", "source": "current_update", "section": "veggie", "targets": ["양파"]}
```

```json
{"type": "recommendation", "candidate": {}, "reason_tags": []}
```

현재 pending 정책은 종류별로 다르게 처리한다.

- execution pending은 `accept/reject`에서 해소하고, order/restriction 변경 뒤에는 final canonical order 기준으로 대상을 갱신한다. preference-only와 상태 질문은 그대로 유지한다.
- recommendation pending은 실제 order/restriction/preference 순변화가 생기면 stale 후보로 보고 제거한다. 상태 질문, general conversation, no-op mutation에서는 유지한다.
- recommendation `request/revise`는 기존 후보를 Recommendation Agent 입력까지 보존한다. 새 proposal이 성공하면 교체하고, 새 유효 후보 생성에 실패하면 이전 pending도 `None`으로 정리한다.

### 1.5 최종 한 턴 흐름

```text
STT
→ ROS Node
→ External sparse Decision
→ normalize_decision
→ one-shot structural repair
→ pending resolution
→ final deterministic validation
→ Recommendation(optional)
→ read-only Response
→ ROS publish
```

`llm_langgraph.py`는 이 순서를 보여주는 orchestration 파일로 단순화했다.

---

## 2. `cancel` 제거와 삭제 contract 통합

### 2.1 Decision `cancel` field 제거

과거 `cancel`은 “무엇을 취소하는가”를 충분히 표현하지 못했다. 현재는 모든 선택 해제를 실제 field mutation으로 표현한다.

```json
{
  "route": "task",
  "order": {
    "sauce": "none",
    "noodle_type": "none",
    "noodle_portion": "none",
    "toppings": {
      "버섯": "none"
    }
  }
}
```

**`none`은 저장값이 아니라 삭제 operation을 나타내는 contract 값이다.**

- scalar의 `none`은 canonical `None`으로 적용한다.
- topping의 `none`은 해당 key를 삭제한다.
- `noodle_type=none`이면 의미가 사라지는 `noodle_portion`도 `None`으로 맞춘다.
- 문자열 `"none"`은 최종 canonical order에 저장하지 않는다.

### 2.2 삭제도 physical protection을 통과한다

`none`이라고 final validator를 우회하지 않는다.

```text
사용자 삭제 의도
→ Decision은 none mutation 생성
→ final validator가 past/completed/active 검사
→ 가능하면 삭제, 이미 실행됐으면 차단
```

즉 모델은 사용자의 삭제 의도를 그대로 표현하며, 코드는 물리적으로 되돌릴 수 있는지만 판단한다.

```text
Meaning        = Model
Physical truth = Code
```

### 2.3 “처음부터 다시” 정책

포괄적인 취소 표현은 기본적으로 수정 가능한 order selection만 다시 선택하게 한다.

```text
order        → none mutation으로 재선택
preferences  → 유지
restrictions → 유지
```

알레르기(allergy), 섭취 불가(cannot_eat), 식이 규칙(dietary_rule), 비선호(dislike)는 사용자가 해당 항목을 명시적으로 철회해야만 remove한다.

이 정책은 Python의 `"처음부터"` keyword 검사로 구현하지 않았다. Decision Agent가 history와 state를 읽고 필요한 구체 field에 `none`을 출력한다.

### 2.4 XGrammar key order

Decision grammar 생성은 다음 설정을 사용한다.

```python
compiled_grammar = compiler.compile_json_schema(
    DECISION_SCHEMA,
    any_order=True,
)
```

`any_order=True`의 효과:

- top-level key 순서가 자유롭다.
- 중첩 `order` 내부 key 순서도 자유롭다.
- unknown field는 계속 거절한다.
- 잘못된 type과 enum value도 계속 거절한다.

**JSON object의 key 순서는 의미가 아니며, schema 엄격성과 key order 자유화는 서로 다른 문제이다.**

---

## 3. one-shot structural repair

### 3.1 repair가 필요한 이유

XGrammar는 JSON 문법과 schema를 보장하지만 field끼리 의미상 동시에 성립할 수 없는 조합까지 막지는 않는다.

현재 repair 대상은 구조적 자기모순으로 제한한다.

```text
route=general + task mutation/action
clarify=true + 이미 확정된 mutation/action
noodle_type=none + noodle_portion=low/normal/high
```

unsupported 메뉴, 현재 section, 완료된 작업, restriction 충돌은 repair 대상이 아니다. 이들은 final validator가 처리한다.

### 3.2 호출 계약

```python
def build_decision_model_input(
    session: SessionState,
    user_text: str,
    robot_state: dict,
    repair: dict | None = None,
) -> dict
```

파라미터 동작:

| 파라미터 | 동작 |
|---|---|
| `session` | 현재 확정 주문, 취향, pending, recent history를 제공한다. |
| `user_text` | 현재 발화를 변경하지 않고 전달한다. |
| `robot_state` | 현재 물리 section과 실행 상태를 전달한다. |
| `repair=None` | 일반 Decision 호출이다. 입력에 repair key 자체가 없다. |
| `repair=dict` | 이전 structural contradiction을 한 번만 수정하도록 전달한다. |

repair payload는 다음 모양이다.

```json
{
  "reason": "structural_contradiction",
  "errors": [],
  "previous_output": {},
  "instruction": "서로 모순되는 Decision field만 최소한으로 수정하라."
}
```

`previous_output`에는 normalized 내부 이름이 아니라 모델이 실제 출력한 external sparse JSON을 넣는다.

```text
허용: order, restrictions, preferences, commit
금지: order_patch, restriction_options, preference_options
```

### 3.3 한 번만 repair한다

```text
첫 Decision 정상          → repair 없음
첫 Decision 구조 모순     → 두 번째 호출 1회
두 번째도 구조 모순       → safe clarify
duplicate JSON key        → 즉시 safe clarify
```

세 번째 모델 호출은 하지 않는다. repair가 새로운 의미를 발명하는 반복 루프가 되지 않게 했다.

(내가 의역해보자면 repair는 사용자의 말을 Python이 고치는 과정이 아니라, 모델에게 자기 JSON의 앞뒤만 한 번 맞추라고 돌려보내는 과정이다.)

---

## 4. final validator로 검증 수렴

### 4.1 Python policy의 최종 역할

현재 `llm_policy.py`는 다음만 담당한다.

```text
supported domain validation
physical protection
restriction/safety enforcement
canonical state consistency
execution feasibility
applied/future changes 계산
```

다음은 담당하지 않는다.

```text
user_text parsing
reference interpretation
STT semantic correction
keyword/regex intent detection
```

### 4.2 multi-error aggregation과 partial success

여러 validator에서 나온 실패를 `policy.issues` 하나로 모은다.

```json
{
  "unsupported": [],
  "invalid": [],
  "protected": [],
  "restriction_conflicts": [],
  "physical_conflicts": []
}
```

정상 field는 적용하고 잘못된 field만 거절하는 partial success를 유지한다. 단, issue가 하나라도 있으면 같은 턴의 `commit=true` 실행은 막는다.

예:

```text
양파 high + 햄 high + 게살 normal + commit
```

```text
양파 high     → 정상 반영
햄 high       → unsupported
게살 normal   → allergy conflict
commit        → 이번 턴 실행 차단
Response      → 정상 반영과 두 실패를 모두 설명
```

### 4.3 restriction net change

같은 턴의 restriction add/remove는 중간 operation이 아니라 시작 상태와 최종 상태의 순변화(net change)로 계산한다.

```text
없음 → add → remove = 최종 추가 없음
있음 → remove → add = 최종 제거 없음
```

이렇게 해야 잠깐 추가됐다가 사라진 restriction이 downstream safety enforcement에 새 제한으로 잘못 전달되지 않는다.

### 4.4 commit의 역할

`commit=true`의 자연어 의미는 Decision Model이 판단한다. Python은 다음 실행 조건만 마지막에 검사한다.

```text
현재 section 필수 선택 누락 여부
active_task 존재 여부
task_queue 존재 여부
domain/safety/physical issue 존재 여부
```

adapter 학습 여부와 runtime contract는 분리했다. 이후 adapter를 다시 학습하더라도 `commit=true` contract 자체는 유지한다.

---

## 5. contextual semantic grounding prompt

dataset 생성은 이후 작업으로 미뤘고, 오늘은 `DECISION_SYSTEM`에 grounding 정책을 반영했다.

### 5.1 Level A — Explicit

메뉴, 재료, 양을 직접 말하면 그대로 mutation으로 변환한다.

```text
“양파 많이”       → 양파 high
“버섯 빼줘”       → 버섯 none
“면 많이 줘”      → noodle_portion high
“페퍼론치노 조금” → 페퍼론치노 low
```

### 5.2 Level B — Strong contextual grounding

메뉴명을 직접 말하지 않아도 history와 현재 state에서 실제 조작 대상이 사실상 하나이면 구체화할 수 있다.

```text
“배부르게 먹고 싶어”
→ preference “푸짐하게”
→ noodle_portion high
```

```text
“조금 매콤하게”
→ preference “매콤하게”
→ 페퍼론치노 low
```

다만 “배부르게”만으로 모든 topping을 high로 바꾸지 않는다.

### 5.3 애매하면 preference로 보존한다

```text
“물컹한 식감이 싫어”
→ 특정 재료 restriction으로 추측하지 않음
→ preference로 보존
```

현재 선택과 recent history에서 버섯 하나가 명확한 상태에서 “물컹한 건 빼줘”라고 하면 버섯 `none`으로 grounding할 수 있다.

### 5.4 description과 request를 구분한다

```text
“이거 좀 꾸덕하네” → 감상, 자동 mutation 없음
“꾸덕하게 해줘”    → 요청, 문맥이 명확할 때만 grounding
“매운 음식이 뭐야?” → 질문, 페퍼론치노 자동 추가 없음
```

**phrase dictionary를 runtime Python 코드에 추가하지 않았다.**

---

## 6. VLM UI 유지 및 단순화

### 6.1 UI 발행 실패 격리

`3626aa7`에서 VLM UI snapshot 발행에 timestamp와 frame id를 넣고 예외를 격리했다.

```text
header.stamp    = 현재 ROS 시간
header.frame_id = vlm_ui_snapshot
format          = jpeg
```

UI 이미지 합성이나 publish가 실패해도 실제 VLM 판정은 계속 진행한다.

### 6.2 `vlm_ui.py` 병합

`vlm_ui.py`는 `build_vlm_ui_jpeg()` 함수 하나만 가지고 있었고 `llm_langgraph_node.py` 한 곳에서만 사용했다. 최종적으로 이 함수를 `vlm.py`에 합치고 원본 파일을 삭제했다.

```python
def build_vlm_ui_jpeg(
    request: dict,
    panel_size=(480, 360),
) -> bytes
```

파라미터 동작:

| 파라미터 | 동작 |
|---|---|
| `request` | `ui_panels` 세 개를 가진 VLM request이다. 정확히 세 개가 아니면 `ValueError`이다. |
| `panel_size=(480, 360)` | 패널 하나의 크기이다. 최종 기본 JPEG는 `1440×396`이다. |

화면 계약은 그대로 유지했다.

```text
REFERENCE | COMPARISON | CURRENT OBSERVATION
```

VLM 모델에 들어가는 원본 이미지 배열은 수정하지 않고 UI용 JPEG만 별도로 만든다.

---

## 7. legacy runtime 제거

### 7.1 삭제한 runtime 파일

현재 실행 경로에서 사용하지 않는 다음 파일을 제거했다.

```text
decision_overrides.py
dialogue_questions.py
task_response.py
dialogue_focus.py
vlm_ui.py
```

`task_response.py`의 응답 기능은 현재 실제 경로인 `agent_prompts.py`의 response prompt, `response_model.py`, `llm_langgraph.py`가 담당한다.

### 7.2 `dialogue_focus.py` offline 격리

과거 v2의 다음 지시어 해석 규칙은 runtime package에서 제거했다.

```text
그거
둘 다
첫 번째 거
두 번째 거
아까 그거
```

과거 dataset migration을 재현하는 데 필요한 최소 helper와 commit phrase는 다음 offline 영역으로 이동했다.

```text
soomac_decision_migration_merge_bundle/
└── route_rebalance/
    └── legacy_dialogue_focus.py
```

현재 runtime은 `recent_history`와 `pending`을 Decision Agent가 직접 읽는다.

### 7.3 legacy field 자체 감사

현재 external Decision, normalized Decision, SessionState에는 다음 field가 없다.

```text
mentions
queries
cancel
pending_question
pending_confirmation
dialogue_focus
reference_context
```

`stt_node.py`의 `grpc_call.cancel()`은 Decision field가 아니라 네트워크 stream 종료 API이므로 유지했다.

### 7.4 제거한 legacy 테스트

삭제된 runtime 코드 전용 테스트 여러 개를 제거하고 현재 계약 테스트로 통합했다.

주요 삭제 대상:

```text
test_decision_operation_grounding.py
test_decision_semantic_fail_safe_red.py
test_duplicate_decision_key_fallback.py
test_explicit_commit_gate.py
test_guard_cleanup_regression.py
test_noodle_grounding.py
test_order_completion.py
test_question_lifecycle.py
test_runtime.py
test_thin_python_refactor.py
```

삭제는 테스트 범위를 없앤 것이 아니라 현재 계약 기준의 `test_natural_multiturn_contract.py`로 의미를 다시 고정한 것이다.

---

## 8. contract 단일 정본

### 8.1 `agent_contract.py` 역할

최종 파일 구조는 다음과 같다.

```text
agent_contract.py
├── runtime log contract version
├── 핵심 runtime TypedDict 4개
├── JSON schemas
├── reusable constructors
└── normalize_decision
```

JSON Schema:

```text
FLEXIBLE_ORDER_SCHEMA
ORDER_PATCH_SCHEMA
DECISION_SCHEMA
RECOMMENDATION_SCHEMA
RESPONSE_SCHEMA
```

핵심 runtime TypedDict:

```text
NormalizedDecision
SessionState
RobotState
TurnState
```

### 8.2 제거한 과도한 TypedDict

한 번은 작은 nested dict까지 총 15개 TypedDict로 이름을 붙였지만 최종적으로 핵심 boundary 4개만 남겼다.

제거한 타입:

```text
ExternalDecision
OrderPatch
Restriction
Preference
OrderState
ExecutionPending
RecommendationPending
PolicyIssues
Policy
RecommendationResult
ResponseInput
```

작은 구조는 plain `dict`, `list[dict]`, `dict | None`로 표현하며 실제 key와 runtime value는 바꾸지 않았다.

**타입 정의를 제거한 것이지 contract field를 제거한 것이 아니다.**

### 8.3 constructor API

```python
def empty_order_patch() -> dict
def new_order_state() -> dict
def new_session_state() -> SessionState
def new_decision() -> NormalizedDecision
def normalize_decision(sparse_decision: dict) -> NormalizedDecision
def new_turn_state(
    session: SessionState,
    user_text: str,
    robot_state: RobotState | dict,
) -> TurnState
```

동작 차이:

| API | 역할 | 주의점 |
|---|---|---|
| `empty_order_patch()` | scalar가 `None`이고 toppings가 빈 patch를 만든다. | canonical order가 아니라 이번 턴 mutation 후보이다. |
| `new_order_state()` | restrictions를 포함한 빈 canonical order를 만든다. | 추천 candidate나 거절된 값은 넣지 않는다. |
| `new_session_state()` | 정확히 다섯 key의 멀티턴 state를 만든다. | 새로운 legacy state를 추가하지 않는다. |
| `new_decision()` | Graph용 full Decision을 만든다. | 외부 모델 출력 형식이 아니다. |
| `normalize_decision()` | sparse external Decision을 full internal Decision으로 바꾼다. | deepcopy하여 입력 객체와 state 공유를 막는다. |
| `new_turn_state()` | session과 robot state 복사본으로 한 턴 state를 만든다. | Graph 실패가 실제 Node state를 오염시키지 않는다. |

(여기서 “단일 정본”은 모든 dict에 타입 이름을 붙인다는 뜻이 아니라, 외부 schema와 핵심 runtime boundary가 한 파일에서 보인다는 뜻이다.)

---

## 9. runtime log contract 정리

현재 로그 식별자는 다음과 같다.

```python
RUNTIME_LOG_SCHEMA_VERSION = 2
RUNTIME_CONTRACT_VERSION = "natural_multiturn_v3"
```

`agent_turn`과 `runtime_event` row 모두 다음 field를 기록한다.

```json
{
  "schema_version": 2,
  "contract_version": "natural_multiturn_v3"
}
```

기존 로그는 자동 migration하지 않는다. dataset 생성 시 version으로 legacy log와 current native log를 나눈다.

### 9.1 repository runtime log 제거

source repository에 커밋되어 있던 실제 실행 로그를 제거했다.

```text
logs/                                  162 files
src/soomac_irc/soomac_runtime_logs/     58 files
합계                                   220 files
용량                                   약 27.4 MiB
```

`.gitignore`에는 다음 경로를 추가했다.

```gitignore
/logs/
/src/soomac_irc/soomac_runtime_logs/
```

Git history에는 과거 로그가 남으므로 필요하면 복구할 수 있다.

---

## 10. ROS·STT·VLM·UI 보존 범위

`src/soomac_irc/setup.py`의 console entrypoint를 기준으로 다음 실행 파일을 유지했다.

```text
stt_node           → soomac_irc.stt_node:main
stt_nemotron_node  → soomac_irc.stt_nemotron_node:main
tts_node           → soomac_irc.tts_node:main
llm_node           → soomac_irc.llm_langgraph_node:main
llm_debug          → soomac_irc.llm_langgraph_debug:main
ui_node            → soomac_irc.ui_node:main
```

Python import가 적다는 이유만으로 ROS entrypoint를 삭제하지 않았다.

보존한 동작:

- 로봇 작업 중 STT gating을 유지한다.
- task response 경로를 유지한다.
- VLM 판정과 UI snapshot topic을 유지한다.
- `/ui/reset`, `/llm/reset`과 section progression을 유지한다.
- ROS topic 이름과 message 흐름을 변경하지 않는다.

두 STT 구현 중 실제 배포에서 하나가 미사용인지 여부는 launch/deployment 근거가 없어 자동 삭제하지 않았다.

---

## 11. 회귀 테스트

최종 실행 환경은 `gemma4_env`이다. 이 환경은 XGrammar의 `any_order` API를 지원한다.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=/usr/lib/python3/dist-packages:src/soomac_irc \
/home/roma/miniconda3/envs/gemma4_env/bin/python \
-m pytest -q src/soomac_irc/test
```

최종 결과:

```text
PASS       60
FAIL       0
SKIP       0
compileall PASS
diff check PASS
```

주요 contract 회귀 범위:

```text
External Decision legacy key 금지
JSON key 순서 자유화와 schema strictness
NormalizedDecision와 SessionState exact key
RobotState와 TurnState exact key
execution/recommendation pending shape
one-shot repair와 sparse previous_output
restriction net change
multi-error aggregation과 partial success
scalar none canonicalization
completed/past/active physical protection
commit missing/busy protection
runtime log schema/contract version
VLM UI 세 패널과 원본 모델 입력 보존
```

`ros_env`에 설치된 구형 XGrammar는 `compile_json_schema(..., any_order=True)`를 지원하지 않았다. 같은 코드가 `gemma4_env`에서는 전체 통과했으므로 코드 회귀가 아니라 환경 package version 차이로 확인했다.

---

## 12. 최종 파일별 역할

| 파일 | 현재 책임 |
|---|---|
| `agent_contract.py` | JSON Schema, 핵심 runtime 타입 4개, constructor, normalization |
| `agent_prompts.py` | Decision/Recommendation/Response의 자연어 정책 |
| `decision_model.py` | model input, XGrammar, sparse JSON parse, normalized Decision 생성 |
| `llm_langgraph.py` | repair, pending, policy, recommendation, response orchestration |
| `llm_policy.py` | domain·restriction·physical·execution deterministic validation |
| `recommendation_model.py` | 원문과 state 기반 recommendation proposal 생성 |
| `response_model.py` | Python이 확정한 사실만 자연어 응답으로 변환 |
| `llm_langgraph_node.py` | ROS state, worker, topic publish, STT/VLM lifecycle |
| `vlm.py` | VLM request, verdict, retry outcome, UI JPEG 생성 |
| `llm_runtime_logger.py` | schema 2 / natural_multiturn_v3 JSONL 기록 |

최종 runtime 구조는 다음과 같다.

```text
STT
→ Node
→ Decision
→ one-shot repair
→ Policy
→ Recommendation(optional)
→ Response
→ ROS
```

---

## 13. 오늘 확정했지만 이후로 미룬 작업

### 13.1 새 Decision Adapter dataset

contextual semantic grounding 정책은 prompt에 반영했지만 dataset 생성은 이후 작업으로 남겼다.

향후 dataset은 다음 contrast를 포함해야 한다.

```text
description vs request
preference vs direct order
preference vs dislike restriction
weak implication vs strong implication
current selection vs future selection
question vs mutation
negative sentence vs positive request
hypothetical vs actual request
single intent vs compound intent
```

phrase dictionary를 Python runtime 코드로 옮기지 않는다.

### 13.2 offline legacy evaluation asset

`soomac_decision_migration_merge_bundle`의 v2 migration 자료와 `evaluation_any_order_true_20260930`의 과거 field는 offline asset으로 남아 있다. 현재 ROS/runtime import 경로에는 들어오지 않는다.

### 13.3 실제 하드웨어 통합 시험

오늘 검증은 unit/regression test, compileall, import, ROS entrypoint 존재 확인까지이다. 실제 카메라·로봇 arm·마이크를 연결한 end-to-end 시험은 별도 실행이 필요하다.

---

## 14. Natural Multiturn V3 dataset semantic cleanup과 runtime contract 최종 검증

참고 bundle: [data_natural_multiturn_v3_rebuild](../data_natural_multiturn_v3_rebuild), [BUILD_REPORT.md](../data_natural_multiturn_v3_rebuild/BUILD_REPORT.md), [DATA_QUALITY_REPORT.md](../data_natural_multiturn_v3_rebuild/DATA_QUALITY_REPORT.md)

→ runtime `25ce994`를 정본으로 Decision v3와 Response v1 dataset을 재구축하고, 학습 전에 semantic gold와 실제 Response 입력 구조를 최종 검증했다.

이번 단계에서는 runtime source와 schema를 수정하지 않았다. 학습, `git add`, commit도 실행하지 않았다.

### 14.1 Semantic cleanup

schema·duplicate·contamination 검사만으로 잡히지 않던 semantic 오류를 generator 원인에서 수정했다.

| 항목 | 발견 | 수정 | 잔여 |
|---|---:|---:|---:|
| Decision fake mutation | 68 | 68 | 0 |
| Response completed/pending 모순 | 60 | 60 | 0 |
| 부자연스러운 한국어 영향 row | 1,246 | 1,246 | 0 |

Decision의 원인은 `unsupported_mutation()`이 주문용 `해줘`와 설명용 `설명해줘`를 같은 동작으로 취급한 것이었다. 재료명 뒤에 실제 추가·삭제·변경 동작이 있는 경우만 unsupported mutation으로 복원하도록 수정했다.

```text
설명 대상: 로봇, 갑각류, 유제품, 육류
→ order mutation 생성 금지

명시적 요청: 베이컨 많이 넣어줘
→ toppings.베이컨=high 유지
```

Response의 `pending_status_question`은 선택 상태를 `task_completed`로 기록해 같은 대상을 완료와 실행 대기로 동시에 만들고 있었다. 이를 실제 session event인 `turn_applied.order_changes`로 바꿨다.

한국어 생성은 받침 기반의 주제격·목적격·주격·방향격과 `이에요/예요`를 사용하도록 통합했다.

```text
버섯는          → 버섯은
소시지이        → 소시지가
페퍼론치노이에요 → 페퍼론치노예요
많이로          → 많이
```

Decision high-risk 200행과 Response high-risk 200행에는 다음 review metadata를 기록했다.

```json
{
  "semantic_review": "pass | fixed | discard",
  "review_reason": "...",
  "error_class": []
}
```

이번 표본은 수정 영향 row를 우선 추출했으므로 Decision 200행과 Response 200행 모두 `fixed`, discard는 0이다.

### 14.2 Response runtime contract 전수 검사

정본은 현재 [llm_langgraph.py](../src/soomac_irc/soomac_irc/llm_langgraph.py), [llm_langgraph_node.py](../src/soomac_irc/soomac_irc/llm_langgraph_node.py), [response_model.py](../src/soomac_irc/soomac_irc/response_model.py) 세 파일이다.

Response 전체 1,225행에서 runtime contract mismatch 693행을 발견했고 모두 해당 synthetic family 또는 reviewed fixture 생성 단계에서 수정했다.

| 수정 범위 | row |
|---|---:|
| validation issue 5개 family | 300 |
| 상태 조회·실행 불가능 family | 240 |
| current-update pending 구조 | 75 |
| recommendation unavailable | 60 |
| section transition | 15 |
| reviewed fixture | 3 |

주요 차이는 다음과 같다.

```text
policy.status=partial          → warning
unsupported/restriction/physical의 blocked → warning
recommendation_unavailable    → clarify
policy.reason=state_query      → state_update_only
next_prompt=section_selection  → section_prompt
개별 issue next_prompt         → validation_issues
robot task {class, amount}     → {class, repeat_count}
```

`active_execution`은 Node가 `active_task` 또는 `task_queue`가 존재할 때 STT turn을 Graph에 전달하지 않으므로 실제 Response 입력으로 생성될 수 없다. 해당 60행은 실제 가능한 `understanding_clarify` family로 교체했다.

current-update execution pending은 session에는 `items`를 저장하지 않고, `_pending_prompt()`가 Response 입력의 `items={}`를 만든다. preselected pending만 `items`와 `candidate`를 유지하도록 dataset을 맞췄다.

section transition은 Node가 실제로 넣는 다음 정보를 모두 포함하도록 수정했다.

```text
policy.reason=section_transition
next_prompt.type=section_prompt
robot_state.section_transition
completed_tasks
robot_started=true
recent_action_history의 section_transition event
```

### 14.3 Decision 9행 감소 원인

semantic cleanup 전 Decision train/validation/test 합계는 3,174행, cleanup 후 합계는 3,165행이다.

**감소한 9행이 전부 재라벨 이후 중복 제거된 것은 아니다.**

```text
4행 → 한국어 정규화 후 input/target이 같아져 duplicate 제거
5행 → 정규화 후 frozen evaluation 발화와 같아져 training에서 격리
임의 discard → 0
```

추가로 격리된 5행은 `DV3-L01096`, `DV3-L01097`, `DV3-L01098`, `DV3-L01099`, `DV3-L01466`이다.

### 14.4 최종 dataset과 검증 결과

| dataset | train | validation | test | challenge | holdout | total |
|---|---:|---:|---:|---:|---:|---:|
| Decision v3 | 2,595 | 285 | 285 | 495 | 377 | 4,037 |
| Response v1 | 898 | 101 | 86 | 75 | 65 | 1,225 |

Response는 family 교체 뒤 balanced split을 다시 계산해 train 899→898, test 85→86으로 한 행 이동했지만 전체 1,225행은 유지했다.

최종 검증 결과:

```text
Decision schema / semantic       PASS
Response schema / fact / prompt  PASS
Response runtime contract        PASS
Response lifecycle consistency   PASS
Korean language quality          PASS
duplicate                        0
contamination                    0
compileall                       PASS
git diff --check                 PASS
```

검증 결과는 [response_runtime_contract.json](../data_natural_multiturn_v3_rebuild/reports/response_runtime_contract.json)과 [quality_report.json](../data_natural_multiturn_v3_rebuild/quality_report.json)에 저장했다.

**최종 판정은 `READY FOR TRAINING`이다. 이 판정 이후에는 추가적인 dataset 구조 재검토 없이 학습 단계로 이동한다.**

Cause: synthetic Response가 과거 policy status와 가상의 prompt/task 구조를 일부 유지하고 있었다.

Effect: Response 1,225행은 현재 Graph와 ROS Node가 실제 생성할 수 있는 입력 구조만 사용한다.

---

## 15. Decision v3 + Response v1 Dual Adapter 학습 패키지 제작

### 15.1 작업 목적

Natural Multiturn V3 runtime contract와 semantic cleanup이 끝난 뒤, Decision과 Response를 같은 원격 학습 작업에서 순차 학습할 수 있는 Docker image를 제작했다.

이번 패키지의 핵심은 **하나의 base model을 사용하되 Decision LoRA와 Response LoRA를 서로 독립적으로 학습하는 것**이다.

```text
Gemma 4 12B base
├── Decision Adapter: base에서 독립 학습
└── Response Adapter: base에서 독립 학습
```

Decision 학습 결과를 Response 학습의 초기값으로 사용하지 않았다. 두 Adapter가 서로의 역할을 섞어 배우지 않도록 각 학습 단계가 끝날 때 모델과 CUDA cache를 해제하고, 같은 base model에서 새 LoRA를 붙였다.

### 15.2 학습 입력

| Adapter | train | validation | epoch | learning rate |
|---|---:|---:|---:|---:|
| Decision v3 | 2,595 | 285 | 6 | `1e-4` |
| Response v1 | 898 | 101 | 4 | `5e-5` |

공통 LoRA 설정은 다음과 같다.

```text
Quantization: NF4 4-bit
Batch size: 1
Gradient accumulation: 16
LoRA rank: 16
LoRA alpha: 32
LoRA dropout: 0.05
Target: language model attention + MLP projection
```

optimizer step은 Decision 978, Response 228, 총 1,206 step으로 계산했다.

### 15.3 원격 학습 이미지 구조

생성한 작업 디렉터리와 TAR은 다음과 같다.

```text
soomac_dual_adapter_training_v2_9/
soomac_decision_response_dual_v2_9.tar
soomac_decision_response_dual_v2_9.tar.sha256
```

최종 TAR SHA-256은 다음과 같다.

```text
c1d8d4413c907dbb471163d90474594a50ee29de29a045f9a2b39cf5869ef92c
```

이미지는 Training Image Spec 2.9의 progress와 resume 계약을 구현했다.

```text
/checkpoints/
├── decision/
│   ├── checkpoint-*
│   ├── epoch_1_adapter/ ... epoch_6_adapter/
│   └── best_adapter/
├── response/
│   ├── checkpoint-*
│   ├── epoch_1_adapter/ ... epoch_4_adapter/
│   └── best_adapter/
├── dual_training_complete.json
└── .lab-monitor/
    ├── progress.json
    └── resume.json
```

각 epoch마다 배포 가능한 Adapter를 별도로 저장하고, `checkpoint-*`에는 Adapter뿐 아니라 optimizer, scheduler, Trainer state, RNG state를 함께 저장하도록 했다.

Decision이 끝난 뒤 Response 단계에서 실패하면 완료된 Decision을 다시 학습하지 않는다. Response checkpoint가 있으면 해당 optimizer step부터 복원한다.

### 15.4 입력 contract 고정

`prepare_bundle.py`가 다음 항목을 runtime 정본에서 직접 복사하고 hash를 manifest에 기록하도록 했다.

- Decision/Response train과 validation JSONL
- `DECISION_SYSTEM`
- route별 `TASK_RESPONSE_SYSTEM`
- Decision/Response JSON schema

따라서 원격 학습 시 로컬 runtime과 다른 prompt나 schema가 조용히 섞이지 않는다.

Cause: Decision과 Response는 같은 base model을 공유하지만 학습 목적과 출력 계약이 다르다.

Effect: 원격 작업 한 번으로 두 Adapter를 연속 학습하되, 각 Adapter는 base model에서 독립적으로 생성된다.

---

## 16. Decision v3 + Response v1 Adapter 반영과 Offline GPU 평가

### 16.1 Adapter 추출 및 runtime 구성

원격 학습 결과에서 다음 Adapter를 로컬에 배치했다.

```text
models/decision_adapter_v3_best_epoch3
models/response_adapter_v1_best_epoch1
```

실제 runtime은 한 base model에 두 LoRA를 등록하고 호출 단계마다 이름으로 전환하도록 구성했다.

```text
Decision node       → decision Adapter
Recommendation node → base model
Response node       → response Adapter
```

이 구성은 Decision의 JSON 출력 성향이 Response 문장 생성에 섞이거나, Response의 자연어 출력 성향이 Decision schema 출력을 오염시키는 문제를 막는다.

### 16.2 평가 환경

초기 dual Adapter 평가 기준은 다음과 같다.

```text
Git HEAD: 509be6035b7732b8f0d6258dc2f73a465936da2a
Base model: gemma-4-12B-it
Quantization: NF4
GPU: NVIDIA RTX PRO 4500 Blackwell, 32623 MiB
Decision: decision_adapter_v3_best_epoch3
Response: response_adapter_v1_best_epoch1
```

평가 산출물은 `evaluation_dual_adapter/`에 저장했다.

```text
decision_predictions.jsonl
decision_metrics.json
response_predictions.jsonl
response_metrics.json
e2e_sessions.jsonl
e2e_turns.jsonl
adapter_switch_report.json
latency_report.json
E2E_EVALUATION_REPORT.md
```

### 16.3 Decision frozen split 결과

학습에 사용하지 않은 test/challenge/holdout 1,157행을 실제 GPU로 추론했다.

| 지표 | 결과 |
|---|---:|
| Schema valid | 100% |
| Route accuracy | 100% |
| Order precision | 90.49% |
| Order recall | 96.34% |
| Order value accuracy | 98.83% |
| Restriction precision / recall | 98.96% / 98.96% |
| Recommendation accuracy | 99.91% |
| Confirmation accuracy | 99.57% |
| Commit accuracy | 99.57% |
| Clarify accuracy | 96.46% |

주요 Decision 실패 family는 질문·설명 경계, compound intent, unsupported target, reference였다.

### 16.4 Response 결과

Response deterministic 평가에서는 RED failure가 없었다.

```text
fact consistency       100%
next_prompt correctness 100%
pending correctness    100%
execution correctness  100%
hallucination free     100%
naturalness average    3.0 / 3.0
```

자연스러움 점수는 같은 base model judge를 사용한 보조 지표이다. 사실성, pending, execution 관련 항목은 별도 deterministic 검사도 통과했다.

### 16.5 E2E와 Adapter switching

```text
sessions: 50
turns: 150
완전 성공 session: 49 / 50
성공 turn: 149 / 150
Decision failure: 1
Policy/runtime failure: 0
Response failure: 0
```

Adapter 전환 event 1,921회를 검사했고 잘못된 Adapter 활성화는 0회였다.

```text
Response → Decision         PASS
Recommendation → Response  PASS
VLM → Decision              PASS
```

### 16.6 Latency

| 구간 | median | p95 |
|---|---:|---:|
| Decision | 1,807.97 ms | 2,178.45 ms |
| Response | 1,552.92 ms | 2,283.35 ms |
| Recommendation | 4,046.18 ms | 4,287.37 ms |
| 전체 model turn | 3,447.85 ms | 7,549.79 ms |

Offline 판정은 `READY FOR ROBOT TEST`였다. 이는 실제 로봇 시연 완료가 아니라 **로봇에 연결해 제한된 smoke test를 시작할 수 있는 상태**라는 뜻이다.

Cause: schema 단위 정확도만으로는 두 Adapter가 한 프로세스에서 서로 오염되지 않는지 확인할 수 없다.

Effect: 실제 GPU에서 Decision/Response 전환 1,921회와 150턴 E2E를 통과해 dual Adapter runtime 구성을 검증했다.

---

## 17. Long-Horizon 30세션 평가와 Evaluator 교정

### 17.1 로봇 전체 대신 LLM-only 평가를 먼저 수행한 이유

35세션 이상의 물리 로봇 회귀 계획을 세웠으나, runtime과 Adapter의 의미 오류를 먼저 저비용으로 분리하기 위해 **로봇 전체를 켜지 않고 LLM graph만 실행하는 long-horizon 평가**를 우선했다.

이 단계에서는 ROS motor 실행, VLM 물리 결과, 실제 재료 투입 성공을 검증하지 않았다. 따라서 `READY FOR DEMO` 판정을 내리지 않았다.

### 17.2 평가 규모

```text
sessions: 30
turns: 360
평가 대상: Decision + Policy + Recommendation + Response + 누적 state
물리 로봇 실행: 미수행
```

multi-turn에서 이전 오류가 뒤 턴 상태에 누적되므로 다음 두 점수를 분리했다.

- strict turn success: 이전 턴 오류가 남긴 state 차이까지 포함한다.
- root-cause clean turn: 현재 턴에서 새로 발생한 직접 오류만 본다.

### 17.3 기존 evaluator의 contract 오류 수정

초기 평가 결과를 그대로 재학습 근거로 쓰지 않고, 현재 runtime contract와 맞지 않는 gold/evaluator 기준부터 교정했다.

| 교정 항목 | 행 |
|---|---:|
| 명확한 unsupported 주문 gold | 3 |
| 허용 가능한 `confirmation=accept + commit=true` | 12 |
| Recommendation criteria gold | 1 |
| 자유 preference 표면형 정규화 | 1 |

초기 fake/extra mutation 후보 70건은 다음과 같이 재분류했다.

```text
TRUE_FAILURE      34
CONTRACT_CORRECT  30
AMBIGUOUS_EVAL     6
```

unsupported target은 Decision이 사용자 표현을 보존하고, Policy/final validator가 실제 반영과 실행을 막는 것이 현재 계약이다. 이를 Decision 단계에서 target을 지우지 않았다는 이유로 실패 처리하지 않도록 수정했다.

### 17.4 교정된 Decision v3 baseline

| 지표 | 결과 |
|---|---:|
| 완전 성공 session | 14 / 30 |
| strict turn success | 283 / 360 |
| root-cause clean turn | 326 / 360 |
| Order precision | 91.0714% |
| Order recall | 100.0000% |
| Order value accuracy | 100.0000% |
| Fake mutation turn | 14 |
| Missed mutation turn | 5 |
| Over-clarify / Under-clarify | 1 / 0 |
| Reference failure | 0 |
| Pending failure | 0 |
| State drift | 0 |
| Policy failure | 0 |

실제 over-mutation 원인은 다음 네 family로 좁혀졌다.

```text
pending 거절 → stale order field까지 삭제       5
preference grounding → extra order mutation     4
question → fake mutation                         3
description/impression → fake mutation           2
```

Response 오류 20건 중 11건은 앞선 Decision 오류의 downstream 결과로 재분류했다. Response Adapter 자체의 직접 실패는 9건, Recommendation Agent 자체 실패는 6건이었다.

### 17.5 Decision v3.1 candidate 생성

교정된 실제 실패만 보강하기 위해 Decision v3 train 2,595행은 그대로 두고 targeted supplement 420행을 만들었다.

```text
base train: 2,595
targeted supplement: 420
candidate train: 3,015
```

주요 family는 pending confirmation, question/description boundary, preference grounding, ambiguous reference, category field, noncanonical amount, unsupported substitution, general interlude이다.

검증 결과는 다음과 같다.

```text
Decision schema                    420 / 420 PASS
supplement id/input/message 중복   0
base train overlap                 0
frozen split overlap               0
한국어 조사·양 표현 오류           0
training started                   false
```

산출물은 `evaluation_long_horizon_corrected/decision_v3_1_candidate/`에 저장했다.

Cause: 기존 evaluator가 contract-correct 출력을 실패로 계산하면 잘못된 정답을 재학습하게 된다.

Effect: 평가 오류 36건을 실제 Decision 오류와 분리하고, 실제 실패 family만 담은 420행 supplement를 만들었다.

---

## 18. Decision v3.1 원격 재학습 TAR 제작

### 18.1 학습 범위 결정

Decision v3.1 candidate는 Decision 오류만 보강하므로 이번 TAR은 Decision만 재학습하도록 만들었다. Response v1은 새 데이터로 재학습하지 않고 기존 Adapter를 유지했다.

```text
Decision v3.1 → 새로 학습
Response v1   → 기존 best epoch 1 유지
```

### 18.2 최종 학습 설정

```text
Base model: google/gemma-4-12B-it
Train / validation: 3,015 / 285
Quantization: NF4 4-bit
Epochs: 6
Optimizer steps: 1,134 (epoch당 189)
Learning rate: 1e-4
Batch size: 1
Gradient accumulation: 16
LoRA: r=16, alpha=32, dropout=0.05
Checkpoint/evaluation: 매 epoch
```

기존 Decision v3 Adapter에서 이어 학습하지 않고 Gemma 4 12B base에서 새 LoRA를 만들었다. v3와 v3.1 비교에서 초기 Adapter 차이가 개입하지 않게 하기 위함이다.

### 18.3 입력 hash와 산출물

```text
Decision v3.1 train SHA-256:
d011cff5dc0460ec55bfd4c7b458fcc88b184064dab69ae0e3e4a963b7f895c4

Frozen validation SHA-256:
c87ab3908b34c22d8f765a5771187f6557fa98268521ab49841330157bb321df

Candidate ZIP SHA-256:
bc31fe6e253a4c6beaac0fbfb3f0c3bc0c9e107622b3373b1754089b44a654b6
```

독립 `decision_v3_1_train.jsonl`과 candidate ZIP 내부 `train.jsonl`의 hash가 같은지 확인했다.

최종 원격 학습 파일은 다음과 같다.

```text
soomac_decision_v3_1_training_v2_9.tar
size: 6.6 GiB
SHA-256: fe146c1f4702cf0ab18034fb8f5c4c5a4f8c1f8a5a24f4047da7477393a1fb3f
```

### 18.4 실제 이미지 GPU 저장·실패·재개 시험

TAR을 만들기 전에 빌드된 image 자체를 RTX PRO 4500 32GB에서 실행했다.

```text
1차 실행: step 1 → 5
checkpoint-5 저장
의도적 실패

2차 실행: checkpoint-5 복원
step 6 → 10
checkpoint-10 저장
```

| 검사 | 결과 |
|---|---:|
| NF4 model load | PASS |
| LoRA trainable parameter | 65,568,768 |
| finite loss / gradient | PASS |
| nonzero gradient | PASS |
| skipped optimizer update | 0 |
| checkpoint-5 checksum 불변 | PASS |
| checkpoint-10 전체 상태 | PASS |
| 재개 후 train loss | 0.1836494857 |
| 재개 후 eval loss | 0.1590311080 |
| peak reserved VRAM | 22.494 GiB |

Spec 2.9 label, entrypoint, linux/amd64, UID:GID 1000 import, dataset preflight, Python unit test 5/5, `compileall`, `git diff --check`를 모두 통과했다.

Cause: 원격 학습은 중간 실패 시 epoch 1부터 다시 시작하면 시간과 비용 손실이 크다.

Effect: 실제 GPU에서 checkpoint 저장, 의도적 실패, optimizer/scheduler/RNG 포함 재개가 이어지는 것을 확인한 TAR을 전달했다.

---

## 19. Decision v3.1 결과 Adapter 추출·비교·채택

### 19.1 원격 결과 수신과 학습 완료 확인

수신한 원격 결과 TAR은 다음과 같다.

```text
/home/roma/Downloads/f77bddde4aca432dacf8ec5434698cf4-result.tar
size: 10,920,173,056 bytes
SHA-256: 36124ce7046db9de108a35a5683cf717d08447a4d3051e0fbbe391ede80bb644
OCI image: local/training-result:fc3c4385674ca4b93a5e
```

원격 학습 완료 상태는 다음과 같다.

```text
global/local step: 1,134
completed epochs: 6
train loss: 0.024848956466711276
samples seen: 18,090
tokens seen: 39,539,568
training time: 37,224.4998 seconds
skipped optimizer update: 0
peak allocated VRAM: 21.102 GiB
peak reserved VRAM: 22.582 GiB
gradient finite/nonzero: PASS
safety gate: PASS
```

validation loss가 가장 낮은 checkpoint는 `checkpoint-567`, 즉 epoch 3이었다. 마지막 checkpoint는 `checkpoint-1134`, 즉 epoch 6이었다.

전체 checkpoint를 로컬에 중복 추출하지 않고 배포에 필요한 best Adapter만 꺼냈다.

```text
models/decision_adapter_v3_1_best_epoch3
adapter_model SHA-256:
050bd821cdbb8feb1ad8cefcb6492fd5b070e32e1c921efa96182cee35bff5e0
```

기존 Response Adapter는 그대로 유지했다.

```text
models/response_adapter_v1_best_epoch1
adapter_model SHA-256:
be64d1512f231873468a80eb5cf7cad54b33c11e973879af340c2add5503ab96
```

학습 image는 PEFT 0.21.0, 로컬 runtime은 PEFT 0.19.1이었다. 로컬에서 미래 config field 일부를 무시한다는 warning은 있었지만, 실제 LoRA tensor 656개를 읽고 GPU 추론과 Adapter 전환까지 성공했다.

### 19.2 실제 GPU smoke test

격리된 runner `evaluation_decision_v3_1/run_evaluation.py`로 다음 순서를 실행했다.

```text
Decision → Response → Decision → Response
```

검증 시나리오는 다음과 같다.

```text
"양파 많이 넣어줘"
→ Decision: 양파 high
→ execution pending 생성
→ Response: 실행 확인 질문

"아니 실행은 취소해"
→ Decision: confirmation reject
→ pending 제거
→ 실행 미승인
```

두 named Adapter 등록, 네 번의 Adapter activation, order mutation, pending 생성과 거절을 모두 통과했다.

### 19.3 Frozen split에서 v3와 v3.1 비교

| 지표 | v3 | v3.1 | 변화 |
|---|---:|---:|---:|
| Schema valid | 100.00% | 100.00% | 유지 |
| Route accuracy | 100.00% | 98.70% | -1.30%p |
| Order precision | 90.49% | 93.67% | +3.18%p |
| Order recall | 96.34% | 97.75% | +1.41%p |
| Order value accuracy | 98.83% | 98.85% | +0.02%p |
| Fake mutation | 70 | 47 | -23 |
| Missed mutation | 26 | 16 | -10 |
| Wrong reference | 12 | 9 | -3 |

v3.1 route 오류 15건은 전부 다음과 같은 동일 synthetic template family였다.

```text
"오일 활용이 시작된 배경을 알려줘"
"양파 활용이 시작된 배경을 알려줘"
"새우 활용이 시작된 배경을 알려줘"
```

gold는 `general`이지만 v3.1은 mutation 없는 `task`를 출력했다. 이 오류는 route 분류 오류이지만 order mutation이나 물리 실행으로 이어지지 않았다.

### 19.4 Long-horizon에서 v3와 v3.1 비교

| 지표 | v3 | v3.1 | 변화 |
|---|---:|---:|---:|
| 완전 성공 session | 14 / 30 | 12 / 30 | -2 |
| strict turn success | 283 / 360 | 314 / 360 | +31 |
| root-cause clean turn | 326 / 360 | 330 / 360 | +4 |
| Decision 직접 실패 | 34 | 10 | -24 |
| Fake mutation turn | 14 | 4 | -10 |
| Missed mutation turn | 5 | 2 | -3 |
| Order precision | 91.07% | 96.19% | +5.12%p |
| Order recall | 100.00% | 99.02% | -0.98%p |
| Order value accuracy | 100.00% | 100.00% | 유지 |
| Wrong reference | 0 | 0 | 유지 |
| Pending failure | 0 | 0 | 유지 |
| State drift | 0 | 0 | 유지 |
| Policy failure | 0 | 0 | 유지 |

완전 성공 session 수는 2개 줄었지만 개별 turn 성공은 31개 늘었다. session 지표는 한 턴 실패만 있어도 전체 session을 실패로 잡으므로, 실제 개선 정도는 direct failure와 turn 단위 지표를 함께 봤다.

Preference persistence mismatch 8건은 독립 오류 8개가 아니다. Recommendation revise 한 턴에서 `매콤하게` preference를 한 번 놓친 뒤 같은 session의 뒤 7턴에 state 차이가 이어진 결과이다.

### 19.5 남은 Decision 직접 실패 10건

```text
pending accept에서 기존 order field 불필요 반복       4
recommendation revise에서 preference 누락             1
recommendation pending reject를 clarify로 처리         3
compound/preselected에서 recommendation spillover      2
```

Response 직접 실패는 v3 baseline 9건에서 v3.1 조합 15건으로 늘었다. 대부분 pending을 유지한 채 일반/status 질문에 답한 뒤 실행·추천 확인 질문을 다시 붙이지 않은 경우였다. Response Adapter는 동일하므로 이 결과는 다음 Response targeted supplement 대상으로 분리했다.

Recommendation Agent 실패는 5건이었다.

### 19.6 채택 결정과 runtime 변경

최종 결정은 다음과 같다.

```text
ADOPT DECISION V3.1
KEEP RESPONSE V1
READY FOR LLM-ONLY / ROBOT SMOKE TEST
```

`src/soomac_irc/soomac_irc/model_runtime.py`의 기본 Decision Adapter 경로를 다음과 같이 변경했다.

```text
decision_adapter_v3_best_epoch3
→ decision_adapter_v3_1_best_epoch3
```

`models/MODEL_MANIFEST.md`에는 평가 당시 Git HEAD, GPU/library 버전, 두 Adapter 경로와 hash, generation 설정을 기록했다.

검증 결과는 다음과 같다.

```text
actual GPU smoke                 PASS
safetensors integrity           PASS, 656 tensors
adapter switching               PASS
manual runtime routing assert   PASS
compileall                      PASS
git diff --check                PASS
pytest                          SKIP
```

`pytest`는 사용 가능한 `gemma4_env`와 `ros_env` 모두에 설치되어 있지 않아 실행하지 못했다. 대신 실제 GPU graph smoke와 명시적 assertion으로 Adapter routing과 artifact 경로를 확인했다.

Cause: v3.1은 route 15건이 퇴행했지만 핵심 목표인 과잉 mutation, 누락 mutation, Decision 직접 실패를 크게 줄였고 safety/state 지표를 유지했다.

Effect: runtime 기본 Decision은 v3.1 epoch 3을 사용하며 Response는 검증된 v1 epoch 1을 계속 사용한다.

---

## 20. Decision v3.2 + Response v1.1 Targeted Dataset 생성

### 20.1 작업 범위

Decision v3.1 평가에서 남은 실패와 Response v1의 pending reminder 실패만 보강했다.

이번 단계에서는 다음을 변경하지 않았다.

```text
runtime source
Decision/Response prompt
Decision/Response schema
현재 Adapter
기존 base dataset
```

학습도 시작하지 않았다. 별도 후보 데이터셋만 `data_targeted_upgrade/`에 생성했다.

### 20.2 공통 생성 원칙

새 문장을 무작위로 늘리지 않고 실제 평가 failure ID를 provenance에 연결했다.

```text
실제 failure family 확인
→ family별 positive/negative contrast 설계
→ supplement 생성
→ schema/runtime lifecycle 검사
→ base/frozen contamination 검사
→ 고위험 행 층화 검토
→ base + supplement candidate 생성
```

candidate의 행 순서는 기존 base 전체 뒤에 supplement를 붙이는 방식으로 고정했다. 기존 학습 표본을 수정하거나 다시 섞지 않았다.

### 20.3 Decision v3.2 candidate

```text
base: 3,015
supplement: 450
candidate: 3,465
route task/general: 351 / 99
preference target rows: 45
unsupported rows: 65
```

family 배분은 다음과 같다.

| family | 행 |
|---|---:|
| general question vs task | 90 |
| pending accept/reject boundary | 70 |
| recommendation confirmation/reject | 55 |
| recommendation revise + preference | 45 |
| question vs mutation | 45 |
| description/impression vs mutation | 35 |
| ambiguous reference | 35 |
| compound intent spillover | 25 |
| field/category/amount | 20 |
| regression preservation | 30 |

`semantic_review.jsonl`에는 고위험 220행을 family와 subtype별로 층화해 담았다. 단순히 생성 파일 앞 220행을 고른 것이 아니라 general, pending, recommendation, question, description, reference, compound intent를 고르게 포함했다.

### 20.4 Decision 생성 중 발견하고 고친 데이터 오류

첫 생성 검사에서 한국어 조사와 양 표현 오류 14건을 잡았다.

```text
버섯를
게살를
페퍼론치노이니
많이로
```

조사 선택 helper와 자연스러운 양 표현으로 수정한 뒤 전체를 다시 생성했다.

수동 검토에서는 다음 semantic 오류도 수정했다.

- 양을 말한 explicit noodle request에 `noodle_portion`이 빠진 사례
- pending reject와 별도 topping 삭제의 의미가 섞인 문장
- noncanonical amount를 부자연스럽게 설명한 문장
- reset 예시의 current order가 비어 있어 실제 삭제를 검증하지 못한 사례
- 고위험 검토 표본이 앞쪽 family에 편중된 사례

reset 예시는 current order가 실제로 채워진 상태에서 order scalar와 topping에 `none` mutation을 출력하고 restriction/preference는 유지하도록 구성했다.

### 20.5 Decision 검증 결과

```text
schema invalid                         0
legacy field                           0
id/input/message/input+target duplicate 0
base overlap                           0
frozen overlap                         0
malformed Korean                       0
unsupported preservation violation     0
general/task semantic error            0
pending semantic error                 0
semantic suspicious                    0
near-template excess cluster           0
near-template max cluster size          5
```

hash는 다음과 같다.

```text
base train:
d011cff5dc0460ec55bfd4c7b458fcc88b184064dab69ae0e3e4a963b7f895c4

supplement:
c81fbca9ecbdf990cd46fba38fb8f2c53ebced8060df629aa7513cbbc9539c3c

candidate train:
76b2123c0183fcb703163cc5cc0e937d8c97e070c9d0efa3ee417b245e0a5060
```

### 20.6 Response v1.1 candidate

```text
base: 898
supplement: 200
candidate: 1,098
```

family 배분은 다음과 같다.

| family | 행 |
|---|---:|
| pending + general question | 70 |
| pending + status question | 35 |
| pending + unrelated explanation | 25 |
| pending + preference statement | 20 |
| pending recommendation context | 20 |
| no-pending control | 20 |
| active/completed execution control | 10 |

핵심 positive target은 **pending이 남아 있는 일반 질문에 먼저 정확히 답하고, 같은 응답 안에서 아직 필요한 확인을 자연스럽게 다시 묻는 것**이다.

반대로 pending이 없거나 task가 이미 active/completed인 control에서는 불필요한 확인 질문을 붙이지 않도록 negative pair를 넣었다.

`semantic_review.jsonl`에는 family별로 층화한 120행을 담았다.

### 20.7 Response runtime contract 검증

Response 입력은 새 임의 validator가 아니라 기존 정본을 직접 재사용했다.

```text
data_natural_multiturn_v3_rebuild/
scripts/validate_response_runtime_contract.py:row_failures
```

추가로 completed/active/pending 모순과 관련 evidence 존재 여부를 기존 lifecycle validator와 같은 기준으로 확인했다.

```text
invalid structured input     0
empty reply                  0
runtime contract failure     0
fact failure                 0
pending failure              0
execution failure            0
next_prompt failure          0
duplicate                    0
base/frozen overlap          0
malformed Korean             0
near-template excess         0
near-template max cluster     5
```

hash는 다음과 같다.

```text
base train:
b258566aa4ebcdc4ac18b106d1dbc2fabf454f8b75bf1606fc64d435354a9f45

supplement:
e7f7faa056a48d6c34533cb06308bb17e01c26398832044666350fa631d8965d

candidate train:
e16b007a0a3c78711951e8b061322fb656f4c09be410f361522aba6d1803d558
```

### 20.8 생성 파일

두 candidate는 동일한 파일 구성을 가진다.

```text
data_targeted_upgrade/
├── build_targeted_upgrade.py
├── generation_summary.json
├── DATASET_GENERATION_REPORT.md
├── decision_v3_2_candidate/
│   ├── train.jsonl
│   ├── supplement_train.jsonl
│   ├── supplement_provenance.jsonl
│   ├── supplement_design.md
│   ├── validation_report.json
│   ├── manifest.json
│   └── semantic_review.jsonl
└── response_v1_1_candidate/
    ├── train.jsonl
    ├── supplement_train.jsonl
    ├── supplement_provenance.jsonl
    ├── supplement_design.md
    ├── validation_report.json
    ├── manifest.json
    └── semantic_review.jsonl
```

candidate train이 정확히 `base + supplement` 순서인지, manifest hash가 실제 파일과 일치하는지, provenance source ID가 실제 v3.1 평가 failure인지 독립적으로 다시 확인했다.

```text
Decision v3.2 validation     PASS
Response v1.1 validation     PASS
manifest/hash verification  PASS
provenance verification     PASS
review row count/status      PASS
compileall                   PASS
git diff --check             PASS
training_started             false
```

최종 판정은 다음과 같다.

```text
Decision v3.2 candidate: READY
Response v1.1 candidate: READY
Training: NOT STARTED
```

Cause: v3.1의 남은 오류는 전체 architecture 문제가 아니라 route/pending/recommendation 경계와 Response pending reminder에 집중되어 있었다.

Effect: 기존 데이터와 runtime을 건드리지 않고, 실제 실패 provenance를 가진 Decision 450행과 Response 200행 supplement를 별도 후보로 만들었다.

---

## 21. 현재 기준 상태와 아직 하지 않은 작업

> 이 절은 2026-10-10 targeted candidate 생성 직후의 상태이다. 학습·선정·실물 로봇 검증까지 반영한 최신 상태는 22절을 따른다.

### 21.1 현재 runtime 기준

```text
Git HEAD: 9332c069fa22bad8db49db27ba6ee13a12aac4ee
Decision runtime Adapter: decision_adapter_v3_1_best_epoch3
Response runtime Adapter: response_adapter_v1_best_epoch1
Decision v3.2 training: NOT STARTED
Response v1.1 training: NOT STARTED
```

현재 working tree에는 Adapter 평가 산출물, 학습 TAR, targeted dataset과 `model_runtime.py`의 v3.1 경로 변경이 아직 commit되지 않은 상태로 존재한다.

### 21.2 검증된 범위

```text
Decision/Response dual Adapter GPU load
Adapter switching
frozen split inference
30-session / 360-turn LLM-only graph
Decision v3.1 runtime path
v3.2 / Response v1.1 dataset static validation
```

### 21.3 아직 검증하지 않은 범위

```text
Decision v3.2 Adapter 학습
Response v1.1 Adapter 학습
새 두 Adapter의 GPU regression
35-session 실제 물리 로봇 regression
READY FOR DEMO 판정
```

`physical_robot_evaluation/rosbag_first_pass/`에는 초기 rosbag scaffold가 있으나, 요구된 규모의 물리 로봇 테스트 결과로 간주하지 않는다.

Cause: LLM-only 평가와 targeted dataset 생성까지만 완료했고, 새 candidate 학습과 실제 로봇 물리 검증은 별도 단계이다.

Effect: 현재 운영 기준과 다음 학습 후보를 혼동하지 않고, v3.1+Response v1은 실행 가능한 baseline으로, v3.2+Response v1.1은 미학습 candidate로 구분한다.

---

## 22. 2026-10-11 실물 로봇 통합 검증과 자동 전환 응답 고정

참고 코드: [Qwen3-ASR 노드](../src/soomac_irc/soomac_irc/stt_qwen3_asr_node.py), [공통 STT 수음·게이트](../src/soomac_irc/soomac_irc/stt_nemotron_node.py), [단계 전환](../src/soomac_irc/soomac_irc/llm_langgraph_node.py), [단계 전환 회귀 테스트](../src/soomac_irc/test/test_section_question_events.py)

→ 실제 마이크·TTS·VLM·로봇을 연결한 한 주문을 끝까지 실행하고, 수음 경로와 뚜껑·소스 자동 전환 응답을 점검한 기록이다.

참고 로그: [12턴 실물 로봇 세션의 Agent turn](../logs/llm/2026-10-11/session_20261011_123013_105237_pid26271/agent_turns.jsonl), [같은 세션의 runtime event](../logs/llm/2026-10-11/session_20261011_123013_105237_pid26271/runtime_events.jsonl), [선정 Adapter 스모크 보고서](../training_v3_2_v1_1/SELECTED_PAIR_12TURN_SMOKE_REPORT.md)

→ 자연어 입력, 확정 상태, 로봇 작업, 단계 전환, VLM 판정과 실제 발행 응답을 같은 세션 ID로 추적할 수 있다.

### 22.1 최신 runtime 기준

```text
Git HEAD: b17f54f
Git branch: flexible_llm
Remote sync: origin/flexible_llm과 동일
Decision runtime Adapter: decision_adapter_v3_2_best_epoch4
Response runtime Adapter: response_adapter_v1_best_epoch1
Qwen3-ASR: Qwen3-ASR-1.7B-hf, bfloat16, CUDA
```

Decision v3.2 epoch 4는 채택했다. Response v1.1 epoch 4는 targeted train-fit은 개선했지만 frozen RED 2건과 long-horizon 실패 54건으로 회귀하여 채택하지 않았다. 따라서 실제 로봇 세션은 **Decision v3.2 + Response v1** 조합으로 수행했다.

### 22.2 수음 개선 완료 상태

Qwen3-ASR 노드는 Nemotron STT의 ROS 토픽, Silero 음성 활동 검출(voice activity detection, VAD), TTS 게이트와 세션 무효화 구조를 재사용하고 추론부만 교체한다.

실제 시작 로그에서 다음 상태를 확인했다.

```text
Silero VAD load: 0.03 s, CPU
Qwen3-ASR load: 1.18 s, CUDA, bfloat16
GPU memory: allocated 3.80 GiB / reserved 3.80 GiB
Capture: PulseAudio, 16,000 Hz, mono, 16-bit
Source: alsa_input.usb-MATA_MATA_STUDIO_C10-00.analog-stereo
Speech probability threshold: 0.50
Utterance close silence: 0.8 s
```

`SOOMAC_AUDIO_SOURCE`를 지정하면 해당 PulseAudio source를 사용한다. 지정하지 않으면 시작 시점의 default source를 조회한다. source가 없거나 출력 monitor이면 조용히 잘못된 장치를 쓰지 않고 시작을 실패시킨다.

TTS가 시작되면 STT 세션 번호를 올리고 gate를 닫는다. 추론이 끝나더라도 캡처 당시 세션과 현재 세션이 다르면 결과를 버린다. 실제 로봇 작업 중에는 TTS 종료 뒤에도 STT를 다시 열지 않았다.

**현장 확인 결과 USB 마이크 수음, 발화 시작·종료 검출, TTS 잔향 차단과 로봇 작업 중 수음 차단이 정상 동작했다.**

(내가 의역해보자면 이제 “어떤 마이크를 듣고 있는가”와 “지금 들어도 되는가”가 둘 다 명시적인 상태가 되었다.)

Cause: 기본 입력 장치에 의존하면 USB 연결 상태나 PulseAudio 설정에 따라 출력 monitor 또는 다른 입력을 잡을 수 있었다.

Effect: 지정한 MATA USB 마이크만 사용하고, TTS·로봇 작업 중 들어온 음성이나 늦은 추론 결과가 다음 대화 세션을 오염시키지 않는다.

### 22.3 실물 로봇 한 사이클 결과

12개 사용자 turn으로 다음 작업을 완료했다.

```text
얇은면 2회
버섯 1회
소시지 3회
치즈 3회 — 첫 VLM fail 뒤 robot_retry, 두 번째 pass
페퍼론치노 3회
뚜껑 1회 — VLM pass
토마토 소스 1회 — VLM pass
최종 session_end 정상 발행
```

로봇 작업 순서, task queue, VLM 재시도, 완료 기록과 최종 종료는 정상 동작했다. 치즈 첫 판정 실패도 오류로 묻히지 않고 같은 작업을 한 번 재시도하여 통과했다.

대화 의미에서는 두 failure를 확인했다.

| 위치 | 실제 입력·상태 | 문제 |
|---|---|---|
| Turn 1 | STT 결과 `저 먹어를 수 있어요.` | 근거 없는 `게살 cannot_eat`을 Decision이 생성했다. 대상이 불명확하므로 clarify가 맞다. |
| Turn 5 | `바로 시작해 줘`, `execution_authorized=true`, `starting_now=[버섯 low]` | Response가 실제 실행과 반대로 `실행 확인이 남아 있어요`라고 말했다. |

이 두 건은 로봇 상태 머신이나 수음 gate 문제가 아니다. Turn 1의 근거 없는 대상 생성은 Decision 재학습과 grounding 검사가 필요하다. 반면 Turn 5는 후속 전수 감사에서 **실행이 이미 시작된 안전 상태를 Response가 반대로 말한 runtime 정합성 문제**로 확인되었다. 이 문제는 재학습만으로 맡기지 않고 상태 기반 응답으로 막아야 한다. 자세한 연쇄 원인과 수정 지점은 23절에 기록한다.

### 22.4 뚜껑·소스 자동 전환 응답 고정

추가 재료 완료 뒤 뚜껑을 시작할 때와 뚜껑 완료 뒤 소스를 시작할 때는 사용자의 새 발화를 받지 않는다. STT도 계속 닫혀 있고 다음 robot task가 이미 확정되어 있다.

수정 전에는 이 두 전환도 Response Adapter를 호출했다.

```text
extra → lid
예상: 이제 뚜껑을 닫을게요.
실제: 직전 치즈·페퍼론치노 실행 문장 반복

lid → sauce
예상: 마지막으로 고르신 소스를 올릴게요.
실제: 같은 직전 실행 문장 반복
```

Decision/Response train·validation과 targeted candidate에는 `lid`, `cover`, `뚜껑` 전환 행이 0개였다. 그러나 이 구간은 사용자 의미를 생성할 필요가 없는 deterministic transition이므로 학습 행을 추가하는 것보다 runtime 소유권을 명확히 하는 편이 맞다.

`LLMLangGraphNode._advance_after_section(completed_section, outcome="completed")`은 이제 다음처럼 동작한다.

```text
next section = lid
→ cover task 시작
→ "이제 뚜껑을 닫을게요." 직접 발행
→ Response 모델 호출 안 함

next section = sauce
→ 선택된 sauce task 시작
→ "마지막으로 고르신 소스를 올릴게요." 직접 발행
→ Response 모델 호출 안 함

next section = veggie / meat / extra
→ 기존 next_prompt 구성
→ Response 모델 호출 유지
```

실패 fallback으로만 존재하던 `next_reply`를 뚜껑·소스의 정상 경로로 사용했다. runtime event와 history 기록, task 시작, VLM, STT gate는 기존 경로를 유지한다.

직전 실측에서 두 Response 호출은 각각 3.978초와 3.967초였다. 따라서 모델 추론만 기준으로 주문 한 건당 약 **7.945초**를 줄인다. TTS 재생 시간과 실제 로봇 동작 시간은 이 수치에 포함하지 않는다.

회귀 테스트는 다음 계약을 고정한다.

```text
extra → lid: 정확한 고정 문구, call_response 미호출, STT 미개방
lid → sauce: 정확한 고정 문구, call_response 미호출, STT 미개방
veggie / meat / extra: 기존 Response model 호출 유지
target test: 11 passed
git diff --check: PASS
```

Cause: 이미 다음 작업과 안내 문구가 확정된 자동 전환까지 생성 모델에 맡겼다.

Effect: 로봇 상태와 음성 안내가 항상 일치하고, stale 대화 반복 가능성과 GPU 추론 2회가 제거된다.

### 22.5 향후 계획

| 우선순위 | 작업 | 완료 기준 |
|---|---|---|
| P0 | 실제 task 시작 응답과 모든 system-driven section 전환 응답을 structured state 기반으로 고정한다. | `execute=true`이면 확인 질문이 나오지 않고, 전환 시 새 section 안내만 한 번 기록된다. |
| P0 | unresolved pending과 빈 optional section의 bare commit을 차단한다. | pending 후보가 사라지지 않으며 `진행해`만으로 선택 없는 다음 section을 건너뛰지 않는다. |
| P1 | Decision/Recommendation 실패 family를 relabel하고 독립 train 행으로 보강한다. | 불명확한 발화가 메뉴 대상을 만들지 않고, 추천 후보 부분 수정이 나머지 후보를 보존한다. |
| P1 | 수정된 runtime과 새 모델로 동일 물리 시나리오를 다시 실행한다. | 실행 확인, section 전이, VLM retry, lid/sauce와 최종 session_end가 한 로그에서 일치한다. |
| P2 | 실제 로봇 세션 로그를 세션 단위로 계속 보존한다. | `agent_turns.jsonl`과 `runtime_events.jsonl`이 같은 session ID로 GitHub에 추적된다. |

뚜껑·소스뿐 아니라 사용자 입력 없이 발생하는 모든 section 전환 안내는 다음 재학습 범위에서 제외한다. 해당 문구는 모델 일반화 문제가 아니라 이미 결정된 상태를 알리는 deterministic UI/TTS 계약이기 때문이다.

(여기서 hardcoding이 필요한 경계가 보인다. 사용자 의미나 메뉴 선택은 모델이 맡지만, 이미 확정된 로봇 단계의 상태 안내는 코드가 맡는다.)

## 23. 2026-10-11 주문 흐름 꼬임 전수 감사

참고 로그:

- `logs/llm/2026-10-11/session_20261011_122351_025654_pid14055/agent_turns.jsonl`
- `logs/llm/2026-10-11/session_20261011_123013_105237_pid26271/agent_turns.jsonl`
- `logs/llm/2026-10-11/session_20261011_130433_286854_pid47658/agent_turns.jsonl`
- `logs/llm/2026-10-11/session_20261011_131341_544954_pid55977/agent_turns.jsonl`
- `logs/llm/2026-10-11/session_20261011_132944_155638_pid65476/agent_turns.jsonl`

→ **핵심은 실행 확인 상태가 사라진 것이 아니라, 이미 실행된 뒤에도 Response가 “확인이 남았다”고 거짓 안내하고 그 문장이 다음 단계까지 복제된 것이다.**

주문·시나리오에 직접 영향을 주지 않은 잡음 발화와 일반대화는 이번 감사 범위에서 제외했다.

### 23.1 가장 쉽게 설명한 실제 고장 순서

정상이라면 다음 순서여야 한다.

```text
사용자: 양파 조금, 버섯 많이 줘
시스템: 선택을 저장하고 실행 확인 pending 생성
사용자: 버섯을 조금으로 바꿔서 바로 담아
시스템: pending 수락 → 양파·버섯 실행 → pending 종료
로봇: 야채 완료 → 육류 선택 안내
```

실제로는 다음처럼 진행되었다.

```text
1. 사용자가 "버섯 조금으로 바꿔서 바로 담아"라고 말했다.
2. structured state는 요청을 정상 처리했다.
   - policy.execute = true
   - pending = null
   - 양파 task 시작, 버섯 task queue 등록
3. Response만 반대로 "실행 확인이 남아 있어요. 담기를 진행할까요?"라고 말했다.
4. 로봇은 이미 야채를 담고 육류 단계로 이동했다.
5. 사용자는 방금 들은 질문에 답하려고 "어 진행해"라고 말했다.
6. 시스템은 이 말을 육류 단계의 새 commit으로 해석했다.
7. 선택된 육류가 없어서 육류를 통째로 건너뛰고 추가 재료 단계로 이동했다.
```

(내가 의역해보자면 로봇의 실제 장부에는 “이미 실행함”이라고 적혀 있었는데, 입으로는 “아직 확인 안 됨”이라고 말했다. 사용자가 그 입말에 답하자 시스템은 이미 다음 페이지로 넘어간 장부에 답을 적용했다.)

Cause: structured state와 사용자에게 읽어 주는 Response 문장이 서로 다른 사실을 말했다.

Effect: 정상적인 `어 진행해`가 다음 section의 실행 명령으로 오해되어 육류 단계가 건너뛰어졌다.

### 23.2 문제 1 — 실행 중인데 다시 실행 확인을 요구함

같은 failure가 세 세션에서 재현되었다.

| 세션 | state와 실제 동작 | 잘못된 응답 |
|---|---|---|
| `122351`, Turn 5 | `execute=true`, 버섯 task 시작 | `버섯 조금 담기를 진행할까요?` |
| `123013`, Turn 5 | `execute=true`, `pending=null`, 버섯 task 시작 | `실행 확인이 남아 있어요` |
| `131341`, Turn 5 | `execute=true`, 양파 시작·버섯 queue, `pending=null` | `실행 확인이 남아 있어요` |

`response_model.py`는 Response 입력에 다음 사실을 정확히 넣고 있다.

```text
execution_authorized = true
starting_now = 현재 실제로 시작할 재료 목록
```

그러나 `agent_prompts.py`에는 `execution_authorized=false`일 때 시작했다고 말하지 말라는 규칙만 있다. `true`일 때 확인 질문을 금지하는 대칭 규칙과 runtime 출력 검사가 없다.

**수정 위치와 내용**

| 위치 | 수정할 내용 |
|---|---|
| `src/soomac_irc/soomac_irc/llm_langgraph_node.py`, `_process_turn()`의 `_start_current_section()` 성공 분기 | Graph의 자유 생성 문장을 그대로 발행하지 말고, 실제 `active_task`와 `task_queue`로 시작 멘트를 만든다. |
| 같은 분기의 `graph_state["history"]` | Graph가 미리 기록한 assistant 응답도 실제 시작 멘트로 교체하여 다음 턴에 거짓 질문이 남지 않게 한다. |
| `src/soomac_irc/soomac_irc/agent_prompts.py`, `TASK_RESPONSE_SYSTEM` | 보조 방어선으로 `execution_authorized=true`이면 확인 질문과 `확인이 남아 있다`는 표현을 금지한다. |

Cause: 실제 task 시작 여부가 확정되기 전에 Response 모델이 문장을 만들고, 노드는 그 문장을 사후 검증 없이 발행한다.

Effect: 실제 시작 task 기반 deterministic 응답을 사용하면 로봇 동작과 TTS가 반드시 일치한다.

### 23.3 문제 2 — section 전환 모델이 새 안내 대신 직전 문장을 복사함

`LLMLangGraphNode._advance_after_section()`은 이미 정확한 `next_reply`를 만든다.

```text
미리 고른 재료가 있음 → build_preselected_confirmation_reply(pending)
미리 고른 재료가 없음 → _section_prompt()
```

하지만 현재는 lid와 sauce에서만 이 값을 직접 사용한다. veggie, meat, extra 전환은 `user_text=""`로 Response 모델을 다시 호출한다. 문제가 발생한 세션에서는 모델이 `next_prompt=meat` 또는 `next_prompt=extra`보다 recent history의 잘못된 실행 확인 문장을 따라 했다.

또한 `_advance_after_section()`은 그 잘못된 문장을 history에 다시 append한다. 빈 section을 skip하는 `_process_turn()` 경로에서도 Graph의 임시 응답과 전환 응답이 둘 다 history에 남을 수 있다.

**수정 위치와 내용**

| 위치 | 수정할 내용 |
|---|---|
| `llm_langgraph_node.py`, `_advance_after_section()`의 `if self.section in ("lid", "sauce")` 분기 | 조건을 없애고 모든 system-driven 전환에서 `reply = next_reply`를 사용한다. |
| 같은 함수의 `call_response(...)` 경로 | 정상 경로에서는 제거한다. 자연화가 꼭 필요하면 새 section/type 검증에 통과한 출력만 채택한다. |
| `_process_turn()`의 optional section skip 분기 | `_advance_after_section()` 호출 전에 Graph가 history에 넣은 임시 assistant 응답을 제거하고, 전환 응답 하나만 남긴다. |

Cause: 이미 정답이 있는 상태 전환을 history 영향을 받는 생성 모델에 다시 물었다.

Effect: 야채·육류·추가 재료·뚜껑·소스 전환이 모두 현재 section과 일치하며 stale 문장이 다음 턴으로 전파되지 않는다.

### 23.4 문제 3 — unresolved recommendation이 section skip과 함께 사라짐

`130433` 세션에서는 추천 pending에 양파와 버섯이 들어 있었다. 사용자는 `양파만 빼고 바로 해줘`라고 말했다. 의도는 **추천 후보에서 양파만 제거하고 버섯을 실행하는 것**이다.

Decision은 `양파=none`, `commit=true`를 만들었지만 `confirmation=accept`를 누락했다. 따라서 추천 후보의 버섯은 canonical order로 이동하지 않았다. 그 상태에서 commit이 승인되었고, 현재 야채 task가 0개라 veggie 전체가 skip되었다. section 전환 시 pending도 `null`로 제거되었다.

**수정 위치와 내용**

| 위치 | 수정할 내용 |
|---|---|
| `src/soomac_irc/soomac_irc/llm_langgraph.py`, `_resolve_pending()` | execution pending에서 `commit=true`가 명확하면 실행 수락으로 정규화한다. recommendation pending은 `confirmation=accept` 없이 실행하지 않는다. |
| 같은 파일 `process_task()`의 commit 분기 | unresolved recommendation pending이 있으면 `execute=false`로 두고 후보 수락·수정 여부를 다시 확인한다. |
| `llm_langgraph_node.py`, optional section skip 직전 | `graph_state.pending is not None`이면 section을 이동하지 않는 마지막 가드를 둔다. |

Cause: commit 경로가 unresolved pending 존재 여부를 검사하지 않고, section 전환이 pending을 무조건 제거한다.

Effect: 추천 후보의 일부만 수정해도 나머지 후보가 보존되며, 확인되지 않은 추천 때문에 section 전체가 사라지지 않는다.

### 23.5 문제 4 — 선택 없는 optional section에서 bare commit이 skip으로 바뀜

현재 veggie, meat, extra는 필수값이 없다. 따라서 해당 section에 선택 재료가 없어도 `missing_current_section()`은 빈 목록을 반환한다. Decision이 어떤 이유로든 `commit=true`를 만들면 policy는 실행을 허용하고, node는 task가 없다는 이유로 section을 자동 skip한다.

이 구조는 다음 두 상황을 구분하지 못한다.

```text
"어 진행해"              → 앞의 실행 확인 질문에 답한 것일 수 있음
"아무것도 안 넣고 넘어가" → 현재 section을 명시적으로 skip한 것임
```

**수정 위치와 내용**

| 위치 | 수정할 내용 |
|---|---|
| Decision contract와 학습데이터 | `commit`과 별도로 명시적 section skip 의미를 표현하는 `skip_current` 계열 field를 추가한다. |
| `llm_langgraph.py`, commit 분기 | order 변경 없는 bare commit은 현재 section과 일치하는 execution pending이 있을 때만 허용한다. |
| `llm_langgraph_node.py`, optional skip 분기 | 명시적 skip이거나 검증된 skip policy일 때만 `_advance_after_section(..., outcome="skipped")`를 호출한다. |

Cause: 실행 승인과 현재 단계 생략을 하나의 `commit` 의미로 처리한다.

Effect: 불명확한 `진행해`나 약한 ASR 결과 한 번으로 재료 단계가 사라지거나 뚜껑이 시작되지 않는다.

### 23.6 재학습이 필요한 부분과 하드코딩할 부분

| 구분 | 처리 | 이유 |
|---|---|---|
| 실제 task 시작 멘트 | runtime deterministic | 이미 시작한 task라는 단일 사실만 전달한다. 틀리면 실제 동작과 음성이 충돌한다. |
| system-driven section 전환 | runtime deterministic | `next_reply`가 이미 계산되어 있고 새 사용자 의미가 없다. |
| 불명확한 발화의 메뉴 target 생성 | Decision 재학습 + grounding 검사 | `저 먹어를 수 있어요`에서 게살 제한을 만든 사례처럼 의미 판단이 필요하다. |
| 추천 후보 부분 수정 | Decision/Recommendation 재학습 + pending guard | `양파만 빼고` 같은 참조 해석은 모델이 필요하지만 후보 유실은 코드가 막아야 한다. |
| 자연스러운 주문 설명 | Response 재학습 | 크림 추천 수락을 답변에서 누락한 것처럼 안전 상태를 바꾸지 않는 표현 품질 문제이다. |

Response v1의 전체 split 1,225건에서 실행 허용 예제 61건과 section 전환 예제 16건은 모두 `recent_history=[]`이다. `주문 변경 + 즉시 실행 + 기존 실행 확인 history` 조합은 0건이다. v1.1 candidate에도 같은 공백이 있으므로 Adapter만 교체해서는 이번 문제가 해결되지 않는다.

(여기서 모델 재학습과 하드코딩의 경계가 명확해진다. **사용자의 뜻을 알아내는 일은 모델이 하고, 이미 확정된 로봇의 실제 상태를 말하는 일은 코드가 한다.**)

### 23.7 필수 회귀 테스트

1. stale history에 `실행 확인이 남아 있어요`가 있어도 `execute=true`이면 실제 시작 재료만 말한다.
2. veggie 완료 뒤에는 history 내용과 무관하게 meat 안내, meat 완료·skip 뒤에는 extra 안내를 말한다.
3. recommendation pending `{양파, 버섯}`에서 `양파만 빼고 바로 해줘`는 버섯을 보존하며, 확인이 불명확하면 section을 유지한다.
4. pending 없는 빈 meat/extra에서 `어 진행해`만으로 section을 skip하지 않는다.
5. `아무것도 안 넣고 다음 단계로 넘어가`는 명시적 skip으로 정상 처리한다.

정상 대조군인 `session_20261011_132944_155638_pid65476`의 5개 turn은 STT, Decision, pending 생성·소비, 실제 실행이 모두 일치했다. 따라서 전체 상태 머신이 항상 실패하는 문제가 아니라 **변경과 즉시 실행이 한 문장에 있고, 오염된 history 또는 unresolved pending이 함께 있는 경계 조건 문제**이다.

Cause: 학습데이터에 단순한 단일 턴은 있지만 실제 장기 대화에서 생기는 `변경 + 즉시 실행 + 기존 확인 + section 전환` 조합이 없다.

Effect: 위 5개 회귀 시나리오를 고정하면 모델 교체와 runtime 수정 뒤에도 오늘의 연쇄 오류가 다시 발생하는지 자동 검출할 수 있다.

### 23.8 적용 순서와 커밋 경계

이번 수정은 로그 원본 보존과 runtime 패치를 분리한다. 그래야 문제를 재현한 원본 로그와 문제를 고친 코드를 같은 커밋에서 섞지 않고 비교할 수 있다.

```text
1. 이 문서와 2026-10-11 실물 로그를 먼저 커밋한다.
2. 커밋 메시지는 "로그 백업 및 터지는 부분 수정 예정"으로 남긴다.
3. 해당 커밋을 GitHub에 push하여 수정 전 증거를 원격에 보존한다.
4. 재학습을 제외한 runtime 안전 패치와 회귀 테스트를 적용한다.
5. 테스트 결과를 확인한 뒤 패치 커밋·push 여부를 별도로 결정한다.
```

이번 비재학습 패치 범위는 다음과 같다.

| 순서 | 패치 | 이번 완료 기준 |
|---|---|---|
| 1 | execution pending과 `commit=true` 정규화 | 실행 승인과 pending이 동시에 남지 않는다. |
| 2 | unresolved recommendation 실행 차단 | 추천 후보가 확인 없이 실행·skip·삭제되지 않는다. |
| 3 | 실제 task 기반 실행 시작 멘트 | `execute=true` 뒤 재확인 질문이 발행되지 않는다. |
| 4 | 모든 system-driven section 전환을 `next_reply`로 고정 | stale history가 다음 section 안내를 덮지 않는다. |
| 5 | 빈 optional section의 bare commit 차단과 history 중복 제거 | `어 진행해`만으로 section이 넘어가지 않고 전환 응답은 한 번만 남는다. |

명시적 `skip_current` contract 추가와 Decision·Recommendation·Response Adapter 재학습은 이번 패치에서 제외한다. 새 contract를 학습하지 않은 현재 Decision 모델은 해당 field를 안정적으로 생성할 수 없기 때문이다. 재학습 전 임시 안전 정책은 **선택도 execution pending도 없는 optional section의 bare commit을 실행하지 않고 현재 section에 머무르는 것**이다. 이 정책에서는 명시적 자연어 skip도 새 contract가 도입되기 전까지 다시 물을 수 있지만, 의도하지 않은 재료 단계 생략과 뚜껑 실행을 우선 차단한다.

Cause: 로그 백업과 수정 코드를 한 커밋에 넣으면 수정 전 재현 증거와 수정 결과의 경계가 흐려진다.

Effect: GitHub에 원본 실패 세션을 먼저 고정한 뒤, runtime 패치의 효과를 같은 로그 기반 회귀 테스트로 독립 검증할 수 있다.

### 23.9 비재학습 runtime 패치 적용 결과

수정 전 증거는 다음 커밋으로 먼저 GitHub에 보존했다.

```text
commit: 5483fd2
message: 로그 백업 및 터지는 부분 수정 예정
remote: origin/flexible_llm
```

그 뒤 재학습 없이 다음 runtime 패치를 적용했다.

| 적용 위치 | 적용 결과 |
|---|---|
| `llm_langgraph.py`, `_resolve_pending()` | execution pending 뒤 `commit=true`이고 confirmation이 누락되어 `none`이어도 실행 수락으로 소비한다. 명시적 reject는 유지하며, 다른 section의 오래된 확인은 실행하지 않는다. |
| `llm_langgraph.py`, commit policy | unresolved pending과 빈 optional section의 bare commit을 `execute=false`로 차단한다. |
| `llm_langgraph.py`, `generate_response()` | `execute=true`이면 Response 모델을 호출하지 않고 현재 section 작업 목록으로 시작 문장을 만든다. |
| `llm_langgraph_node.py`, `_process_turn()` | 실제 task 시작 성공 뒤 같은 deterministic 문장을 발행하고 history도 교체한다. Graph 오류가 있어 pending이 남으면 node가 한 번 더 실행을 차단한다. |
| `llm_langgraph_node.py`, `_advance_after_section()` | veggie·meat·extra·lid·sauce 전부 기존 `next_reply`를 직접 사용하고 전환 Response 모델 호출을 제거했다. |

현재 동작은 다음처럼 달라졌다.

```text
변경 + 바로 실행
→ pending 소비
→ 실제 current section task 계산
→ "양파 적게, 버섯 많이 담기를 시작해요."
→ Response 모델 미호출

야채 완료
→ history 내용과 무관하게 육류 안내 직접 발행

빈 meat에서 "어 진행해"
→ meat 유지
→ 육류 선택 안내 다시 발행
→ extra로 자동 skip하지 않음
```

재학습 전 임시 제한도 있다. 선택과 pending이 모두 없는 optional section에서는 bare commit을 막으므로, Decision이 명시적 skip을 별도 구조로 표현하지 못하면 `다음 단계로 넘어가`도 현재 section 안내를 다시 받을 수 있다. 기존 선택 재료를 `none`으로 명시적으로 제거하면서 commit한 경우는 current-section 변경 사실이 있으므로 정상 skip을 허용한다.

검증 결과는 다음과 같다.

```text
section/node regression: 14 passed
natural multiturn contract: 59 passed
전체 src/soomac_irc/test: 96 passed
py_compile: PASS
git diff --check: PASS
```

Cause: 안전에 직접 영향을 주는 실행·전환 문장을 생성 모델에서 제거하고, pending과 section의 일치 여부를 실행 직전에 다시 검사했다.

Effect: 오늘 발생한 `거짓 실행 확인 → 후속 진행 답변 → 다음 section skip` 경로가 코드와 회귀 테스트 양쪽에서 차단된다.

---

## 24. 2026-10-11 최신 세션: 취향은 기억했지만 추천 질문에 같은 답을 반복한 문제

참고 로그: `logs/llm/2026-10-11/session_20261011_135637_747313_pid87462/agent_turns.jsonl`

→ 추가 재료 단계에서 `매운 것 못 먹음` 취향은 structured state에 정상 보존됐지만, Decision·Recommendation·Response가 각각 다른 방식으로 실패하면서 같은 과거 요약이 세 번 반복되었다.

### 24.1 실제 대화와 관측 결과

```text
사용자: 나 매운 거 못 먹어.
로봇:   매운 것을 못 드시는 취향을 저장했어요.

사용자: 뭘뭘 하든.
로봇:   지금까지 얇은면과 양파, 버섯, 소시지, 게살을 담았고
        매운 것을 못 드시는 취향을 저장했어요.

사용자: 그럼 토핑 뭐 담아?
로봇:   지금까지 얇은면과 양파, 버섯, 소시지, 게살을 담았고
        매운 것을 못 드시는 취향을 저장했어요.

사용자: 매운 거 못 먹는데 토핑 뭐 넣을까?
로봇:   지금까지 얇은면과 양파, 버섯, 소시지, 게살을 담았고
        매운 것을 못 드시는 취향을 저장했어요.
```

문제가 발생한 시점에 실제로 전달된 state는 다음과 같았다.

```json
{
  "preferences": [{"value": "매운 것 못 먹음"}],
  "robot_state": {"section": "extra"}
}
```

도메인 정의상 `extra` 단계에서 선택 가능한 항목은 `치즈`, `페퍼론치노`이고, 그 뒤에는 자동 작업인 `lid`와 아직 실행되지 않은 `sauce`가 있다. 현재 입력에는 현재 단계부터 미래 단계까지의 선택 가능 범위를 나타내는 별도 field가 없고, 전체 topping 목록과 `section=extra`만 따로 전달되었다.

`preferences`는 Decision, Recommendation, Response 입력에 모두 전달되었다. 따라서 이 현상은 **기억이 삭제된 문제가 아니다. 저장된 취향을 현재 단계의 추천 판단에 사용하지 못한 의미 연결 실패**이다.

Cause: 세 모델 모두 취향 state를 입력으로 받았지만, `매운 것 못 먹음 → extra 단계 → 페퍼론치노 제외 → 치즈 추천` 연결을 안정적으로 수행하지 못했다.

Effect: 사용자 입장에서는 앞에서 말한 내용을 잊은 것처럼 보였고, 같은 과거 요약만 반복됐다.

### 24.2 `그럼 토핑 뭐 담아?`에서 발생한 Decision 실패

실제 Decision 출력은 다음과 같았다.

```json
{"route":"task"}
```

이 출력에는 `recommendation={"action":"request"}`가 없었다. Graph는 이를 추천 질문이 아닌 변경 없는 task로 받아 `policy.reason=state_update_only`를 만들었다. Response는 명확한 다음 행동이 없는 상태에서 과거 주문을 요약했다.

정상 Decision은 다음이어야 한다.

```json
{"route":"task","recommendation":{"action":"request"}}
```

Cause: 현재 Decision 학습과 프롬프트가 `추천해 줘` 같은 직접 표현보다 `뭐 담아?`, `뭐 넣을까?` 같은 간접 추천 질문을 안정적으로 분류하지 못한다.

Effect: Recommendation Agent 자체가 호출되지 않아 현재 단계에 맞는 제안을 만들 기회가 사라졌다.

### 24.3 `매운 거 못 먹는데 토핑 뭐 넣을까?`의 의미

이 턴에서는 Decision이 추천 요청을 정상 인식했다. 문제는 그 다음 Recommendation 출력이었다.

```json
{
  "proposal": {
    "sauce": "크림",
    "noodle_type": "얇은면",
    "noodle_portion": "low",
    "toppings": {
      "버섯": "low",
      "양파": "low",
      "소시지": "low",
      "게살": "low"
    }
  },
  "reason_tags": ["매운맛 배제", "기존 선택 유지"]
}
```

당시 로봇은 `extra` 단계였으므로 새로 선택할 수 있는 것은 `치즈`, `페퍼론치노`뿐이었다. 그러나 모델은 이미 지나간 단계에서 담은 재료를 다시 추천했고, 정작 현재 가능한 치즈는 출력하지 않았다.

Python validator는 물리적으로 다시 선택할 수 없는 필드를 모두 거절했다.

```text
Recommendation의 accepted_fields = []
policy.reason = recommendation_unavailable
pending = null
```

여기서 `recommendation_unavailable`은 **추천 기능이 없다는 뜻이 아니다. 모델이 낸 후보 중 지금 적용 가능한 후보가 하나도 남지 않았다는 뜻**이다. 마지막으로 Response 프롬프트에 이 policy 전용 처리 규칙이 없어, 모델은 다시 과거 상태 요약을 출력했다.

Cause: Recommendation 입력에는 전체 지원 메뉴와 `section=extra`가 들어갔지만, 현재 단계 선택지만 별도 필드로 강제하지 않아 모델이 이미 완료한 재료를 재추천했다. Response에는 `recommendation_unavailable` 전용 안내 규칙도 없었다.

Effect: 안전 validator는 잘못된 재추천을 막았지만, 사용자에게는 실패 이유나 가능한 선택지를 알려주지 못했다.

### 24.4 프롬프트 즉시 보강안

프롬프트 보강은 재학습 전 오답률을 낮추는 임시 조치로 적용할 가치가 있다. 다음 규칙을 각 Agent prompt에 추가한다.

Decision prompt 추가안:

```text
- 특정 문구만 보고 추천 요청으로 고정하지 않는다. current message, recent_history,
  robot_state.section, 직전 assistant 질문을 함께 읽고 질문의 목적을 판단한다.
- "뭐 담을 수 있어?", "여기서 고를 수 있는 게 뭐야?"는 현재 선택지 조회이다.
  추천 요청으로 만들지 않고 현재 section의 가능한 항목을 설명한다.
- "뭐가 좋아?", "뭘 넣는 게 나아?", "매운 거 못 먹는데 뭐 넣을까?"는
  선택 조언을 요구하므로 recommendation={"action":"request"}로 출력한다.
- "지금 뭐 담고 있어?", "아까 뭐 담았어?"는 실행/완료 상태 조회이다.
  recommendation을 만들지 않고 robot_state와 action history를 조회한다.
- "뭐 담아?"처럼 문장만으로 여러 의미가 가능하면 직전 대화로 해석한다.
  직전 대화까지 읽어도 선택지 조회·추천·진행 상태 중 하나로 정할 수 없을 때만 clarify=true를 출력한다.
```

Recommendation prompt 추가안:

```text
- 현재 section부터 아직 실행되지 않은 미래 section까지를 추천 가능 범위로 본다.
- 이미 완료했거나 지나간 section은 추천 범위에서 제외한다.
- 자동 실행 section인 lid는 사용자 선택 후보로 만들지 않는다.
- 제공된 recommendation_scope 안에서 current_user_text와 recent_history를 읽고
  어떤 section과 항목을 제안할지는 모델이 판단한다.
- 사용자가 현재 단계의 토핑만 물었다면 미래 소스까지 불필요하게 추천하지 않는다.
- 이미 선택된 값을 그대로 proposal에 반복하지 않는다. 다른 값으로 바꾸는 추천은
  해당 section이 아직 실행 전이고 사용자가 변경·대안을 요구한 경우에만 만든다.
- 현재 section이 extra이고 사용자가 매운 것을 못 먹는다고 하면
  페퍼론치노를 추천하지 않고 치즈를 우선 검토한다.
- 적용 가능한 새 후보가 있으면 최소 한 개를 proposal에 포함한다.
```

Response prompt 추가안:

```text
- policy.reason=recommendation_unavailable이면 과거 주문 요약을 반복하지 않는다.
- 현재 단계에서 추천 가능한 후보를 만들지 못했다고 알리고,
  현재 선택 가능한 항목을 다시 안내한다.
```

Recommendation이 현재 단계와 취향을 올바르게 해석했을 때의 기대 문장은 다음과 같다.

```text
지금은 추가 재료 단계예요. 매운 것을 못 드시면 페퍼론치노 대신 치즈를 추천해요. 치즈를 담을까요?
```

여기서 필요한 값은 `current_section_options`보다 범위가 넓은 `recommendation_scope`이다. 자연어 의미를 대신 판단하는 keyword 규칙이 아니라, runtime이 현재 단계부터 미래의 사용자 선택 단계까지 물리적으로 변경 가능한 범위를 계산해 모델 입력에 넣는 structured fact이다.

```json
{
  "current_section": "extra",
  "recommendation_scope": [
    {
      "section": "extra",
      "options": ["치즈", "페퍼론치노"],
      "selected": []
    },
    {
      "section": "sauce",
      "options": ["오일", "토마토", "크림"],
      "selected": ["크림"]
    }
  ],
  "excluded_past_sections": ["noodle", "veggie", "meat"]
}
```

`lid`는 자동 실행 단계이므로 선택 후보에서 빠진다. 이 값의 역할은 질문 분류가 끝난 뒤 모델이 검토할 수 있는 물리적 범위를 제한하는 것이다. 어느 section을 실제 답에 사용할지는 모델이 사용자 질문으로 판단한다.

```text
extra에서 "토핑 뭘 넣는 게 좋아?" → extra 후보만 검토
extra에서 "앞으로 뭐 고를 수 있어?" → extra와 아직 실행 전인 sauce 안내
extra에서 "소스 바꾸는 게 좋을까?" → 미래 sauce 후보 검토
extra에서 이미 끝난 "버섯 바꿀까?" → past section이므로 추천하지 않음
```

`recommendation_unavailable` 고정 멘트는 정상 추천을 대신하는 하드코딩이 아니다. Recommendation이 또 지나간 재료만 출력하여 validator가 후보를 전부 거절했을 때, Response 모델이 과거 답을 반복하지 않도록 사용하는 마지막 실패 안내이다. 이때 취향을 임의 해석해 특정 재료를 확정하지 않고 현재 선택지만 안내한다.

```text
추천 후보를 만들지 못했어요. 지금 고를 수 있는 재료는 치즈와 페퍼론치노예요. 이후 소스도 바꿀 수 있어요.
```

현재 `validate_recommendation_proposal()`은 사용자가 이미 확정한 값이면, 아직 실행되지 않은 미래 section이어도 `already_selected`로 거절한다. 따라서 scope만 현재 이후로 넓혀도 `크림 → 오일` 같은 미래 소스 변경 추천은 막힌다. 추천은 pending 후보로만 저장되고 사용자가 수락하기 전에는 주문을 바꾸지 않으므로, 향후에는 **동일 값을 반복한 no-op은 거절하되 아직 실행 전인 미래 값의 대안 제안은 허용**하는 방향으로 validator를 조정할 수 있다.

### 24.5 재학습 필요 판단

**재학습이 필요하다.** 이번 문제는 단순히 예외 문장 하나가 빠진 것만이 아니라 자연어 의미 분류와 단계 기반 추천이 함께 실패한 사례이다. 프롬프트만 추가하면 같은 표현에는 나아질 수 있지만, 표현이 바뀌면 다시 실패할 가능성이 있다.

재학습 데이터에는 다음 세 failure family를 보강한다.

| Agent | 보강할 의미 | 정답 예시 |
|---|---|---|
| Decision | 선택지 조회·추천·진행 상태 구분 | `뭘 넣는 게 좋아?` → recommendation request, `뭐 담을 수 있어?` → 선택지 조회, `지금 뭐 담고 있어?` → 진행 조회 |
| Recommendation | 현재 section과 취향을 함께 반영 | extra + 매운맛 회피 → 치즈 제안, 페퍼론치노 제외 |
| Response | 추천 실패 policy 설명 | `recommendation_unavailable` → 반복 요약 금지, 현재 선택지 안내 |

현재 Recommendation Agent는 `recommendation_model.py`에서 Decision Adapter를 비활성화하고 base model로 추론한다. 따라서 **기존 Decision/Response Adapter만 다시 학습해도 Recommendation 오답은 직접 고쳐지지 않는다.** Recommendation까지 학습으로 해결하려면 전용 학습 데이터와 Adapter를 새로 연결해야 한다. 그 전에는 현재 이후의 `recommendation_scope` 명시, prompt 보강, validator, deterministic fallback을 함께 사용한다.

실제 평가 문장 자체를 그대로 train에 복사하지 않고, 의미가 같은 다양한 표현으로 targeted supplement를 만든다. 현재 로그 원문은 frozen regression으로 남겨 재학습 뒤 동일 실패가 사라졌는지 확인한다.

프롬프트와 재학습만으로 안전을 맡기지는 않는다. runtime에도 현재 이후의 `recommendation_scope`를 Recommendation 입력으로 명시하고, `recommendation_unavailable`에는 deterministic fallback을 둔다.

```text
1차: Decision/Recommendation이 자연스럽게 의미를 해석한다.
2차: validator가 지나간 단계의 후보를 거절한다.
3차: 후보가 모두 거절돼도 deterministic 문장으로 현재 선택지를 안내한다.
```

Cause: 의미 다양성은 모델 학습 대상이고, 현재 물리 단계와 실패 시 사용자 안내는 deterministic runtime 책임이다.

Effect: 재학습 후에는 다양한 추천 질문을 이해하고, 모델이 다시 틀리더라도 같은 과거 답변을 반복하지 않게 된다.

### 24.6 미래 section 추천과 진입 시 재확인

사용자가 현재보다 뒤 section의 추천안을 수락하면 추천 후보는 canonical order에 미리 저장된다. 이후 로봇이 해당 section에 도착했을 때의 동작은 section별로 다르다.

| 도착 section | 현재 동작 | 재확인 여부 |
|---|---|---|
| `veggie` | 미리 선택한 양파·버섯을 execution pending으로 다시 만든다. | 확인 후 실행 |
| `meat` | 미리 선택한 소시지·게살을 execution pending으로 다시 만든다. | 확인 후 실행 |
| `extra` | 미리 선택한 치즈·페퍼론치노를 execution pending으로 다시 만든다. | 확인 후 실행 |
| `lid` | 사용자 선택 항목이 아닌 자동 작업이다. | 즉시 실행 |
| `sauce` | 주문 초기에 선택된 소스를 마지막 단계에서 자동 시작한다. | 재확인 없음 |

현재 deterministic 확인 문장은 다음 형식이다.

```text
치즈이(가) 미리 선택되어 있어요. 이대로 진행할까요?
```

따라서 “미래 값을 미리 추천·확정하되, 실제 section에 왔을 때 다시 알려주고 실행 확인을 받는다”는 흐름은 `veggie`, `meat`, `extra`에 이미 구현되어 있다. 소스도 같은 정책을 적용하려면 `build_preselected_section_confirmation()`의 대상과 `_advance_after_section()`의 sauce 자동 시작 경로를 조정해야 한다.

`already_selected`는 로봇이 이미 재료를 물리적으로 담았다는 뜻이 아니다. canonical order에 값이 이미 들어 있다는 뜻이다. 현재 Recommendation validator는 두 경우를 구분하지 않는다.

```text
현재 extra, sauce는 아직 실행 전, order.sauce="크림"

Recommendation이 "크림" 제안
→ 기존 값과 동일한 no-op
→ already_selected 거절이 맞음

Recommendation이 "오일" 제안
→ 아직 실행 전인 미래 값의 대안
→ 현재 코드도 order.sauce가 이미 채워졌다는 이유로 already_selected 거절
```

두 번째 경우는 과도한 차단이다. 추천은 즉시 반영되지 않고 recommendation pending으로 남으며, 사용자가 수락해야 canonical order가 바뀐다. 따라서 다음 정책이 적절하다.

```text
기존 값과 동일한 추천                 → no-op 거절
실행 전 current/future 값의 다른 대안 → recommendation pending 허용
실행 중·완료·과거 값의 변경 추천      → 물리 보호로 거절
```

이 흐름을 runtime에서 계산하고 재확인하는 부분은 재학습이 필요하지 않다. 반면 모델이 `recommendation_scope`를 읽고 사용자 질문에 맞는 현재·미래 section만 골라 자연스럽게 추천하려면 Recommendation 전용 학습이 필요하다. 기존 Decision/Response Adapter 재학습만으로는 base model로 실행되는 Recommendation Agent가 바뀌지 않는다.

Cause: 기존 코드는 미래 토핑의 section 진입 재확인은 구현했지만, Recommendation validator가 “동일 값 반복”과 “아직 실행 전인 값의 대안”을 모두 `already_selected`로 묶었다.

Effect: validator를 구분해서 고치면 미래 추천을 안전하게 pending으로 보관하고, 실제 section 도착 시 한 번 더 사용자 확인을 받은 뒤 실행할 수 있다.

### 24.7 적용 예정 패치

이번 패치는 자연어 판단을 Python keyword 규칙으로 옮기지 않는다. runtime은 물리적으로 가능한 범위와 확정 상태만 만들고, 사용자가 어느 범위를 묻는지와 어떤 항목이 적절한지는 모델이 판단한다.

#### 24.7.1 현재 이후 추천 범위 생성

`llm_policy.py`에 현재 section부터 미래의 사용자 선택 section까지를 계산하는 helper를 추가한다. 지나간 section과 실행 중·완료된 항목은 제외하고, 자동 작업인 `lid`도 추천 후보에서 제외한다.

```json
{
  "current_section": "extra",
  "recommendation_scope": [
    {
      "section": "extra",
      "options": ["치즈", "페퍼론치노"],
      "selected": []
    },
    {
      "section": "sauce",
      "options": ["오일", "토마토", "크림"],
      "selected": ["크림"]
    }
  ],
  "excluded_past_sections": ["noodle", "veggie", "meat"]
}
```

이 구조를 `recommendation_model.py`와 `response_model.py` 입력에 전달한다. Recommendation은 추천 범위로 사용하고, Response는 선택지 조회와 추천 실패 안내에 사용한다.

#### 24.7.2 Agent prompt 보강

`agent_prompts.py`를 다음 책임으로 보강한다.

```text
Decision:
- 선택지 조회, 추천 요청, 진행 상태 조회를 문구 하나가 아니라 문맥으로 구분한다.
- 끝까지 모호한 "뭐 담아?"만 clarify로 보낸다.

Recommendation:
- recommendation_scope 밖의 과거 항목은 제안하지 않는다.
- scope 전체를 무조건 채우지 않고 사용자 질문과 취향에 필요한 항목만 고른다.
- 현재 토핑 질문에는 미래 소스를 불필요하게 끼워 넣지 않는다.

Response:
- 선택지 조회에는 scope를 설명한다.
- recommendation_unavailable이면 과거 요약을 반복하지 않는다.
```

프롬프트는 오답률을 낮추는 장치이며 물리 보호를 대신하지 않는다.

#### 24.7.3 Recommendation validator 수정

`llm_policy.py`의 `validate_recommendation_proposal()`에서 `already_selected`를 다음처럼 나눈다.

```text
기존 값과 제안 값이 같음
→ no-op으로 거절

기존 값과 다르고 해당 section이 아직 실행 전임
→ recommendation pending 후보로 허용

해당 field가 과거·실행 중·완료 상태임
→ 기존 physical protection으로 거절
```

추천 후보는 즉시 주문을 덮어쓰지 않는다. 사용자가 recommendation pending을 수락한 뒤에만 기존 검증 경로를 다시 통과해 canonical order에 반영한다.

#### 24.7.4 미래 선택의 section 진입 재확인

`veggie`, `meat`, `extra`는 기존 `build_preselected_section_confirmation()` 경로를 유지한다. 미래 추천을 수락해 주문에 저장했더라도 해당 section에 도착하면 execution pending으로 바꿔 다시 확인한다.

```text
미래 추천 수락
→ canonical order에 미리 저장
→ 해당 section 도착
→ "미리 고른 치즈를 담을까요?"
→ accept 뒤 실제 로봇 실행
```

기존 `치즈이(가) 미리 선택되어 있어요` 문장은 조사 문제가 없도록 item 목록 기반 문장으로 교체한다.

소스는 현재 section 도착 즉시 자동 실행된다. 일관된 확인 정책을 적용하기 위해 `build_preselected_section_confirmation()`이 sauce도 만들 수 있게 확장하고, `_advance_after_section()`의 sauce 자동 시작을 execution pending 생성으로 바꾼다.

```text
현재: sauce 도착 → 즉시 실행
변경: sauce 도착 → "미리 고른 크림 소스를 올릴까요?" → accept → 실행
```

이 방식은 선택 출처가 추천인지 직접 선택인지와 무관하게, 실제 실행 직전에 최종 확인을 한 번 받는다. 별도의 추천 출처 추적 state를 추가하지 않아 canonical state를 단순하게 유지한다.

#### 24.7.5 deterministic fallback과 회귀 테스트

`llm_langgraph.py`는 `recommendation_unavailable`일 때 Response 모델의 자유 생성을 사용하지 않고 scope 기반 실패 안내를 만든다. 특정 재료를 임의로 추천하지 않고 현재 이후의 가능한 범위를 안내한다.

```text
추천 가능한 후보를 만들지 못했어요. 현재 단계부터 선택 가능한 항목을 다시 알려드릴게요.
```

추가할 회귀 테스트는 다음과 같다.

| 테스트 | 기대 결과 |
|---|---|
| extra의 recommendation scope | 과거 noodle·veggie·meat 제외, extra·sauce 포함, lid 제외 |
| 기존 크림에 크림 추천 | no-op 거절 |
| 실행 전 크림에 오일 대안 추천 | recommendation pending 허용 |
| 완료된 버섯 변경 추천 | physical protection 거절 |
| 미래 치즈를 수락하고 extra 진입 | 치즈 실행 전 재확인 |
| sauce 진입 | 소스 실행 전 재확인 |
| 추천 후보 전부 거절 | 과거 답 반복 없이 deterministic fallback |

검증 순서는 targeted unit test, 전체 `src/soomac_irc/test`, `py_compile`, `git diff --check`이다.

#### 24.7.6 재학습 범위

runtime 패치는 학습 없이 적용할 수 있다. 다만 모델이 새 `recommendation_scope`를 다양한 사용자 표현과 함께 안정적으로 사용하는 것은 별도 학습 대상이다.

```text
Decision Adapter 재학습:
선택지 조회 / 추천 요청 / 진행 상태 조회 대조 데이터

Response Adapter 재학습:
추천 성공 / 추천 실패 / 선택지 안내 대조 데이터

Recommendation 전용 Adapter 신규 학습:
현재~미래 scope 중 질문에 필요한 section 선택
취향·제한을 후보에 반영
과거 항목 재추천 금지
```

기존 학습 pipeline은 Decision과 Response Adapter만 다룬다. Recommendation은 base model로 실행되므로 Recommendation 문제를 학습으로 해결하려면 전용 dataset, Adapter, runtime activation 경로가 추가로 필요하다.

Cause: runtime 변경은 가능한 범위와 실행 확인을 결정하지만, scope 안에서 사용자의 의도를 일반화하는 일은 모델 학습의 책임이다.

Effect: 먼저 runtime 패치로 과거 항목 재추천과 무응답 반복을 차단하고, 후속 재학습으로 표현이 달라져도 올바른 현재·미래 추천을 만들 수 있다.

---

## 용어 정리

### Sparse Decision

현재 발화와 관련된 field만 포함하는 외부 JSON이다. `route`는 항상 있고 나머지는 필요할 때만 존재한다.

### Normalized Decision

Graph가 매번 같은 key를 읽을 수 있도록 기본값을 채운 내부 Decision이다.

### Grounding

자연어 표현을 현재 history, order, robot state에 근거해 실제 조작 가능한 field 의미로 연결하는 과정이다. 단순 keyword 치환과 다르다.

### Canonical state

runtime이 최종 사실로 보관하는 표준 상태이다. 삭제 의미인 문자열 `"none"`을 저장하지 않고 scalar는 `None`, topping은 key 부재로 표현한다.

### Structural contradiction

각 field의 값은 schema에 맞지만 한 Decision 안에서 동시에 참일 수 없는 조합이다. 자연어 의미나 domain 지원 여부와는 다르다.

### One-shot repair

구조적 자기모순이 있을 때 모델을 정확히 한 번만 다시 호출하는 절차이다. 두 번째도 실패하면 safe clarify로 끝낸다.

### Final validator

지원 메뉴, restriction, 물리 완료 상태, 실행 가능성을 최종 판단하는 deterministic Python 경로이다.

### Partial success

한 Decision의 일부 field가 실패해도 정상 field는 반영하는 정책이다. 다만 issue가 있으면 같은 턴의 실행은 허용하지 않는다.

### Pending

아직 사용자가 확정하지 않은 실행 확인 또는 추천 후보이다. canonical order와 구분한다.

### Physical protection

이미 완료했거나 현재 실행 중이거나 지나간 section의 값을 소프트웨어 state에서 거짓으로 되돌리지 않는 규칙이다.

### Adapter

하나의 base model에 붙여 특정 역할을 학습시키는 작은 추가 가중치이다. 현재 Decision과 Response는 같은 Gemma 4 12B를 쓰지만 서로 다른 Adapter를 활성화한다.

### QLoRA

양자화된 base model에 LoRA를 붙여 GPU memory 사용량을 줄이는 미세조정 방식이다. 이번 학습은 NF4 4-bit base와 rank 16 LoRA를 사용했다.

### Checkpoint

학습을 중단한 지점부터 재개하기 위한 저장본이다. Adapter 외에도 optimizer, scheduler, Trainer state, RNG state가 있어야 동일 step에서 안전하게 이어갈 수 있다.

### Frozen split

학습 데이터에 넣지 않고 모델 세대 간 성능 비교를 위해 고정한 validation/test/challenge/holdout 자료이다.

### Root-cause clean turn

이전 턴 오류의 누적 영향은 제외하고 현재 턴에서 Decision, Recommendation, Response의 새 직접 오류가 생겼는지만 보는 지표이다.

### Downstream failure

현재 단계 자체의 잘못이 아니라 앞 단계의 오류가 입력 state에 남아서 뒤 단계 결과까지 틀어진 경우이다.

### Targeted supplement

전체 데이터를 다시 만들지 않고 실제 실패 family만 보강한 추가 학습 자료이다. 원본 base 뒤에 붙여 candidate train을 만든다.

### Contamination

평가용 frozen 문장이나 동일 입력이 학습 자료에 섞여 성능이 실제보다 높게 측정되는 문제이다.

### Contract version

로그가 어떤 runtime 계약에서 생성됐는지 나타내는 식별자이다. 현재 값은 `natural_multiturn_v3`이다.

---

## 최종 자체 점검

**현재 runtime source에는 Natural-language-first 외의 실질적인 Decision/state 계약이 남아 있지 않다.**

```text
자연어 의미 해석        → Decision Agent
구조적 자기모순        → one-shot repair
지원·안전·물리 사실    → final validator
추천                    → Recommendation Agent + final validator
사용자 답변             → confirmed structured facts
```

코드가 단순해진 이유는 검사를 없앴기 때문이 아니다. 같은 사실을 여러 중간 계층에서 반복 해석하던 구조를 없애고, 각 책임을 한 경로에 모았기 때문이다.

(내가 의역해보자면 이번 작업은 “모델을 더 믿자” 하나가 아니라, 의미 판단은 모델에 맡기고 로봇 안전과 실제 상태는 코드가 더 명확하게 책임지도록 경계를 다시 그은 작업이다.)
