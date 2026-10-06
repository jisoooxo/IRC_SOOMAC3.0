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

상태 조회처럼 새 mutation이 없는 턴은 기존 pending을 유지한다. 새로운 명시적 주문이나 추천 요청이 들어오면 이전 pending을 닫고 새 의미를 우선한다.

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
