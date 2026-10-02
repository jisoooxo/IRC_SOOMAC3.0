# IRC_SOOMAC3.0 — Decision Routing + Multi-turn Focus 구현 계획

> 기준 저장소: `/home/roma/IRC_SOOMAC3.0`  
> 기준 브랜치: `flexible_llm`  
> 기준 커밋: `5215b5102336a838916ecb0c71477fa0d377a345` (`판단 필드 추가 전 백업`)  
> 작성일: 2026-10-02  
> 목표: 기존 주문·안전·로봇 실행 의미를 보존하면서 `task / general / mixed` routing과 안전한 멀티턴 지시어 처리를 추가한다.

참고 문서:

- [기존 초안](/home/roma/Downloads/IRC_SOOMAC3_DECISION_ROUTING_FOCUS_PLAN.md)  
  → 최초 아키텍처·데이터셋·협업 방향을 정의한 문서이다.
- [현재 코드 지도](./IRC_SOOMAC3_CODE_MAP.md)  
  → 현재 파일 역할, 상태 소유자, ROS 연결을 실제 코드 기준으로 정리한 문서이다.

**핵심 결론: 현재 구조를 폐기하거나 다시 설계할 필요는 없다. 기존 task pipeline 앞에 route/focus 검증을 추가하고, 응답과 턴 마감을 분리하면 구현할 수 있다.**

---

## 0. 확정된 범위

### 0.1 이번 작업에서 한다

1. Decision 출력에 required `route`와 optional `mentions`를 추가한다.
2. 주문 선택 대화 구간에서 `general`, `task`, `mixed`를 분기한다.
3. `그거`, `방금 그거`, `아까 그거`, `둘 다`, `첫 번째 거`, `두 번째 거`를 `dialogue_focus`와 Python resolver로 처리한다.
4. unsupported 대상과 모호한 지시어가 주문·로봇 실행으로 이어지지 않게 한다.
5. task/general/mixed가 같은 history 저장 경로를 사용하게 한다.

### 0.2 이번 작업에서 하지 않는다

1. 로봇 동작 중 자유대화는 지원하지 않는다. 기존 `robot_busy` 차단과 STT gate를 유지한다.
2. 기존 Decision Adapter는 연결하지 않는다. 새 데이터로 재학습한 뒤 연결한다.
3. 로봇 제어 계약과 MAIN 코드는 수정하지 않는다. 실제 제어 정본은 메인 브랜치에서 확인한다.
4. legacy `test_runtime.py`는 현재 구조로 이식하지 않는다. 새 LangGraph 기준 테스트를 별도로 만든다.
5. 별도 Router LLM, Response LoRA, vector DB memory, 장기 memory summarizer는 추가하지 않는다.

### 0.3 수정 금지 경계

**이번 작업은 LLM routing·focus 계층만 수정한다. 다음 영역은 절대 수정하지 않는다.**

| 보호 영역 | 수정 금지 대상 |
| --- | --- |
| STT | `stt_node.py`, `stt_nemotron_node.py`, `stt_hotword.py` |
| TTS | `tts_node.py` |
| VLM | `vlm.py`, `vlm_prompts.py`, `vlm_rag.py`, `vlm_reference_embedding.py`, VLM 관련 scripts |
| UI | `ui_node.py`, `static/` 아래 UI asset |
| HTML | `templates/index.html`과 다른 HTML 파일 |
| 로봇 제어 | `src/irc_control_pkg/` 전체 |

테스트나 full-cycle 확인 중 보호 영역에서 문제가 발견되어도 이번 branch에서 고치지 않는다. 위치·원인·영향만 기록하고 별도 작업으로 분리한다.

허용된 변경 대상은 다음 LLM 계층으로 제한한다.

```text
agent_contract.py
agent_prompts.py
decision_model.py
decision_overrides.py
dialogue_focus.py (신규)
llm_langgraph.py
llm_policy.py
response_model.py
llm_langgraph_node.py (필요한 최소 연결 변경만)
LLM routing/focus 전용 신규 테스트
```

Cause: 이번 목표는 Decision routing과 multi-turn focus이며 음성·화면·시각 판정·로봇 제어 변경이 아니다.  
Effect: 기존 STT/TTS/VLM/UI/HTML/로봇 동작을 그대로 둔 상태에서 LLM 계층 회귀만 분리해 검증할 수 있다.

Cause: 현재 목표는 주문 선택 중 자연대화와 지시어 안전성을 추가하는 것이다.  
Effect: STT·TTS·UI·VLM·로봇 제어 동시성 문제를 이번 변경에서 분리할 수 있다.

---

## 1. 역할 규약

```text
Decision
= 현재 발화의 route와 sparse task semantics 추출
+ 현재 발화에 직접 등장한 mention을 순서대로 추출

Python
= dialogue_focus를 사용한 reference resolution
+ route·supported 여부 검증
+ canonical state·restriction·물리 상태·실행 권한 관리

Response
= Python이 확정한 사실과 일반 질문을 자연어로 표현

ROS
= Python policy가 execute=True로 허용한 task만 실행
```

**Decision은 최소 의미만 출력하고, 지시어 해석은 `dialogue_focus`와 Python resolver가 담당한다.** resolver는 합의된 지시어 표현만 판별하며 자연어 전체를 다시 이해하는 별도 LLM이나 범용 semantic parser가 아니다.

(내가 의역해보자면, Decision은 `그거 넣어줘`가 task라는 최소 의미를 출력하고, Python이 `그거`의 후보를 focus에서 찾아 기존 주문 의미와 결합한다.)

---

## 2. 목표 아키텍처

```text
User utterance
      ↓
Decision Agent 1회
  route + mentions + existing sparse semantics
      ↓
Decision contract + Python reference resolver
  route consistency + focus candidate 해석·검증
      ↓
route_by_decision
  ├─ general ─→ no-op policy ─┐
  │                            │
  └─ task/mixed ─→ 기존 task pipeline ─┤
                                      ↓
                              generate_response
                                      ↓
                               finalize_turn
                  focus 갱신 + history/action 기록
                                      ↓
                                     END
```

별도 Router LLM을 호출하지 않는다. route는 기존 Decision inference의 JSON 출력에 포함한다.

---

## 3. Decision 외부 계약

### 3.1 `route`

`route`는 top-level required field이다.

```json
{"route":"general","mentions":["치즈"]}
```

허용값:

| 값 | 의미 | 예시 |
| --- | --- | --- |
| `task` | 주문·추천·확인·상태 조회·실행 요청 | `치즈 많이 넣어줘` |
| `general` | 주문 상태나 로봇 동작을 요구하지 않는 대화 | `치즈가 뭐야?` |
| `mixed` | task와 general이 한 발화에 함께 있음 | `치즈 빼고 치즈가 뭐야?` |

unsupported 메뉴라도 주문 의도이면 `task`이다.

```text
떡볶이가 뭐야? → general
떡볶이 줘      → task + clarify
```

### 3.2 `mentions`

`mentions`는 현재 사용자 발화에서 직접 언급한 대상을 등장 순서대로 담는 optional list이다.

```json
{"route":"general","mentions":["치즈","버섯"]}
```

규칙:

1. supported와 unsupported 대상을 모두 허용한다.
2. `그거`, `아까 그거` 자체는 mention으로 넣지 않는다.
3. Assistant가 혼자 말한 대상은 현재 mention으로 복사하지 않는다.
4. 명시적 alias만 허용하며 fuzzy·semantic nearest-neighbor 치환은 금지한다.
5. 배열 순서는 user 발화를 그대로 보존한다.

`mentions`를 포함해 이번에 새로 추가하는 Decision field는 `route`, `mentions` 두 개뿐이다. `reference`, `reference_type`, `reference_index`, `issue`, `supported`, `unsupported`, `confidence`, `intent_type`, `topic`, `section`, `current/future`, `general_query`, `mixed_query`, `ordinal`, `temporal_reference`, `target_source` 같은 별도 field는 추가하지 않는다.

새 edge case는 먼저 `route` + `mentions` + 기존 semantic fields + `dialogue_focus` + Python resolver/validation + 기존 canonical state의 조합으로 처리한다. 이 계약으로 의미를 표현할 수 없다는 실패 test가 생겨도 field를 바로 추가하지 않고, 이유와 대안을 설명하여 사용자 승인을 먼저 받는다.

### 3.3 내부 Decision 기본값

모델을 건너뛰는 deterministic override가 있으므로 internal Decision은 다음 안전 기본값을 사용한다.

```json
{"route":"task","mentions":[]}
```

`task` 기본값은 기존 confirmation·recommendation override가 general 경로로 잘못 빠지는 것을 막는다.

### 3.4 XGrammar

현재 `decision_model.py`의 `compile_json_schema(..., any_order=True)`를 유지한다.

XGrammar는 JSON syntax, required, enum, key 제한만 담당한다. route 의미, reference, supported 여부, 실제 state 변경은 Prompt·Dataset·Python이 담당한다.

---

## 4. `dialogue_focus` 계약

### 4.1 SessionState 구조

```json
{
  "dialogue_focus": {
    "current": {
      "mentions": ["치즈"],
      "history_turn": 11
    },
    "recent": []
  }
}
```

`dialogue_focus`는 실제 주문이나 별도 대화 이력이 아니다. 실제 주문 정본은 계속 `SessionState.order`이고, 전체 자연어 대화 정본은 `SessionState.history`이다.

### 4.2 멀티턴 번호 공유

focus 전용 `turn_index`를 만들지 않는다. 성공한 user/assistant 한 쌍을 기존 멀티턴의 한 턴으로 사용한다.

```python
completed_history_turn = len(session["history"]) // 2
current_history_turn = completed_history_turn + 1
```

`generate_response()`에서 현재 응답이 확정된 뒤 `current_history_turn`을 focus event에 기록하고, 그 다음 user/assistant 두 항목을 history에 추가한다. 다음 Decision 턴에는 `len(session["history"]) // 2`가 저장된 `history_turn`과 같아진다.

- 같은 사용자 발화를 Decision repair로 다시 호출해도 focus 턴은 증가하지 않는다.
- graph가 실패하면 history와 focus를 모두 확정하지 않는다.
- `action_history`의 로봇·VLM event는 대화 턴으로 계산하지 않는다.

Cause: focus가 독립 카운터를 가지면 Decision repair, graph 실패, 로봇 event 때문에 실제 대화 이력과 어긋날 수 있다.  
Effect: focus TTL과 `SessionState.history`가 같은 성공한 사용자 턴을 기준으로 계산된다.

### 4.3 history와 focus의 역할

```text
history        = 실제 user/assistant 자연어 대화 정본
dialogue_focus = 최근 user mentions와 해당 history 턴을 가리키는 작은 index
```

`dialogue_focus.py`는 Decision이 출력한 `mentions` 문자열과 순서를 그대로 저장하고, `current`·`recent`·TTL·문자열 제거만 담당한다.

`mentions`의 구조 제약은 `list[str]`뿐이다. 메뉴 enum, supported 여부, 주문 field, 물리 section, robot task mapping을 focus 안에 넣지 않는다. 따라서 `치즈` 같은 지원 메뉴뿐 아니라 `떡볶이`, `로봇` 같은 unsupported·일반대화 대상도 사용자 발화 그대로 들어갈 수 있다.

다음 작업은 담당하지 않는다.

1. alias 변환과 fuzzy matching
2. supported/unsupported 메뉴 판정
3. 메뉴를 주문 field나 물리 section으로 변환
4. 주문 state 수정과 로봇 실행 결정

Python reference resolver는 user utterance의 합의된 지시어와 focus를 읽어 후보 개수·등장 순서·시간 위치를 해석한다. supported 여부, 실제 주문 변경, 물리 실행 가능 여부는 기존 domain validation과 `llm_policy.py`가 계속 담당한다.

unsupported mention도 별도 구조로 바꾸지 않고 사용자가 말한 문자열로 저장한다.

```json
{"mentions":["떡볶이"],"history_turn":11}
```

Cause: focus가 메뉴 정규화와 주문 field 변환까지 수행하면 Decision 및 `llm_policy.py`와 역할이 겹친다.  
Effect: focus는 기억만 담당하고 의미 해석과 실행 검증은 기존 소유자에게 남는다.

### 4.4 Decision 입력

`SessionState`에 필드를 추가하는 것만으로는 모델 입력에 자동 포함되지 않는다. `decision_model.py::build_decision_inputs()`가 명시적으로 만드는 `model_input`에도 `dialogue_focus`를 추가한다.

### 4.5 focus 갱신 시점과 출처

`generate_response()`에서 policy와 최종 응답이 확정된 뒤, 현재 user utterance에서 Decision이 직접 추출한 `mentions`만 focus에 기록한다.

- task/general/mixed는 같은 focus 갱신 및 history 저장 경로를 사용한다.
- mention이 없는 턴은 current를 교체하지 않고 history 턴만 진행한다.
- Assistant 자유응답에만 등장한 대상은 focus에 넣지 않는다.
- 사용자가 선택하지 않은 추천 후보를 focus에 넣지 않는다.

### 4.6 초기 TTL

초기 상수는 다음으로 시작하고 hard eval 결과로만 조정한다.

```text
FOCUS_RECENT_LIMIT = 3
FOCUS_MAX_TURN_GAP = 3
```

- `현재 history 턴 - event.history_turn`이 3보다 크면 단독 지시어로 실행하지 않고 clarify한다.
- `recent`는 이전 focus event를 최신순으로 최대 3개 보관한다.
- TTL이 지나도 자연어 history를 삭제하지 않는다. 실행 referent로 신뢰하지 않을 뿐이다.

### 4.7 로봇 실행 시작 시 active task mention 제거

`llm_langgraph_node.py::_publish_next_task()`에서 queue의 task를 `active_task`로 확정하는 순간 `task["class"]`와 정확히 같은 mention만 `current`와 `recent`에서 제거한다. 다른 mention은 유지한다.

현재 runtime에서 `active_task`는 완료된 task가 아니라 **queue에서 꺼내 로봇에 전달한 현재 실행 중 task**이다.

```text
task_queue에서 꺼냄
→ active_task 할당
→ /llm/plan 발행
→ 로봇 작업
→ /llm/next 수신
→ active_task 해제 후 completed_tasks에 기록
```

focus 정리 기준은 `/llm/next`의 완료 시점이 아니라 `active_task` 할당 시점이다. 이는 runtime이 관측할 수 있는 실행 시작 경계이다.

```python
self.graph_state["dialogue_focus"] = remove_focus_mentions(
    self.graph_state["dialogue_focus"],
    [task["class"]],
)
```

focus helper는 active task나 메뉴 의미를 알지 않고 전달받은 문자열과 정확히 같은 mention만 제거한다. mention이 모두 제거된 event도 삭제하지 않고 `mentions=[]`로 남겨 `recent`가 더 오래된 event로 당겨지는 것을 막는다.

`active_task`가 이미 있거나 `task_queue`가 비어 있으면 `_publish_next_task()`가 먼저 `False`를 반환하므로 focus를 변경하지 않는다. 로봇 동작 중에는 기존 `robot_busy` gate가 사용자 발화를 무시하므로 focus도 새로 갱신하지 않는다.

Cause: focus 전체 초기화는 아직 실행하지 않은 다른 mention과 일반대화 대상까지 지운다.  
Effect: 실행을 시작한 task만 참조 대상에서 빠지고 나머지 focus와 event 시간 순서는 유지된다.

---

## 5. 지시어 처리 규약

### 5.1 current reference

```text
그거 / 이거 / 방금 그거 / 지금 말한 거
```

- `current.mentions`에 후보가 정확히 1개여야 한다.
- Python resolver가 그 후보를 기존 semantic field의 target으로 사용한다.
- 후보가 없거나 여러 개면 임의 선택하지 않고 `clarify=true`로 바꾼다.

### 5.2 plural·ordinal reference

```text
둘 다 / 두 개 다 / 첫 번째 거 / 두 번째 거
```

- `둘 다`, `두 개 다`: `current.mentions`가 정확히 2개일 때 두 대상을 등장 순서대로 선택한다.
- `첫 번째 거`: `current.mentions[0]`이 있을 때 첫 대상을 선택한다.
- `두 번째 거`: `current.mentions[1]`이 있을 때 두 번째 대상을 선택한다.
- 필요한 후보 개수나 순서가 없으면 임의 선택하지 않고 `clarify=true`로 바꾼다.

### 5.3 temporal reference

```text
아까 그거 / 전에 말한 거
```

`current` 바로 이전의 `recent[0]` focus event만 대상으로 삼는다. 그 event의 `mentions`가 정확히 1개일 때만 허용한다.

```text
U1: 치즈가 뭐야?   → current=치즈
U2: 버섯은?        → current=버섯, recent[0]=치즈
U3: 아까 그거 넣어줘 → 치즈
```

여러 event를 건너뛰며 임의로 가장 그럴듯한 대상을 찾지 않는다.

### 5.4 unsupported reference

```text
U1: 떡볶이가 뭐야?
U2: 그거 줘
```

결과:

```text
route=task
clarify=true
state mutation 없음
robot execution 없음
지원하지 않는 메뉴임을 알리고 지원 메뉴를 다시 질문
```

### 5.5 reference field는 추가하지 않음

별도의 `reference` JSON field를 만들지 않는다. Python resolver가 user utterance의 지시어와 `dialogue_focus`를 해석하고 기존 semantic fields 및 canonical state와 결합한다.

Cause: 기존 schema와 focus로 task semantics 및 reference 후보를 표현할 수 있다.
Effect: Decision schema를 늘리지 않고 plural·ordinal·temporal reference까지 처리할 수 있다.

---

## 6. route consistency guard

기존 `validate_and_repair` 단계에 route 계약 검사를 합치거나 바로 다음 단계로 분리한다. repair는 기존처럼 최대 1회만 허용한다.

| 모순 | 처리 |
| --- | --- |
| `general` + order/restriction/commit 등 mutation semantic | Decision repair 1회 |
| `mixed`인데 task semantic 없음 | repair 1회 |
| `task`인데 query·order·recommendation·confirmation·commit·clarify가 전부 없음 | repair 후에도 같으면 clarify |
| reference target이 focus와 불일치 | Python이 clarify로 변경 |
| unsupported reference가 supported order로 치환됨 | mutation 제거 후 clarify |

repair 뒤에도 모순이 남으면 안전한 internal Decision으로 교체한다.

```text
route=task
understanding=clarify
mutation 없음
commit=false
```

---

## 7. LangGraph 변경

### 7.1 목표 graph

```text
START
  ↓
interpret_decision
  ↓
validate_decision_contract
  ↓
validate_reference
      ↓
route_by_decision
  ├─ general
  │    ↓
  │  build_general_noop_policy
  │    ↓
  │  generate_response
  │
  └─ task / mixed
       ↓
     resolve_confirmation
       ↓
     apply_workers
       ↓
     run_recommendation
       ↓
     check_policy
       ↓
     generate_response
       ↓
finalize_turn
  ↓
END
```

### 7.2 general no-op policy

현재 ROS Node는 모든 graph 결과에 `policy.execute`가 있다고 가정한다. general 경로도 동일한 반환 계약을 유지한다.

```json
{"status":"pass","reason":"general","execute":false}
```

general 경로는 다음 값을 변경할 수 없다.

- `session.order`
- `session.preferences`
- `session.recommendation`
- `session.pending_confirmation`
- robot state

`dialogue_focus`와 자연어 `history`만 갱신할 수 있다.

### 7.3 mixed 경로

mixed 전용 worker는 만들지 않는다.

```text
기존 task pipeline으로 task 부분 확정
→ mixed response에서 task 결과 설명
→ 같은 발화의 일반 질문에도 답변
```

Response가 Python이 확정하지 않은 주문 사실을 추가하면 안 된다.

### 7.4 `finalize_turn`

현재 `generate_response()`에 들어 있는 공통 턴 마감 로직을 분리한다.

`finalize_turn` 역할:

1. 최종 Decision의 `mentions`를 현재 history 턴 번호와 함께 `dialogue_focus`에 기록한다.
2. 실제 변경이 있으면 `action_history`에 기록한다.
3. user 발화를 `session.history`에 기록한다.
4. assistant reply를 `session.history`에 기록한다.
5. 최종 `session`, `reply`, `policy`를 반환한다.

task/general/mixed가 모두 이 노드를 거쳐야 한다.

---

## 8. Response 변경

Response model wrapper는 하나를 유지하고 `route`에 따라 system prompt를 선택한다.

```text
TASK_RESPONSE_SYSTEM
GENERAL_RESPONSE_SYSTEM
MIXED_RESPONSE_SYSTEM
```

### Task

- 기존 응답 규칙과 Python 확정 사실을 유지한다.
- 주문·추천·확인·실행 결과를 새로 만들어내지 않는다.

### General

- 자연스러운 자유대화를 제공한다.
- 주문·추천·로봇 state를 변경했다고 주장하지 않는다.
- 주문 선택 구간이라는 현재 시스템 문맥을 벗어난 실행 약속을 하지 않는다.

### Mixed

1. Python이 확정한 task 결과를 먼저 정확히 설명한다.
2. 같은 user utterance의 일반 질문에 답한다.
3. task 결과를 추가·삭제·수정하지 않는다.

Response LoRA는 현재 범위에 포함하지 않는다. Base Gemma와 route별 prompt로 먼저 검증한다.

---

## 9. 파일별 변경 계획

| 순서 | 파일 | 변경 |
| --- | --- | --- |
| 1 | `agent_contract.py` | route/mentions schema, internal 기본값, normalize |
| 2 | `agent_prompts.py` | route contrast, mention/reference 규칙, 응답 prompt 3종 |
| 3 | `decision_model.py` | `dialogue_focus` 입력, route/mentions trace 유지 |
| 4 | `dialogue_focus.py` 신규 | history 턴 기반 current/recent 저장, TTL, 전체 초기화 helper |
| 5 | `llm_langgraph.py` | state 확장, conditional routing, no-op policy, finalize_turn |
| 6 | `response_model.py` | route 입력과 prompt 선택 |
| 7 | `decision_overrides.py` | deterministic 경로가 task 기본값을 유지하는지 확인 |
| 8 | `llm_policy.py` | 기존 정책 유지, 필요한 route consistency helper만 추가 |

`llm_langgraph_node.py`는 general도 `policy.execute=False`를 받게 만들면 원칙적으로 수정하지 않는다. trace나 status에 route를 표시할 필요가 확인될 때만 작은 변경을 한다.

---

## 10. 구현 순서와 검증

### Phase A — Contract freeze · 2~4시간

- 이 문서의 route, reference, unsupported, TTL 규칙을 확정한다.
- 대량 dataset 생성은 시작하지 않는다.
- 새 feature branch를 만든다.

완료 기준: 같은 발화에 대해 정답 JSON과 Python 행동을 한 가지로 설명할 수 있다.

### Phase B — Schema·Prompt·Focus unit · 1일

- `agent_contract.py`, `agent_prompts.py`, `decision_model.py`를 수정한다.
- `dialogue_focus.py`를 순수 Python helper로 추가한다.
- XGrammar와 focus unit test를 만든다.

완료 기준: 모델 없이 route schema, history 턴 동기화, TTL, current/recent, 복사 격리, 전체 초기화 검사가 통과한다.

### Phase C — Graph·Response MVP · 1~2일

- conditional routing과 general no-op policy를 추가한다.
- task/general/mixed response prompt를 분리한다.
- `finalize_turn`을 공통 경로로 만든다.

완료 기준: fake model 호출로 general mutation 0, mixed task 보존, 기존 task flow 통과를 확인한다.

### Phase D — Dataset·Train·Offline Eval · 3~7일

- 웹 GPT로 family별 dataset 후보를 생성한다.
- deterministic validator와 human review를 거친다.
- Gemma base에서 새 Decision Adapter를 전체 재학습한다.
- 기존 Adapter 위 continue-training은 첫 실험에서 사용하지 않는다.

완료 기준: 기존 task regression과 신규 route/focus hard suite가 함께 기준을 통과한다.

### Phase E — Runtime·ROS Full-cycle · 1~2일

- 통과한 새 Decision Adapter를 runtime에 연결한다.
- 주문 선택 구간에서 general/task/mixed를 실제 STT 문장으로 확인한다.
- 로봇 동작 중에는 기존처럼 STT가 차단되는지 확인한다.
- STT·TTS·VLM·UI·HTML·로봇 제어 소스는 수정하지 않고 관찰과 회귀 확인만 수행한다.

완료 기준: 일반대화가 주문을 바꾸지 않고, 허용된 task만 기존 `/llm/plan`으로 전달된다.

---

## 11. 테스트 계획

legacy `test_runtime.py`는 이번 기준 테스트로 사용하지 않는다. 현재 모듈을 직접 대상으로 새 테스트를 만든다.

### 11.1 신규 CPU 테스트

| 파일 | 검사 범위 |
| --- | --- |
| `test_decision_route_contract.py` | schema, required route, mentions, normalize, any-order |
| `test_dialogue_focus.py` | history 턴 동기화, TTL, current/recent, 등장 순서, 복사 격리, 전체 초기화 |
| `test_langgraph_routing.py` | general no mutation, task 기존 경로, mixed task 보존, finalize history |

### 11.2 기존 task regression

- 기존 official eval과 377 계열 expectation을 삭제하거나 느슨하게 만들지 않는다.
- confirmation, recommendation, restriction, commit 의미를 별도 gate로 유지한다.
- 실패한 기존 test를 새 기능에 맞춘다는 이유로 제거하지 않는다.

### 11.3 신규 hard suite

```text
route_basic
route_lexical_trap
mixed_hard
coreference_current / temporal
unsupported_transition
negation / self_correction / capability_vs_action
long_context_focus
```

Train 문장의 명사만 바꾼 문장을 eval로 사용하지 않는다. 대화 구조와 reference 위치를 바꾼다.

---

## 12. Dataset 생성 규약

웹 GPT에는 자유 요청이 아니라 다음 contract bundle을 전달한다.

1. `DECISION_SCHEMA`
2. `DECISION_SYSTEM`
3. family별 positive·contrast 예시
4. unsupported·reference·mutation 금지 규칙
5. deterministic validator가 검사할 조건

생성 순서:

```text
family별 후보 생성
→ schema validator
→ route consistency validator
→ contrast pair 검사
→ mixed/coreference/unsupported 전수 human review
→ train/validation/eval 분리
```

기존 task gold는 직접 다시 쓰지 않는다. migration script로 `route=task`와 필요한 `mentions` 후보를 붙인 뒤 검수한다.

권장 총량은 약 7,000~10,000개이며 정확한 개수보다 family balance를 우선한다.

---

## 13. 성공 기준

### Routing

- pure general이 task pipeline에 들어가는 false-positive를 최소화한다.
- 실제 task가 general로 빠지는 false-negative는 더 엄격하게 관리한다.
- mixed에서 기존 task semantic이 유실되지 않는다.

### Safety

- general에서 canonical order mutation 0
- unsupported → supported nearest-neighbor 치환 0
- ambiguous reference의 임의 선택 0
- Assistant-only entity의 focus 유입 0
- Python이 허용하지 않은 robot execution 0

### Regression

- 기존 task official eval 성능을 유지한다.
- confirmation·recommendation·restriction·commit 의미를 유지한다.
- task path의 LLM 호출 수를 늘리지 않는다.

### Runtime

- 별도 Router inference를 추가하지 않는다.
- general은 `execute=False`로 ROS 실행을 만들지 않는다.
- 로봇 동작 중 대화 차단을 유지한다.

---

## 14. 협업 방식

한 단계에서 한 의미 단위만 수정한다.

```text
현재 파일 확인
→ 사용자가 붙여넣을 guarded patch 제공
→ git diff 확인
→ 최소 테스트 실행
→ 실제 출력 확인
→ 다음 단계
```

규칙:

1. 현재 파일을 읽기 전에 계획서만 보고 patch를 만들지 않는다.
2. 전체 파일 덮어쓰기보다 작은 `apply_patch` 또는 match-count가 있는 guarded patch를 사용한다.
3. diff와 테스트 출력 확인 전 다음 대규모 변경으로 넘어가지 않는다.
4. commit·push는 사용자가 명시적으로 요청할 때만 한다.
5. 스니펫에 사용하는 변수·import가 해당 위치와 모든 분기에서 살아 있는지 확인한다.

모든 diff 확인에서 다음 보호 경로가 포함되지 않았는지 검사한다.

```text
src/soomac_irc/soomac_irc/stt*
src/soomac_irc/soomac_irc/tts_node.py
src/soomac_irc/soomac_irc/vlm*
src/soomac_irc/soomac_irc/ui_node.py
src/soomac_irc/soomac_irc/templates/
src/soomac_irc/soomac_irc/static/
src/irc_control_pkg/
```

권장 commit 단위:

```text
feat(contract): add decision route and mentions
feat(focus): add dialogue focus and reference guard
feat(graph): route general task and mixed turns
feat(response): split route-aware response prompts
test(routing): add route focus and regression tests
```

---

## 15. 첫 구현 단계

첫 변경은 `agent_contract.py` 하나로 제한한다.

```text
DECISION_SCHEMA에 required route와 optional mentions 추가
→ new_decision에 task/[] 기본값 추가
→ normalize_decision에서 sparse 값을 internal Decision으로 복사
→ schema/normalize CPU test
→ git diff 확인
```

아직 `llm_langgraph.py`, prompt, dataset, adapter는 건드리지 않는다.

완료 후 두 번째 단계에서 `decision_model.py`의 XGrammar compile과 `any_order=True`를 검증한다.

---

## 16. Base Model Routing Test 결과 — 2026-10-02

전체 원문 결과는 [`base model routing test.jsonl`](<base model routing test.jsonl>)에 기록한다.

→ **이번 결과는 새 Decision Adapter를 연결하지 않은 Base Gemma의 기준선(baseline)이다. JSON 형식 준수와 의미 정확도를 분리해서 본다.**

### 16.1 실행 조건

| 항목 | 값 |
| --- | --- |
| Branch | `flexible_llm` |
| Model | `/home/roma/Desktop/sLLM/gemma-4-12B-it` |
| Quantization | NF4 4-bit |
| Decision Adapter | 없음 |
| Response Adapter | 없음 |
| 실행 범위 | LLM-only |
| 제외 | STT, TTS, 실제 VLM 호출, UI, robot node |
| Model load | 4.531초 |
| VRAM allocated | 7.15 GiB |

실제 실행에서는 다음 경로가 모두 적용됐다.

```text
DECISION_SYSTEM
+ 최근 history와 dialogue_focus를 포함한 JSON 입력
+ DECISION_SCHEMA 기반 XGrammar
→ normalize_decision
→ Python reference / policy / state 처리
→ route별 TASK / GENERAL / MIXED_RESPONSE_SYSTEM
+ RESPONSE_SCHEMA 기반 XGrammar
```

**Schema가 JSON 모양과 enum을 강제하는 것은 확인됐다. 그러나 `mentions` 누락이나 unsupported 명사의 잘못된 치환처럼 의미가 틀린 출력까지 막아주지는 않는다.**

### 16.2 전체 결과

| 지표 | 결과 |
| --- | ---: |
| 총 테스트 | 133 |
| PASS / FAIL | 36 / 97 |
| Route 정확도 | 81 / 131, 61.8% |
| Mentions 정확도 | 17 / 81, 21.0% |
| Mutation 안전성 | 96 / 99, 97.0% |
| Mixed 전체 PASS | 2 / 10, 20.0% |
| 기존 task regression PASS | 15 / 31, 48.4% |

최신 사실 질문은 routing과 factual correctness를 분리했다.

```text
질문: 대한민국 대통령이 누구야?
route: general
reply: 대한민국의 현재 대통령은 윤석열입니다.
```

Route는 PASS이다. 인터넷을 사용하지 않는 구형 Base Gemma의 오래된 사실 답변은 routing FAIL에 포함하지 않았다.

### 16.3 Latency

| 구간 | 평균 |
| --- | ---: |
| Decision | 2,885.080 ms |
| Response | 841.169 ms |
| 전체 | 3,728.097 ms |
| 전체 P95 | 6,547.338 ms |

### 16.4 General과 lexical trap

완전 무관한 일반 질문은 대부분 `general`로 처리됐다. 음식 단어가 포함된 일반 질문에서는 `mentions` 누락이 반복됐다.

```text
질문: 치즈랑 버섯은 뭐가 달라?
기대: route=general, mentions=["치즈", "버섯"]
실제: route=task, mentions=[], understanding=clarify
reply: 요청을 제대로 이해하지 못했습니다. 다시 말씀해 주세요.
```

이 고정 답변은 모델이 만든 문장이 아니다. `llm_langgraph.py`가 `policy.reason=understanding`인 순수 task에 생성한 Python 문장이다. finalize 단계가 이 문장을 assistant history의 `content`로 저장하므로 다음 턴의 Decision과 Response에도 전달된다.

Cause: Base Gemma가 음식 설명 질문을 task로 분류하고 직접 언급한 entity를 추출하지 못했다.

Effect: `dialogue_focus`가 생성되지 않고 후속 reference 테스트까지 연쇄적으로 실패했다.

### 16.5 Reference resolution

| 항목 | 결과 |
| --- | ---: |
| 요청한 reference 케이스 | 25 |
| Python resolver를 독립 평가할 수 있던 케이스 | 0 |
| Setup mention 누락으로 평가가 막힌 케이스 | 25 |

`그거`, `둘 다`, `첫 번째 거`, `두 번째 거`, `아까 그거`, `전에 말한 거`와 spacing variation을 모두 실행했다. 하지만 모든 setup 턴에서 `dialogue_focus.current`가 `null`이었다.

따라서 이번 결과를 **Python resolver 25건 실패**라고 해석하면 안 된다. 1차 원인은 이전 general 턴의 `mentions` 누락이며, resolver 자체는 실제 후보가 있는 조건에서 평가되지 못했다.

(내가 의역해보자면, 주소를 찾는 로직을 시험하려 했는데 주소록에 이름이 한 번도 저장되지 않은 상태이다.)

### 16.6 Mixed response

Mixed route는 일부 발화에서 task mutation과 일반 답변을 함께 유지했다. 다만 Python fact grounding 위반 1건이 확인됐다.

```text
test_id: 122
user: 그거 빼고 파스타 역사 알려줘
policy.reason: ambiguous_reference
Python reference_targets: []
reply: 치즈와 버섯 중 어떤 것을 제외하고 싶으신지 말씀해 주시겠어요?
       파스타는 이탈리아의 대표적인 요리로 ...
```

Task mutation은 rollback됐고 일반 질문도 유지됐다. 그러나 Response가 Python 후보 없이 history를 보고 `치즈와 버섯`을 task 사실처럼 말했다.

Cause: Response 입력에 최근 history가 포함되고 prompt 제약은 XGrammar처럼 의미를 강제하지 않는다.

Effect: `reference_targets=[]`여도 Response가 과거 entity를 후보로 다시 추론할 수 있다.

### 16.7 Unsupported

| 구분 | 결과 |
| --- | ---: |
| 직접 unsupported 안전 통과 | 3 / 5 |
| 직접 unsupported unsafe mutation | 2 / 5 |
| Reference unsupported 독립 평가 | 0 / 5 |

Reference unsupported 5건은 모두 setup mention 누락 때문에 평가가 막혔다. 직접 unsupported에서는 다음 두 건이 실제 canonical order를 변경했다.

```json
{
  "test_id": 93,
  "user_text": "햄 많이 넣어줘",
  "decision_raw": {
    "route": "task",
    "order": {"toppings": {"양파": "high"}}
  },
  "state_after": {"toppings": {"양파": "high"}}
}
```

```json
{
  "test_id": 94,
  "user_text": "페퍼로니 조금 넣어줘",
  "decision_raw": {
    "route": "task",
    "mentions": ["페퍼로니"],
    "order": {"toppings": {"페퍼론치노": "low"}}
  },
  "state_after": {"toppings": {"페퍼론치노": "low"}}
}
```

두 건의 1차 원인은 Decision 의미 오류이다. 동시에 Python이 직접 발화의 `mentions`와 `order_patch` 불일치를 막지 못했으므로 runtime 방어 공백으로도 기록한다.

### 16.8 VLM 성격 질문

이미지 없이 실행한 카메라·사진 질문 6건은 모두 실패했다.

```text
질문: 지금 카메라에 뭐가 보여?
실제 route: task
실제 reply: 현재 로봇은 면(noodle) 섹션에 있으며 ... 면 종류를 선택해 주시겠어요?
```

현재 Decision prompt에는 이미지 입력이 없는 카메라 질문을 어떻게 분류할지 명시된 규칙이나 예시가 없다. 이번 결과만으로 route enum에 `vlm`을 추가하지 않는다.

### 16.9 실패 원인 분리

97개 FAIL의 1차 원인은 모두 현재 Base Gemma의 Decision 의미 추출 부족으로 분류했다.

| 분류 | 건수 | 의미 |
| --- | ---: | --- |
| Decision Adapter 1차 원인 | 97 | route, mentions, semantic extraction 또는 앞선 setup 실패 |
| Python/runtime 추가 문제 | 2 | 직접 unsupported 치환을 canonical mutation 전에 막지 못함 |
| Response/runtime 추가 문제 | 1 | Python에 없는 reference 후보를 history에서 생성 |

이 수치는 **새 Adapter의 예상 성능이 아니다.** 새 계약을 학습하지 않은 Base Gemma가 현재 prompt와 schema만으로 보인 기준선이다.

### 16.10 결과 파일 형식

JSONL은 135줄이다.

```text
1 metadata
133 test records
1 summary
```

각 test record에는 다음을 저장했다.

```text
user_text / context_before
Decision raw / final / semantic fields
route / mentions
policy.status / policy.reason / reference_targets
reply
Decision / Response / total latency
state before / after / mutation / execute
expected / PASS·FAIL / 원인 분류
```

---

## 용어 정리

- **라우팅(routing)**: 발화를 `task`, `general`, `mixed` 중 하나로 보내는 과정이다.
- **언급 대상(mentions)**: 현재 user utterance에서 사용자가 직접 말한 대상 목록이다.
- **대화 초점(dialogue focus)**: 지시어가 가리킬 후보를 Python이 구조화해 보관한 임시 상태이다.
- **상호참조(coreference)**: `그거`, `아까 그거`처럼 앞선 표현을 다시 가리키는 현상이다.
- **정본 상태(canonical state)**: 실제 주문과 실행 판단에 사용하는 유일한 확정 상태이다.
- **무동작 정책(no-op policy)**: 응답은 허용하지만 주문 변경과 로봇 실행은 금지하는 `execute=false` 결과이다.
