# 2026-10-05 Thin Python 리팩토링 계획

[가드 정리 작업 기록](2026-10-05_guard_cleanup_worklog.md)

- 기준: `flexible_llm` HEAD `fc07f04`
- 작업 브랜치: `refactor/thin-python-guards`

**원칙: Decision 이 맞게 낸 의미를 Python 이 다시 해석해서 뒤집지 않는다. Python 은 스키마·물리·안전·완료 보호·실행 게이트만 맡는다. 코드는 최소 변경한다.**

```text
Decision LLM   = 말뜻 해석
Python         = 스키마 / 명백한 모순 / 안전 / 물리 검증
LangGraph      = 질문·확인·section 상태 수명
Response       = 확정된 사실을 표현 (질문의 존재와 의미는 state 가 정함)
```

이번 범위 밖: 대기 상태 4종(`pending_confirmation` 2종, `pending_question`, `recommendation.proposed`)을 하나로 합치는 재구성. 별도로 논의한다.

---

## 1. 실제 로그 failure trace (session_20261005_221135)

| 턴 | 발화 | Decision | 결과를 바꾼 곳 | 분류 |
|---|---|---|---|---|
| #10 | 소시지한 계살 둘 다 많이줘 | `소시지 high, 게살 high` (정답) | `resolve_reference` 가 "둘다"를 보고 focus(빈 event)를 참조 → `ambiguous_reference` | **Code** |
| #9→#10 | 지수야 머먹고 싶어 | `robot_status` query | `next_prompt=recommendation_offer` → `question_from_prompt` 가 저장 안 함 → 소시지/게살 선택 질문 소실 | Decision 은 **Model**, 질문 소실은 **Code** |
| #8→#9 | 바로 시작 (양파 실행) | commit | 노드 `remove_focus_mentions` 가 양파 제거 후 `current.mentions=[]` 빈 event 유지 → #10 resolver 가 빈 current 를 참조 | **Code** |
| #13 | 나는 치즈가 싫어. 패포론 지눈만 줘. | 치즈 dislike + 페퍼론치노 normal | 자모 0.45 → `menu_confirmation` 후보 보류 (의도된 동작) | — |
| #14 | 그럼 그럼 바로 시작해. | `commit=true`, confirmation 없음 | `interpret_decision` 이 accept 일 때만 후보 병합 → 후보 버림 → 빈 extra 실행 → skip → lid | **Code** (P0) |
| #13 응답 | — | — | Response 에 `proposal` 의 null 필드(sauce/noodle)까지 전달 → "소스, 면 종류, 면 양은 기본 설정대로…" | contract **Code** + Response **Model** |

뚜껑 (session_20261005_213903, 노드 ROS 로그):

```text
21:51:27 /llm/plan cover → 21:52:16 confirm_start(VLM) → 21:52:23 /llm/next → /llm/plan sauce_tomato → 21:53:44 VLM pass → 21:53:59 주문 완료
```

- 실물 로봇은 이미 "LangGraph 가 cover 를 소유하고, VLM 확인 후 `/llm/next`" 계약으로 동작했다.
- 소스 후 15초 만에 reset 이 왔다. cover 동작은 49초 걸리므로 소스 뒤 cover 중복 실행은 없었다.
- **레포의 `main_vlm.py` 는 옛 계약**("cover 는 VLM 없이 home", "소스 끝나면 cover 자동 실행")으로 남아 있다.

## 2. 수정 항목

### P0-1 미확정 후보가 있으면 실행·skip 금지
- 위치: `llm_langgraph.check_policy`
- 활성 `menu_confirmation` 질문이 있고 이번 턴에 accept/reject 가 없으면, `execute=false` 로 하고 질문을 유지한다.
- Python 이 후보를 임의로 accept 하지 않는다.
- Decision 이 accept + commit 을 함께 내면 후보를 반영하고 실행한다. 현재는 accept 시 commit 을 지운다 → **유지하도록 변경**. 실행 문구 게이트는 그대로 적용한다.

### P0-2 cover 실행 소유자 단일화 = LangGraph `lid`
- LLM 쪽만 다룬다. `main_vlm.py` 는 다른 브랜치 소유라 이 브랜치에서 수정하지 않는다.
- 주석(`domain.py`, `llm_langgraph_node.py`)을 실제 계약(lid 가 cover 소유)에 맞춘다.

### P1-1 발화에 직접 나온 대상 > focus resolver
- 위치: `llm_langgraph.resolve_reference`, `guard_decision_grounding` 의 지시어 문맥
- Decision 의 대상이 모두 이번 발화에 근거가 있으면(자모 근거), "둘 다/그거"를 다시 해석하지 않는다.
- resolver 에 한국어 표현을 추가하지 않는다.

### P1-2 pending_question 수명
- 위치: `dialogue_questions.question_after_turn`
- `next_prompt` 가 저장 대상 타입이 아니면(예: `recommendation_offer`) 기존 질문을 **유지**한다.
- 닫는 경우: 답변됨 / 다른 대상 명시 / 실행 시작 / section 전환(노드) / 거절 / 확인 우선권.

### P1-3 빈 focus event 제거
- 위치: `dialogue_focus.remove_focus_mentions`
- mention 이 0개가 된 event 는 current/recent 에 남기지 않는다.

### P1-4 "반영했어요 → 바로 담을까요?" (`execution_offer`)
- `build_next_prompt`: 현재 section 에 실행할 재료가 있고, 이번 턴에 그 section 을 바꿨고, 실행하지 않았고, 다른 질문이 없으면 `{"type": "execution_offer", "section": ...}`
- `pending_question` 으로 저장한다.
- 다음 턴 accept("응" 정확 일치 또는 Decision confirmation=accept) → commit 을 state 근거로 허용해 실행한다. 실행 문구 게이트 예외는 이 상태 하나뿐이다.
- reject → 질문만 닫는다.
- Response 는 `question_to_ask` 를 표현만 한다.

### P1-5 Response contract
- `unconfirmed_candidate` / `question_to_ask.proposal` 에서 null·빈 필드를 뺀다.

## 3. Guard 분류

| 분류 | guard |
|---|---|
| KEEP | 스키마/XGrammar, `find_invalid_queries`, `protected_order_keys`·완료 보호, restriction 충돌 HITL, `robot_busy`, 실행 문구 게이트, 메뉴 근거(자모) |
| REDUCE | `resolve_reference` (직접 대상이 있으면 off), focus 빈 event, `question_from_prompt` 의 질문 덮어쓰기 |
| REMOVE 후보 (이번엔 보류) | 게이트가 지운 commit 으로 인한 repair 재호출, `BLOCKED_COMMIT_MARKERS` 의 알러지 해제 재사용 |

## 4. 테스트 (RED 먼저)

`test_thin_python_refactor.py`

| | 시나리오 | 기대 |
|---|---|---|
| A | 소시지랑 게살 둘 다 많이 (focus 빈 상태) | 둘 다 high, clarify 아님 |
| B | 후보 확인 → 응 → 바로 시작 | 후보 반영 → extra 실행 |
| B' | 후보 대기 + commit (accept 없음) | execute=false, 질문 유지, skip 없음 |
| C | 후보 대기 + accept + commit | 반영 + 실행 |
| D | 선택 질문 중 상태 질문 (`recommendation_offer` 상황) | 질문 유지 |
| E | 실행 대상 focus 제거 | 빈 event 없음 |
| F | 양파 보통 추가 → execution_offer → 응 | 실행 |
| G | lid section 작업 = 뚜껑 1회, sauce section 에는 cover 없음 (LLM 쪽만) | — |

실행 순서: 신규 → 기존 dialogue/reference → 전체 → 신규 재실행(3회). 기존 known failure(10 + import 2)는 구분해서 보고한다.

## 5. 결과

### 변경 파일

| 파일 | 함수 | 변경 |
|---|---|---|
| `llm_langgraph.py` | `interpret_decision` | 후보 accept 시 commit 을 지우지 않음. `execution_offer` accept/reject 처리 |
| `llm_langgraph.py` | `apply_explicit_commit_gate` | 예외 1개: `execution_offer` 수락 state 이면 문구 없이 실행 허용 |
| `llm_langgraph.py` | `resolve_reference` | Decision 대상이 모두 발화에 직접 있으면 지시어 해석 생략 |
| `llm_langgraph.py` | `check_policy` | 미확정 `menu_confirmation` 이 남아 있으면 `execute=false` |
| `llm_policy.py` | `decision_targets_are_direct`, `compact_order_patch` (신규) | 직접 대상 판정, Response 용 빈 필드 제거 |
| `llm_policy.py` | `build_next_prompt` | `execution_offer` 생성 |
| `dialogue_questions.py` | `question_from_prompt`, `question_after_turn` | `execution_offer` 저장. 저장 안 하는 prompt 가 기존 질문을 덮지 않음 |
| `dialogue_focus.py` | `remove_focus_mentions` | 빈 event 제거 |
| `decision_overrides.py` | `pre_decision_override` | `execution_offer` 에도 응/아니 단답 적용 |
| `response_model.py` | `call_response` | 후보 빈 필드 제거, `execution_offer` 표현 규칙 1줄 |
| `domain.py`, `llm_langgraph_node.py` | 주석 | cover 소유자를 lid 로 명시 |

### 테스트

| 단계 | 결과 |
|---|---|
| 1. 신규 `test_thin_python_refactor.py` | 수정 전 12개 중 8개 실패 (main_vlm 테스트 1개는 범위 밖이라 삭제)(RED) → 11/11 통과 |
| 2. 기존 dialogue/reference | `test_question_lifecycle` 4개는 계약 변경(후보 확인 뒤 `execution_offer` 생성)으로 기대값을 수정. `test_section_question_events` 1개는 lid 커밋 때부터 깨져 있던 stale 테스트(extra 다음 = lid)로 기대값을 수정 |
| 3. 전체 106개 | 실패 10 + import 오류 2. 기존 known failure(실행 문구 보정 기대 테스트, 없는 모듈 import)와 같은 목록이다. 새 회귀 0 |
| 4. 신규 재실행 | 3/3 통과 |

로그 순차 재생 (session_20261005_221135, 모델 출력 고정):

| 턴 | 기존 `fc07f04` | 수정 후 |
|---|---|---|
| #10 소시지한 계살 둘 다 많이줘 | `clarify/ambiguous_reference` | 소시지·게살 high 반영 + `execution_offer` |
| #14 그럼 그럼 바로 시작해. | `execution_allowed` → extra skip → lid | `clarify/menu_confirmation`, 후보 유지, 실행 안 함 |

### 남은 모델/데이터 문제 (코드로 고치지 않음)

- "지수야 머먹고 싶어" → `robot_status` (Decision 분류)
- "베프론치노…패플론치노 담아줘" → 주문 없이 clarify (Decision)
- 후보 확인 대기 중 "그럼 바로 시작해" → Decision 이 accept 를 내지 않음. 지금은 다시 묻는다. 학습 데이터에 "후보 대기 + 실행 = accept+commit" 추가가 필요하다
- Response 의 엉뚱한 질문, 실행하지 않았는데 실행했다고 말하는 문제 (Response adapter SFT 예정)
- STT 일본어 출력("トマト"), 짧은 메뉴명 오탐
- 런타임 Decision 프롬프트(9.2k자)와 학습 프롬프트(5.8k자) 불일치

### 확인 필요

- `main_vlm.py` 는 다른 브랜치 소유다. 이 브랜치의 옛 버전(소스 뒤 cover)은 실물과 다르지만 수정하지 않았다.
- 실제 모델 재실행 완료 (6장 참고).
- Codex 교차검증은 하지 않았다.

---

## 6. 실제 모델 재실행 (2026-10-05 22:53, gemma + decision adapter)

로그: `logs/llm/2026-10-05/session_20261005_225333_*` (22:11 세션 발화), `225450_*` (18:18 세션 발화), `225547_*` (반복 요청)

| 장면 | 전 | 후 |
|---|---|---|
| "소시지한 계살 둘 다 많이줘" | "어느 쪽이요?" (`ambiguous_reference`) | "소시지와 게살을 많이로 반영했어요. 지금 바로 담기 시작할까요?" |
| 페퍼론치노 확인 중 "그럼 그럼 바로 시작해" | 후보를 버리고 실행 → extra skip → lid | 실행하지 않고 "페퍼론치노를 보통 양으로 담아드릴까요?" 재확인 |
| 채소 단계에서 상태 질문 2번 | 양파/버섯 선택 질문 소실 | 선택 질문 유지 |
| "양파만 적당히 추가해줘" | "반영했어요."에서 끝 | "…지금 바로 양파를 담기 시작할까요?" → "바로 시작" → 실행 |
| "벗어선 제외해 달라니까 바로 시작해" | 되물음 | 실행 |
| 이미 있는 소스·면·알러지 반복 요청 | 직전 "반영했어요" 반복 | "이미 반영되어 있어요" |

## 7. 왜 "응"으로만 수락되나 (다음 작업)

확인 대기 중 "그럼 그럼 바로 시작해"가 수락되지 않고 다시 묻는다. 수락이 처리되는 길이 두 개인데 둘 다 막혀 있다.

1. **Python 단답 처리** (`decision_overrides.py` `ACCEPT_CONFIRMATIONS`)
   - 발화 **전체**가 `응/네/예/그래/좋아/오케이/ok` 중 하나와 정확히 같아야 한다.
2. **Decision 모델**
   - 22:53 #14 입력에 `pending_question = {type: menu_confirmation, targets: [페퍼론치노]}`가 정확히 들어갔는데, 출력은 `{"route":"task","commit":true}`였다.
   - 학습 데이터(`decision_train.jsonl` 3,049줄)를 확인한 결과:

     | 항목 | 등장 횟수 |
     |---|---|
     | `pending_question` 입력 | **0** |
     | `menu_confirmation` | **0** |
     | `confirmation: accept` 정답 | 50 (전부 알러지·미리 고른 재료·추천 확인) |

   - Cause: `pending_question` 은 어댑터 학습 이후에 추가된 기능이다.
   - Effect: 모델은 이 필드가 "답을 기다리는 질문"인 줄 모르고, "바로 시작해"를 실행 요청으로만 읽는다.

### 해결안

| | 방법 | 비고 |
|---|---|---|
| A (근본) | 학습 데이터에 "메뉴 확인 대기 + 동의/실행 발화 → accept (+commit)" 추가 후 재학습 | 알러지 해제·분류 오류 튜닝과 함께 진행 |
| B (임시, 추천) | `decision_model.decision_pending_view` 에서 메뉴 확인을 모델이 학습한 확인 형식으로 보여줌 (예: `preselected_section` 과 같은 "넣을까요?" 확인) | 단어 추가 없음. 수락 판단은 모델이 함. restriction 확인을 학습 형식으로 바꿔 accept 가 나오기 시작한 전례가 있음. 효과는 실제 모델로 확인 필요 (추측) |
| C | `ACCEPT_CONFIRMATIONS` 에 표현 추가 | 단어 하드코딩이라 하지 않음 |

같은 이유로 `execution_offer` ("바로 담을까요?") 도 학습 데이터에 없다. 지금은 "응" 정확 일치나 실행 문구("바로 시작")로만 진행된다. A 를 할 때 함께 넣는다.
