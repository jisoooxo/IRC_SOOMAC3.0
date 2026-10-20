# IRC_SOOMAC 3.0
# Runtime Safety v4 + Decision v3.3 + Recommendation v1 + Response v1.2
# Contract Migration → Dataset Rebuild → Validation → Training → Best Epoch → A/B → Runtime Adoption

[2026-10-20 최종 작업 기록](2026-10-20_full_agent_runtime_retrain_worklog.md)

→ 전체 데이터 감사·fresh training·전 epoch/실로그/1,000턴 평가까지 완료했으며, 장기 gate 실패로 신규 후보는 미채택하고 기존 stack을 유지했다.

현재 `flexible_llm` 브랜치 최신 HEAD를 기준으로 작업한다.

이번 작업은 단순 supplement 추가가 아니다.

현재 runtime과 모델 입력 계약이 기존 Decision v3.2 / Response v1 학습 시점 이후 많이 변경되었고, 이번 작업에서도 추가 contract 변경이 예정되어 있다.

따라서:

```text
기존 데이터
+ 새 데이터 몇백 행
```

방식으로 바로 학습하지 않는다.

반드시:

```text
1. 최신 runtime/source 감사
2. 새 contract 확정
3. 기존 Decision / Response dataset 전체 migration
4. Recommendation dataset 신규 생성
5. 오늘 실제 robot log 기반 failure family 반영
6. 데이터 검증
7. PASS 시 즉시 학습
8. epoch별 validation
9. best epoch 자동 선택
10. 기존 모델과 A/B
11. 좋은 Adapter만 runtime 채택
```

순서로 진행한다.

사용자에게 중간 승인 요청 없이 진행하되,
검증 gate가 실패한 상태에서 학습을 강행하지 않는다.

---

# 0. 현재 baseline 보존

현재 안정 baseline:

```text
Base:
gemma-4-12B-it

Decision:
decision_adapter_v3_2_best_epoch4

Recommendation:
Base Gemma

Response:
response_adapter_v1_best_epoch1
```

절대 삭제하거나 overwrite하지 않는다.

탈락했던:

```text
decision v3.1
response v1.1
```

도 그대로 보존한다.

현재 baseline은 rollback 기준이다.

---

# 1. 이번 목표 architecture

최종 candidate:

```text
Gemma 4 12B Base
│
├── Decision Agent
│   └── decision_adapter_v3_3_best_epochX
│
├── Recommendation Agent
│   └── recommendation_adapter_v1_best_epochY
│
├── Response Agent
│   └── response_adapter_v1_2_best_epochZ
│
└── VLM
    └── existing Base / existing path
```

세 Adapter는 절대 합치지 않는다.

호출별 adapter routing:

```text
Decision
→ Decision v3.3

Recommendation
→ Recommendation v1

Response
→ Response v1.2

VLM
→ 기존 path
```

---

# 2. 가장 먼저 현재 HEAD를 감사하라

먼저 실제 source를 전부 읽는다.

최소:

```text
src/soomac_irc/soomac_irc/
  agent_contract.py
  agent_prompts.py
  decision_model.py
  recommendation_model.py
  response_model.py
  adapter_runtime.py
  llm_langgraph.py
  llm_policy.py
  llm_langgraph_node.py
  domain.py
```

그리고:

```text
docs/
logs/llm/2026-10-11/
training_v3_2_v1_1/
data_targeted_upgrade/
기존 training scripts
기존 evaluation scripts
```

를 확인한다.

이 프롬프트의 설명보다 **현재 HEAD source가 구조적 사실에서는 우선**한다.

단 아래의 목표 semantics는 유지한다.

산출물:

```text
upgrade_v4/CURRENT_CONTRACT_AUDIT.md
```

---

# 3. 오늘 실제 로그를 전수 분석한다

`logs/llm/2026-10-11/` 아래의:

```text
agent_turns.jsonl
runtime_events.jsonl
```

을 모든 세션에 대해 분석한다.

다음 failure source를 agent별로 분류한다.

```text
Decision
Recommendation
Response
Policy/runtime
ASR/noisy input
downstream failure
already-fixed-by-latest-runtime
evaluation ambiguity
```

중요:

오늘 로그는 모두 같은 source revision에서 생성된 것이 아니다.

따라서 commit timestamp와 runtime log timestamp를 비교해서:

```text
이미 이후 commit에서 해결된 문제
vs
현재 HEAD에서도 남아 있는 문제
```

를 구분한다.

이미 deterministic runtime으로 해결된 문제를
새 Adapter 학습 데이터에 과도하게 넣지 않는다.

산출물:

```text
upgrade_v4/REAL_LOG_FAILURE_AUDIT.md
upgrade_v4/real_log_failure_provenance.jsonl
```

---

# 4. Runtime P0 — strong restriction removal safety

다음 strong restriction:

```text
allergy
cannot_eat
dietary_rule
```

은 Decision이 remove를 출력했다고 즉시 삭제하지 않는다.

예:

```json
{
  "restrictions": [
    {
      "target": "게살",
      "reason": "allergy",
      "action": "remove"
    }
  ]
}
```

현재처럼 canonical restriction을 바로 삭제하면 안 된다.

대신:

```json
{
  "type": "restriction_removal",
  "candidate": {
    "target": "게살",
    "reason": "allergy"
  }
}
```

pending을 만든다.

사용자 응답:

```text
accept
→ 실제 restriction 삭제

reject
→ 기존 restriction 유지
```

---

# 5. strong restriction 삭제는 안전 정책이지 keyword guard가 아니다

Python에서:

```text
"알레르기 없어"
"먹어도 돼"
"괜찮아"
```

를 regex로 판단하지 않는다.

자연어 의미는 Decision이 판단한다.

Python은:

```text
strong restriction removal
→ 추가 확인이 필요한 safety-sensitive mutation
```

이라는 구조적 정책만 담당한다.

---

# 6. skip_current를 Decision contract에 정식 추가

현재 `commit=true`에:

```text
현재 선택 실행
현재 optional section skip
```

이 섞이지 않게 한다.

새 field:

```json
{
  "skip_current": true
}
```

추가.

의미:

```text
commit=true
→ 현재 section의 확정 selection 실행

skip_current=true
→ 현재 optional section에서 아무것도 실행하지 않고 다음 단계
```

---

# 7. skip_current 적용 대상

기본:

```text
veggie
meat
extra
```

예:

```text
"육류는 안 먹을래"
→ skip_current=true

"여긴 그냥 넘어가"
→ skip_current=true

"추가 토핑 필요 없어"
→ skip_current=true

"소시지랑 게살 다 빼고 넘어가"
→ 소시지=none
→ 게살=none
→ skip_current=true
```

대조:

```text
"지금 고른 거 담아"
→ commit=true

"이대로 시작"
→ commit=true
```

---

# 8. skip_current와 commit의 contradiction 검사

다음 조합은 기본적으로 구조적 모순으로 본다.

```text
commit=true
skip_current=true
```

단 contract상 정말 동시에 필요한 특수 의미가 없다면 허용하지 않는다.

one-shot repair 대상에 포함한다.

---

# 9. optional empty-section 처리

현재 temporary `empty_section_commit` guard를
새 contract에 맞게 재정리한다.

```text
empty optional + commit
→ 실행 금지

empty optional + skip_current
→ section advance

selection 있음 + commit
→ execution

selection 있음 + skip_current
→ 필요하면 selection clear semantics와 함께 처리
```

Python이 자연어를 다시 판단하지 않는다.

---

# 10. Runtime P0 — section transition atomicity

현재 section transition 중:

```text
section change
pending clear
rail publish
task creation
plan publish
history update
```

가 반쯤 적용된 상태에서 exception이 날 수 있는지 검사한다.

목표:

```text
prepare
→ validate
→ side effect
→ commit state
```

또는 실패 시 안전 rollback.

---

# 11. transition fault injection

다음 오류를 강제로 발생시킨 테스트를 작성한다.

```text
rail publisher failure
plan publisher failure
task build failure
lid task failure
sauce task failure
reply publish failure
history write failure
transition 중 generic exception
```

검사:

```text
section drift 없음
pending silent loss 없음
duplicate execution 없음
active_task / queue contradiction 없음
STT stuck/open error 없음
canonical state와 robot execution state 모순 없음
```

---

# 12. contract version 갱신

새 schema가 적용되면:

```text
natural_multiturn_v3
```

를 그대로 사용하지 않는다.

예:

```text
natural_multiturn_v4
```

실제 repository convention에 맞춰 version을 갱신한다.

migration manifest에:

```text
old contract
new contract
schema changes
pending changes
runtime behavior changes
```

를 기록한다.

---

# 13. 핵심 신규 개념 — Contextual Carry-over

이번 upgrade의 핵심 목표 중 하나다.

사용자가 이전 턴에서 말한:

```text
restriction
preference
long-term conversational condition
```

을 단순 저장하는 것에서 끝내지 않는다.

**나중에 관련 section에 도착했을 때 자연스럽게 다시 활용한다.**

예:

```text
초반:
User:
"나 매운 거 못 먹어."

→ state에 매운맛 회피 정보 저장

여러 턴 뒤:
robot_state.section = extra

현재 선택지:
치즈
페퍼론치노

Expected behavior:
이전에 말한 매운맛 회피를 current section과 연결
→ 페퍼론치노 회피
→ 치즈 또는 safe alternative 안내
```

예상 자연어:

```text
"아까 매운 걸 못 드신다고 하셨는데,
여기서는 페퍼론치노를 빼고 치즈로 드릴까요?"
```

---

# 14. Contextual carry-over는 단순 history copy가 아니다

금지:

```text
매 section마다 모든 preference 반복
직전 user sentence 그대로 재출력
과거 정보를 무조건 설명
```

목표:

```text
현재 section과 관련 있을 때만 꺼내 사용
```

예:

```text
게살 allergy
→ meat에서 relevant

매운맛 회피
→ extra의 페퍼론치노에서 relevant

꾸덕한 식감 선호
→ extra의 치즈 또는 sauce와 relevant

버섯 dislike
→ veggie에서 relevant

다이어트/가벼운 양 선호
→ portion/recommendation 관련 상황에서 relevant
```

---

# 15. Restriction과 preference carry-over를 다르게 처리

Restriction:

```text
강한 금지/제약
```

예:

```text
"게살 알레르기가 있다고 하셨으니 게살은 제외할게요.
소시지는 드릴까요?"
```

Preference:

```text
선호 기반 제안
```

예:

```text
"꾸덕한 식감을 좋아한다고 하셨으니
치즈를 넣어보는 건 어떠세요?"
```

즉:

```text
restriction = must-not / safety constraint
preference = soft recommendation signal
```

을 유지한다.

---

# 16. carry-over를 Python keyword mapping으로 만들지 않는다

금지:

```text
매운 → extra → 치즈
꾸덕 → 치즈
다이어트 → low
```

같은 Python keyword mapping.

의미 연결은 Recommendation Agent와 Response Agent가 수행한다.

Runtime은:

```text
확정 preferences
restrictions
current section
recommendation_scope
```

라는 structured fact만 제공한다.

---

# 17. carry-over relevance를 테스트하라

다음 예를 평가한다.

```text
매운맛 회피
→ noodle에서는 불필요하게 말하지 않음
→ extra에서는 활용

게살 allergy
→ veggie에서는 불필요
→ meat에서는 활용

꾸덕함
→ meat에서 무리하게 소시지 추천하지 않음
→ extra/sauce에서 활용

버섯 싫음
→ meat에서 반복하지 않음
→ veggie에서 사용
```

---

# 18. Decision v3.3 — 기존 v3.2 dataset 전체 migration

기존 Decision v3.2 candidate를 그대로 append-only 방식으로 사용하지 않는다.

모든 row를 새 contract 기준으로 검사한다.

각 row:

```text
UNCHANGED
MIGRATED
RELABEL_REQUIRED
AMBIGUOUS
DROP
```

중 하나로 분류한다.

---

# 19. sparse contract 유지

새 field가 생겼다고:

```json
"skip_current": false
```

를 모든 row에 넣지 않는다.

의미가 있을 때만 field 출력.

---

# 20. 기존 skip semantics 재라벨

기존 데이터에서:

```text
다음 단계
패스
넘어가
안 넣을래
아무것도 안 넣을게
```

의 의미를 전수 검색한다.

기존 gold가:

```text
commit=true
```

였으면 새 contract에 따라 migration한다.

---

# 21. restriction removal multi-turn migration

첫 턴:

```text
"게살 알레르기 이제 없어"
```

Decision semantic:

```text
restriction remove
```

Runtime outcome:

```text
restriction_removal pending
```

다음:

```text
"응 해제해"
→ confirmation accept
```

Decision과 runtime 책임을 섞지 않는다.

---

# 22. Decision v3.3 failure family — preference polarity

오늘 실제 로그에서:

```text
"매운 걸 별로 안 좋아해"
```

를:

```text
매콤하게 preference add
```

로 저장한 문제가 있었다.

다음 minimal pair를 충분히 생성한다.

```text
"매운 거 좋아해"
"매운 건 별로야"

"맵게 먹고 싶어"
"매운 건 피하고 싶어"

"오늘은 매콤하게"
"오늘은 안 맵게"

"매운 건 못 먹는데 꾸덕한 건 좋아"
```

negative polarity를 positive preference로 뒤집지 않는다.

---

# 23. Decision v3.3 — noisy ASR target hallucination

오늘 실제 로그에서 target이 명확하지 않은 깨진 발화에
없는 restriction target을 만든 사례가 있었다.

훈련:

```text
history로 target이 명확
→ resolve 가능

history에도 target 불명
→ clarify

없는 메뉴/재료 창작 금지
```

예:

```text
history 없음
"그거 못 먹어"
→ clarify

직전:
"게살 드실 수 있나요?"
User:
"그거 못 먹어"
→ 게살 cannot_eat
```

---

# 24. Decision v3.3 — option / recommendation / status contrast

충분한 contrastive data를 만든다.

```text
"뭐 담을 수 있어?"
→ selection/options query

"뭘 넣는 게 좋아?"
→ recommendation.request

"지금 뭐 담고 있어?"
→ status

"아까 뭐 담았어?"
→ past execution query

"앞으로 뭐 고를 수 있어?"
→ future option query
```

---

# 25. Decision v3.3 — recommendation partial modification

예:

```text
pending recommendation:
양파 normal
버섯 high
```

User:

```text
"버섯만 적게 하고 그대로 가자"
```

Expected:

```text
confirmation accept
버섯=low
양파는 candidate 유지
필요하면 commit
```

다음도 포함:

```text
"양파만 빼고 나머지는 그대로"
"치즈만 빼고 추천대로"
"버섯은 조금으로 바꾸고 바로 시작"
```

---

# 26. Decision v3.3 — pending reject + replacement

예:

```text
pending:
양파 normal
버섯 normal

User:
"아니 그냥 둘 다 조금만 줘"
```

Expected:

```text
confirmation=reject

order:
양파=low
버섯=low
```

runtime은 기존 pending을 닫고
새 execution pending을 만든다.

---

# 27. Decision v3.3 신규 supplement

migration base 위에
실제 failure 기반 supplement 추가.

권장:

```text
700~1000 rows
```

예상 family:

```text
skip_current / commit                 150
restriction removal lifecycle        100
preference polarity                  100
ASR noisy / hallucination            100
recommendation partial modification  120
query intent contrast                100
pending replacement                   80
reference/long context                70
carry-over semantic state             80
regression rehearsal                 100
```

실제 분석 결과에 따라 조정.

---

# 28. Recommendation Adapter v1 신규 생성

현재 Recommendation은 Base Gemma를 사용한다.

이번에 전용 Adapter를 만든다.

Recommendation의 역할:

```text
사용자 질문
+ current order
+ restrictions
+ preferences
+ pending
+ robot_state
+ recommendation_scope
```

를 읽고:

```text
현재/미래에서 실제 가능한 후보 중
사용자가 원하는 최소 proposal
```

을 만든다.

---

# 29. Recommendation v1 핵심 — contextual carry-over

예:

```text
Turn 1:
"매운 거 못 먹어"

... 8 turns later ...

current section = extra
scope = 치즈 / 페퍼론치노

User:
"여기 뭐 넣는 게 좋아?"
```

Gold:

```text
치즈 proposal
페퍼론치노 제외
```

Reason:

```text
매운맛 회피
현재 extra 단계
```

---

# 30. Recommendation은 관련 없는 preference를 억지로 사용하지 않는다

예:

```text
preference:
매운맛 회피

current section:
veggie

user:
"양파랑 버섯 중 뭐가 좋아?"
```

매운맛 회피를 억지로 양파/버섯 선택 근거로 사용하지 않는다.

즉 carry-over는:

```text
relevance-aware
```

이어야 한다.

---

# 31. Recommendation v1 — one-of constraint

오늘 실제 로그:

```text
"둘 중 하나만 추천해"
```

인데 양파와 버섯 둘 다 proposal에 넣은 사례가 있었다.

다음 constraint를 학습한다.

```text
하나만
한 개만
둘 중 하나
하나 골라줘
```

→ exactly one relevant candidate.

---

# 32. Recommendation v1 — current/future scope

현재 extra:

```text
extra:
치즈
페퍼론치노

future sauce:
오일
토마토
크림
```

User:

```text
"토핑 뭐 넣는 게 좋아?"
```

→ extra만.

User:

```text
"소스는 뭘로 바꿀까?"
```

→ future sauce만.

User:

```text
"앞으로 전체 조합 추천해줘"
```

→ current + future 고려 가능.

---

# 33. Recommendation v1 — past section 금지

이미 완료된:

```text
noodle
veggie
meat
```

를 current recommendation proposal에 다시 넣지 않는다.

`recommendation_scope` 밖은 제안 금지.

---

# 34. Recommendation v1 — no-op vs alternative

현재 future sauce:

```text
크림
```

User:

```text
"다른 소스 추천"
```

Gold:

```text
크림 그대로 ❌
오일/토마토 등 적절한 alternative ✅
```

---

# 35. Recommendation v1 dataset 규모

권장:

```text
800~1300 rows
```

family:

```text
current recommendation
future recommendation
carry-over preference
carry-over restriction
one-of
multi-choice
partial revise
pending revise
no-op avoidance
past section exclusion
scope boundary
conflicting preference
long-context carry-over
```

---

# 36. Response v1.2 신규 학습

기존 Response v1을 그대로 유지하지 않는다.

단 v1.1 dataset을 그대로 확장하지 않는다.

v1.1은 특정 pending family overfit으로
전체 일반화가 회귀한 전례가 있다.

Response v1.2는 **현재 전체 Response contract를 균형 있게 다시 학습**한다.

---

# 37. Response dataset도 전체 migration

현재 Response input은 과거 v1 학습 당시와 다르다.

현재 HEAD의 실제 input을 기준으로 migration한다.

최소 현재 존재하는:

```text
user_text
recent_history
confirmed_order
preferences
pending
policy
applied_this_turn
future_changes
recommendation_result
next_prompt
robot_state
recommendation_scope
recent_action_history
execution_authorized
starting_now
```

를 확인한다.

실제 HEAD와 차이가 있으면 HEAD를 따른다.

---

# 38. Response v1.2 핵심 — current turn priority

Response는:

```text
현재 user_text
현재 policy
현재 structured facts
```

를 가장 먼저 따른다.

recent_history의 이전 assistant 답변을
현재 답변 대신 복사하지 않는다.

---

# 39. 이전 답변 재탕 금지의 정확한 의미

금지:

```text
직전 reply와 문자열이 같으면 다른 표현을 사용
```

같은 superficial anti-repeat rule.

목표는:

```text
현재 질문 목적이 다르면
현재 질문에 맞는 새로운 factual content를 답한다.
```

이다.

예:

```text
User:
"지금까지 뭐 담았어?"
→ completed history 설명

다음:
"여기서 뭐 고를 수 있어?"
→ 현재 option 설명

다음:
"뭐 추천해?"
→ recommendation 결과 설명
```

세 질문에 같은 과거 주문 요약을 반복하면 FAIL.

---

# 40. Response v1.2 — no-op factuality

`applied_this_turn`이:

```json
{
  "order_changes": {},
  "restriction_added": [],
  "restriction_removed": [],
  "preference_added": [],
  "preference_removed": []
}
```

이면:

```text
"저장했어요"
"변경했어요"
"적용했어요"
```

라고 말하면 안 된다.

오늘 실제 로그에서 이런 오류가 여러 번 있었다.

---

# 41. Response v1.2 — structured value fidelity

정확히 읽는다.

```text
low
→ 적게 / 조금

normal
→ 보통

high
→ 많이
```

예:

```text
recommendation_result:
치즈=high
```

Response:

```text
"치즈를 많이..."
```

여야 한다.

```text
"치즈를 보통..."
```

이면 FAIL.

---

# 42. multi-item amount attribution

예:

```text
양파=low
버섯=high
```

이면:

```text
양파 조금, 버섯 많이
```

라고 각각 정확하게 말한다.

```text
양파와 버섯을 보통
```

처럼 하나의 양으로 뭉개지 않는다.

---

# 43. Response v1.2 — clarify priority

Decision:

```text
clarify=true
```

Policy:

```text
reason=understanding
```

이면 먼저:

```text
현재 발화를 이해하지 못했다는 사실
```

을 설명한다.

pending이 남아있더라도
pending 질문만 기계적으로 반복하지 않는다.

예:

```text
"방금 말씀은 정확히 이해하지 못했어요.
다시 말씀해 주세요."
```

필요한 경우만:

```text
"기존 양파/버섯 실행 확인은 그대로 남아 있어요."
```

를 추가 가능.

---

# 44. Response v1.2 — rejected pending factuality

pending이 실제로:

```text
null
```

이 됐으면:

```text
"확인이 남아 있어요"
```

라고 말하면 안 된다.

현재 structured state가 history보다 우선이다.

---

# 45. Response v1.2 — contextual carry-over verbalization

네가 원하는 핵심.

예:

```text
earlier:
"매운 걸 못 먹어"

current:
extra

Recommendation:
치즈
```

Response:

```text
"아까 매운 걸 못 드신다고 하셨는데,
여기서는 페퍼론치노 대신 치즈가 좋을 것 같아요.
치즈를 넣어드릴까요?"
```

처럼 자연스럽게 연결한다.

---

# 46. carry-over verbalization은 매번 반복하지 않는다

같은 preference를 3턴 연속:

```text
"아까 ~~라고 하셨는데"
```

반복하면 FAIL.

필요할 때 한 번 자연스럽게 연결하고
이후에는 context상 불필요하면 생략한다.

---

# 47. restriction carry-over verbalization

예:

```text
earlier:
게살 allergy

meat 진입:
```

Response:

```text
"게살 알레르기가 있다고 하셨으니 게살은 제외할게요.
소시지는 드릴까요?"
```

이런 식의 안전한 recall을 학습한다.

단 canonical restriction이 존재할 때만 말한다.

---

# 48. preference carry-over verbalization

예:

```text
earlier:
꾸덕한 걸 좋아함

extra:
```

Response:

```text
"꾸덕한 식감을 좋아한다고 하셨으니
치즈를 추가해보는 건 어떠세요?"
```

단 Recommendation 결과/structured facts가 그 제안을 실제로 뒷받침할 때만 말한다.

Response가 새로운 추천을 창작하지 않는다.

---

# 49. Response는 Recommendation 결과를 설명할 뿐 새 proposal을 만들지 않는다

매우 중요.

Response:

```text
Recommendation Agent proposal
+ confirmed structured context
```

를 자연스럽게 설명.

Response 스스로:

```text
치즈 추천
크림 추천
양파 추천
```

을 새로 결정하지 않는다.

---

# 50. Response general factuality

일반대화에서도 없는 관계/상황을 창작하지 않는다.

실제 로그:

```text
"박지수가 말을 안 들어"
```

→

```text
"육아는 힘들죠"
```

같은 근거 없는 relation inference는 피한다.

정상:

```text
"말을 잘 안 들어서 답답하신가 보네요."
```

처럼 사용자가 말한 범위에서 답한다.

---

# 51. Response dataset 규모

v1 base를 새 input contract로 migration한 뒤
balanced supplement를 만든다.

권장 supplement:

```text
700~1000 rows
```

family:

```text
current-turn priority           120
no-op factuality               100
structured amount fidelity     100
clarify + pending               80
pending reject/state truth      70
status/options/recommendation   90
conversation recall             70
contextual carry-over          120
general factuality              60
regression rehearsal           100
```

---

# 52. Long-context carry-over dataset은 세 Agent에 모두 포함

다음 길이 다양성을 가진다.

```text
1 turn delay
3 turn delay
5 turn delay
8~12 turn delay
```

중간에:

```text
unrelated conversation
other preference
restriction add
preference removal
recommendation
execution
section transition
```

이 섞이게 한다.

---

# 53. carry-over invalidation도 학습

예:

```text
Turn 1:
"매운 거 못 먹어"

Turn 6:
"이제 매운 거 괜찮아"
→ restriction/preference semantics 변경

Turn 10:
extra
```

old preference를 계속 사용하면 안 된다.

최신 confirmed state가 항상 우선.

---

# 54. conflicting preference

예:

```text
"꾸덕한 건 좋아하지만 치즈는 싫어"
```

이면:

```text
꾸덕함 때문에 치즈를 추천
```

하면 안 된다.

restriction/specific dislike가 generic preference보다 우선.

---

# 55. Dataset contamination 방지

기존:

```text
validation
test
challenge
holdout
long-horizon
real robot failure exact utterance
```

를 train에 그대로 복사하지 않는다.

실제 failure의 의미만 추출해서
paraphrase/minimal pair를 만든다.

원문은 frozen real-world regression으로 남긴다.

---

# 56. real robot frozen suite 생성

오늘 로그에서:

```text
success
Decision failure
Recommendation failure
Response failure
ASR noisy
carry-over
pending
restriction
amount mismatch
repeated response
```

를 추출한다.

파일:

```text
real_robot_frozen_v1.jsonl
```

실제 원문은 train 금지.

---

# 57. Decision validation gate

학습 전:

```text
schema invalid = 0
legacy field = 0
contract mismatch = 0
duplicate = 0
frozen overlap = 0
migration unresolved = 0
```

semantic:

```text
skip/commit contradiction = 0
strong restriction lifecycle contradiction = 0
preference polarity suspicious = 0
target hallucination suspicious = 0
pending contradiction = 0
```

---

# 58. Recommendation validation gate

```text
schema invalid = 0
scope violation = 0
past-section proposal = 0
restriction conflict = 0
exact duplicate = 0
frozen overlap = 0
one-of violation = 0
no-op-only proposal = 0
```

---

# 59. Response validation gate

```text
schema invalid = 0
frozen overlap = 0
fact contradiction = 0
amount mismatch = 0
false applied claim = 0
pending contradiction = 0
execution contradiction = 0
```

추가 static audit:

```text
same-input duplicate target
stale-history target
unsupported invented fact
```

---

# 60. high-risk review

학습 전 최소:

```text
Decision 300 rows
Recommendation 300 rows
Response 300 rows
```

고위험 샘플 수동/semantic review.

집중:

```text
restriction
skip
carry-over
ASR
pending
amount
recommendation scope
no-op
response repetition
```

---

# 61. validation PASS면 즉시 학습

다음이 모두 PASS:

```text
RUNTIME V4 PASS
DECISION DATA PASS
RECOMMENDATION DATA PASS
RESPONSE DATA PASS
```

이면 추가 사용자 승인 요청 없이 바로 학습한다.

---

# 62. 기존 training environment 재사용

기존 Decision v3.2 / Response v1 성공 학습 설정과
Docker/TAR/scripts를 먼저 읽는다.

임의로 새 환경을 만들지 않는다.

Base:

```text
/home/roma/Desktop/sLLM/gemma-4-12B-it
```

기존 실제 successful config를 우선.

---

# 63. 세 Adapter 독립 학습

각각:

```text
Decision v3.3
Recommendation v1
Response v1.2
```

독립 QLoRA.

checkpoint 공유/혼합 금지.

---

# 64. Epoch

각 role:

```text
최대 6 epoch
```

각 epoch 저장.

validation metric + loss 기록.

---

# 65. Decision best epoch

priority:

```text
schema
contract safety
fake mutation
precision
recall
skip_current
commit
restriction semantics
preference polarity
ASR hallucination
pending
route/query intent
validation loss
```

---

# 66. Recommendation best epoch

priority:

```text
scope compliance
restriction compliance
past-section violation
one-of constraint
current/future relevance
carry-over relevance
preference alignment
partial revise
no-op
validation loss
```

---

# 67. Response best epoch

validation loss만으로 고르지 않는다.

priority:

```text
1. factual contradiction = 0
2. execution contradiction = 0
3. pending contradiction = 0
4. amount fidelity
5. no-op factuality
6. current-turn relevance
7. contextual carry-over correctness
8. stale/repeated response rate
9. general factuality
10. naturalness
11. validation loss
```

---

# 68. Response stale/repetition metric

단순 exact string repeat만 보지 않는다.

다음 각각 측정:

```text
exact stale repeat
semantic stale repeat
history-copy response
current-question unanswered
false repeated state update
```

예:

```text
Q1 상태 질문
Q2 선택지 질문

두 답변이 같은 과거 요약
→ stale repeat
```

---

# 69. Decision A/B

```text
v3.2
vs
v3.3
```

동일 migrated frozen + real robot frozen.

---

# 70. Recommendation A/B

```text
Base Gemma
vs
Recommendation v1
```

동일 structured input.

---

# 71. Response A/B

```text
Response v1
vs
Response v1.2
```

반드시 **동일 structured upstream input을 replay**해서 평가한다.

Decision/Recommendation 결과를 매번 새로 생성해서
Response 비교를 오염시키지 않는다.

---

# 72. Agent isolation A/B

Decision 비교:

```text
Decision v3.2 + Recommendation Base + Response v1
vs
Decision v3.3 + Recommendation Base + Response v1
```

Recommendation 비교:

```text
same Decision replay
Recommendation Base
vs
Recommendation v1
```

Response 비교:

```text
same upstream structured context
Response v1
vs
Response v1.2
```

각 component의 효과를 분리한다.

---

# 73. 최종 E2E 조합

최종 candidate:

```text
Decision winner
+
Recommendation winner
+
Response winner
```

Baseline:

```text
Decision v3.2
+
Recommendation Base
+
Response v1
```

---

# 74. Offline long-horizon 평가

실물 테스트가 당분간 어렵기 때문에
long-horizon을 강화한다.

최소:

```text
50+ sessions
각 10~20 turns
```

다음 포함:

```text
long-context carry-over
restriction add/remove
preference add/remove
multiple pending
recommendation revise
skip
ASR noise
general interlude
state query
option query
recommendation query
```

---

# 75. carry-over 전용 long-horizon suite

예:

```text
Turn 1:
매운 거 못 먹어

Turn 2~7:
다른 주문 진행

Turn 8:
extra 진입

Turn 9:
뭐 넣는 게 좋아?
```

기대:

```text
매운맛 회피 기억
페퍼론치노 제외
치즈 suggestion
Response가 자연스럽게 과거 context 연결
```

---

# 76. carry-over overuse test

반대도 평가한다.

```text
매운맛 회피 기억
+
현재 veggie 질문
```

에서:

```text
"아까 매운 걸 못 먹는다고..."
```

를 불필요하게 꺼내면 penalty.

즉:

```text
remember when relevant
ignore when irrelevant
```

둘 다 평가.

---

# 77. Response repeat regression

long-horizon에서:

```text
연속 exact duplicate
semantic duplicate
same stale summary repeated
```

를 계산한다.

단 실제로 같은 질문을 다시 했고
같은 답이 정답인 경우는 오류로 치지 않는다.

질문의 intent가 바뀌었는데 같은 답을 재사용했을 때 failure.

---

# 78. Runtime adapter switching smoke

최종 candidate:

```text
Decision
→ Recommendation
→ Response
→ Decision
→ Response
→ VLM
→ Recommendation
→ Response
```

반복.

adapter activation errors = 0.

---

# 79. Adoption은 독립적으로

가능:

```text
ADOPT Decision v3.3
ADOPT Recommendation v1
ADOPT Response v1.2
```

또는:

```text
ADOPT Decision v3.3
KEEP Recommendation Base
ADOPT Response v1.2
```

등 독립 판정.

새 버전이라는 이유로 강제 채택 금지.

---

# 80. Response v1.2 adoption gate

최소:

```text
frozen factual failure <= v1
pending contradiction <= v1
execution contradiction = 0
amount fidelity > v1
stale repeat < v1
no-op false claim < v1
carry-over correctness 유의미한 향상
long-horizon direct Response failure < v1
```

v1.1처럼 targeted set만 크게 좋아지고
independent frozen/long-horizon이 나빠지면 탈락.

---

# 81. Recommendation v1 adoption gate

```text
scope violation = 0
past-section violation = 0
restriction violation = 0
one-of accuracy > Base
carry-over relevance > Base
partial revise > Base
overall usefulness > Base
```

---

# 82. Decision v3.3 adoption gate

최소 기존 v3.2의 장점을 잃지 않는다.

baseline:

```text
Frozen precision ≈ 94.82%
Frozen recall ≈ 97.89%
Long precision ≈ 98.08%
Long recall = 100%
Decision direct failure = 6
```

새 contract 때문에 단순 숫자 직접 비교가 어려운 항목은
migrated matched subset에서 비교한다.

목표:

```text
old semantics regression 최소
+
new skip/restriction/carry-over semantics 안정적
```

---

# 83. Runtime default는 A/B 통과 후 변경

학습 완료만으로:

```text
model_runtime.py
```

default adapter를 즉시 바꾸지 않는다.

A/B PASS 후에만 selected Adapter를 연결한다.

---

# 84. 최종 runtime smoke

최종 조합으로:

```text
정상 order
skip
restriction add
restriction remove pending
restriction remove accept/reject
recommendation
recommendation partial revise
carry-over
status query
option query
general question
ASR unclear
no-op turn
```

을 테스트.

---

# 85. 반드시 만들어야 하는 최종 산출물

```text
upgrade_v4/
├── CURRENT_CONTRACT_AUDIT.md
├── REAL_LOG_FAILURE_AUDIT.md
├── contract_migration.md
├── evaluation_contract_migration.jsonl
├── real_robot_frozen_v1.jsonl
│
├── decision_v3_3/
│   ├── dataset/
│   ├── validation_report.md
│   ├── training/
│   ├── epoch_metrics.json
│   └── best_checkpoint.json
│
├── recommendation_v1/
│   ├── dataset/
│   ├── validation_report.md
│   ├── training/
│   ├── epoch_metrics.json
│   └── best_checkpoint.json
│
├── response_v1_2/
│   ├── dataset/
│   ├── validation_report.md
│   ├── training/
│   ├── epoch_metrics.json
│   └── best_checkpoint.json
│
├── runtime/
│   ├── fault_injection_report.md
│   └── runtime_regression.json
│
├── evaluation/
│   ├── decision_ab.json
│   ├── recommendation_ab.json
│   ├── response_ab.json
│   ├── carry_over_eval.json
│   ├── long_horizon_eval.json
│   ├── e2e_ab.json
│   └── failure_analysis.jsonl
│
└── FINAL_UPGRADE_REPORT.md
```

---

# 86. FINAL_UPGRADE_REPORT

반드시 간단한 최종 표를 먼저 제공한다.

```text
Runtime v4:
PASS / FAIL

Decision:
baseline = v3.2
candidate = v3.3 epochX
verdict = ADOPT / KEEP

Recommendation:
baseline = Base
candidate = v1 epochY
verdict = ADOPT / KEEP

Response:
baseline = v1
candidate = v1.2 epochZ
verdict = ADOPT / KEEP

Final combination:
...

READY FOR ROBOT TEST:
YES / NO
```

---

# 87. 최종적으로 반드시 답할 질문

### Decision

```text
사용자 의미를 v3.2보다 더 정확히 이해하는가?
skip_current를 안정적으로 구분하는가?
negative preference를 뒤집지 않는가?
깨진 ASR에서 없는 target을 창작하지 않는가?
```

### Recommendation

```text
현재 section과 미래 scope를 제대로 사용하는가?
과거 section을 다시 추천하지 않는가?
restriction을 지키는가?
사용자의 오래된 관련 preference를 적절한 순간에 활용하는가?
```

### Response

```text
현재 질문에 실제로 답하는가?
직전 답변을 재탕하지 않는가?
state 변화가 없는데 변경했다고 거짓말하지 않는가?
low/normal/high를 정확히 말하는가?
과거의 관련 preference/restriction을 자연스럽게 다시 연결하는가?
관련 없는 과거 기억은 굳이 꺼내지 않는가?
```

---

# 88. 가장 중요한 최종 원칙

```text
Meaning
→ Model

Recommendation reasoning
→ Recommendation Model

User-facing natural explanation
→ Response Model

Physical truth / safety / irreversible state
→ Python Runtime
```

그리고 이번 upgrade의 핵심 UX는:

```text
사용자가 전에 말한 것을
단순히 저장하는 시스템
```

이 아니라:

```text
사용자가 전에 말한 것을 기억하고,
그 정보가 실제로 필요한 순간이 왔을 때
자연스럽게 다시 활용하는 시스템
```

이다.

예:

```text
"아까 매운 걸 못 드신다고 하셨는데,
지금 추가 재료에서는 페퍼론치노를 빼고
치즈로 드릴까요?"
```

이런 대화가 특별한 hardcoded 예외가 아니라
일반화된 multi-turn behavior가 되도록 학습한다.

---

# 89. 작업 종료 조건

다음까지 완료해야 작업 완료다.

```text
runtime contract 변경
dataset migration
dataset generation
validation
training
best epoch selection
component A/B
long-horizon
carry-over evaluation
runtime adapter routing
final smoke
final adoption report
```

Dataset만 만들고 종료하지 않는다.

Validation PASS 후에는 바로 학습까지 수행한다.

단 실패한 Adapter를 억지로 runtime에 채택하지 않는다.
