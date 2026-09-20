# 2026-09-20 Tool·Reply 어댑터 분리 계획

참고:

- [`call_model_v7.py`](../src/soomac_irc/soomac_irc/call_model_v7.py) — 현재 Tool 어댑터 호출과 base model 자유응답 생성 코드이다.
- [`agent_v7.py`](../src/soomac_irc/soomac_irc/agent_v7.py) — Tool 결과를 검증하고 주문 상태 후보를 만드는 코드이다.
- [`llm_node_v7.py`](../src/soomac_irc/soomac_irc/llm_node_v7.py) — 실운영과 Debug가 함께 호출하는 최종 응답 함수와 자유응답 claim 방어 코드이다.
- [`llm_debug_v7.py`](../src/soomac_irc/soomac_irc/llm_debug_v7.py) — ROS·카메라 없이 Tool, 정책, Reply, 상태 변화를 기록하는 평가 환경이다.

→ **Tool과 Reply는 연결하되 한 번의 모델 출력으로 합치지 않는다. Tool 검증이 끝난 뒤 Reply 어댑터가 검증된 사실만 자연어로 표현하게 한다.**

## 1. 현재 구조

```text
사용자 발화
  → Tool LoRA: 행동과 인자를 JSON으로 추출
  → Python: Tool 검증·주문 상태 후보 생성
  → respond일 때 base model: 자유응답 생성
  → Python: 정책 문장·다음 행동 결합 및 claim 방어
  → 상태 commit
  → 발행
```

현재 `TOOL_SYSTEM`은 여덟 Tool 중 하나와 인자를 고르는 프롬프트이다. `FREE_REPLY_SYSTEM`은 `respond` Tool에서 질문 답변·설명·공감·잡담을 만드는 프롬프트이다. `TOOL_PROMPT_VERSION`과 `FREE_REPLY_PROMPT_VERSION`은 이 두 프롬프트를 실행 로그에서 구분하는 사람이 읽는 라벨이며 모델 동작을 바꾸지 않는다.

`llm_debug_v7`은 별도 Reply 구현을 갖지 않는다. VLM·카메라 경로만 사용하지 않고 `llm_node_v7.build_turn_reply()`와 `call_model_v7.make_generate_reply()`를 실운영과 동일하게 호출한다.

## 2. 목표 구조

```text
사용자 발화
  → Tool adapter
  → Python validation
  → response plan 생성
  → Reply adapter
  → Python claim guard
  → 상태 commit
  → 한 번 발행
```

Reply adapter에는 원시 Tool 출력이 아니라 다음 검증 완료 자료만 전달한다.

```json
{
  "user_text": "양파 적게 하고 진행해",
  "validated_transaction": {},
  "state_after_candidate": {},
  "mandatory_facts": ["양파를 적게로 반영함"],
  "forbidden_claims": ["실행되지 않은 작업을 완료했다고 말하지 않음"],
  "authoritative_next_action": "야채 담기 시작"
}
```

`authoritative_next_action`은 Python이 정본으로 보유한다. Reply adapter가 이 문장을 자연스럽게 만들더라도 최종 claim guard가 누락·변조를 검사하고, 실패하면 Python 고정 문장으로 대체한다.

Cause: Tool과 Reply를 한 번에 생성하면 Python이 Tool을 거부하거나 수정하기 전에 Reply가 반영·실행을 주장할 수 있다.

Effect: Tool 의미 분류 오류와 Reply 표현 오류를 따로 측정하면서도 최종 응답은 검증된 상태에 맞출 수 있다.

## 3. 어댑터 구성

같은 base model에 두 LoRA를 별도 이름으로 로드한다.

| 어댑터 | 입력 | 출력 | 책임 |
|---|---|---|---|
| `tool_adapter` | 현재 상태, 최근 대화, 사용자 발화 | Tool JSON | 행동과 인자 해석 |
| `reply_adapter` | 검증된 response plan | 구조화된 응답 초안 | 자연스러운 한국어 표현 |

Reply 출력은 자유 문자열 하나보다 역할을 나눈 JSON이 적합하다.

```json
{
  "acknowledgement": "양파를 적게로 반영했어요.",
  "explanation": "",
  "transition": "이제 야채 담기를 시작할게요."
}
```

Python은 필수 사실과 다음 행동이 보존됐는지 검사한 뒤 사용자에게는 세 필드를 합친 한 문장만 발행한다.

## 4. 데이터와 평가 계획

1. `llm_debug_v7`에서 실제 오분류와 할루시네이션을 `OK/BAD`로 표시한다.
2. BAD 턴마다 원래 발화, 상태, Tool 정답, response plan, Reply 정답을 저장한다.
3. Tool 오류와 Reply 오류를 분리해 각각의 학습 데이터로 만든다.
4. 고정 회귀 세트에서 Tool 정확도, 금지 claim 발생률, 필수 사실 누락률을 측정한다.
5. 회귀 세트를 통과한 뒤 실제 ROS 노드에 어댑터 전환을 연결한다.

초기에는 데이터 개수 자체보다 현재 여덟 Tool, 추천 확인, 부정·모호 응답, 이미 담긴 재료, 미래 section 추천 등 위험 상황이 모두 포함됐는지가 중요하다.

## 5. 구현 단계와 예상 시간

| 단계 | 산출물 | 예상 시간 |
|---|---|---:|
| 1 | `ResponsePlan` 계약과 claim guard 테스트 | 1~2시간 |
| 2 | 독립 LLM Debug UI와 `OK/BAD` 기록 | 1~2시간 |
| 3 | 로그를 Tool·Reply 학습 레코드로 변환 | 2~4시간 |
| 4 | Reply LoRA 학습·오프라인 회귀 평가 | 반나절~2일 |
| 5 | V7 runtime 어댑터 전환 및 통합 테스트 | 2~4시간 |

## 용어 정리

- **Tool adapter**: 사용자 발화를 여덟 Tool 중 하나와 구조화된 인자로 바꾸는 LoRA이다.
- **Reply adapter**: 검증된 처리 결과를 사용자에게 들려줄 자연스러운 문장으로 바꾸는 LoRA이다.
- **response plan**: Reply가 반드시 말할 사실, 말하면 안 되는 주장, 다음 행동을 Python이 구조화한 데이터이다.
- **claim guard**: 생성 문장이 실제 상태보다 앞서 반영·시작·완료를 주장하지 않는지 검사하는 최종 방어 계층이다.
- **authoritative next action**: 다음 로봇 행동에 관한 정본이며 모델이 아니라 Python 정책이 결정한다.
