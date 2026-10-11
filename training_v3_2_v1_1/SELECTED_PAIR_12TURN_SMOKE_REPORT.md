# Decision v3.2 + Response v1 — 12턴 멀티턴 Smoke Test

- 실행일: 2026-10-11
- 실행 환경: 실제 GPU inference
- Decision: `decision_adapter_v3_2_best_epoch4`
- Response: `response_adapter_v1_best_epoch1`
- Recommendation / VLM: base model
- 시나리오: `pending_interlude`, 단일 세션 12턴

## 결론

```text
Decision 의미 해석        12 / 12
주문·취향 상태 보존       12 / 12
자동 Response 판정        10 / 12
수동 Response 판정         9 / 12
Adapter switching error    0
턴 latency median          3111.421 ms
턴 latency max             3936.973 ms
```

**Decision, canonical state, pending accept/reject는 정상이다.**

Response에서는 pending reminder 2건과 주문 상태 항목 누락 2건이 확인되었다. 같은 턴에 두 문제가 함께 발생한 경우가 있어 실패 턴은 총 3개이다.

## 12턴 결과

| Turn | User | Decision / 상태 변화 | Response | 판정 |
|---:|---|---|---|---|
| 1 | 양파 많이 넣어줘 | `양파=high`, execution pending 생성 | 양파를 많이 선택했어요. 지금 담을까요? | PASS |
| 2 | 근데 지금까지 뭐 들어갔어? | 상태 조회, pending 유지 | 지금까지 양파를 많이 선택했고 현재 채소 단계예요. 실행 확인이 남아 있어요. 지금 담을까요? | PASS |
| 3 | 양파는 어떻게 보관해야 해? | `route=general`, pending 유지 | 양파는 서늘하고 통풍이 잘되는 곳에 보관하는 게 좋아요. | **FAIL: pending reminder 누락** |
| 4 | 응 아까 그걸로 해 | pending accept, `execute=true` | 양파 많이 담기를 진행해요. | PASS |
| 5 | 버섯 조금 넣어줘 | `버섯=low`, execution pending 생성 | 버섯을 조금 선택했어요. 지금 담을까요? | PASS |
| 6 | 아니 버섯 실행은 취소해 | pending reject, 선택값은 유지 | 버섯 담기는 진행하지 않아요. | PASS |
| 7 | 담백한 맛을 좋아해 | preference `담백하게` 추가 | 담백하게 선호도를 저장했어요. | PASS |
| 8 | 현재 주문 상태 알려줘 | 상태 조회, canonical order에 `양파=high`, `버섯=low` 존재 | 현재 양파 많이 선택과 담백한 선호도가 저장되어 있고 채소 단계예요. | **FAIL: 버섯 선택 누락** |
| 9 | 양파는 조금으로 바꿔줘 | `양파=low`, execution pending 생성 | 양파를 조금으로 바꿨어요. 지금 담을까요? | PASS |
| 10 | 현재 주문도 알려주고 버섯 특징도 설명해줘 | `route=mixed`, pending 유지 | 현재 양파 조금 선택과 담백한 선호도가 저장되어 있고 채소 단계예요. 버섯은 식재료에 따라 맛과 식감이 달라져요. | **FAIL: 버섯 선택 및 pending reminder 누락** |
| 11 | 응 그 양파로 진행해 | pending accept, `execute=true` | 양파 조금 담기를 진행해요. | PASS |
| 12 | 파스타라는 말의 뜻이 뭐야? | `route=general`, 상태 변경 없음 | 파스타는 이탈리아 요리로 면과 소스를 곁들여 먹는 음식을 뜻해요. | PASS |

## 오류 1 — pending 중 일반 질문 후 확인 복귀 누락

Turn 3에서 execution pending은 정상적으로 보존되었다.

```text
pending_after.type = execution
pending_after.targets = [양파]
```

하지만 Response는 양파 보관법만 답했고, 남아 있는 실행 확인 질문을 다시 제시하지 않았다.

Expected:

```text
양파는 서늘하고 통풍이 잘되는 곳에 보관하는 게 좋아요.
아까 선택한 양파를 지금 담을까요?
```

Cause: Response v1이 general 답변 뒤 `next_prompt`의 execution 목적을 응답에 반영하지 않았다.

Effect: pending은 안전하게 유지되지만 사용자는 확인이 남아 있다는 사실을 놓칠 수 있다.

## 오류 2 — 상태 조회에서 선택 재료 누락

Turn 8의 canonical order에는 다음이 모두 존재했다.

```json
{
  "toppings": {
    "양파": "high",
    "버섯": "low"
  }
}
```

Response는 양파와 preference만 말하고 버섯을 누락했다.

Cause: Response v1이 상태 조회에서 canonical order 전체를 열거하지 않았다.

Effect: 내부 상태는 맞지만 사용자에게 불완전한 주문 상태를 전달한다.

## 오류 3 — mixed 질문에서 상태와 pending을 동시에 누락

Turn 10은 다음 두 요청을 포함한 mixed 발화이다.

```text
1. 현재 주문 상태 조회
2. 버섯 특징 설명
```

Response는 버섯 특징에는 답했지만 다음을 누락했다.

```text
canonical order의 버섯 low
남아 있는 execution pending 확인 질문
```

Cause: Response v1이 mixed 답변에서 설명 내용은 생성했지만 주문 상태와 `next_prompt`를 완전하게 결합하지 못했다.

Effect: 복합 질문에서 대화 진행 목적과 일부 canonical fact가 응답에서 빠진다.

## Safety 및 상태 일관성

다음은 모두 정상이다.

- pending 중 general/mixed 질문이 들어와도 pending state는 삭제되지 않았다.
- accept 전에는 `execute=false`, accept 후에만 `execute=true`였다.
- reject 후에는 실행되지 않았다.
- preference와 order state는 12턴 동안 유지되었다.
- Adapter switching error는 0건이다.

따라서 이번 실패는 **Decision 또는 physical safety 오류가 아니라 Response progression/fact completeness 오류**이다.

## 테스트 한계

이번 검사는 LLM-only 실행이다. ROS task lifecycle과 실제 로봇 완료 이벤트는 연결하지 않았다.

실제 로봇에서는 작업 실행 중 STT가 차단되므로 Turn 4 직후 바로 다음 사용자 발화가 들어오는 흐름은 제한된다. 또한 robot completed state가 갱신되지 않았으므로 물리 완료 이후 mutation protection은 이번 세션의 검사 범위가 아니다.

## 판정

```text
Decision / canonical state: PASS
Pending lifecycle:          PASS
Physical execution gate:    PASS
Response completeness:      FAIL (3 / 12 turns)
```

현재 조합은 주문 상태와 실행 안전성은 유지하지만, **pending 중 일반·mixed 질문 및 전체 주문 조회 Response를 보완해야 한다.**
