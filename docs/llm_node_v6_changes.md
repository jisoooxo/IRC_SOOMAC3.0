# LLM Node V6 변경 기록

참고:

- [`llm_node_v5_tool_behavior_and_followup.md`](./llm_node_v5_tool_behavior_and_followup.md) — V5의 8개 Tool 동작과 확인된 개선 항목이다.
- [`agent_v6.py`](../src/soomac_irc/soomac_irc/agent_v6.py) — V6 transaction, section skip proposal, commit 로직이다.
- [`call_model_v6.py`](../src/soomac_irc/soomac_irc/call_model_v6.py) — V6 Tool adapter와 구조화된 자유응답 생성 코드이다.
- [`llm_node_v6.py`](../src/soomac_irc/soomac_irc/llm_node_v6.py) — V6 응답 조립과 사용자 재확인 처리 코드이다.
- [`test_tool_transitions_v6.py`](../src/soomac_irc/test/test_tool_transitions_v6.py) — 8개 Tool과 새 skip 분기의 상태 전이 테스트이다.
- [`test_response_followup_v6.py`](../src/soomac_irc/test/test_response_followup_v6.py) — 응답 조립과 section 제외 승인·거부 테스트이다.

→ **V6는 V5를 보존한 별도 파일이며, 확인 전 상태 변경과 선택지 0개 dead-end를 제거하고 자유응답 형식을 한 필드로 제한한다.**

## 1. V6 파일 분리

V5 파일은 수정하지 않았다. 다음 파일을 V6로 따로 만들었다.

| V6 파일 | 기준 파일 | 역할 |
|---|---|---|
| `order_v6.py` | `order_v5.py` | 메뉴와 restriction 정본이다. 현재 동작 변경은 없다. |
| `agent_v6.py` | `agent_v5.py` | section skip proposal과 commit, 모든 선택지 제한 처리를 추가했다. |
| `call_model_v6.py` | `call_model_v5.py` | 자유응답 JSON schema와 parser를 추가했다. |
| `llm_node_v6.py` | `llm_node_v5.py` | V6 module import, pending skip 승인·거부 처리를 적용했다. |

## 2. `refuse_section` 확인 전 상태 변경 수정

### V5 문제

```text
사용자: 야채는 전부 싫어요.
로봇: 현재 단계 재료를 모두 제외하고 다음 단계로 넘어갈까요?
```

V5는 질문을 하기 전에 양파·버섯 dislike restriction과 기존 야채 선택 제거를 먼저 적용했다. 사용자가 `아니`라고 해도 적용된 상태가 원복되지 않았다.

### V6 동작

```text
1. refuse_section 검증
2. section_skip proposal만 저장
3. 주문·restriction·추천 상태는 그대로 유지
4. 사용자 응답이 "응"이면 proposal commit
5. 사용자 응답이 "아니"이면 proposal 폐기
```

예시는 다음과 같다.

```text
기존 주문: 양파 적게
사용자: 야채는 전부 싫어요.
로봇: 현재 단계 재료를 모두 제외하고 다음 단계로 넘어갈까요?
사용자: 아니
결과: 양파 적게 유지, dislike restriction 없음, 야채 단계 유지
```

```text
기존 주문: 양파 적게
사용자: 야채는 전부 싫어요.
로봇: 현재 단계 재료를 모두 제외하고 다음 단계로 넘어갈까요?
사용자: 응
결과: 양파·버섯 dislike 적용, 양파 선택 제거, 야채 skip, 육류 단계 이동
```

Cause: V5의 `refuse_section`이 재확인 여부를 계산하기 전에 restriction과 주문 제거를 commit했다.

Effect: V6의 dislike 전체 제외는 사용자 승인 전까지 `order`, `execution`, `recommendation`을 바꾸지 않는다.

## 3. 모든 선택지가 restriction으로 사라진 경우

### V5 문제

양파와 버섯을 각각 제한하면 안내는 `야채 단계를 제외할지 말씀해 주세요`라고 했지만 실제 `awaiting_confirm` 상태가 없었다. 사용자가 `응`이라고 말하면 일반 `confirm_section`으로 들어가 필수 야채 누락으로 막힐 수 있었다.

### V6 동작

현재 section에 선택된 재료가 없고 모든 선택지가 제한되면 `all_options_restricted` skip을 생성한다.

| 제한 구성 | V6 처리 |
|---|---|
| 모든 재료가 알레르기·섭취 불가·식단 규칙으로 차단 | 안전상 고를 수 없으므로 현재 section을 즉시 skip한다. |
| 하나라도 dislike로 차단 | 다시 선택할 수 있는 소프트 제약이므로 section skip을 사용자에게 재확인한다. |
| 현재 section에 이미 명시적으로 선택한 재료가 있음 | 자동 skip을 만들지 않고 현재 선택 확정 흐름을 유지한다. |

예시는 다음과 같다.

```text
사용자: 양파 알레르기고 버섯도 못 먹어요.
결과: 두 restriction 적용 → 야채 즉시 skip → 육류 단계 이동
```

```text
사용자: 양파도 싫고 버섯도 싫어요.
로봇: 현재 선택 가능한 야채가 없어요. 야채 단계를 제외할지 말씀해 주세요.
사용자: 응
결과: 야채 skip → 육류 단계 이동
```

```text
사용자: 아니
결과: 야채 단계 유지 → dislike 재료를 다시 명시적으로 선택할 수 있다고 안내
```

## 4. `cancel_order` dead branch 제거

현재 제품 정책은 전체 주문 취소 미지원이다. 따라서 `cancel_order`는 계속 다음처럼 동작한다.

```text
빈 주문: 아직 선택된 주문이 없어요.
선택 후: 전체 주문 취소는 지원하지 않아요.
로봇 시작 후: 이미 담기 시작해서 전체 주문을 취소할 수 없어요.
```

V6에서는 절대로 생성되지 않는 `awaiting_confirm.type == "cancel_order"` 처리와 UI 상태 문구를 제거했다. 실제 취소 기능은 새 제품 정책 없이는 추가하지 않았다.

Cause: `cancel_validation.needs_confirmation`은 항상 `False`인데 과거 확인 분기만 남아 있었다.

Effect: 현재 계약은 “취소 요청을 안전하게 거절하고 상태를 유지한다”로 코드와 일치한다.

## 5. 구조화된 자유응답

V6의 자유응답 모델은 다음 schema만 생성할 수 있다.

```json
{
  "explanation": "토마토 소스는 산뜻한 맛이에요."
}
```

`explanation` 외의 필드는 허용하지 않는다. XGrammar가 생성 단계에서 JSON 구조를 제한하고, `parse_free_reply()`가 정확히 한 필드인지 다시 확인한다.

최종 발행 전에는 V5의 문장 필터도 유지한다.

```text
자유응답 JSON 생성
  → explanation 추출
  → 질문 문장 제거
  → 반영·확정·작업 시작·다음 단계 주장 제거
  → Python이 계산한 실제 다음 행동 추가
```

Cause: 프롬프트 지시만으로는 자유응답 모델의 출력 역할을 100% 고정할 수 없다.

Effect: V6는 출력 구조를 `explanation`으로 제한하고, 의미상 운영 문장은 기존 중앙 후처리에서 한 번 더 제거한다.

## 6. 8개 Tool 상태 전이 테스트

`test_tool_transitions_v6.py`는 모델을 호출하지 않고 다음을 직접 검증한다.

| Tool | 검증 내용 |
|---|---|
| `set_order` | 주문값만 반영하고 작업을 시작하지 않는다. |
| `set_order_and_confirm` | 현재 section이 완전하고 차단이 없을 때만 확정을 허용한다. |
| `refuse_section` | dislike는 확인 전 상태를 보존하고 승인 후에만 commit한다. |
| `recommend_order` | 실제 주문을 바꾸지 않고 proposal만 만든다. |
| `confirm_section` | 필수값이 갖춰진 현재 section만 허용한다. |
| `cancel_order` | pending 확인 없이 차단하고 주문을 보존한다. |
| `respond` | 주문과 실행 상태를 바꾸지 않는다. |
| `describe_scene` | 주문과 실행 상태를 바꾸지 않는다. |

추가로 모든 hard restriction 즉시 skip, 모든 dislike 재확인, 자유응답 parser의 추가 필드 거부를 검증한다.

`test_response_followup_v6.py`는 V5의 응답 회귀 8개에 다음 두 사례를 추가한다.

- dislike skip 거부 시 원래 주문과 restriction 유지
- dislike skip 승인 시 proposal commit 후 다음 section 이동

## 7. V6 실행 방법

V6도 package module로 실행해야 한다.

```bash
source /opt/ros/humble/setup.bash
cd /home/roma/IRC_SOOMAC3.0/src/soomac_irc
/home/roma/miniconda3/envs/gemma4_env/bin/python -m soomac_irc.llm_node_v6
```

`llm_node_v6.py`의 기본 기능 플래그는 V5와 동일하다.

```python
ENABLE_VLM = True
ENABLE_VLM_UI_IMAGES = True
ENABLE_TOOL_LORA = True
ENABLE_RUNTIME_LOG = True
```

## 8. 이번 V6 범위에 포함하지 않은 항목

- 실제 전체 주문 취소 기능은 제품 정책이 정해지지 않아 구현하지 않았다.
- `describe_scene`의 camera frame freshness 검사는 현재 주문 응답 문제와 별도이므로 구현하지 않았다.
- GPU 모델을 실제로 올리는 통합 실행은 사용자가 GPU를 건드리지 말라고 지정했으므로 수행하지 않았다.

## 용어 정리

- **proposal**: 사용자 확인 전에는 실제 상태에 반영되지 않는 변경 후보이다.
- **commit**: 검증과 사용자 확인이 끝난 변경을 실제 주문 상태에 한 번에 반영하는 단계이다.
- **hard restriction**: 알레르기, 섭취 불가, 식단 규칙처럼 임의로 재선택하면 안 되는 제한이다.
- **soft restriction**: 사용자가 명시적으로 다시 선택하면 허용할 수 있는 dislike 제한이다.
- **dead branch**: 현재 상태 계약에서는 진입할 수 없지만 과거 구현에서 남은 조건 분기이다.
- **구조화된 출력(structured output)**: 모델이 정해진 JSON schema 형태로만 출력하도록 생성 문법을 제한하는 방식이다.
