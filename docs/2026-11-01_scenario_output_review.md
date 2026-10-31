# 2026-11-01 closed-loop 시나리오 출력 검수

참고: [`manual_scenario_stack_probe.py`](../src/soomac_irc/test/manual_scenario_stack_probe.py), `/tmp/soomac_closed_loop_20261101_v2/{PLAN.json,turns.jsonl,RESULT.json,EXIT.json}`  
→ 실제 D/Q/P 누적 대화의 첫 턴에서 추천 계약이 깨져 두 에피소드가 중단되었다.

## 실행 결과와 판정 범위

- `EXIT.json`: `exit_code=0`, GPU 추론 종료. `RESULT.json`: `field_execution_passed=false`.
- `compound`: 계획 6턴 중 1턴 관측, `initial` 실패. `split_control`: 계획 11턴 중 1턴 관측, 같은 `initial` 실패. **분리 발화 대조군은 복합 발화 성공으로 계산하지 않는다.**
- 각 에피소드는 실패한 턴 직후 멈춘다. 두 행의 D/P 출력은 실제 모델 출력이며 Q는 호출되지 않았다. 물리 완료는 실행이 승인된 뒤에만 주입하는 오프라인 fixture이지만, 이번 실행에는 승인된 턴과 주입된 물리 완료가 모두 0개다. 실제 ROS·VLM 관측은 없다.

## 행별 판정

| 에피소드·행 | 요구 | 관측·판정 |
| --- | --- | --- |
| compound `initial` | 얇은면 `normal` 확정, 느끼한 소스 추천으로 크림 미확정 후보 보관, 실행 없음 | **실패.** 면 종류·양 확정과 실행 없음은 통과했으나 크림 후보·추천 pending이 없다. |
| compound `cream_execute` | 크림 선택과 즉시 실행 | **미도달.** |
| compound `vegetables` | 양파·버섯 모두 `low`, 실행 없음 | **미도달.** |
| compound `mushroom_execute` | 버섯만 `normal`로 변경·즉시 실행, 양파 `low` 유지 | **미도달.** |
| compound `allergy_sausage_execute` | 게살 알레르기와 소시지 `high` 보관·즉시 실행 | **미도달.** |
| compound `cheese_execute` | 페퍼론치노 제외, 치즈 `low` 선택·즉시 실행 | **미도달.** |
| split_control `initial` | 첫 턴은 compound와 동일 | **실패.** 동일한 D/P 출력과 상태. 나머지 10턴은 모두 **미도달**. |

## 첫 턴의 D → 상태 → P 증거

1. D 입력 `message`에는 “면은 얇은면으로 보통 양으로 주고, 소스는 느끼한 걸로 추천해줘.”가 온전히 들어갔다. `recent_history`에는 인사만 있고, 초기 `order`와 `pending`은 비어 있었다. 따라서 **입력 전달 누락은 아니다.**
2. D raw는 `{"order":{"noodle_portion":"normal","noodle_type":"얇은면"},"route":"task"}`이다. 정규화 후 `recommendation.action="none"`, `commit=false`다. **원인:** D가 명시적 소스 추천 요청을 출력에 담지 않았다. **효과:** Q 호출 0회, 크림 후보와 추천 pending 모두 생성되지 않았다.
3. 실제 상태는 `noodle_type="얇은면"`, `noodle_portion="normal"`, `sauce=null`, `pending=null`이다. 적용 이력에는 면 종류·양만 기록되었다. `policy.execute=false`, `next_prompt=null`, `execution_authorized=false`다. 면 수량은 `normal`로 보존되었고, 토핑 수량·제한은 해당 턴에 요청되지 않았다.
4. P 입력에는 `recommendation_result=null`, `pending=null`, `next_prompt=null`, `confirmed_order.sauce=null`이 명시된다. 그런데 P raw 답변은 “느끼한 소스는 크림을 추천해요. 이대로 진행할까요?”이다. **원인:** P가 검증된 추천 후보가 없는데 크림을 새로 제안하고, 유효한 확인 대상 없이 진행 질문을 만들었다. **효과:** 발행 답변과 실제 상태가 어긋나 다음 턴의 지시 대상을 오염시킬 수 있다. 이 턴에서 “이미 그 양”, 실제 담기, 완료를 주장한 출력은 관측되지 않았다.

## 용어 정리

- **확정 주문(confirmed order)**: 현재 저장된 면·소스·재료·수량이다.
- **추천 pending**: 아직 확정하지 않은 후보와 그 수락 질문이다. 이번 실행에서 생성되지 않았다.
- **오프라인 물리 완료 fixture**: 승인된 실행 뒤 시험 코드가 `completed_tasks`와 `task_completed`를 넣는 방식이다. 실제 로봇 완료 증거가 아니다.

판정: **복합 closed-loop 불합격.** 현재 관측으로 즉시 실행·양파 수량 유지·알레르기 적용·페퍼론치노 제외의 성공이나 실패를 주장할 수 없다. 다음 검증은 D의 첫 턴 추천 누락과 P의 후보 없는 추천·확인 질문을 고친 뒤 동일 시나리오를 처음부터 다시 실행하는 것이다.

## 후속: 17개 발화 전행 측정 (`soomac_closed_loop_all_20261101_v2`)

참고: `/tmp/soomac_closed_loop_all_20261101_v2/{PLAN.json,turns.jsonl,RESULT.json,EXIT.json}`  
→ 아래 내용은 위의 **첫 실패에서 중단한 실행**과 별개다. `--measure-all`로 모든 원래 발화를 넣었으며, 이전 모델의 실제 주문·history·로봇 단계만 다음 턴에 넘겼다. 기대 주문은 검사 기준으로만 계산하고 모델 상태에 주입하지 않았다.

`EXIT.json`은 `exit_code=0`, `RESULT.json`은 `compound=6/6`·`split_control=11/11` 발화 관측, 두 에피소드 `passed=false`, `field_execution_passed=false`다. **17개 모두 D 1회, Q 0회, P 1회**였고 모델 오류는 없었다. 17개 모두 `pending` 전후 `null`, P 입력의 `recommendation_result=null`, `execution_authorized=false`, `starting_now=[]`, 실제 실행 계획 `[]`였다. history 접두와 그 턴의 사용자·assistant 쌍은 모두 보존되었다. 첫 면 실행 실패 뒤 **실제 로봇 section은 17개 내내 `noodle`**이었다. 따라서 뒤의 `veggie/meat/extra` 발화는 예상 단계와 다르게 처리되었으며, 이 차이를 고쳐 넣지 않았다.

뒤 행에서 관측한 실행 실패와 D의 commit 누락은 **면 단계에 계속 머문 상태에서 나온 결과**다. 특히 육류·추가 토핑의 본래 단계에서 같은 실패가 독립적으로 발생한다고 입증한 시험은 아니다.

**크림을 자연어로 제안한 문장 자체는 사용자 요구에 맞고 유용하다.** 문제는 D가 추천 요청을 Q로 보내지 않아 후보·pending이 저장되지 않은 것이다. 응답을 억제하는 방향으로 해석하지 않는다. 아래의 `담을까요/진행할까요` 문제는 해당 질문의 구조화된 확인 대상이 없다는 뜻이다.

### 복합 발화 6행

각 행의 `D raw`는 실제 모델 JSON의 의미 필드를 그대로 적었다. `실제`의 주문값은 그 턴 **이후** 값이며, 적지 않은 기존값은 그대로다. `계획`은 기대 실행 대상 → 실제 생성 계획이다.

| # · 실제 발화 | D raw / 요청과 실제 주문·실행·계획 | P 실제 응답 / 의미 판정 |
| --- | --- | --- |
| C1 `initial`: “면은 얇은면으로 보통 양으로 주고, 소스는 느끼한 걸로 추천해줘.” | `order={noodle_type:얇은면,noodle_portion:normal}`, `route=task`; 추천·commit 없음. 실제 면 `얇은면/normal`, 소스 `null`, 후보·pending `null`, 실행 `false`; 계획 `[]→[]`. | “면은 얇은면으로 보통 양을 선택했어요. 느끼한 소스는 크림을 추천해요. 이대로 진행할까요?” 면 사실과 크림 제안의 자연스러움은 적절하나, 크림 후보가 저장되지 않았고 확인 대상도 없다. |
| C2 `cream_execute`: “크림으로 선택할게. 바로 담아.” | `commit=true`만 출력, `order.sauce` 없음. 요청 크림 선택·면 2회 즉시 실행 → 실제 소스 `null`, 면 `normal` 유지, `missing_order(sauce)`, 실행 `false`; 계획 `noodle_thin×2→[]`. | “소스는 크림으로 선택했어요. 이대로 담을까요?” **크림 확정 주장이 거짓**이며 P 입력은 소스 `null`, `next_prompt=missing_order(sauce)`였다. |
| C3 `vegetables`: “양파랑 버섯 둘 다 적게 담아줘.” | `order.toppings={양파:low,버섯:low}`, commit 없음. 두 수량은 실제 `low/low`, 실행 `false`; 계획 `[]→[]`. 예상 `veggie`, 실제 `noodle`; 소스는 계속 `null`. | “양파와 버섯 모두 적은 양으로 선택했어요. 이대로 담을까요?” 선택 사실은 맞지만 실행 pending·`next_prompt` 없는 담기 확인 질문이다. |
| C4 `mushroom_execute`: “버섯만 보통 양으로 바꿔서 바로 담아.” | `order.toppings.버섯=normal`, `commit=true`. 실제 버섯 `low→normal`, 양파 `low` 유지, 소스 `null`, `missing_order(sauce)`, 실행 `false`; 예상 `veggie`/실제 `noodle`, 계획 `onion×1 + mushroom×2→[]`. | “버섯은 보통 양으로 바꿨어요. 아직 소스가 선택되지 않았어요. 소스를 말씀해주세요.” 변경·소스 누락 설명은 실제 상태와 맞고 담기 완료 주장은 없다. |
| C5 `allergy_sausage_execute`: “게살 알레르기가 있어서 소시지만 많이 바로 담아줘.” | `restrictions=add(게살,allergy)`, `order.toppings.소시지=high`, **commit 없음**. 제한·`high`는 실제 반영, 소스 `null`, 실행 `false`; 예상 `meat`/실제 `noodle`, 계획 `sausage×3→[]`. | “게살 알레르기 제한을 추가했어요. 소시지는 많이 선택되어 있어요. 이대로 담을까요?” 제한·수량은 맞지만 이번 턴 새 선택임을 드러내지 않는다. 구조화된 실행 질문도 없다. |
| C6 `cheese_execute`: “페퍼론치노는 빼고 치즈만 적게 골라서 바로 담아.” | `order.toppings={페퍼론치노:none,치즈:low}`, **commit 없음**. 페퍼론치노는 전에도 미선택, 실제 치즈 `low`, 기존 양파 `low`·버섯 `normal`·소시지 `high`·게살 제한 유지, 실행 `false`; 예상 `extra`/실제 `noodle`, 계획 `cheese×1→[]`. | “페퍼론치노는 제외하고 치즈를 적은 양으로 선택했어요. 이대로 담을까요?” 주문 선택 설명은 맞다. 실행 pending 없이 담기 확인을 묻고 즉시 실행 요청은 충족하지 못한다. |

### 분리 발화 대조군 11행

대조군은 **복합 발화의 성공을 대체하지 않는다.** 여기서도 실제 section은 전행 `noodle`, 소스는 전행 `null`이다. 첫 행의 자연스러운 크림 제안과 백엔드 후보 누락은 C1과 같다.

| # · 실제 발화 | D raw / 요청과 실제 주문·실행·계획 | P 실제 응답 / 의미 판정 |
| --- | --- | --- |
| S1 `initial`: “면은 얇은면으로 보통 양으로 주고, 소스는 느끼한 걸로 추천해줘.” | `order={noodle_type:얇은면,noodle_portion:normal}`, 추천·commit 없음. 실제 면 `얇은면/normal`, 소스·후보·pending `null`, 실행 `false`; 계획 `[]→[]`. | “면은 얇은면으로 보통 양을 선택했어요. 느끼한 소스는 크림을 추천해요. 이대로 진행할까요?” 자연스러운 크림 제안이 구조화된 후보와 맞지 않는다. |
| S2 `cream_select`: “크림으로 선택할게.” | `confirmation=accept`만 출력, `order.sauce` 없음. 수락할 pending도 없으므로 실제 소스 `null`, 실행 `false`; 계획 `[]→[]`. | “소스는 크림으로 선택했어요. 이대로 진행할까요?” **확정 주문에 없는 크림을 선택했다고 잘못 말한다.** 실행 pending도 없다. |
| S3 `cream_execute`: “바로 담아.” | `commit=true`. 실제 소스 `null`, 면 `normal`, `missing_order(sauce)`, 실행 `false`; 계획 `noodle_thin×2→[]`. | “아직 소스가 선택되지 않았어요. 원하는 소스를 말씀해주세요.” 현재 상태 설명은 맞고, 앞선 잘못된 선택 주장과 충돌한다. |
| S4 `vegetables`: “양파랑 버섯 둘 다 적게 담아줘.” | `order.toppings={양파:low,버섯:low}`, commit 없음. 실제 `low/low`, 실행 `false`; 예상 `veggie`/실제 `noodle`, 계획 `[]→[]`. | “양파와 버섯 모두 적은 양으로 선택했어요. 이대로 담을까요?” 선택은 맞고, 담기 질문에는 pending·`next_prompt`가 없다. |
| S5 `mushroom_change`: “버섯만 보통 양으로 바꿔줘.” | `order.toppings.버섯=normal`, commit 없음. 실제 버섯 `low→normal`, 양파 `low` 유지, 실행 `false`; 예상 `veggie`/실제 `noodle`, 계획 `[]→[]`. | “버섯을 보통 양으로 바꿨어요. 양파는 적은 양 그대로예요. 이대로 담을까요?” 수량 보존 설명은 정확하다. 담기 질문은 pending이 없다. |
| S6 `mushroom_execute`: “바로 담아.” | `order.toppings.버섯=normal`, `commit=true`. 기존 수량 유지, `missing_order(sauce)`, 실행 `false`; 예상 `veggie`/실제 `noodle`, 계획 `onion×1 + mushroom×2→[]`. | “아직 소스가 선택되지 않았어요. 원하는 소스를 말씀해주세요.” 실제 차단 원인을 맞게 설명한다. |
| S7 `allergy`: “게살 알레르기가 있어요.” | `restrictions=add(게살,allergy)`, `order.toppings.게살=none`. 실제 게살은 애초 미선택이고 강한 제한 추가, 실행 `false`; 예상 `meat`/실제 `noodle`, 계획 `[]→[]`. | “게살 알레르기 제한을 추가했어요. 현재 주문에는 게살을 피해야 해요.” 제한 사실과 일치한다. |
| S8 `sausage_select`: “소시지만 많이 골라줘.” | `order.toppings.소시지=high`, 기존 `게살 allergy`를 `add`로 재출력. 실제 소시지 `high` 신규 선택, 제한은 중복 없이 유지, 실행 `false`; 예상 `meat`/실제 `noodle`, 계획 `[]→[]`. | “소시지를 많이 선택했어요. 이대로 담을까요?” 새 선택 설명은 맞다. pending 없는 담기 질문이다. |
| S9 `sausage_execute`: “바로 담아.” | `commit=true`. 소시지 `high`·게살 제한 유지, `missing_order(sauce)`, 실행 `false`; 예상 `meat`/실제 `noodle`, 계획 `sausage×3→[]`. | “아직 소스가 선택되지 않았어요. 원하는 소스를 말씀해주세요.” 상태 설명은 맞다. |
| S10 `cheese_select`: “페퍼론치노는 빼고 치즈만 적게 골라줘.” | `order.toppings={페퍼론치노:none,치즈:low}`, commit 없음. 페퍼론치노는 이전에도 미선택, 치즈 `low` 신규 선택, 다른 수량·제한 유지, 실행 `false`; 예상 `extra`/실제 `noodle`, 계획 `[]→[]`. | “페퍼론치노는 선택에서 제외했고 치즈는 적은 양으로 선택했어요. 이대로 담을까요?” 선택은 맞고 담기 질문에는 pending이 없다. |
| S11 `cheese_execute`: “바로 담아.” | `order.toppings.치즈=low`, `commit=true`. 치즈 `low` 및 기존 수량·제한 유지, `missing_order(sauce)`, 실행 `false`; 예상 `extra`/실제 `noodle`, 계획 `cheese×1→[]`. | “아직 소스가 선택되지 않았어요. 원하는 소스를 말씀해주세요.” 실제 차단 원인을 설명한다. |

**공통 판정:** 두 에피소드의 최종 주문은 모두 `소스=null, 얇은면/normal, 양파=low, 버섯=normal, 소시지=high, 치즈=low, 게살 allergy`다. 페퍼론치노는 미선택이며 기존 양파 수량과 게살 제한은 보존되었다. 응답에 실제 투입·완료를 주장한 문장은 없다. C5의 “선택되어 있어요”는 이번 턴 새 선택이라는 시간 관계를 흐리고, C2/S2의 크림 확정은 실제 주문과 다르다. 사용자가 요청하지 않은 새 추천은 관측되지 않았다. 첫 크림 추천은 요청된 자연어 제안이며, 문제는 Q 미호출과 후보·pending 누락이다. 명시적 실행 요청 8건(C2/C4/C5/C6, S3/S6/S9/S11)은 모두 실행되지 않았고, 생성 계획·오프라인 물리 완료·단계 전환은 0건이다. 이 실행의 `RESULT.json`은 모든 행을 **측정했다**는 증거이지, 복합 또는 분리 경로를 통과했다는 증거가 아니다.
