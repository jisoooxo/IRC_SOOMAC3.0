# D→Q→P 대화의 실제 흐름 독립 검수

[17턴 계측 JSON](2026-11-01_scenario_measurements.json) · [시나리오 회귀 기록](2026-11-01_llm_scenario_regression.md) · [출력 검수](2026-11-01_scenario_output_review.md) · [대화 그래프](../src/soomac_irc/soomac_irc/llm_langgraph.py)
→ **첫 추천 요청이 D 출력에서 사라져 Q가 호출되지 않았고, P의 말과 저장된 주문이 갈라진 뒤 실행이 막혔다.**

작성일: 2026-11-01 KST. 기존 `/tmp/soomac_closed_loop_all_20261101_v2/` 결과와 현재 관련 LLM 코드를 읽은 독립 검수이다. 새 추론·전수감사·데이터 수정·학습·ROS 실행은 하지 않았다. 아래 `C`는 복합 6턴, `S`는 분리 대조 11턴이다. 두 에피소드는 각각 새 세션이며 각 에피소드 안에서 실제 주문과 대화 이력이 누적됐다.

## 1. 사용자 요청과 합격 기준

사용자가 기대한 대화는 `얇은면/보통 양 선택 + 느끼한 소스 추천 요청 → Q의 크림 후보 저장 → 크림 수락 → 현재 주문 실행`이다. 크림을 자연스럽게 추천하는 말은 요구에 맞는다. **문제는 그 말에 대응하는 검증된 후보와 다음 수락 대상이 실제 상태에 없었다는 점**이다. `담아줘`를 선택 요청으로 해석해 실행을 보류할 수 있는 턴과, `바로 담아`라는 명시적 즉시 실행을 한 턴에 처리해야 하는 경우도 구분한다.

원인: [D 입력 생성](../src/soomac_irc/soomac_irc/decision_model.py)은 현재 발화·이력·주문·pending을 함께 전달하지만, C1/S1의 실제 D raw에는 면 선택만 있었다.

결과: [그래프의 추천 분기](../src/soomac_irc/soomac_irc/llm_langgraph.py)는 `recommendation.action=request/revise`일 때만 Q를 호출하므로 첫 요청에서 Q가 호출되지 않았다.

## 2. 시험 경계와 확인한 기술 상태

`docs/2026-11-01_scenario_measurements.json`의 17행 모두 실제 `robot section=noodle`, `pending` 전후 `null`, D 1회·Q 0회·P 1회, `execution_authorized=false`, `starting_now=[]`, `actual_plans=[]`이다. 원본 `EXIT.json`은 `exit_code=0`, 모델 오류 검사는 17행 모두 `true`이다. 이는 **계측이 기술적으로 끝났다는 뜻**이며 대화 의미가 통과했다는 뜻은 아니다. 합성 물리 완료와 단계 전환도 0건이고 실물 ROS 호출은 없었다.

후반 예상 단계 `veggie/meat/extra`는 오라클의 목표 단계다. 실제 단계는 끝까지 `noodle`이었다. 그러므로 아래 후반 행은 **면 단계에 머문 상태에서의 진단**이지 각 재료의 정상 단계 성능을 단정하는 근거가 아니다. [회귀 기록 §2·§5](2026-11-01_llm_scenario_regression.md)은 실패 후 정답 주문·강제 `commit`·강제 단계 이동을 모델 세션에 넣지 않았다고 명시한다.

병렬 작업 중 [대화 뷰 HTML](2026-11-01_conversation_view.html)과 [빌더](build_conversation_view.py)가 생성됐다. 이 검수에서는 읽기만 했다. HTML의 내장 `measurement-data`는 계측 JSON과 **객체 전체가 일치**하고, 내장 `supplement-data`는 [별도 보강 뷰 JSON](2026-11-01_scenario_supplement_view.json)과 **객체 전체가 일치**한다. 내장 JavaScript의 `node --check`는 exit 0이고 미치환 템플릿 표식도 없다. 화면의 정적 요약 `D17/Q0/P17`, `execute 0/17`, 실제 `noodle`·소스 없음은 17턴과 일치한다. **읽기 검수에서 HTML의 생성·자료 연결·스크립트 문법 오류는 발견되지 않았다.** 브라우저 상호작용·렌더링 검증은 이 작업 범위가 아니며 주담당이 별도로 진행한다.

## 3. 전체 17턴의 필드·발화 대조

표의 필드는 실제 `before_order → after_order`이고, `∅`는 해당 확정값 또는 수락 가능한 pending이 없다는 뜻이다. 수량은 실제 저장값 `low/normal/high`를 그대로 적었다. 모든 행의 `sauce`는 `null`, `pending`은 `null→null`, 실제 plan은 `[]`이므로 차이가 생기는 곳에 다시 적었다. D의 `commit`이 raw에 없으면 정규화 후 `false`이다. 인용한 P 문장은 실제 발행 답변이다.

### 복합 발화 C1~C6

| 턴·실제 발화 요지 | D와 실제 필드 전→후 | 실행·Response 대조 |
| --- | --- | --- |
| **C1** 얇은면 보통 + 느끼한 소스 추천 | D raw `order={noodle_type:얇은면,noodle_portion:normal}`, 추천 없음. 면 `∅→얇은면/normal`; 소스 `null→null`; 후보·pending 없음 | Q 미호출·실행 없음. P “크림을 추천해요. 이대로 진행할까요?”는 자연스러운 제안이지만 저장 후보·확인 대상이 없다. |
| **C2** “크림으로 선택할게. 바로 담아.” | D raw `commit=true`만 있음. 소스 `null→null`, 면 `얇은면/normal` 유지 | `missing_order(sauce)`로 실행 차단, 기대 면 계획 `thin×2→[]`. P “소스는 크림으로 선택했어요”는 실제 확정 주문과 다르다. |
| **C3** 양파·버섯 둘 다 적게 | D `양파=low,버섯=low`, `commit` 없음. 둘 다 `∅→low` | 선택값 반영·실행 없음은 목표와 맞는다. 실제 단계는 `noodle`; P의 “이대로 담을까요?”에는 execution pending·`next_prompt`가 없다. |
| **C4** 버섯만 보통으로 바꿔 바로 담아 | D `버섯=normal`+`commit=true`. 버섯 `low→normal`, 양파 `low→low` | `missing_order(sauce)`로 차단, 기대 채소 계획 `onion×1+mushroom×2→[]`. P는 변경과 소스 누락을 정확히 말한다. 실제 단계는 `noodle`이다. |
| **C5** 게살 알레르기 + 소시지만 많이 바로 | D `게살 allergy` 추가·`소시지=high`, **`commit` 누락**. 제한 `∅→게살 allergy`, 소시지 `∅→high` | 즉시 실행 의도가 구조화되지 않았다. 실행 없음·실제 `noodle`; 기대 `sausage×3→[]`. P는 제한·수량을 맞게 말하지만 “선택되어 있어요”가 이번 새 선택을 흐리고 확인 대상 없는 담기 질문을 한다. |
| **C6** 페퍼론치노 제외 + 치즈 적게 바로 | D `페퍼론치노=none,치즈=low`, **`commit` 누락**. 치즈 `∅→low`, 페퍼론치노는 계속 미선택 | 실행 없음·실제 `noodle`; 기대 `cheese×1→[]`. P의 선택 설명은 맞지만 pending 없는 “담을까요?”가 즉시 실행 요청을 대신했다. |

### 분리 대조 S1~S11

| 턴·실제 발화 요지 | D와 실제 필드 전→후 | 실행·Response 대조 |
| --- | --- | --- |
| **S1** C1과 같은 첫 요청 | 면 `∅→얇은면/normal`, 소스 `null→null`; D 추천 없음 | Q 미호출·후보 없음. P는 크림을 추천하고 진행을 묻는다. |
| **S2** “크림으로 선택할게.” | D raw `confirmation=accept`만 있음. 소스 `null→null`, 수락할 pending 없음 | 실행 없음. P “소스는 크림으로 선택했어요”는 실제 주문과 다르다. |
| **S3** “바로 담아.” | D `commit=true`; 면 유지, 소스 `null→null` | `missing_order(sauce)`, 기대 `thin×2→[]`. P는 소스가 없다고 정확히 설명한다. |
| **S4** 양파·버섯 둘 다 적게 | D 두 재료 `∅→low`, `commit` 없음 | 선택값 보존·실행 없음. P 수량은 맞고 담기 질문에는 pending이 없다. 실제 단계 `noodle`. |
| **S5** 버섯만 보통으로 변경 | D 버섯 `low→normal`; 양파 `low→low` | P “버섯을 보통 양으로 바꿨어요. 양파는 적은 양 그대로예요”는 변경과 유지 모두 맞다. 담기 질문은 pending 없음. |
| **S6** “바로 담아.” | D `버섯=normal` 재출력+`commit=true`; 실제 버섯·양파 수량은 그대로 | `missing_order(sauce)`, 기대 `onion×1+mushroom×2→[]`; P는 소스 누락을 정확히 설명한다. |
| **S7** “게살 알레르기가 있어요.” | D `게살=none`+제한 추가; 제한 `∅→게살 allergy`, 게살은 미선택 유지 | 제약 보존·실행 없음. P는 게살을 피해야 한다고 맞게 말한다. 실제 단계 `noodle`. |
| **S8** 소시지만 많이 선택 | D 소시지 `∅→high`; 게살 allergy를 재출력했지만 제한은 한 번만 저장 | P “소시지를 많이 선택했어요”는 새 선택과 양이 맞다. 담기 질문은 pending 없음. |
| **S9** “바로 담아.” | D `commit=true`; 소시지 `high→high`, 게살 allergy 유지 | `missing_order(sauce)`, 기대 `sausage×3→[]`; P는 소스 누락을 정확히 설명한다. |
| **S10** 페퍼론치노 제외·치즈 적게 선택 | D `페퍼론치노=none,치즈=low`; 치즈 `∅→low`, 다른 수량·제약 유지 | P의 선택 설명은 맞다. 담기 질문은 pending 없음. 실제 단계 `noodle`. |
| **S11** “바로 담아.” | D `치즈=low` 재출력+`commit=true`; 치즈 `low→low`, 페퍼론치노 미선택 | `missing_order(sauce)`, 기대 `cheese×1→[]`; P는 소스 누락을 정확히 설명한다. |

양쪽 최종 확정 주문은 `얇은면/normal`, `소스=null`, `양파=low`, `버섯=normal`, `소시지=high`, `치즈=low`, `게살 allergy`로 같다. 페퍼론치노는 선택되지 않았다. 원시 D·P와 필드별 근거는 [계측 JSON의 `turns`](2026-11-01_scenario_measurements.json) 및 [출력 검수 §5](2026-11-01_scenario_output_review.md)에 있다.

## 4. 처음 어긋난 지점과 독립 누락

```text
사용자: 면 선택 + 느끼한 소스 추천
  ↓ D: 면 patch만 출력, recommendation.action=none
  ↓ Q: 호출되지 않음 → 크림 candidate/pending 없음
  ↓ P: 크림을 말하고 진행 질문 → 말과 구조 상태 불일치
사용자: 크림 수락(+즉시 실행)
  ↓ D: C2는 commit만, S2는 confirmation=accept만 → sauce patch 없음
  ↓ Graph: pending 없음 → S2 수락 대상 없음; sauce=null 유지
  ↓ commit 턴: missing_order(sauce) → execute=false → section=noodle 유지
```

**첫 원인과 연쇄:** C1/S1에서 추천 요청이 D raw에 없었다. [그래프의 `route_prepared_task`](../src/soomac_irc/soomac_irc/llm_langgraph.py)는 D가 `request/revise`를 내야 Q로 간다. [Q 결과 적용](../src/soomac_irc/soomac_irc/llm_langgraph.py)은 검증된 proposal만 `recommendation` pending에 둔다. Q가 0회였으므로 P가 언급한 크림은 저장된 후보가 아니다. C2는 `commit`만 내고 명시적 크림 patch를 놓쳤다. S2는 `accept`를 냈지만 [pending 해소](../src/soomac_irc/soomac_irc/llm_langgraph.py)는 pending이 `null`이면 바로 반환한다. 이후 `commit=true`가 있어도 [필수 주문 검사](../src/soomac_irc/soomac_irc/llm_langgraph.py)가 `sauce` 누락을 막는다.

원인: D의 첫 추천 누락과 후속 크림 선택 누락이 합쳐져 필수 소스가 끝까지 비었다.

결과: C2/C4/S3/S6/S9/S11의 `commit=true` 6건도 안전 검사에서 실행되지 않았고, 뒤 턴이 실제로는 면 단계에 쌓였다. **기존 `commit` 처리 로직의 수정으로 첫 D 추천 누락이 해결됐다고 해석할 수 없다.** [pending Runtime 수정 기록](2026-11-01_llm_pending_runtime_fix.md)은 올바른 Decision 주입에 대한 코드 회귀이며 이 실제 D raw 실패를 해결했다는 증거가 아니다.

**별도 관측 누락:** C5와 C6에는 `sauce=null`이라는 연쇄와 별개로, 명시적 “바로 담아”가 있는데 D raw에 `commit`이 없다. 이 사실은 실제 `noodle` 문맥에서의 D 필드 누락이다. 두 턴에 `commit`만 보충해도 같은 상태의 소스 누락과 단계 불일치가 남으므로 **그 시나리오가 실행됐을 것이라고 단정하지 않는다.** C3·S4의 두 `low` 반영, C4·S5의 `low→normal` 변경과 양파 `low` 보존, S7의 알레르기 저장은 독립적인 긍정 사례다.

**Response의 별도 불일치:** C1/S1에서 자연어 크림 추천은 사용자 요구에 맞지만 `recommendation_result=null`, `pending=null`, `next_prompt=null`인 P 입력과 연결되지 않았다. C2/S2의 “크림으로 선택했어요”는 `confirmed_order.sauce=null`과 명백히 충돌한다. C3/C5/C6/S4/S5/S8/S10의 “담을까요?”에는 실행 확인 대상이 없다. 반면 C4 및 S3/S6/S9/S11은 실제 `missing_order(sauce)`를 설명했다. **실제 투입·완료를 허위로 주장한 문장은 관측되지 않았다.** [P 입력 생성](../src/soomac_irc/soomac_irc/llm_langgraph.py)은 확정 주문과 pending/next_prompt를 넘겼으며, [P prompt](../src/soomac_irc/soomac_irc/agent_prompts.py)도 후보 없는 새 추천·대상 없는 실행 질문·확정되지 않은 주문 주장을 금지한다. 따라서 C2/S2는 P 입력 누락이 아니라 P의 사실 준수 실패로 분류한다. C1/S1의 추천 질문은 D→Q 연결 실패와 P의 구조 사실 준수 실패가 함께 있다.

## 5. 수량·데이터 보강 판단과 불확실성

**확인된 수량:** 면 양 `normal`, 양파 `low`, 버섯 `low→normal`, 소시지 `high`, 치즈 `low`는 실제 주문 상태에 반영됐다. 토핑의 `low/normal/high`를 틀린 값으로 덮거나 담기 완료라고 말한 사례는 이 17턴에서 보이지 않는다. S5의 변경/유지 설명과 S8의 새 `high` 설명도 맞다. 이는 이전 다른 세션의 수량 오류가 완전히 해결됐다는 뜻은 아니다.

**보강 필요 판단:** 이 실제 실패는 D→Q→P 연결 후보와 D/P 대조 데이터를 **국소적으로 보강해 검수할 근거**가 된다. 우선 같은 첫 발화에서 `면 patch+추천 request`, 후보가 있을 때와 없을 때의 크림 수락, 크림 patch+즉시 실행, `low→normal` 변경+실행, 알레르기+소시지 선택+실행, 없는 pending에서의 P 답변을 양성·음성 쌍으로 검수하는 것이 타당하다. [잔여 작업 계획 §3·§4](<IRC LLM Final remaining work.md>)의 기존 ledger·split·HOLD를 재사용하는 범위이며 이 문서에서 새 학습행을 만들지 않는다. 병렬 생성된 HTML에는 별도 오라클 리플레이 13턴·22개 타깃이 **비활성·미학습 자료**로 표시돼 있고, 내장 JSON에도 `training_run=false`, `loader_connected=false`, `model_success=false`라고 적혀 있다. 이것은 실제 17턴 모델 출력을 고친 증거가 아니다. **이 증거만으로 재학습이 반드시 필요하다거나 문제가 전부 학습 데이터 때문이라고 확정할 수 없다.** 보강안 또는 비활성 후보가 있다는 사실은 현재 adapter의 재학습·교체·성능 개선 완료를 뜻하지 않는다. 현재 17턴에 사용된 것은 지정된 기존 best adapter이고 이 실행의 `training=false`이다.

**아직 모르는 것:** Q는 한 번도 호출되지 않아 이 입력에서 Q가 실제로 크림을 제안·검증할지 관측하지 못했다. C5/C6을 정상 `meat/extra` 단계와 소스가 채워진 주문에서 다시 평가한 결과도 없다. D가 첫 추천과 해당 복합 `commit`을 왜 놓쳤는지의 내부 모델 원인은 **원인 미상**이다. 프롬프트, 데이터 양, 이전 assistant 발화의 영향은 분리 실험이 없어 특정하지 못한다. 새 대조 데이터를 학습하면 개선되는지도 미측정이다.

원인: 현재 증거는 실제 실패 지점과 성공 필드를 보여 주지만, 교정된 D/Q/P 흐름이나 새 adapter의 출력을 포함하지 않는다.

결과: 이 문서의 결론은 **실패 분석과 보강 후보 판단**까지이며 새 계약·모델의 수락 판정은 아니다.

## 6. 검수 종료 기록

계측 JSON의 17행 원시 호출·실제 전후 필드·P 입력 핵심 사실, 회귀 기록, 관련 LLM 코드 경로를 읽었다. `RESULT.json`/`EXIT.json`과 새 HTML의 내장 JSON·정적 표시·스크립트 문법을 확인했다. 이 검수에서는 파일 `docs/2026-11-01_conversation_visual_review.md`만 생성했다. 실제 모델·GPU·ROS·학습·원격 변경은 실행하지 않았다.

## 용어 정리

| 용어 | 여기서의 뜻 |
| --- | --- |
| D / Q / P | 결정(Decision) / 추천(Recommendation) / 응답(Response) 모델이다. D가 추천을 요청해야 사용자 턴의 Q 경로가 열린다. |
| 후보(candidate) | Q가 검증을 거쳐 제안한 미확정 주문 변경이다. P의 자연어 문장만으로 후보가 생기지 않는다. |
| 확정 주문(confirmed order) | 그래프가 검증·저장한 현재 주문이다. `sauce=null`이면 크림이 아직 확정되지 않았다. |
| 대기 확인(pending) | 추천 수락이나 현재 주문 실행처럼 다음 사용자 답을 연결할 구조화된 대상이다. `null`이면 `accept`가 적용될 후보가 없다. |
| 실행 의도(commit) / 실행 허가(execute) | D의 `commit=true`는 지금 실행 요청이라는 뜻이다. 필수 주문·제약·pending·로봇 상태 검사를 통과해야 `execute=true`가 된다. |
| 예상 단계 / 실제 단계 | 대본의 목표 구간과 로봇이 실제 도달한 구간이다. 이번 17턴의 실제 단계는 모두 `noodle`이다. |

(내가 의역해보자면, 손님에게는 “크림 어때요?”라고 말했지만 계산서에는 크림 후보가 쓰이지 않았다. 그래서 “그걸로요”가 가리킬 항목이 없었고, “바로 담아”도 소스 없는 주문이라 시작할 수 없었다.)
