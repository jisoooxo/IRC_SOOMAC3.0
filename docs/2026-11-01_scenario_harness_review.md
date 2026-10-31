# 시나리오 하네스 독립 검토 (2026-11-01)

대상: `src/soomac_irc/test/manual_scenario_stack_probe.py`의 정적 검토. 모델·GPU·ROS는 실행하지 않았다. 기준은 현재 `llm_langgraph.py`, `llm_policy.py`, `domain.py`와 실제 `llm_langgraph_node.py`의 작업 변환이다.

## 확인된 계약

- 누적 compound 입력은 면 얇은면 보통·크림 추천 → 크림 수락과 즉시 실행 → 양파·버섯 적게 선택만 → 버섯 보통으로 변경하고 즉시 실행 → 게살 알레르기와 소시지 많이 즉시 실행 → 페퍼론치노 제외와 치즈 적게 즉시 실행 순서이다. 별도 split control도 존재한다. 근거: `manual_scenario_stack_probe.py:45-68`.
- 각 턴은 이전 `session`을 다음 `graph.invoke`에 넘기고, 기대 주문은 이전 주문의 사본에 해당 턴의 변경만 적용한다. 첫 실패에서 해당 에피소드를 중단하며 주문을 gold 값으로 덮지 않는다. 근거: `manual_scenario_stack_probe.py:71-88,114-143`; `agent_contract.py:265-280`.
- 면·야채·육류·추가 재료의 계획은 `section_execution_items`와 실제 노드의 `AMOUNT_TO_COUNT`·`TOPIC_CLASS_NAMES`를 사용한다. 실제 노드도 같은 함수를 통해 작업을 만들고 동일한 클래스·횟수로 plan을 발행한다. 근거: `manual_scenario_stack_probe.py:37-42,145-152`; `llm_policy.py:95-113`; `llm_langgraph_node.py:64-83,1153-1182`.
- 완료 통지와 단계 전환은 하네스가 삽입하는 **합성 물리 fixture**이며, 결과에 `actual_ros: false`가 남는다. 인사와 단계 안내는 실제 노드가 남기는 assistant history를 재현한다. 합성 user event는 없다. 근거: `manual_scenario_stack_probe.py:104-108,145-161,176-190`; `llm_langgraph_node.py:648-650,742-752,878-884`.

## 합격 판정의 빈틈

1. **정확한 수량·plan은 판정하지 않는다.** `plans`는 현재 주문과 같은 변환 함수로 생성해 기록할 뿐 기대 목록과 비교하지 않는다(`manual_scenario_stack_probe.py:123-126,145-150,162`). 원인: order 비교는 양 문자열을 확인하지만 실제 계획의 `noodle_thin ×2`, `onion ×1`, `mushroom ×2`, `sausage ×3`, `cheese ×1`을 별도 oracle로 검사하지 않는다. 효과: 계획 생성 결과가 비어 있지 않기만 하면 잘못된 횟수·순서에도 `passed: true`가 가능하다.
2. **pending/commit의 단계별 의미를 판정하지 않는다.** 처음 크림 후보의 `type`과 `sauce`만 확인하고, 나머지 턴은 `policy.execute`만 비교한다(`manual_scenario_stack_probe.py:123-130`). 원인: 야채 선택 뒤 execution pending의 대상과 확정 주문 수량, split control의 선택 뒤 pending, 즉시 실행 턴의 Decision `commit`, 실행 후 pending 해소에 대한 assert가 없다. 실제 그래프는 pending을 만들고 갱신한다(`llm_langgraph.py:477-501,999-1013`). 효과: 실행 보류 이유가 잘못되거나 실행 의도 추출이 틀려도 주문과 execute bool만 맞으면 통과할 수 있다.
3. **대화와 응답의 의미 검토는 아직 수동이다.** 검사는 이전 history의 prefix와 trace의 error 유무만 본다(`manual_scenario_stack_probe.py:123-126,131-137`). 원인: 이번 user/assistant 쌍과 추천·수량·실행 안내 문장의 내용은 합격 조건이 아니다. 그래프는 쌍을 추가하지만(`llm_langgraph.py:1112-1120`), 모델 응답이 실제 주문·실행과 일치하는지는 별도 확인이 필요하다. 효과: `field_execution_passed: true`는 D/Q/P 자연어 정확도 통과로 읽을 수 없다. `RESULT.json`의 `response_semantics: REQUIRES_INDEPENDENT_REVIEW` 표시는 이 한계를 정확히 드러낸다(`manual_scenario_stack_probe.py:189-190`).

**판정:** 입력·상태 누적·실패 중단·합성 물리 경계·노드 계획 변환은 요구와 일치한다. 현재 `passed`는 주문 상태와 실행 허용의 좁은 검사이며, 정확한 pending/commit/plan 수량 및 Response 의미까지 포함한 전체 시나리오 합격 판정은 아니다.

## 추가 검토: `--measure-all` 변경분

이 절은 위의 초기 검토 뒤 추가된 측정 모드만 다룬다. 특히 위의 “정확한 수량·plan은 판정하지 않는다”와 “이번 user/assistant 쌍을 검사하지 않는다”는 지적은 **새 측정 모드에서 보완되었다**.

1. **의미 실패 뒤에도 원래 발화를 계속 측정한다.** `--measure-all`이면 실패를 `failed_turns`에 남기고 다음 `steps` 항목으로 진행한다. 실제 `session`과 `robot`을 다음 `graph.invoke`에 전달하며, 기대 주문 `oracle_order`는 별도 객체로 누적한다. 기대 section으로 로봇을 강제 이동하거나 gold 주문을 주입하는 대입은 없다. 원인: 반환은 strict 모드 분기에만 있고 물리 전환은 실제 `policy.execute`와 실제 주문의 실행 항목에 달려 있다(`manual_scenario_stack_probe.py:111-127,163-186`). 효과: 미실행으로 인한 단계 이탈을 `section_matches` 실패로 기록하면서 후속 발화의 실제 입력 상태를 유지한다(`manual_scenario_stack_probe.py:130-135`).
2. **수량 oracle과 계획 비교가 추가되었다.** 기대 주문은 발화 태그별로 누적하고, 기대 plan은 `noodle_thin ×2`, `onion ×1`·`mushroom ×2`, `sausage ×3`, `cheese ×1`의 명시적 목록이다. 실제 plan은 현재 주문·현재 section에서 실제 노드와 같은 `section_execution_items`·class·count 변환으로 만들어 `plans_match`에서 비교한다(`manual_scenario_stack_probe.py:71-88,111,121,141-153`; `llm_langgraph_node.py:1153-1182`). 원인: 기대값과 실제 변환 경로가 분리되었다. 효과: 주문 수량 또는 실제 계획의 항목·횟수·순서가 다르면 해당 턴이 실패로 기록된다. ROS 발행 검증은 아니다(`manual_scenario_stack_probe.py:159-160`).
3. **각 턴의 실행 시간과 대화 쌍을 기록·검사한다.** `latency_ms`는 `graph.invoke` 앞뒤의 `perf_counter` 차이이며 모델 로드나 합성 물리 처리 시간은 포함하지 않는다. 이전 history prefix와 이번 user/assistant 쌍을 검사한다(`manual_scenario_stack_probe.py:119-136,154-161`). 원인: 측정 경계가 그래프 호출에만 놓였다. 효과: D/Q/P 턴 지연과 history 전달 오류를 볼 수 있다. 그래프 호출·plan 변환 등에서 예외가 나면 모든 발화 측정은 중단되고 외부 `except`가 `EXIT.json`에 기술 실패를 기록한다(`manual_scenario_stack_probe.py:125,148-152,210-222`).

**남은 backend 합격 조건:** 초기 크림 추천 후보 외에는 단계별 pending 종류·대상, Decision `commit`, 실행 후 pending 해소를 직접 판정하지 않는다(`manual_scenario_stack_probe.py:130-140`). 또한 단계 이탈로 `extra`가 일찍 완료되면 fixture는 `lid` section만 설정하고 다음 사용자 발화를 계속 그래프에 넣는다(`manual_scenario_stack_probe.py:180-186`). 실제 노드는 `lid` 작업을 즉시 시작하며 작업 중 STT를 무시하므로(`llm_langgraph_node.py:624-627,1281-1285`), 그 이후 행은 오류 진단용 측정이지 실제 사용자 턴의 재현이 아니다.

**크림 제안의 유용성은 별도 판정:** 현재 검사는 크림이 미확정 추천 후보에 들어갔는지만 확인한다(`manual_scenario_stack_probe.py:137-140`). 왜 자연어 “느끼한” 요청에 유용한 제안인지, 사용자에게 어떻게 설명했는지는 `response_semantics: REQUIRES_INDEPENDENT_REVIEW`대로 사용자가 응답 원문을 보고 판단해야 한다(`manual_scenario_stack_probe.py:216-217`).
