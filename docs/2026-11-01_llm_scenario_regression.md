# 사용자 시나리오 — 최신 best D/Q/P 누적 GPU 회귀

[prompt 변경 Worklog](2026-11-01_llm_prompt_alignment_worklog.md) · [시험 코드](../src/soomac_irc/test/manual_scenario_stack_probe.py)
→ **자연스러운 주문이 목표다. 복합 발화의 주문·변경·제약·즉시 실행을 한 턴에 처리하는지 실제 모델로 검증한다.**

2026-11-01 KST. 최신 상태: **사용자 지시에 따라 중간 의미 오류로 멈추지 않고 전체17턴 계측을 완료했다(exit0).** `llm_final_2`의 수정 prompt v2와 `model_runtime.py`에 지정된 최신 best D/Q/P 세 adapter를 사용했다. 이전 개별 입력20회 A/B와 구분하며, 예고만 했던 추가 D7회는 대신 실행하지 않았다. 최종 수락은 사용자가 실제 출력/계측을 보고 판단한다. 전체 수락·운영 채택을 자동 선언하지 않는다.

## 1. 사용자 요청과 합격 기준

1. 면 얇은면 normal 선택+느끼한 소스 추천 → 실제 Q가 크림 후보를 제시. “크림으로 선택할게. 바로 담아.”에 크림 확정+실행 허가가 함께 필요하다.
2. “양파랑 버섯 둘 다 적게 담아줘.” → 둘 다 low, execute=false. 재확인 전 자동 실행 금지.
3. “버섯만 보통 양으로 바꿔서 바로 담아.” → 버섯 normal, 언급하지 않은 양파 low 보존, 실행 허가.
4. “게살 알레르기가 있어서 소시지만 많이 바로 담아줘.” → 게살 allergy 실제 state 반영, 소시지 high, 게살 실행 제외, 실행 허가.
5. “페퍼론치노는 빼고 치즈만 적게 골라서 바로 담아.” → 치즈 low, 페퍼론치노 없음, 실행 허가. Response가 실제 선택/변경/시작/완료 상태와 수량을 다르게 말하면 별도 실패다.

## 2. 시험 경계

주 시험 `compound`는 위6개 사용자 발화의 누적 흐름이다. `split_control`은 선택·변경 후 “바로 담아”를 분리한11턴 대조다. 각 episode는 새 session으로 시작하지만 **episode 안에서는 모델의 실제 after state/history를 그대로 다음 입력으로 전달**한다. 정답 Decision, 주문 patch 보정, 강제 commit, 실패 뒤 임의 section 전환, 가짜 사용자 발화는 주입하지 않는다.

주문·실행 gate가 틀리면 해당 episode를 즉시 중단하고 이후 단계는 NOT_REACHED로 남긴다. 분리 대조가 성공해도 주 복합 시나리오 성공으로 세지 않는다. 모든 턴의 before/after·D/Q/P trace·policy·reply·field check를 보존하고 주담당과 독립 작업자가 검수한다.

**로봇 물리 완료만 offline fixture다.** 실제 execute=true 뒤에만 현재 order를 node와 같은 task/plan mapping으로 읽어 가상의 task_completed/section_transition을 기록한다. 이 표시는 실제 ROS 발행·제조 완료·VLM 관측 성공의 증거가 아니다. 단계를 넘길 때는 현재 node의 임시 고정 안내를 AST로 읽어 assistant history에 추가하고 새 user message를 만들지 않는다. ROS/rclpy/VLM 실행 코드는 import하지 않는다.

이번 목표는 면·채소·육류·치즈 주문 구간까지다. 뚜껑 닫기·실제 소스 투입·최종 완성까지 실물 검증했다고 하지 않는다. 크림은 주문에 확정되는지를 검사하며 면 실행 시 크림까지 물리 투입했다고 말하지 않는다.

## 3. 실행 identity와 멀티에이전트

- GPU 산출물: `/tmp/soomac_closed_loop_20261101_v2/PLAN.json`, `IDENTITY.json`, `turns.jsonl`, `RESULT.json`, `EXIT.json`.
- Base 한 번 로드, 최신 best D/Q/P 모두 GPU에 붙이고 실제 각 역할 호출에 해당 adapter를 활성화한다. 경로·weight/config hash와 코드 hash를 기록한다. 학습·운영 경로 변경·commit/push는 하지 않는다.
- Orca Run `run_5dcc02059ec1`, 이 작업 전용 coordinator `term_5204c04f-7c3e-441b-abe8-d5fae9f747b5`. VS Code 호출자 env에 Orca handle이 없어 새 전용 shell에서 Run을 만들었다. 기존 사용자 pane의 발신자를 빌리지 않았다.
- 검수A: task `task_7b482bb84e99`, dispatch `ctx_6cb9b003819d`, 실제 launch `gpt-6-sol/high`. 하니스·정답·state 누적·물리 fixture·실패 중단 검수, 쓰기 소유권은 `2026-11-01_scenario_harness_review.md`만이다.
- 검수B: 별도 출력 검수 작업자. D/Q/P 의미·수량·허위 이미/완료·불필요 추천을 검수하며 GPU를 실행하지 않는다. 쓰기 소유권은 `2026-11-01_scenario_output_review.md`만이다. 작업자 receipt와 최종 상태는 종료 시 추가한다.

Cause: 앞선 D/P 독립 입력 시험은 Q의 실제 추천과 이전 모델 출력이 누적되는 전체 흐름을 검증하지 않았다.

Effect: 이번 결과는 자연 누적 시나리오의 최초 실패 턴과 실제 도달 범위를 구분한다. 구조 gate 통과와 Response 의미 합격도 따로 보고한다.

## 4. 사용자 정정 반영 — 전체 계측을 먼저 한다

앞선 strict 실행(`/tmp/soomac_closed_loop_20261101_v2/`)은 양쪽 episode의 첫 턴 뒤 중단했다. 면 thin/normal은 반영됐으나 D recommendation 누락으로 Q를 호출하지 않았고, P가 크림을 제안했다. **크림 추천을 자연스럽게 말하는 것 자체는 사용자가 원하는 행동이다.** 후보 state/후속 선택과의 연결 누락과 발화의 유용성을 구분하며, “그 멘트를 없애야 한다”는 방향으로 해석하지 않는다. strict 실행은 보존하되 이2행으로 전체 시나리오 검증 완료라고 하지 않는다.

사용자 요청: “시나리오 다 돌려보고 계측부터 한 다음 내가 accept”. 이에 `--measure-all`을 추가했고 production prompt/Runtime/adapter는 추가 변경하지 않았다. 새 산출물은 `/tmp/soomac_closed_loop_all_20261101_v2/`이다. 6+11개 원래 발화를 모두 입력하며, 의미 불일치는 기록하되 episode를 중단하지 않는다. 기술 오류/OOM은 별도 실패로 기록한다. 추가 확인 발화·gold 주문·강제 commit·강제 단계 전환은 넣지 않는다.

**실제 robot section도 모델 실행 결과에만 따라간다.** 예를 들어 버섯 단계 commit이 빠지면 육류 발화를 하더라도 robot section은 여전히 veggie일 수 있다. 이런 stage drift를 숨기지 않고 expected_section/robot_before.section과 actual_plans에 기록한다. execute=true일 때의 물리 완료만 기존처럼 합성 fixture로 처리한다. 따라서 후반 발화의 실패는 최초 누락의 연쇄 영향인지 독립 모델 오류인지 나눠 설명한다.

예상 주문은 별도의 oracle 사본으로 누적하며, 실제 잘못된 이전 수량을 다음 턴 기대값으로 삼지 않는다. oracle은 모델 session에 주입하지 않는다. 계획은 node mapping으로 만든 actual 목록과 독립적인 기대 목록(thin×2, onion×1+mushroom×2, sausage×3, cheese×1)을 기록·대조한다. 턴 벽시계시간과 D/Q/P 각각의 raw·토큰·지연, pending 전후, 실제 응답도 보존한다.

새 측정 동작 CPU2건 통과: 의미 오류가 모든 턴에 나도compound6턴을 끝까지 기록하며, 실행 없는 robot section을 강제로 넘기거나 실제 주문을 oracle로 바꾸지 않는 것을 확인했다. 이는 모델 성능 시험과 구분한다. 최초 live 확인은 PID85399, GPU process 메모리17014MiB였다.

검수A 후속 task `task_72d11aa15148`, dispatch `ctx_e6e73146e4a5`; 검수B 후속 task `task_11a70daa7129`, dispatch `ctx_29438abd3c73`. 기존 검수 완료 보고2건을 수신·검토한 뒤 같은 proven 터미널을 재사용했고 이전 delivery를 acknowledge했다. 새 agent/GPU를 중복 실행하지 않았다. A는 변경된 하니스만, B는 전체17턴 출력만 후속 검수한다. 사용자 수락을 대신 선언하지 않는다.

## 5. 전체 계측 결과 — 사용자 수락 판단용

[전체17턴의 발화·raw 필드 독립 검수](2026-11-01_scenario_output_review.md) · [지속 보관용 계측 JSON](2026-11-01_scenario_measurements.json) · [계측 하니스 독립 검수](2026-11-01_scenario_harness_review.md)

원본 trace는 `/tmp/soomac_closed_loop_all_20261101_v2/turns.jsonl`이다. 재부팅 등에 대비해17턴의 실제 사용자 발화/응답·전후 주문/pending·D/Q/P raw·토큰·지연·P 입력 핵심 사실·adapter/code identity·계획을 위 JSON에도 저장했다. system全文/history가 포함된 원본 trace와 줄인 계측 JSON을 구별한다. 어느 쪽도 학습 데이터로 편입하지 않았다.

### 복합 발화6턴 — 실제 결과

| 턴 | 실제 발화 | 실제 응답 | 주문·commit·실행 관측 | 그래프 시간 |
| --- | --- | --- | --- | ---: |
| C1 | 면은 얇은면으로 보통 양으로 주고, 소스는 느끼한 걸로 추천해줘. | 면은 얇은면으로 보통 양을 선택했어요. 느끼한 소스는 크림을 추천해요. 이대로 진행할까요? | 얇은면 normal 반영. D 추천 요청 누락/Q 미호출. 크림 후보/소스/pending=null | 13.33초 |
| C2 | 크림으로 선택할게. 바로 담아. | 소스는 크림으로 선택했어요. 이대로 담을까요? | D raw는 commit=true만 출력. 크림 patch 없음. 소스는 여전히null, execute=false/missing_order | 10.65초 |
| C3 | 양파랑 버섯 둘 다 적게 담아줘. | 양파와 버섯 모두 적은 양으로 선택했어요. 이대로 담을까요? | 두 재료low 반영, execute=false. 실제 section은 아직noodle, pending=null | 12.18초 |
| C4 | 버섯만 보통 양으로 바꿔서 바로 담아. | 버섯은 보통 양으로 바꿨어요. 아직 소스가 선택되지 않았어요. 소스를 말씀해주세요. | 버섯normal 변경/양파low 유지/commit=true. 소스 누락으로 execute=false | 12.83초 |
| C5 | 게살 알레르기가 있어서 소시지만 많이 바로 담아줘. | 게살 알레르기 제한을 추가했어요. 소시지는 많이 선택되어 있어요. 이대로 담을까요? | 게살allergy/소시지high 반영. commit 누락/execute=false | 14.11초 |
| C6 | 페퍼론치노는 빼고 치즈만 적게 골라서 바로 담아. | 페퍼론치노는 제외하고 치즈를 적은 양으로 선택했어요. 이대로 담을까요? | 치즈low 반영, 페퍼론치노없음. commit 누락/execute=false | 13.50초 |

크림을 제안하는 문장 자체는 사용자 의도에 부합한다. **C2의 “크림으로 선택했어요”는 실제 `order.sauce=null`과 다르다.** 이 사실 오류와 C1 추천 문장의 유용성을 같은 문제로 취급하지 않는다. 이 실행에서 담기 완료를 주장한 응답은 없었다. 다수의 “담을까요”에는 실제 execution pending이 없었다는 연결 상태를 함께 제시한다.

### 분리 확인11턴 — 시간과 주요 관측

전체 실제 발화/응답은 독립 검수 문서와 JSON에 있다. 아래 S1~S11은 그 문서의 동일 행이다.

| 턴 | 발화 요지 | 그래프 시간 | 주요 관측 |
| --- | --- | ---: | --- |
| S1 | 면 normal+느끼한 소스 추천 | 12.99초 | 면 반영, Q 미호출/P 크림 제안 |
| S2 | 크림으로 선택할게 | 10.84초 | confirmation=accept만 출력, pending이 없어 소스null 유지. 응답은 크림 선택됐다고 말함 |
| S3 | 바로 담아 | 11.05초 | commit=true, 소스 누락으로 차단 |
| S4 | 양파·버섯low | 12.52초 | 두 재료low 반영 |
| S5 | 버섯만normal로 변경 | 12.68초 | 버섯normal/양파low 유지. 응답도 두 수량을 정확히 설명 |
| S6 | 바로 담아 | 12.35초 | commit=true, 소스 누락으로 차단 |
| S7 | 게살 알레르기 | 14.13초 | 게살allergy 추가 |
| S8 | 소시지만high 선택 | 13.50초 | 소시지high 신규 선택. 게살allergy 재출력은 중복 저장되지 않음 |
| S9 | 바로 담아 | 11.80초 | commit=true, 소스 누락으로 차단 |
| S10 | 페퍼론치노 제외·치즈low | 14.04초 | 치즈low/기존량·제약 유지 |
| S11 | 바로 담아 | 12.85초 | commit=true, 소스 누락으로 차단 |

### 수량·실행·성능 계측의 범위

최종 주문은 양쪽 episode 모두 `얇은면/normal`, `소스=null`, `양파low/버섯normal/소시지high/치즈low`, `게살allergy`다. 게살·페퍼론치노는 실행 후보에 없으며 **양파low 유지와 버섯low→normal 변경 자체는 관측됐다.** 이17턴에서 수량 값을 잘못 normal로 덮거나 실제로 담았다고 말한 사례는 관측되지 않았다. 앞선 독립 control의 기존low→normal 오류가 해결됐다는 뜻은 아니다.

commit=true는 C2/C4와 S3/S6/S9/S11, 총6턴이다. C5/C6의 명시 즉시 실행은 commit이 누락됐다. 17턴 모두 execute=false, 생성 plan0, 합성 task_completed0, 단계 전환0이다. 모든 턴의 실제 section은noodle이었다. 후반 육류·토핑의 결과를 각각 정상 meat/extra 단계에서 독립 검증한 성적으로 보고하지 않는다.

| 항목 | 관측값 |
| --- | --- |
| 실제 모델 호출 | D17 / Q0 / P17. Q adapter도 로드했으나 D가 추천 요청을 내지 않아 조건부 경로에서 호출되지 않음 |
| 복합6턴 모델 대화 처리 합계 / 중앙값 | 76.59초 / 13.08초 |
| 분리11턴 합계 / 중앙값 | 138.75초 / 12.68초 |
| 전체17턴 시간 범위 / 중앙값 | 10.65~14.13초 / 12.83초 |
| D 호출 중앙값 / P 호출 중앙값 | 6.39초 / 6.26초 |
| D 입력 / P 입력 토큰 범위 | 8092~8492 / 7574~8315 |
| 최대 PyTorch allocated / GPU | 13549817344bytes(약12.62GiB) / RTX PRO4500 Blackwell |

시간은 `graph.invoke` 경계이며 모델 로드·STT·TTS·물리 이동은 제외한다. 실제 사용자 체감 전체 지연이나 TTFT/p95를 측정했다고 하지 않는다. 같은 경우를 반복한 안정성 시험도 아니다.

Cause: 첫 추천 요청 누락과 뒤 크림 선택 누락 때문에 canonical 소스가 비어 있었다. Python의 필수 주문 검증이 실제 실행을 막았고, 그 결과 이후 발화에도 면 단계 상태가 누적됐다.

Effect: “추천 문장은 나온다 / 수량·제약은 상당 부분 반영된다 / 소스 확정과 복합commit·응답 사실 연결은 어긋난다”를 분리해 볼 수 있다. **다음 변경이나 운영 수락은 이 계측을 사용자가 확인한 뒤 결정한다.**

## 6. 검수·종료 기록

검수A는 measure-all에서 실제 state/section 유지·독립 수량/plan oracle·대화 쌍·그래프 시간 계측을 확인했다. pending/commit의 자동 의미 채점 및 lid/busy fixture 경계는 전체 합격으로 확대하지 말라는 지적을 남겼다. 이번 실제17턴은 모두noodle/execute=false여서 lid/busy 경계에 도달하지 않았고 물리 완료 fixture도 적용되지 않았다.

검수B는 전체17행과 RESULT/EXIT 및 raw 입력을 대조했다. D/Q/P 호출 수·소스null·수량/제약 누적·후반 단계 이탈·C2/S2 허위 소스 확정 설명을 주담당과 교차 확인했다. 크림 제안 자체는 유용하며 제거할 대상이 아니라는 사용자 정정도 보고에 반영했다. 독립 결과는 사용자 수락을 대체하지 않는다.

두 후속 `worker_done`을 수신·검토·acknowledge했고 공식 `worker-release`로 각각 종료했다. Run `run_5dcc02059ec1`의 reclaimable worker 조회는0개였다. 최초 완료 작업자2명은 같은 터미널에서 후속 Task로 재사용했으며 중복 새 작업자를 만들지 않았다. 모델 프로세스도 exit0으로 끝났고 자동 재시험을 시작하지 않았다.

`git diff --check` 통과, measure-all 하니스 CPU2건 재확인 통과, 보관 JSON17행 확인. 이번 계측을 이유로 production prompt/Runtime/학습 데이터/adapter 경로를 추가 수정하거나 학습·실물 ROS·commit/push를 하지 않았다. `.claude/settings.json`의 다른 변경은 보존했다.
