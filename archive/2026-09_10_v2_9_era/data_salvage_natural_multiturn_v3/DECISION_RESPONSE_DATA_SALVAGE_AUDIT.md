# Decision·Response 데이터 salvage audit

참고 기준: [`agent_contract.py`](../src/soomac_irc/soomac_irc/agent_contract.py), [`decision_model.py`](../src/soomac_irc/soomac_irc/decision_model.py), [`response_model.py`](../src/soomac_irc/soomac_irc/response_model.py)

→ `25ce994`의 `natural_multiturn_v3`를 정답 contract로 고정하고, 과거 자료는 입력·문맥·실패 증거만 salvage한다.

## 1. 발견한 데이터 source

현재 repository에는 물리적으로 22,879개의 JSONL row가 있으나 파생본과 예측 복사본이 많이 포함되어 있다. **파일 row 합계를 학습 가능한 고유 turn 수로 보면 안 된다.**

| source | commit / branch | path | rows | user | history | state | Decision | Response | STT raw | 실행 결과 | Decision 가치 | Response 가치 | grade | notes |
|---|---|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| v2.2.5 train | `25ce994` current | `decision_adapter_v225_root_cause_review_20260930/dataset_contract/decision_train.jsonl` | 2,111 | O | O | O | O | X | X | X | 높음 | 없음 | B / D | route 이전 target과 legacy recommendation/pending 입력이다. |
| v2.2.5 validation | current | 같은 디렉터리 `decision_validation.jsonl` | 159 | O | O | O | O | X | X | X | 높음 | 없음 | B / D | 학습 split과 대화 그룹 중복 여부를 다시 검사해야 한다. |
| frozen evaluation | current | `decision_adapter_v225_root_cause_review_20260930/evaluation_data/` | 377 | O | O | O | O | X | 일부 표면 | X | 평가 전용 | 없음 | B·C / D | challenge 64, hard 220, latest 48, runtime holdout 45이다. training 금지이다. |
| any-order false predictions | current | `decision_adapter_v225_root_cause_review_20260930/results_any_order_false/` | 377 | O | O | O | 예측 | X | 일부 | X | 실패 증거 | 없음 | C / D | gold가 아니라 비교용 model output이다. |
| any-order true predictions | current | `evaluation_any_order_true_20260930/*_predictions.jsonl` | 377 | O | O | O | 예측 | X | 일부 | X | 실패 증거 | 없음 | C / D | review bundle의 temporary prediction과 다수 exact duplicate이다. |
| any-order 파생 subset | current | `evaluation_any_order_true_20260930/{improved,remaining,regressed}_cases.jsonl` | 59 | O | O | O | 비교 | X | 일부 | X | 실패 증거 | 없음 | C / D | improved 50, remaining 9, regressed 0이다. |
| full-cycle 결과 | current | `evaluation_any_order_true_20260930/llm_full_cycle_results.jsonl` | 12 | O | 일부 | O | O | O | O | 일부 | 높음 | 높음 | C / C | PASS 8, FAIL 4이다. 현재 contract로 Decision과 Response를 모두 다시 쓴다. |
| flexible migration | current | `soomac_decision_migration_merge_bundle/output/` | final 2,789 | O | O | O | O | X | X | X | 중간 산출물 | 없음 | B·C / D | 2,111 legacy + 685 supplement - 7 exact duplicate이다. |
| route rebalance | current | `soomac_decision_migration_merge_bundle/route_rebalance/final_train_rebalanced.jsonl` | 3,339 | O | O | O | O | X | 일부 | X | 가장 높은 synthetic 가치 | 없음 | B·C / D | latest legacy Decision source로 사용한다. `mentions` 2,533행, `queries` 203행이다. |
| route hard eval | current | `soomac_decision_migration_merge_bundle/route_rebalance/hard_eval.jsonl` | 495 | O | O | O | O | X | 일부 | X | challenge 전용 | 없음 | B·C / D | training으로 이동하지 않는다. training bundle의 copy와 exact duplicate이다. |
| v2.9 split copy | current | `soomac_decision_training_v2_9_epoch6/.../records/` | 3,339 | O | O | O | O | X | 일부 | X | 없음 | 없음 | D / D | route rebalance를 3,049/290으로 다시 나눈 파생 copy이다. |
| base routing original | current | `docs/base model routing test.jsonl` | test 133 | O | O | O | O | O | X | 일부 | 실패 증거 | 실패 증거 | C / C | 같은 133개 scenario의 이전 실행이다. |
| base routing hardened | current | `docs/base_model_routing_test_runtime_hardened.jsonl` | test 133 | O | O | O | O | O | X | 일부 | 매우 높음 | 매우 높음 | C / C | PASS 39, FAIL 94이다. issue label은 Decision 63, runtime 29, Response 11건이다. |
| 일반 runtime agent turns | `09a93e6` history | `logs/llm/**/agent_turns.jsonl` | 452 | O | O | O | O | 441 | STT 결과만 | O | 매우 높음 | 매우 높음 | C / C | 230개 unique user text이다. schema v1이며 current v3 row는 0개이다. |
| prompt boundary turns | `09a93e6` history | `logs/prompt_boundary_20261005/**/agent_turns.jsonl` | 51 | O | O | O | O | 34 | X | 일부 | 높음 | 높음 | C / C | 15개 unique surface의 반복 실행이므로 session 단위 dedup이 필요하다. |
| prompt review turns | `09a93e6` history | `logs/prompt_review_20261005/**/agent_turns.jsonl` | 17 | O | O | O | O | 15 | X | 일부 | 높음 | 높음 | C / C | guard/prompt failure의 hard negative로 사용한다. |
| legacy tool turns | `llm_vlm`, `llm_langgraph` | `src/soomac_irc/soomac_runtime_logs/tool_turns_*.jsonl` | 136 | O | O | O | tool call | final reply | STT 결과만 | O | 높음 | 중간 | C / C | 109개 unique user text이며 현 contract와 거리가 크다. |
| legacy runtime events | `09a93e6` history | `logs/**/runtime_events.jsonl`, `soomac_runtime_logs/runtime_events_*.jsonl` | 896 | 일부 | 일부 | O | X | X | X | O | 보조 | 보조 | D / C | task_completed 86, robot_task_started 99, VLM result 130, section transition 68이다. |
| current contract tests | `25ce994` | `src/soomac_irc/test/test_natural_multiturn_contract.py`, `test_section_question_events.py` | tests 63 | O | O | O | 기대값 | 응답 scaffold | 일부 | mock | 높음 | 높음 | C / C | 실대화가 아니라 contract edge를 정확히 고정한 fixture이다. |
| deleted legacy tests | `44d90e1^`, `f569d37^`, `3b4ff37^` | 삭제된 `src/soomac_irc/test/test_*.py` 16개 | tests 175 | O | 일부 | O | 기대값 | 일부 | 일부 | mock | 중간 | 중간 | C / C | guard 유도 정답과 legacy field가 있으므로 그대로 gold로 쓰지 않는다. |
| adapter binary | current | `models/decision_adapter_epoch6_best/` | N/A | X | X | X | X | X | X | X | 없음 | 없음 | D / D | dataset이 아니라 학습 결과물이다. |

## 2. 기존 데이터의 가장 큰 문제

### 2.1 Old contract contamination

현재 external Decision은 다음 8개 field만 허용한다.

```text
route, order, restrictions, preferences,
recommendation, commit, confirmation, clarify
```

route rebalance 3,339행에는 `mentions`가 2,533행, `queries`가 203행 포함되어 있다. 입력에도 `pending_confirmation`, legacy recommendation state, `action_history`가 들어 있다. **target에서 legacy key만 삭제하는 것으로 끝나지 않고 current model input부터 다시 구성해야 한다.**

### 2.2 Guard-induced wrong output

과거 runtime은 unsupported target이나 지시어를 Python guard가 먼저 막거나 Decision이 `clarify`하도록 학습했다. 현재 contract는 명확한 사용자 의미를 Decision에 보존하고 domain·physical truth를 final validator가 판단한다.

예:

```text
"햄 빼줘"
legacy → clarify
v3 gold → {"route":"task","order":{"toppings":{"햄":"none"}}}
```

### 2.3 Response hallucination과 진행성 부족

과거 Response는 Decision이 실패했는데도 반영했다고 말하거나, `next_prompt` 없이 임의 질문을 붙이거나, 반대로 실제 선택 질문을 생략한 사례가 있다.

Response gold는 다음 진행 규칙을 함께 지켜야 한다.

- `next_prompt`가 있으면 목적에 맞는 질문이나 행동 유도를 반드시 포함한다.
- pending이 있으면 아직 확정되지 않았음을 지키면서 확인 질문을 한다.
- section 선택이면 지원 선택지와 "넘어가기"를 안내한다.
- `missing_order`이면 부족한 정보를 반드시 묻는다.
- validation issue가 있어도 가능한 다음 선택을 안내한다.
- `next_prompt` 없는 상태/general 질문에는 억지 후속 질문을 붙이지 않는다.
- 실행 시작·완료 안내에는 불필요한 질문을 붙이지 않는다.

### 2.4 History와 state mismatch

schema v1 runtime 로그는 `recommendation`, `pending_confirmation`, `dialogue_focus`, `pending_question`을 사용한다. recent history는 살릴 수 있지만 state는 current `order/preferences/pending/robot_state`로 재구성해야 한다.

### 2.5 Duplicate와 contamination

동일한 evaluation prediction이 review bundle, temporary eval, any-order 결과에 복사되어 있다. route hard eval 495행도 training bundle과 exact duplicate이다. **split은 row가 아니라 `session_id`, `conversation_id`, `semantic_id`, `paraphrase_group`의 우선순위로 묶어야 한다.**

## 3. Decision salvage 예상

최신 unique synthetic source 3,339행을 기준으로 위험 조건을 계산했다.

| 분류 | 예상 rows | 처리 |
|---|---:|---|
| A 그대로 사용 | 0 | 현재 v3 형식으로 만들어진 기존 dataset이 없다. |
| B 부분 relabel | 2,393 | route·핵심 mutation 의미를 유지하고 input shape, legacy key, restriction/recommendation 모양을 변환한다. |
| C 전면 relabel | 946 | clarify, unsupported, reference, confirmation, cancel, STT/noise 계열을 사람이 재검토한다. |
| C runtime 추가 | 최대 656 | agent turn 520 + legacy tool turn 136에서 입력과 실패 사례를 salvage한다. |
| D / 평가 격리 | 최소 872 | frozen eval 377 + route hard eval 495는 training에서 제외한다. |

946행은 다음 위험 조건의 합집합이다: legacy `clarify`, pending confirmation, select/cancel recommendation, unsupported/reference/confirmation/STT/noise 계열이다.

## 4. Response salvage 예상

현재 v3 Response 학습 dataset과 v3 실로그는 모두 0행이다. 따라서 기존 응답을 A gold로 승인하지 않는다.

| 분류 | 예상 rows | 처리 |
|---|---:|---|
| A 그대로 사용 | 0 | current Response model input과 정확히 일치하는 기존 dataset이 없다. |
| B 부분 relabel | 0 | 사실이 맞아도 진행 질문 정책까지 재검토해야 한다. |
| C 전면 relabel | 최대 834 | response trace 490 + tool turn 136 + hardened routing 133 + full-cycle 12 + current fixtures 63이다. |
| D | 최소 163 | Response trace 없는 agent turn 30과 중복 routing original 133이다. |

dedup 뒤 실제 Response 후보는 약 450~700행으로 예상한다. section transition Response는 current 실로그가 없으므로 현재 test fixture에서 시작하고 이후 v3 runtime log로 보강해야 한다.

## 5. Failure source 분석

기존 labeled evidence는 다음과 같다.

| evidence | STT | Decision | Policy/runtime | Response | Ambiguous | Multiple |
|---|---:|---:|---:|---:|---:|---:|
| hardened routing 133 | 미분리 | 63 | 29 | 11 | issue label 내부 | 중복 허용 |
| full-cycle 12 | 1 이상 | 2 이상 | 2 이상 | 0 명시 | 일부 | FAIL 4 |
| review sample 26 | 3 | 5 | 0 | 3 | 1 | 3 |

전체 520 runtime turn은 아직 사람이 failure-source label을 확정하지 않았다. 이번 단계에서는 대표 사례만 라벨링했고, 대량 label 추정은 하지 않았다.

## 6. 현재 contract로 변환한 sample

- canonical intermediate: [`salvage_intermediate.jsonl`](salvage_intermediate.jsonl) 26행
- Decision review sample: [`decision_v3/review_sample.jsonl`](decision_v3/review_sample.jsonl) 14행
- Response review sample: [`response_v1/review_sample.jsonl`](response_v1/review_sample.jsonl) 12행

구성에는 explicit order, compound intent, general/question boundary, restriction, preference, recommendation, confirmation, multi-turn reference, STT noise, execution pending, recommendation pending, missing order, partial success, physical conflict, section transition이 포함된다.

## 7. Dataset build 제안

### 7.1 예상 규모

| dataset | train | validation | test | challenge | holdout |
|---|---:|---:|---:|---:|---:|
| Decision v3 | 2,800~3,400 | 250~350 | 250~350 | 기존 495 유지 | 기존 377 격리 |
| Response v1 | 400~550 | 50~80 | 50~80 | 60~100 | 실제 v3 log 확보 후 50+ |

Response 부족 유형은 `missing_order`, validation issue 이후 선택지, recommendation revise 실패, section transition, pending 중 상태 질문이다. 실제 trace salvage 뒤에도 부족할 때만 80~150행을 제한적으로 합성한다.

### 7.2 생성 순서

1. provenance를 유지한 canonical intermediate로 모든 source를 변환한다.
2. legacy output을 참고 자료로만 두고 current contract gold를 재라벨한다.
3. session/dialogue/semantic group 단위로 dedup과 split을 수행한다.
4. Decision과 Response schema·fact consistency validator를 각각 실행한다.
5. challenge/holdout contamination을 검사한 뒤 training JSONL을 export한다.

## 8. 다음 행동

대표 26행의 contract 해석을 먼저 승인받는다. 승인 후 전체 salvage → dedup → quality validation → Decision v3 → Response v1 순서로 생성한다.

Cause: 기존 자료는 legacy state와 model output이 섞여 있어 파일 수나 PASS label만으로 gold 여부를 결정할 수 없다.

Effect: runtime을 수정하지 않고도 Decision 의미 오류와 Response 표현 오류를 별도 dataset으로 분리할 수 있다.
