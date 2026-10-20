# IRC_SOOMAC 3.0 Runtime v4 · 전체 데이터 감사 · 재학습 작업 기록

[2026-10-05 가드 정리 WORKLOG](2026-10-05_guard_cleanup_worklog.md)

→ 단어 기반 Python 가드를 줄이고 실제 로봇 로그에서 확인된 제약 확인 반복·STT 오인식·거짓 실행 안내를 정리한 이전 기준이다.

[2026-10-06 Natural-language-first WORKLOG](2026-10-06_flexible_llm_natural_multiturn_refactor_worklog.md)

→ 의미 판단을 Decision 모델에 모으고 Python은 domain·safety·physical truth를 검증하도록 정리한 직전 구조 기준이다.

[Full Agent Upgrade 원 계획](full_agent_upgrade.md)

→ Runtime 계약 감사부터 데이터 재구축·fresh LoRA·epoch 평가·실로그·장기 평가·조건부 채택까지 요구한 계획이다.

[최종 감사 보고서](</home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/full_rebuild_v4/FINAL_AUDITED_RETRAIN_REPORT.md>)

→ 신규 후보는 짧은 실로그에서는 개선됐지만 장기 상태 유지 gate에서 탈락하여 기존 Adapter를 유지했다.

---

## 작업 기준

- 작업 브랜치: `flexible_llm`
- 시작 시점의 기존 WORKLOG 기준: 2026-10-06, HEAD `f558965`
- 현재 HEAD: `1637e57`
- Runtime 계약: `natural_multiturn_v3` → `natural_multiturn_v4`
- 학습 워크스페이스: `/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4`
- 런타임 Adapter: `models/runtime_adapters/`
- 런타임 선택 정본: `config/llm_active_stack.json`

**이번 작업의 결론은 “재학습은 완료했지만 장기 상태 유지가 부족하여 신규 Adapter를 채택하지 않는다”이다.**

```text
Runtime code 개선                  = 적용 대상
신규 Decision/Response Adapter     = 평가 보존, 미적용
기존 검증 Adapter                  = 계속 사용
학습 코드·데이터·체크포인트 65GB  = 저장소 밖으로 이동
```

(내가 의역해보자면 코드에서 확실히 막을 수 있는 물리 상태 오류는 고쳤고, 의미 모델은 짧은 대화 성적만 보고 교체하지 않은 것이다.)

---

## 기존 WORKLOG와 현재 결과 비교

| 항목 | 2026-10-06 WORKLOG | 2026-10-20 현재 |
|---|---|---|
| Decision 계약 | `route/order/restrictions/preferences/recommendation/commit/confirmation/clarify` | `skip_current`를 별도 의미로 추가하여 실행과 빈 단계 생략을 분리했다. |
| 제한 삭제 | Decision의 remove를 일반 변경처럼 처리했다. | allergy·cannot_eat·dietary_rule 삭제는 `restriction_removal` pending 확인 뒤 적용한다. |
| 추천 범위 | 현재 상태와 모델 판단 중심이었다. | 현재 section부터 미래의 물리적으로 수정 가능한 선택지만 `recommendation_scope`로 준다. |
| 추천 설명 | 모델의 `reason_tags`를 그대로 사용할 수 있었다. | 확정 state·현재 section·검증된 proposal로 Runtime이 canonical reason을 만든다. |
| 예외 응답 | 대부분 Response 모델 자유 생성이었다. | 실행 거절·제한 삭제 거절·추천 제안/수락·추천 불가처럼 상태 충돌 위험이 큰 event는 확정 사실 문장을 사용한다. |
| section 전환 | 먼저 state를 변경한 뒤 publish 실패 시 일부 꼬일 수 있었다. | task 시작 전 실패는 snapshot으로 rollback하고, 물리 실행 시작 뒤에는 중복 실행 방지를 위해 전환 state를 보존한다. |
| 모델 구성 | Decision v3.2 + Base Recommendation + Response v1 | Decision v3.4 e4 + Recommendation v2 e4 + Response v1.3 e3 |
| 검증 | 코드 회귀 및 제한된 로그 재생 | 전체 corpus 감사, 전 epoch 평가, 실로그 97턴×2방식, 50세션·1,000턴 장기 평가까지 수행했다. |

Cause: 기존 WORKLOG 이후 추천 Agent와 장기 canonical state가 추가되면서 모델 의미와 물리 실행 사이의 경계가 다시 넓어졌다.

Effect: 현재 Runtime은 모델 출력이 이상해도 pending·section·물리 task가 서로 모순된 상태로 조용히 진행되지 않는다.

---

## 1. 실제 로봇 대화에서 확인한 문제

### 1.1 실행 확인과 pending

실제 대화에서 “확인 필요” 상태가 생성된 뒤 다음 발화에서 과거 주문 요약이 반복되거나, 확인 대상과 현재 section이 엇갈리는 문제가 있었다.

- 실행 pending 거절을 일반 `confirmation_rejected`로 뭉치지 않고 `execution_confirmation_rejected`로 분리했다.
- 추천 거절은 `recommendation_rejected`, 강한 제한 삭제 거절은 `restriction_removal_rejected`로 분리했다.
- 거절 뒤 Response 모델이 “남은 확인”이나 과거 실행을 지어내지 못하도록 확정 event 문장을 사용한다.

### 1.2 뚜껑(lid)·소스 자동 단계

뚜껑부터는 사용자 수음을 받을 단계가 아니므로 기존 section 진행 흐름 안에서 자동 task를 생성한다.

- `extra → lid` 전환 시 뚜껑 task 생성 성공을 확인한 뒤 상태를 확정한다.
- `lid → sauce` 전환도 선택된 소스 task 생성 성공을 확인한다.
- task 생성이나 rail publish가 실패하면 section·pending·queue·active task를 전환 전 snapshot으로 복원한다.
- 이미 물리 task가 시작된 뒤 reply/log publish만 실패한 경우에는 state를 되돌리지 않는다.

**물리 실행 뒤 rollback하면 같은 재료나 뚜껑 작업이 재발행될 수 있으므로, 실행 시작 전후의 실패 처리 기준을 다르게 했다.**

### 1.3 추천·선택지·상태 질문 혼동

다음 발화는 서로 다른 의미로 처리한다.

```text
“뭐 넣는 게 좋아?”   → 추천 요청
“뭐 넣을 수 있어?”   → 선택지 조회
“지금 뭐 담고 있어?” → 현재 상태 조회
```

- Decision prompt에 세 발화의 차이와 문맥 사용 기준을 추가했다.
- 추천 범위는 현재 section부터 미래 section까지 허용하고, 이미 지나간 section은 제외한다.
- 미래 선택은 미리 추천할 수 있지만 실제 section에 도달할 때 기존 section 진행 로직으로 처리한다.
- `recommendation_unavailable`이면 과거 주문 요약을 반복하지 않고 추천 가능한 후보가 없음을 알린다.

### 1.4 취향과 제한의 생명주기

- 음식과 무관한 일시적 상황은 장기 preference로 저장하지 않는다.
- canonical state에서 제거된 preference는 오래된 history에 남아 있어도 추천 근거로 부활시키지 않는다.
- 강한 섭취 제한 삭제는 사용자 확인 전까지 실제 state에서 제거하지 않는다.
- dislike 삭제는 안전 확인 없이 일반 변경으로 처리한다.

---

## 2. Runtime v4 코드 변경

### 2.1 `skip_current`

```json
{
  "route": "task",
  "skip_current": true
}
```

`skip_current`는 optional section에서 아무 재료도 담지 않고 다음 단계로 넘어가는 의미이다.

- 허용 section: `veggie`, `meat`, `extra`
- 현재 section에 선택 재료가 있으면 먼저 명시적으로 제거해야 한다.
- `commit=true`와 동시에 사용할 수 없다.
- `noodle`, `lid`, `sauce`에서는 거절한다.

### 2.2 추천 reason의 정본화

Recommendation 모델의 자유 문자열을 사용자 응답 근거로 직접 사용하지 않는다.

```text
proposal
→ validator
→ accepted proposal
→ canonical current state + current section
→ Runtime reason tags
```

예를 들어 추가 재료 단계에서 `맵지 않게` preference와 치즈 proposal이 함께 있으면 `매운맛 회피`, `현재 추가 재료 단계`를 근거로 만든다.

### 2.3 section 전환 transaction

전환 전에 아래 상태를 snapshot한다.

```text
section
graph_state
task_queue
active_task
completed_tasks
robot_started
vlm_confirmed
```

task 시작 전 예외는 snapshot으로 복원한다. task 시작 뒤 reply·log 예외는 실행 state를 유지한다.

Cause: publish 성공 여부보다 먼저 section과 pending을 바꾸면 소프트웨어 state와 실제 로봇 상태가 달라질 수 있었다.

Effect: 실패 재시도에서 section이 건너뛰거나 같은 물리 task가 중복 발행되는 경로가 차단됐다.

---

## 3. 전체 데이터 의미 감사

기존 데이터라는 이유로 정답으로 간주하지 않고 Decision·Recommendation·Response 전체를 최신 Runtime 계약으로 다시 검사했다.

| 구성요소 | 전체 감사 행 | 수정/재라벨 | 제외 | ambiguous | unknown remaining |
|---|---:|---:|---:|---:|---:|
| Decision | 5,831 | 3,796 | 1,113 | 0 | 0 |
| Recommendation | 1,100 | 0 | 441 | 0 | 0 |
| Response | 3,019 | 1,178 | 1,450 | 0 | 0 |

최종 clean 학습 corpus:

| 구성요소 | Train | Validation | Test | Challenge | Holdout |
|---|---:|---:|---:|---:|---:|
| Decision | 4,144 | 293 | 214 | 463 | 361 |
| Recommendation | 612 | 13 | 18 | 15 | 13 |
| Response | 1,183 | 181 | 82 | 75 | 65 |

검사 결과:

- schema invalid: 0
- contract mismatch: 0
- unresolved ambiguous train row: 0
- frozen exact/near overlap: 0
- state propagation contradiction: 0
- preference lifecycle contradiction: 0

과거 장기 평가 67건도 다시 분류했다.

| 원인 | 건수 |
|---|---:|
| STATE_NOT_SAVED | 17 |
| MODEL_IGNORED_VALID_STATE | 0 |
| ACTUAL_MODEL_FAILURE | 17 |
| GOLD/EVALUATOR/FIXTURE | 33 |

Cause: state 전달 오류나 평가 fixture 오류를 모델 정답으로 넣으면 잘못된 행동을 학습한다.

Effect: canonical state가 정상 전달됐는데도 모델이 틀린 사례만 학습 문제로 분리했다.

---

## 4. 폐기한 학습과 fresh training

다음 run은 최종 후보로 사용하지 않았다.

- Decision v3.5 targeted continuation: supplement-only 학습
- Response v1.4 targeted continuation: supplement-only 학습
- Decision v3.6: 장기 state coverage 부족
- near-duplicate contaminated run
- multi-preference active state 누락 run

잘못된 run의 checkpoint·log·manifest는 `invalid_runs/`에 보존했고 resume하지 않았다.

최종 학습:

| 구성요소 | 시작점 | 완료 epoch | optimizer step | 최종 train loss |
|---|---|---:|---:|---:|
| Decision | Gemma 4 12B base | 6 | 1,554 | 0.02216 |
| Response | Gemma 4 12B base | 2, early stop | 148 | 0.58303 |
| Recommendation | 기존 v2 e4 유지 | 4 | frozen | 해당 없음 |

Decision과 Response는 거절된 Adapter를 이어 학습하지 않고 base model에서 fresh LoRA로 시작했다.

---

## 5. 전 epoch 평가와 후보 선택

모든 저장 epoch를 strict suite로 평가하고 실로그 probe를 실행했다.

최종 component 후보:

| 구성요소 | 후보 | Strict | 핵심 결과 |
|---|---|---:|---|
| Decision | `decision_v3_7_state_coverage_audited_epoch4` | 100/132 | held-out 3·5·8·12턴 12/12 |
| Recommendation | `recommendation_v2_epoch4` | 18/19 | 기존 모델 유지 |
| Response | `response_v1_4_audited_epoch2` | 105/125 | 실행 거절·carry-over 안전 gate 통과 |

loss만으로 epoch를 선택하지 않고 semantic error, critical regression, 실로그, 멀티턴 순서로 선택했다.

---

## 6. 실로그 replay와 장기 평가

실제 로그 97턴을 두 방식으로 실행했다.

| 방식 | 기존 모델 실패 | 신규 후보 실패 |
|---|---:|---:|
| turn-isolated | 28 | 21 |
| full-session | 27 | 23 |
| 합계 | 55 | 44 |

개선된 항목:

- 추천 요청 누락: 14 → 4
- 암시적 양 대상 오류: 2 → 1
- 거짓 반복 업데이트: 4 → 3
- inference error: 두 방식 모두 0

남은 핵심 문제:

- wrong-section reference: 15 → 15
- Response 추천 창작: 4 → 4
- 지정 actual failure: 25 → 24로 개선 부족

장기 평가는 50세션 × 20턴, 총 1,000턴으로 실행했다.

| 실패 유형 | 건수 |
|---|---:|
| current section 누락 | 50 |
| 일반 state mutation 오류 | 50 |
| pending과 preference 동시 유지 실패 | 33 |
| 삭제한 preference 재등장 | 33 |
| 합계 | 166 |

실패 원인 분류는 STATE 0, MODEL 166, GOLD/EVAL 0, HARNESS 0이다.

Cause: 신규 후보가 짧은 suite와 실로그 일부는 개선했지만 모든 장기 세션에서 최소 한 번 state를 잘못 사용했다.

Effect: 신규 Decision·Response 후보는 active stack에 채택하지 않았다.

---

## 7. 최종 gate와 활성 모델

통과:

- dataset full audit
- evaluation coverage
- fresh training from base
- strict component·critical regression
- state propagation·multi-turn Decision
- real-log inference clean·adapter smoke·runtime safety

실패:

- actual failure improvement
- long horizon
- preference lifecycle
- recommendation semantics

최종 활성 stack:

```text
Decision       = decision_v3_4_epoch4
Recommendation = recommendation_v2_epoch4
Response       = response_v1_3_epoch3
```

**신규 후보는 `rejected_keep_previous`이며 실제 로봇용 활성 Adapter는 변경하지 않았다.**

---

## 8. 학습 파일 외부화

메인 저장소에 있던 `upgrade_v4`는 Git 미추적 상태로 66GB를 차지했다.

주요 구성:

| 유형 | 대략적 크기 |
|---|---:|
| `.safetensors` | 52.5GB |
| optimizer/checkpoint `.pt` | 11.3GB |
| JSON·tokenizer·trainer state | 6.4GB |

활성 Adapter 3개를 먼저 다음 위치로 이동했다.

```text
models/runtime_adapters/
├── decision_v3_4_epoch4/
├── recommendation_v2_epoch4/
└── response_v1_3_epoch3/
```

그 뒤 전체 학습 트리를 구조 그대로 이동했다.

```text
/home/roma/IRC_SOOMAC3.0/upgrade_v4
→ /home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4
```

이동 직전 기준으로 65GB, 파일 3,425개, 디렉터리 777개이다. 삭제하거나 tar로 중복 복사하지 않았다.

Runtime은 더 이상 `upgrade_v4/full_rebuild_v2/v3/v4`의 adoption manifest를 탐색하지 않는다. `config/llm_active_stack.json` 하나만 읽으며 Adapter 상대경로는 저장소 root 기준으로 해석한다.

---

## 9. 최종 검증

런타임 독립화 뒤 실행한 테스트:

```text
test_adapter_routing.py
test_natural_multiturn_contract.py
test_section_question_events.py
```

결과:

```text
114 passed
2 subtests passed
0 failed
```

확인된 Adapter SHA-256:

| 구성요소 | SHA-256 |
|---|---|
| Decision | `d0d0ce5ad598dc67ee1cf0a4250b07dca307c76f2e5389a7c8c8c5561f344fd9` |
| Recommendation | `65b213714dfa3524a02a479066197a443a2ec3795c6ed1b883c8a5a4709913ad` |
| Response | `83fade2b72d3d444e3f1370fb3ec3a1af8d979912ff39084933516d92ee5e043` |

현재 로봇 실행에는 기존 검증 모델이 연결되며, 외부화된 학습 워크스페이스가 없어도 Adapter 경로 해석이 완료된다.

---

## 10. 남은 계획

다음 재학습은 전체 데이터를 다시 무작정 늘리지 않는다.

1. `current_section_missing`과 `general_state_mutation` 100건의 공통 원인을 분리한다.
2. pending + preference 동시 유지와 preference REMOVE delayed recall을 contrastive family로 보강한다.
3. removed preference를 되살린 Recommendation 1건을 별도 semantic gate로 유지한다.
4. fresh LoRA 후 동일한 97턴×2 replay와 1,000턴 long-horizon을 다시 실행한다.
5. 모든 gate가 통과할 때만 `config/llm_active_stack.json`을 새 Adapter로 변경한다.

---

## 용어 정리

- **canonical state**: 과거 문장을 모델이 기억한다고 가정하지 않고 매 턴 Runtime이 제공하는 현재 확정 주문·취향·제한·pending 상태이다.
- **pending**: 추천·실행·강한 제한 삭제처럼 사용자 확인 전에는 확정할 수 없는 임시 후보이다.
- **turn-isolated replay**: 각 실제 로그 턴을 독립 입력으로 재생하여 모델 자체 의미 오류를 보는 평가이다.
- **full-session replay**: 앞 턴의 state와 history를 이어 실제 대화처럼 재생하는 평가이다.
- **long-horizon evaluation**: 여러 턴 동안 preference·restriction·section·pending을 계속 유지·수정·삭제할 수 있는지 보는 장기 평가이다.
- **fresh LoRA**: 이전 실패 Adapter를 이어 학습하지 않고 base model에서 새 LoRA를 시작하는 방식이다.

(여기서 loss가 낮아도 장기 canonical state를 못 쓰면 채택하면 안 된다는 결론이 나온다.)
