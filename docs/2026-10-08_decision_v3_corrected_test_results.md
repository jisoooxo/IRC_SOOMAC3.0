# Decision v3 Corrected Long-Horizon 테스트 결과

- 참고: [Corrected Long-Horizon Report](../evaluation_long_horizon_corrected/CORRECTED_LONG_HORIZON_REPORT.md)

  → 현재 Decision contract로 기존 360턴을 다시 채점한 정본이다.
- 참고: [Evaluator Contract Changes](../evaluation_long_horizon_corrected/evaluator_contract_changes.md)

  → 기존 평가 기준에서 무엇을 왜 수정했는지 기록한다.
- 참고: [Decision v3.1 Candidate Validation](../evaluation_long_horizon_corrected/decision_v3_1_candidate/validation_report.json)

  → targeted supplement의 schema, 중복, contamination 검사 결과이다.

## 1. 테스트 범위

기준 baseline은 다음과 같다.

```text
Git HEAD: 509be6035b7732b8f0d6258dc2f73a465936da2a
Decision Adapter: decision_adapter_v3_best_epoch3
Response Adapter: response_adapter_v1_best_epoch1
Long-horizon sessions: 30
Long-horizon turns: 360
```

이번 테스트에서는 기존 inference 결과를 그대로 사용했다. GPU inference를 다시 실행하지 않았으며 runtime, prompt, Decision/Response Adapter도 수정하지 않았다.

## 2. Evaluator contract 수정 결과

| 검사 항목 | 결과 |
|---|---:|
| 명확한 unsupported 주문 gold 수정 | 3턴 |
| 허용 가능한 `confirmation=accept + commit=true` 인정 | 12턴 |
| recommendation criteria gold 수정 | 1턴 |
| 자유 preference 표면형 정규화 | 1턴 |

기존 fake/extra mutation 70건은 다음과 같이 재분류했다.

| 분류 | 건수 | 의미 |
|---|---:|---|
| `TRUE_FAILURE` | 34 | 실제 Decision 오류이다. |
| `CONTRACT_CORRECT` | 30 | 현재 contract에서는 올바른 출력이다. |
| `AMBIGUOUS_EVAL` | 6 | 입력만으로 정답을 하나로 확정할 수 없다. |

**명확한 unsupported 주문은 Decision이 대상 문자열을 보존하고, Policy가 실제 반영을 차단하는 것이 현재 contract이다.**

(내가 의역해보자면 모델은 사용자의 뜻을 기록하고, 실제 로봇이 할 수 있는지는 코드가 마지막에 판단한다.)

## 3. Corrected long-horizon 결과

| 지표 | 결과 |
|---|---:|
| 완전 성공 session | 14 / 30 |
| strict turn success | 283 / 360 |
| root-cause clean turn | 326 / 360 |
| Order precision | 91.0714% |
| Order recall | 100.0000% |
| Order value accuracy | 100.0000% |
| Fake mutation turn | 14 |
| Missed mutation turn | 5 |
| Over-clarify / Under-clarify | 1 / 0 |
| Reference / Pending / State drift failure | 0 / 0 / 0 |
| Policy 자체 failure | 0 |

`strict turn success`는 이전 오류가 이후 state에 남긴 영향까지 실패로 계산한다. `root-cause clean turn`은 해당 턴에서 새로 발생한 Decision, Response, Recommendation의 직접 오류만 계산한다.

### 실제 long-horizon over-mutation

| 원인 family | 건수 |
|---|---:|
| pending 거절을 주문 삭제까지 확장 | 5 |
| preference를 특정 order로 과도하게 grounding | 4 |
| 상태 질문을 mutation으로 변환 | 3 |
| 단순 감상을 mutation으로 변환 | 2 |

## 4. Response와 Recommendation 재분류

| 대상 | 분류 | 건수 |
|---|---|---:|
| Response | `RESPONSE_TRUE_FAILURE` | 9 |
| Response | `DOWNSTREAM_OF_DECISION` | 11 |
| Response | `DOWNSTREAM_OF_RECOMMENDATION` | 0 |
| Response | `DOWNSTREAM_OF_EVAL_GOLD_ERROR` | 0 |
| Recommendation | Agent 자체 failure | 6 |

Decision/state가 먼저 틀린 11건은 Response Adapter 오류로 중복 집계하지 않았다.

## 5. Decision v3.1 targeted candidate 검증

기존 Decision v3 train 2,595행은 그대로 유지하고, corrected `TRUE_FAILURE` family만 사용해 supplement 420행을 생성했다.

```text
base train: 2,595 rows
targeted supplement: 420 rows
candidate train: 3,015 rows
training: NOT STARTED
```

| 검증 항목 | 결과 |
|---|---:|
| Decision schema | 420 / 420 PASS |
| Supplement ID 중복 | 0 |
| Supplement input 중복 | 0 |
| Supplement message 중복 | 0 |
| Base train ID/message/input overlap | 0 / 0 / 0 |
| Validation/test/challenge/holdout message overlap | 0 |
| Validation/test/challenge/holdout input overlap | 0 |
| 한국어 조사·양 표현 오류 | 0 |

Execution pending 60행은 실제 `current_update` runtime 구조와 동일하게 다음 필드만 사용한다.

```json
{
  "type": "execution",
  "source": "current_update",
  "section": "veggie",
  "targets": ["양파"]
}
```

## 6. 정적 검증

| 명령 | 결과 |
|---|---|
| `python -m compileall` | PASS |
| `git diff --check` | PASS |
| JSON/JSONL 생성 및 재읽기 | PASS |
| 언어 품질 검사 | 420행, 오류 0 |

검증에는 `/home/roma/miniconda3/envs/gemma4_env/bin/python`을 사용했다. 시스템 기본 `python` 명령은 설치되어 있지 않으므로 재현 시 동일 interpreter를 사용해야 한다.

## 7. 생성 산출물

```text
evaluation_long_horizon_corrected/
├── CORRECTED_LONG_HORIZON_REPORT.md
├── corrected_metrics.json
├── corrected_failure_cases.jsonl
├── corrected_overmutation_analysis.json
├── corrected_turns.jsonl
├── response_failure_reclassification.json
├── recommendation_failure_analysis.json
├── evaluator_contract_changes.md
└── decision_v3_1_candidate/
    ├── supplement_train.jsonl
    ├── supplement_provenance.jsonl
    ├── train.jsonl
    ├── validation_report.json
    ├── manifest.json
    └── supplement_design.md
```

## 8. 판정

```text
CORRECTED BASELINE READY
KEEP DECISION V3
Decision v3.1 training: NOT STARTED
```

Cause: 기존 evaluator가 unsupported 의미 보존과 허용 가능한 pending 출력을 실패로 계산했다.

Effect: 평가 오류를 제거한 뒤 실제 Decision 실패만 v3.1 보강 대상으로 분리했다.

## 용어 정리

- **Contract**: Decision이 어떤 의미와 JSON 구조를 출력해야 하는지 정한 규칙이다.
- **Unsupported target**: 사용자가 요청했지만 현재 로봇 메뉴나 장비가 지원하지 않는 대상이다.
- **Over-mutation**: 사용자가 바꾸라고 하지 않은 field까지 Decision이 추가·삭제·변경한 경우이다.
- **Downstream failure**: 앞 단계의 오류 때문에 뒤 단계가 잘못된 결과를 낸 경우이다.
- **Contamination**: 학습 데이터에 평가용 문장이나 입력이 섞인 상태이다.
