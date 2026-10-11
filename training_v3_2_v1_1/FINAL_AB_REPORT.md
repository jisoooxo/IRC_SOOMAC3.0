# Decision v3.2 + Response v1.1 최종 A/B 평가

## 최종 판정

- **ADOPT DECISION V3.2 (epoch 4)**
- **KEEP RESPONSE V1**
- **Selected pair: Decision v3.2 + Response v1**
- **READY FOR ROBOT TEST**

## Decision

| metric | v3.1 | v3.2 |
|---|---:|---:|
| schema | 100.00% | 100.00% |
| route | 98.70% | 100.00% |
| frozen order precision | 93.67% | 94.82% |
| frozen order recall | 97.75% | 97.89% |
| frozen fake mutation | 47 | 38 |
| frozen missed mutation | 16 | 15 |
| long order precision | 96.19% | 98.08% |
| long order recall | 99.02% | 100.00% |
| long Decision failures | 10 | 6 |
| long state/pending/policy failures | 0/0/0 | 0/0/0 |

Cause: v3.2가 frozen precision과 long-horizon precision/recall을 높이고 extra/missed mutation 및 실제 Decision 실패를 줄였다.

Effect: runtime Decision 기본값을 epoch 4로 교체할 수 있다.

## Response

| metric | v1 | v1.1 |
|---|---:|---:|
| frozen RED failures | 0 | 2 |
| fact consistency | 100.00% | 99.56% |
| pending consistency | 100.00% | 100.00% |
| next prompt | 100.00% | 100.00% |
| hallucination free | 100.00% | 99.56% |
| long Response failures | 15 | 54 |
| targeted train-fit RED | 185 | 2 |

Targeted 200행은 학습행 재검증이므로 일반화 판정에는 사용하지 않았다.

Cause: v1.1은 목표 pending family를 학습했지만 독립 frozen RED 2건과 long-horizon Response 실패 54건으로 회귀했다.

Effect: runtime Response는 기존 v1을 유지한다.

## E2E

| combination | sessions | successful sessions | turns | successful turns | switching failures |
|---|---:|---:|---:|---:|---:|
| A: v3.1 + v1 | 50 | 49 | 150 | 149 | 0 |
| B: v3.2 + v1 | 50 | 49 | 150 | 149 | 0 |
| D: v3.2 + v1.1 | 50 | 49 | 150 | 149 | 0 |

세 조합 모두 같은 1개 Decision 실패를 기록했다. Policy/runtime 및 Adapter switching 실패는 0건이다.

## 무결성 및 적용

- TAR SHA256: `025f763aced60a09cee3ee22334660109be69fcaa37de2e27e8d2c67892ab69f`
- 학습 safety gate: Decision/Response 모두 PASS
- token truncation: Decision/Response train·validation 모두 0행
- runtime Decision: `models/decision_adapter_v3_2_best_epoch4`
- runtime Response: `models/response_adapter_v1_best_epoch1`
- runtime smoke: 8턴 / 20 adapter events / 오류 0건
- rollback Decision v3.1과 Response v1 원본은 보존했다.
