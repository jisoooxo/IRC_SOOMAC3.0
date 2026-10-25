# Natural Multiturn V3 Data Quality Report

참고 결과: [`quality_report.json`](quality_report.json), [`reports/decision_semantic_audit.json`](reports/decision_semantic_audit.json), [`reports/response_runtime_contract.json`](reports/response_runtime_contract.json), [`reports/response_state_consistency.json`](reports/response_state_consistency.json), [`reports/language_quality.json`](reports/language_quality.json)

→ schema, structured fact consistency, next prompt 진행성, duplicate와 split contamination을 검사한다.

## Decision validation

- status: **PASS**
- schema invalid: 0
- legacy field: 0
- duplicate: 0
- contamination: 0
- semantic audit: **PASS**
- unresolved suspicious mutation: 0

## Response validation

- status: **PASS**
- invalid input: 0
- empty reply: 0
- fact-consistency failure: 0
- next_prompt failure: 0
- duplicate: 0
- contamination: 0
- state consistency: **PASS**
- runtime contract: **PASS**
- runtime contract mismatch: 0
- completed/pending contradiction: 0
- language quality: **PASS**

Response 사실성 검사는 structured field와 실행·pending·next_prompt 관련 명백한 문구만 검사한다. 자유로운 자연어 의미를 다시 판정하는 semantic parser는 사용하지 않는다.

## Self-audit

- runtime source 변경: 0
- old model prediction을 gold로 사용: 0
- frozen evaluation train contamination: 0
- duplicate leakage: 0
- Decision high-risk review: 200
- Response high-risk review: 200

## 용어 정리

- **External sparse Decision**: 현재 발화에 필요한 field만 출력하는 8-field 외부 계약이다.
- **Structured fact consistency**: Response가 이미 확정된 Python state와 모순되지 않는지 확인하는 검사이다.
- **Contamination**: 같은 발화·semantic group·paraphrase group이 train과 evaluation에 함께 들어가는 현상이다.
- **High-risk review**: STT, reference, compound, pending, validation conflict처럼 사람 검토 가치가 높은 표본이다.

## 최종 판정

**READY FOR TRAINING**
