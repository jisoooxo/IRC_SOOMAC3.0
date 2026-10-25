# Natural Multiturn V3 Rebuild

참고 계약: [`BUILD_REPORT.md`](BUILD_REPORT.md), [`DATA_QUALITY_REPORT.md`](DATA_QUALITY_REPORT.md)

→ Decision v3와 Response v1을 runtime `25ce994` 기준으로 재구축한 학습 bundle이다.

## 재현

```bash
python3 data_natural_multiturn_v3_rebuild/scripts/build_datasets.py
python3 data_natural_multiturn_v3_rebuild/scripts/validate_decision.py
python3 data_natural_multiturn_v3_rebuild/scripts/validate_response.py
python3 data_natural_multiturn_v3_rebuild/scripts/validate_response_runtime_contract.py
python3 data_natural_multiturn_v3_rebuild/scripts/audit_decision_semantics.py
python3 data_natural_multiturn_v3_rebuild/scripts/validate_response_state.py
python3 data_natural_multiturn_v3_rebuild/scripts/audit_language_quality.py
python3 data_natural_multiturn_v3_rebuild/scripts/check_contamination.py
python3 data_natural_multiturn_v3_rebuild/scripts/report_stats.py
```

## 구조

- `decision_v3/`: train, validation, test, frozen challenge, frozen holdout, high-risk review이다.
- `response_v1/`: train, validation, test, challenge, holdout, high-risk review이다.
- `intermediate/`: current input/target와 audit metadata를 함께 보존한다.
- `provenance/`: source와 relabel 방법을 row ID별로 추적한다.
- `reports/`: validation과 통계의 machine-readable 결과이다.

## 계약

Decision input은 `recent_history`, `order`, `preferences`, `pending`, `robot_state`, `message`만 사용한다. Response input은 runtime의 14개 field와 정확히 일치한다.

최종 상태: **READY FOR TRAINING**
