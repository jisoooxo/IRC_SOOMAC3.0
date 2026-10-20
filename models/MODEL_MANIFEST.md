# Runtime Adapter Model Manifest

- 기준일: 2026-10-20
- 작업 브랜치/HEAD: `flexible_llm` / `1637e57`
- Base: `/home/roma/Desktop/sLLM/gemma-4-12B-it` (`nf4`)
- Runtime 선택 정본: `config/llm_active_stack.json`
- 평가 보고서: `/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/full_rebuild_v4/FINAL_AUDITED_RETRAIN_REPORT.md`

## 활성 Adapter

| 구성요소 | 경로 | SHA-256 |
|---|---|---|
| Decision | `models/runtime_adapters/decision_v3_4_epoch4` | `d0d0ce5ad598dc67ee1cf0a4250b07dca307c76f2e5389a7c8c8c5561f344fd9` |
| Recommendation | `models/runtime_adapters/recommendation_v2_epoch4` | `65b213714dfa3524a02a479066197a443a2ec3795c6ed1b883c8a5a4709913ad` |
| Response | `models/runtime_adapters/response_v1_3_epoch3` | `83fade2b72d3d444e3f1370fb3ec3a1af8d979912ff39084933516d92ee5e043` |

## 2026-10-20 재학습 판정

- Decision 후보: `decision_v3_7_state_coverage_audited_epoch4`
- Recommendation: `recommendation_v2_epoch4` 유지
- Response 후보: `response_v1_4_audited_epoch2`
- 판정: `rejected_keep_previous`
- 원인: 1,000턴 장기 평가에서 166턴의 모델 state 오류가 발생했다.
- 결과: 위 활성 Adapter 세 개를 계속 사용한다.

## Runtime 독립성

Runtime은 외부 학습 워크스페이스의 checkpoint나 adoption manifest를 읽지 않는다. `config/llm_active_stack.json`이 없거나 손상되면 동일한 검증 Adapter 세 개를 rollback 경로로 사용한다.
