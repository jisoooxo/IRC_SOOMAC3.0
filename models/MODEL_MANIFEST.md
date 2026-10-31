# Runtime Adapter Model Manifest

## 2026-10-31 — 사용자 승인 최신 best 실물 테스트 준비

**현재 선택 정본은 `src/soomac_irc/soomac_irc/model_runtime.py`의 직접 경로 상수 3개이다.** `config/llm_active_stack.json`은 같은 경로를 기록하는 비활성 정보용 사본이다. `active=false`, `all_gates_pass=false`는 의미 회귀 미통과/미검증임을 정직하게 표시하며, 현재 직접 경로의 로딩을 막거나 다른 모델로 바꾸지 않는다.

| Agent | 최신 완료 학습의 best | weight SHA-256 |
|---|---|---|
| Decision | 원격 v5, step750 / epoch2.944417 | `c2cf15bb3a351c016974b69e5126b427373c9f06a9dbe367b402daf038c8a3d4` |
| Recommendation | 로컬 v5, step100 / epoch2.642140 | `cda007d8e7e985c483a96821cf32e9b6e0fcb8b713e3305c5c0d905e45c7e9de` |
| Response | 로컬 v5 재개 계보, step50 / epoch0.551724 | `7f528012569df694ecefa0bcb2b816c26dec0875995fdbd300e6fb32dc9a5dff` |

전체 경로와 후속 검수는 [통합 계획](../docs/IRC%20LLM%20Final%20integrated%20validation%20plan.md) §2를 따른다. best는 loss 최저 임시선정이며 실제 전체 대화 통과를 뜻하지 않는다. 10월31일 alignment 보강 데이터는 아직 재학습에 반영되지 않았다. 이번에는 경로/CPU 검사만 수행하고 새 GPU·ROS·학습을 실행하지 않는다.

현재 경로는 외부 학습 작업장에 있는 보존된 export를 직접 읽는다. 아래 10월20일의 “외부 작업장을 읽지 않는다/manifest가 정본” 설명은 **과거 운영 정책**이며 현재 직접 경로 정책과 혼동하지 않는다. 해당 파일을 이동·삭제하면 다음 기동이 실패할 수 있다. 이미 실행 중인 프로세스에는 소스 경로 변경이 소급 적용되지 않는다.

### 복구

기존 `ROLLBACK_DECISION_ADAPTER_PATH`, `ROLLBACK_RECOMMENDATION_ADAPTER_PATH`, `ROLLBACK_RESPONSE_ADAPTER_PATH`와 실제 파일은 보존했다. 되돌릴 때는 **직접 경로 상수 3개를 각각 해당 rollback 상수 값으로 변경하고 프로세스를 사용자 승인하에 재기동한다.** 정보용 manifest만 바꿔서는 현재 선택이 바뀌지 않는다. 자동 복구/자동 재기동은 하지 않는다. VLM/ROS/실행 안전 검증은 변경하지 않았다.

## 2026-10-20 — 이전 운영 기록 (보존)

- 기준일: 2026-10-20
- 작업 브랜치/HEAD: `flexible_llm` / `1637e57`
- Base: `/home/roma/Desktop/sLLM/gemma-4-12B-it` (`nf4`)
- Runtime 선택 정본: `config/llm_active_stack.json`
- 평가 보고서: `/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/full_rebuild_v4/FINAL_AUDITED_RETRAIN_REPORT.md`

### 이전 활성 Adapter (2026-10-20)

| 구성요소 | 경로 | SHA-256 |
|---|---|---|
| Decision | `models/runtime_adapters/decision_v3_4_epoch4` | `d0d0ce5ad598dc67ee1cf0a4250b07dca307c76f2e5389a7c8c8c5561f344fd9` |
| Recommendation | `models/runtime_adapters/recommendation_v2_epoch4` | `65b213714dfa3524a02a479066197a443a2ec3795c6ed1b883c8a5a4709913ad` |
| Response | `models/runtime_adapters/response_v1_3_epoch3` | `83fade2b72d3d444e3f1370fb3ec3a1af8d979912ff39084933516d92ee5e043` |

### 2026-10-20 재학습 판정

- Decision 후보: `decision_v3_7_state_coverage_audited_epoch4`
- Recommendation: `recommendation_v2_epoch4` 유지
- Response 후보: `response_v1_4_audited_epoch2`
- 판정: `rejected_keep_previous`
- 원인: 1,000턴 장기 평가에서 166턴의 모델 state 오류가 발생했다.
- 결과: 위 활성 Adapter 세 개를 계속 사용한다.

### 이전 Runtime 독립성

Runtime은 외부 학습 워크스페이스의 checkpoint나 adoption manifest를 읽지 않는다. `config/llm_active_stack.json`이 없거나 손상되면 동일한 검증 Adapter 세 개를 rollback 경로로 사용한다.
