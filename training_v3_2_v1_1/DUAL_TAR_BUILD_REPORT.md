# Decision v3.2 + Response v1.1 Dual TAR Build Report

→ 단일 원격 작업에서 Decision v3.2 학습 후 Response v1.1 학습이 자동으로 이어지는 이미지이다.

## 결과물

```text
Image: soomac/decision-v3.2-response-v1.1-dual:v2.9
TAR: decision_v3_2_response_v1_1_dual_training_bundle_v2_9.tar
Size: 6,981,790,208 bytes
SHA-256: 879fd14fa1babb847705bbee6cc33dd7e58df88a3ea0e1c7a395fe57d13fcaec
Platform: linux/amd64
```

## 학습 순서

```text
1. Decision v3.2: 3,465 train / 285 validation / 6 epochs / 1,302 optimizer steps
2. Response v1.1: 1,098 train / 101 validation / 4 epochs / 276 optimizer steps
3. Total: 1,578 optimizer steps
```

두 Adapter는 같은 Gemma 4 12B Base Model에서 각각 독립적으로 학습한다. Decision Adapter를 Response 초기값으로 사용하지 않는다.

## 검증 결과

```text
Dataset preflight: PASS
Bundle unit tests: 5 / 5 PASS
compileall: PASS
git diff --check: PASS
Image labels: spec=2.9, progress=1, resume=output-dir-v1
```

실제 GPU smoke는 Decision 10 step을 완료한 뒤 Response 5 step에서 의도적으로 중단했다. 이후 `/checkpoints/response/checkpoint-5`와 누적 step 15로 재개하여 전체 20 step을 완료했다.

```text
Decision → Response 자동 전환: PASS
Response checkpoint 생성: PASS
Response 단계 resume: PASS
최종 dual_training_complete.json 생성: PASS
```

## 출력 구조

```text
/checkpoints/decision/checkpoint-*
/checkpoints/decision/epoch_1_adapter ... epoch_6_adapter
/checkpoints/decision/best_adapter
/checkpoints/response/checkpoint-*
/checkpoints/response/epoch_1_adapter ... epoch_4_adapter
/checkpoints/response/best_adapter
/checkpoints/dual_training_complete.json
```
