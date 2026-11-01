# Recommendation 로컬 학습 속도 설정 후보

참고: [최신 데이터 MANIFEST](/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/candidates/compound_semantics_20261101_r2/dataset/MANIFEST.json), [토큰 영수증](/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/candidates/compound_semantics_20261101_r2/validation/TOKEN_MASK_ROWS.jsonl), [학습 실행기](../training/learning_first_v43/training_release_20261101_r5/common/train_agent.py), [프로파일러](../training/learning_first_v43/training_release_20261101_r5/common/profiling.py)
→ 최신 데이터와 실제 토큰 길이를 유지하면서 Recommendation의 속도 후보를 GPU에서 선택하도록 한다.

설정 파일: [recommendation_speed.json](../training/learning_first_v43/training_release_20261101_r5/local/recommendation_speed.json). **GPU 미측정 후보**이며 `local/launch.py` 연결과 실제 학습 결과를 뜻하지 않는다.
원본 MANIFEST의 `ENCODING_PASS_SEMANTIC_REVIEW_PENDING` 및 `user_training_authorized=false`는 원본 당시 기록이다. 이 JSON의 `requires_gpu_measurement`는 GPU 측정 상태만 나타내며 의미 검토나 학습 승인 상태를 대신하지 않는다.

## 데이터와 길이 근거

- MANIFEST SHA-256 `7850948287237c5ace9d1b0f56931419760fbce3f0afa02cea40681acb6ceffa`; Recommendation 학습 623행, 검증 30행, `max_length=5120`이다. 실제 JSONL 행수도 일치한다.
- 해당 MANIFEST가 지목한 `TOKEN_MASK_ROWS.jsonl` SHA-256은 `013e0f3f1630c8590b28ca0047cf3292acbc05621b9196a19738a0bec333aee3`이다. 학습 최대 4,201토큰, 검증 최대 4,332토큰이며 둘 다 5,120 이내이다. 학습 89행·검증 5행은 4,096을 초과한다. 영수증의 `history_trimmed`는 두 split 모두 0건이다.
- `max_length`, 원본 데이터·history·prompt, 6 epoch, 학습률 `1e-4`, 유효 배치 16, NF4/LoRA 구조는 유지한다. 623행 기준 목표 optimizer update 수는 `ceil(623/16) × 6 = 234`이다.

## 속도 후보와 메모리 경계

- **로컬 Recommendation 후보**의 프로세스 요청 예산은 `32,000,000,000`바이트, allocator 밖 reserve는 `536,870,912`바이트(512MiB)이다. 요청이 GPU·서버 한도에 걸리지 않을 때 PyTorch allocator 한도 후보는 `31,463,129,088`바이트이다. 실제 한도는 `min(요청, GPU 총량, 서버 제한)`으로 정해지므로 사용 가능한 VRAM 전체를 보장하지 않는다. 512MiB는 실행 코드가 허용하는 최소 reserve이며 원격 동결 build05의 1GiB 설정은 변경하지 않는다.
- `gradient_checkpointing=auto`는 OFF와 ON에서 각각 microbatch 1·2·4·8을 유한 탐색한다. 최장/최대 정답 범위 stress와 유효 배치 16의 누적 optimizer update를 재고, 정상 후보 중 실측 `sequences_per_second` 최고값을 고른다. OFF가 더 빠르고 예산에 맞으면 OFF가 선택될 수 있다. batch/모드는 현재 확정하지 않는다.
- 프로파일은 단일 누적 update 표본에 가깝다. Decision의 기존 1GiB reserve 프로파일에서 ON·batch 2가 OOM이었다는 사실은 로컬 512MiB 시험의 동기일 뿐, Recommendation의 적합성이나 속도를 예측하지 않는다. 실제 GPU free, 타 프로세스, allocator 밖 메모리, 최장 역전파의 적합성과 장시간 학습 속도는 **GPU 측정 및 첫 저장 검증 전까지 미확정**이다. 이전 Response batch 1의 26.87 GiB 수치는 Recommendation 측정치로 전용하지 않는다.

## 용어 정리

- **유효 배치(effective batch)**: optimizer update 한 번에 누적하는 예시 16개이다. microbatch 1·2·4·8에 따른 누적 횟수는 16·8·4·2이다.
- **기울기 체크포인팅(gradient checkpointing)**: 일부 중간 활성값을 역전파 때 다시 계산해 메모리를 줄이는 선택이다. `(내가 의역해보자면 메모리와 재계산 시간 사이의 실측 선택이다.)`
- **프로세스 예산(process budget)**: NVML 관측 대상인 학습 프로세스의 요청 상한이다. reserve는 PyTorch allocator 외 사용량을 위해 남긴다.
