# Decision v3.2 + Response v1.1 Training TAR 제작 결과

- 참고 규격: `/home/roma/Downloads/TRAINING_IMAGE_SPEC.ko_v2_9.md`

  → Decision과 Response를 서로 다른 LoRA로 학습하는 서버 업로드용 Docker image TAR 두 개를 제작했다.

## 1. 최종 산출물

| 대상 | Docker image | TAR | bytes | SHA-256 |
|---|---|---|---:|---|
| Decision v3.2 | `soomac/decision-v3.2:v2.9` | `decision_v3_2_training_bundle_v2_9.tar` | 6,981,597,696 | `20aa4e3a59060c8c08a92ca532d0e2a3065c98bfcee4c588f5fb7f1fd83bbb5a` |
| Response v1.1 | `soomac/response-v1.1:v2.9` | `response_v1_1_training_bundle_v2_9.tar` | 6,981,562,880 | `615d8d7fc21510c645f3dd50cd2baa439f92df36ed03424f3187de983dd33360` |

두 파일은 `docker image save`로 만든 압축하지 않은 `linux/amd64` image TAR이다. dataset 묶음용 `tar.gz`가 아니다.

## 2. 학습 입력 고정

### Decision v3.2

```text
train: 3,465
validation: 285
train SHA-256:
76b2123c0183fcb703163cc5cc0e937d8c97e070c9d0efa3ee417b245e0a5060
validation SHA-256:
c87ab3908b34c22d8f765a5771187f6557fa98268521ab49841330157bb321df
```

### Response v1.1

```text
train: 1,098
validation: 101
train SHA-256:
e16b007a0a3c78711951e8b061322fb656f4c09be410f361522aba6d1803d558
validation SHA-256:
531fefdc8ec1e1eeda8f85edce34ca9c24832b22a6b7e6dacdb1812e12acac6f
```

각 candidate의 validation report를 다시 검사했다.

```text
schema invalid             0
duplicate                  0
frozen overlap             0
semantic suspicious        0
malformed Korean           0
```

Response의 동일 structured input에 서로 다른 자연스러운 정답이 있는 행은 `route + input + target` 완전 중복이 아니므로 유지했다. 이는 candidate 정본의 duplicate 정의와 같다.

## 3. 학습 설정

| 설정 | Decision v3.2 | Response v1.1 |
|---|---:|---:|
| Base model | Gemma 4 12B | Gemma 4 12B |
| Quantization | NF4 4-bit | NF4 4-bit |
| Epoch | 6 | 4 |
| Optimizer step | 1,302 | 276 |
| Learning rate | `1e-4` | `5e-5` |
| Batch / accumulation | 1 / 16 | 1 / 16 |
| Scheduler | cosine | cosine |
| Warmup ratio | 0.03 | 0.03 |
| Max sequence | 4,096 | 4,096 |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 | 16 / 32 / 0.05 |

기존 v3.1/Response v1과 같은 base model, tokenizer, chat template, quantization, LoRA target module, optimizer 계열을 유지했다. 이번 실험의 주 변수는 dataset이다.

## 4. Token length audit

실제 `/home/roma/Desktop/sLLM/gemma-4-12B-it`의 `AutoProcessor.apply_chat_template(enable_thinking=False)`를 사용했다.

### Decision train

| 구분 | mean | median | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| input | 2,169.318 | 2,161 | 2,202 | 2,222 | 2,250 | 2,289 |
| output | 17.195 | 18 | 26 | 31 | 41 | 44 |
| total | 2,186.514 | 2,179 | 2,219 | 2,240 | 2,270 | 2,316 |

### Response train

| 구분 | mean | median | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| input | 702.947 | 746 | 858 | 909 | 935 | 941 |
| output | 28.156 | 28 | 39 | 43 | 48 | 53 |
| total | 731.103 | 766 | 892 | 946 | 984 | 994 |

Decision/Response train과 validation 모두 4,096 tokens 초과 0행, history trim 필요 0행, 최종 truncation 0행이다. 따라서 기존 `MAX_LENGTH=4096`을 유지했다.

## 5. Spec 2.9 계약

| 검사 | Decision | Response |
|---|---|---|
| `lab.training.spec=2.9` | PASS | PASS |
| `lab.training.progress=1` | PASS | PASS |
| `lab.training.resume=output-dir-v1` | PASS | PASS |
| `lab.training.inputs` JSON | PASS | PASS |
| Platform `linux/amd64` | PASS | PASS |
| ENTRYPOINT / CMD | PASS | PASS |
| UID:GID 1000 offline import | PASS | PASS |
| Embedded dataset preflight | PASS | PASS |
| Unit test | 5 / 5 PASS | 5 / 5 PASS |
| `compileall` | PASS | PASS |
| `git diff --check` | PASS | PASS |

이미지는 검증 완료된 `soomac/decision-v3.1:v2.9`의 CUDA/Python dependency layer를 재사용하고, 새 학습 코드·contract·dataset만 별도 layer로 교체했다. 최종 image에는 각 package의 pinned version과 context `SHA256SUMS`가 포함된다.

## 6. 실제 GPU 저장·실패·재개 시험

GPU는 NVIDIA RTX PRO 4500 Blackwell 32GB를 사용했다. Docker daemon에 NVIDIA CDI runtime이 없어 device와 host driver library를 읽기 전용으로 연결했다.

두 image 모두 다음 순서를 실제로 수행했다.

```text
step 1 → 5 학습
checkpoint-5 전체 상태 저장
resume.json 원자적 게시
LAB_TEST_FAIL_AFTER_GLOBAL_STEP=5 강제 실패
checkpoint-5 복원
step 6 → 10 재개
checkpoint-10 저장 및 완료
```

| 검사 | Decision | Response |
|---|---:|---:|
| Trainable LoRA parameters | 65,568,768 | 65,568,768 |
| Gradient finite | PASS | PASS |
| Gradient nonzero | PASS | PASS |
| Skipped optimizer update | 0 | 0 |
| Safety gate | PASS | PASS |
| checkpoint-5 hash 불변 | PASS | PASS |
| checkpoint-10 optimizer state | PASS | PASS |
| Peak allocated VRAM | 20.853 GiB | 15.409 GiB |
| Peak reserved VRAM | 22.494 GiB | 16.248 GiB |

Decision checkpoint-5 Adapter hash:

```text
b86b022c0e5026ec655ebb76d6365689eed1cd523a62acd5cc87c078a1d9de27
```

Response checkpoint-5 Adapter hash:

```text
c91a934ed1ac0fdd12ae7ab81d23cb099242c7f1cd048ab587ac777e2a3a8eef
```

## 7. Epoch 저장과 best 선택

Decision은 6개, Response는 4개의 epoch checkpoint와 standalone `epoch_N_adapter`를 저장한다.

학습 직후의 `best_adapter`는 validation loss 최저 checkpoint이다. 최종 채택 epoch는 이 값만 사용하지 않고, 반환된 모든 epoch Adapter를 frozen semantic evaluation에 통과시킨 뒤 별도로 선택한다.

```text
Decision 우선순위:
schema → catastrophic regression → fake mutation → precision → recall → route → loss

Response 우선순위:
RED failure → fact → pending → execution → reminder → unnecessary follow-up → loss
```

## 8. 현재 상태

```text
Decision v3.2 full training: NOT STARTED
Response v1.1 full training: NOT STARTED
Runtime default Adapter: 변경하지 않음
기존 v3.1 / Response v1 Adapter: 보존
TAR 판정: READY FOR REMOTE TRAINING
```

Cause: A/B 평가 전에 runtime을 바꾸면 baseline과 candidate 조건이 섞인다.

Effect: 두 신규 Adapter는 독립 TAR에서 학습되며, 결과 평가가 끝날 때까지 현재 runtime baseline은 유지된다.
