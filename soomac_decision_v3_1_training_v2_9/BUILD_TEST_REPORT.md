# Decision v3.1 재학습 이미지 빌드·검증 결과

## 1. 최종 산출물

```text
Image: soomac/decision-v3.1:v2.9
Platform: linux/amd64
TAR: soomac_decision_v3_1_training_v2_9.tar
TAR size: 6.6 GiB
TAR SHA-256: fe146c1f4702cf0ab18034fb8f5c4c5a4f8c1f8a5a24f4047da7477393a1fb3f
```

## 2. 입력 자료

| 항목 | 행 | SHA-256 |
|---|---:|---|
| Decision v3.1 train | 3,015 | `d011cff5dc0460ec55bfd4c7b458fcc88b184064dab69ae0e3e4a963b7f895c4` |
| Frozen validation | 285 | `c87ab3908b34c22d8f765a5771187f6557fa98268521ab49841330157bb321df` |
| Candidate source ZIP | - | `bc31fe6e253a4c6beaac0fbfb3f0c3bc0c9e107622b3373b1754089b44a654b6` |

독립 `decision_v3_1_train.jsonl`과 candidate ZIP 내부 `train.jsonl`의 hash가 동일함을 확인했다.

## 3. 학습 설정

```text
Base Model: google/gemma-4-12B-it
Quantization: NF4 4-bit
Epochs: 6
Optimizer steps: 1,134 (189 per epoch)
Learning rate: 1e-4
Batch size: 1
Gradient accumulation: 16
LoRA: r=16, alpha=32, dropout=0.05
Checkpoint/Eval: every epoch
```

## 4. Spec 2.9 정적 검증

| 검사 | 결과 |
|---|---|
| `lab.training.spec=2.9` | PASS |
| `lab.training.progress=1` | PASS |
| `lab.training.resume=output-dir-v1` | PASS |
| `lab.training.inputs` JSON | PASS |
| Entrypoint / CMD | PASS |
| linux/amd64 | PASS |
| UID:GID 1000 import | PASS |
| 내장 dataset preflight | PASS, train 3,015 / validation 285 |
| Python unit test | PASS, 5 / 5 |
| `compileall` | PASS |
| `git diff --check` | PASS |

## 5. 실제 이미지 GPU 저장·실패·재개 시험

빌드된 `soomac/decision-v3.1:v2.9` 이미지 자체를 RTX PRO 4500 32GB에서 실행했다.

```text
1차 실행: step 1 → 5
checkpoint-5 저장 완료
resume.json = decision/checkpoint-5, global_step=5
LAB_TEST_FAIL_AFTER_GLOBAL_STEP=5로 의도적 실패

2차 실행: checkpoint-5 복원
start_step=5 확인
step 6 → 10 진행
checkpoint-10 저장 완료
resume.json = decision/checkpoint-10, global_step=10
```

| 검사 | 결과 |
|---|---|
| NF4 모델 로딩 | PASS |
| LoRA trainable parameter | 65,568,768 |
| 유한 loss | PASS |
| 유한 gradient | PASS |
| nonzero gradient | PASS |
| skipped optimizer update | 0 |
| 10-step safety gate | PASS |
| checkpoint-5 원본 checksum 불변 | PASS |
| checkpoint-10 전체 상태 파일 | PASS |
| 재개 후 train loss | 0.1836494857 |
| 재개 후 eval loss | 0.1590311080 |
| peak reserved VRAM | 22.494 GiB |

Docker daemon에 NVIDIA runtime/CDI 등록이 없어 `--gpus all`은 사용할 수 없었다. 시험에서는 동일 Docker 이미지에 GPU device와 host driver library를 읽기 전용으로 직접 연결했다. 이미지 내부 PyTorch는 CUDA 12.8과 RTX PRO 4500을 인식했고 실제 학습·저장·복원을 완료했다.

## 6. 서버 제출 권장값

```text
Model: 이미지 기본값 google/gemma-4-12B-it 또는 동일 파일의 준비된 cache
Dataset: 이미지 내장 dataset
GPU: 공유 1개
GPU memory fraction: 0.90
CPU: 8 cores
RAM: 24 GiB
Shared memory: 4 GiB
```

가장 긴 smoke sample을 포함한 시험에서 peak reserved VRAM이 22.494 GiB였으므로 32GB GPU에서 0.90 예산을 사용한다.

## 7. 판정

```text
READY FOR REMOTE TRAINING
```

Cause: Decision v3.1은 기존 v3와 같은 Base Model·LoRA 조건에서 비교해야 한다.

Effect: epoch별 Adapter 6개, best Adapter, optimizer 포함 checkpoint, `resume.json`을 생성하는 Spec 2.9 학습 TAR이 준비됐다.
