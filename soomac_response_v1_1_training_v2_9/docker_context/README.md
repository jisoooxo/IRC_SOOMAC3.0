# Response v1.1 Training Image

→ Response v1.1 candidate 1,098행을 Gemma 4 12B base에서 Decision과 분리된 NF4 QLoRA로 학습한다.

## 실행 계약

```text
Spec: 2.9
Progress: schema 1
Resume: output-dir-v1
Input model: MODEL_DIR
Embedded dataset: /app/assets/dataset
Output: CHECKPOINT_DIR
Default command: /app/run_training.sh
```

## 학습

```text
epochs: 4
optimizer steps: 276
learning rate: 5e-5
batch / accumulation: 1 / 16
max length: 4,096
LoRA: r=16, alpha=32, dropout=0.05
```

각 epoch에 validation과 checkpoint 저장을 수행한다. `epoch_N_adapter`는 epoch별 standalone Adapter이며 `best_adapter`는 validation loss 기준 임시 선택이다. 최종 best epoch는 pending reminder와 불필요한 follow-up을 포함한 semantic evaluation 뒤 결정한다.

## 재개

`LAB_RESUME=1`, `RESUME_CHECKPOINT_DIR`, `RESUME_STEP`을 받으면 Adapter, optimizer, scheduler, Trainer state, RNG를 복원한다. 체크포인트가 없거나 step이 다르면 처음부터 재시작하지 않고 오류 종료한다.
