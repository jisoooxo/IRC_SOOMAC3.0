# Decision v3.2 Training Image

→ Decision v3.2 candidate 3,465행을 Gemma 4 12B base에서 독립 NF4 QLoRA로 학습한다.

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
epochs: 6
optimizer steps: 1,302
learning rate: 1e-4
batch / accumulation: 1 / 16
max length: 4,096
LoRA: r=16, alpha=32, dropout=0.05
```

각 epoch에 validation과 checkpoint 저장을 수행한다. `epoch_N_adapter`는 epoch별 standalone Adapter이며 `best_adapter`는 validation loss 기준 임시 선택이다. 최종 best epoch는 반환된 모든 epoch Adapter에 frozen semantic evaluation을 실행한 뒤 결정한다.

## 재개

`LAB_RESUME=1`, `RESUME_CHECKPOINT_DIR`, `RESUME_STEP`을 받으면 Adapter, optimizer, scheduler, Trainer state, RNG를 복원한다. 체크포인트가 없거나 step이 다르면 처음부터 재시작하지 않고 오류 종료한다.
