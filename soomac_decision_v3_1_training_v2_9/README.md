# SOOMAC Decision v3.1 Adapter Training Image

- 참고 규격: `/home/roma/Downloads/TRAINING_IMAGE_SPEC.ko_v2_9.md`

→ Decision v3.1 targeted candidate 3,015행을 Gemma 4 12B 원본 모델에서 NF4 QLoRA로 재학습한다.

## 학습 설정

```text
train / validation: 3,015 / 285
epochs: 6
optimizer steps: 1,134 (189 per epoch)
learning rate: 1e-4
batch size: 1
gradient accumulation: 16
LoRA: r=16, alpha=32, dropout=0.05
target: text decoder attention + MLP projections
```

기존 Decision v3 Adapter에서 이어서 학습하지 않는다. 비교 조건을 유지하기 위해 Gemma 4 12B Base Model에서 새 LoRA를 생성한다.

## 체크포인트

```text
/checkpoints/
├── decision/
│   ├── checkpoint-189/ ... checkpoint-1134/
│   ├── epoch_1_adapter/ ... epoch_6_adapter/
│   ├── best_adapter/
│   ├── run_manifest.json
│   └── training_complete.json
├── decision_v3_1_training_complete.json
└── .lab-monitor/
    ├── progress.json
    └── resume.json
```

`checkpoint-*`는 Adapter, optimizer, scheduler, Trainer state, RNG를 포함한다. `epoch_*_adapter`와 `best_adapter`는 배포용 독립 Adapter이다.

## 입력 고정

```text
Decision v3.1 train SHA-256:
d011cff5dc0460ec55bfd4c7b458fcc88b184064dab69ae0e3e4a963b7f895c4

Frozen validation SHA-256:
c87ab3908b34c22d8f765a5771187f6557fa98268521ab49841330157bb321df
```

`prepare_bundle.py`는 독립 train 파일과 candidate ZIP 내부 `train.jsonl`이 같은지 확인하고 현재 runtime의 `DECISION_SYSTEM`과 schema를 내장한다.

## 빌드

```bash
PYTHONPATH=src/soomac_irc \
  /home/roma/miniconda3/envs/gemma4_env/bin/python \
  soomac_decision_v3_1_training_v2_9/prepare_bundle.py

cd soomac_decision_v3_1_training_v2_9/docker_context
docker build --platform linux/amd64 \
  --build-arg LAB_INPUTS="$(python3 -c 'import json; print(json.dumps(json.load(open("lab-inputs.json")), separators=(",", ":")))')" \
  --build-arg TRAIN_MODE=full \
  --build-arg TRAIN_QUANTIZATION=int4 \
  -t soomac/decision-v3.1:v2.9 .

docker image save -o ../../soomac_decision_v3_1_training_v2_9.tar \
  soomac/decision-v3.1:v2.9
```

## 서버 선택값

```text
모델: google/gemma-4-12B-it 또는 동일 파일을 가진 준비된 cache
데이터: 이미지 내장 dataset
GPU: 공유 GPU 1개
CHECKPOINT_DIR: /checkpoints
```

## 재개

`LAB_RESUME=1`, `RESUME_CHECKPOINT_DIR`, `RESUME_STEP`을 모두 받아 마지막 저장 완료 checkpoint부터 복원한다. checkpoint의 Adapter, optimizer, scheduler, Trainer state, RNG 또는 step이 맞지 않으면 처음부터 시작하지 않고 오류 종료한다.
