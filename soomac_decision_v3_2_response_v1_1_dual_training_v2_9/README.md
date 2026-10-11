# SOOMAC Decision v3.2 + Response v1.1 Dual Adapter Training Image

참고 규격: `/home/roma/Downloads/TRAINING_IMAGE_SPEC.ko_v2_9.md`

→ 단일 원격 작업에서 Decision v3.2를 먼저 학습하고, 완료 후 Response v1.1을 자동으로 학습한다.

## 학습 순서

```text
Gemma 4 12B base
├── Decision v3.2: NF4 QLoRA, 6 epochs, LR 1e-4
└── Response v1.1: NF4 QLoRA, 4 epochs, LR 5e-5
```

Response는 Decision Adapter를 이어받지 않는다. Decision 단계가 끝나면 모델과 CUDA 캐시를 해제하고, 같은 Base Model에서 별도 Response LoRA를 생성한다.

## 데이터와 optimizer step

```text
Decision: train 3,465 / validation 285 / 1,302 steps
Response: train 1,098 / validation 101 / 276 steps
Total: 1,578 steps
```

두 데이터셋, 현재 runtime prompt/schema, provenance, tokenizer 길이 검증 결과를 이미지에 내장한다.

## 체크포인트

```text
/checkpoints/
├── decision/
│   ├── checkpoint-*/
│   ├── epoch_1_adapter/ ... epoch_6_adapter/
│   └── best_adapter/
├── response/
│   ├── checkpoint-*/
│   ├── epoch_1_adapter/ ... epoch_4_adapter/
│   └── best_adapter/
└── dual_training_complete.json
```

`checkpoint-*`는 optimizer/scheduler/RNG까지 포함하는 재개용 저장본이다. `epoch_*_adapter`와 `best_adapter`는 각각 독립 배포 가능한 Adapter이다.

## 재개

`.lab-monitor/resume.json`의 step은 두 단계 누적값이다. Response 로컬 step에는 Decision의 1,302 step을 더해 기록한다.

```text
Decision 도중 재개: 해당 Decision checkpoint에서 계속한 뒤 Response 시작
Response 도중 재개: Decision 완료 표시를 확인한 뒤 해당 Response checkpoint에서 계속
```

## 빌드 및 TAR

```bash
cd soomac_decision_v3_2_response_v1_1_dual_training_v2_9/docker_context
docker build \
  --build-arg LAB_INPUTS="$(tr -d '\n' < lab-inputs.json)" \
  --build-arg TRAIN_MODE=full \
  --build-arg TRAIN_QUANTIZATION=int4 \
  -t soomac/decision-v3.2-response-v1.1-dual:v2.9 .

docker save \
  -o ../../decision_v3_2_response_v1_1_dual_training_bundle_v2_9.tar \
  soomac/decision-v3.2-response-v1.1-dual:v2.9
```
