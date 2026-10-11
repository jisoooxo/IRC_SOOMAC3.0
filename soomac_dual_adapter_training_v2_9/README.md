# SOOMAC Decision + Response Dual Adapter Training Image

참고 규격: `/home/roma/Downloads/TRAINING_IMAGE_SPEC.ko_v2_9.md`

→ 하나의 원격 작업에서 Gemma 4 12B 원본 모델을 기준으로 Decision LoRA와 Response LoRA를 독립적으로 순차 학습한다.

## 학습 순서

```text
Gemma 4 12B base
├── Decision: NF4 QLoRA, 6 epochs, LR 1e-4
└── Response: NF4 QLoRA, 4 epochs, LR 5e-5
```

Decision Adapter를 Response 초기값으로 사용하지 않는다. 각 단계 사이에 모델과 CUDA 캐시를 해제한 뒤 같은 Base Model에서 새 LoRA를 생성한다.

## 데이터

- Decision: train 2,595 / validation 285
- Response: train 898 / validation 101
- Decision optimizer steps: 978
- Response optimizer steps: 228
- 전체 optimizer steps: 1,206

데이터와 system prompt/schema는 `prepare_bundle.py`가 현재 runtime 정본에서 복사하고 SHA-256을 manifest에 기록한다.

```bash
PYTHONPATH=src/soomac_irc \
  /home/roma/miniconda3/envs/gemma4_env/bin/python \
  soomac_dual_adapter_training_v2_9/prepare_bundle.py
```

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

`checkpoint-*`는 Adapter뿐 아니라 optimizer, scheduler, Trainer state, RNG를 포함하는 재개용 저장본이다. `epoch_*_adapter`와 `best_adapter`는 배포용 독립 Adapter이다.

## 빌드 및 TAR 생성

```bash
cd soomac_dual_adapter_training_v2_9/docker_context
docker build \
  --build-arg LAB_INPUTS="$(tr -d '\n' < lab-inputs.json)" \
  --build-arg TRAIN_MODE=full \
  --build-arg TRAIN_QUANTIZATION=int4 \
  -t soomac/decision-response-dual:v2.9 .

docker save \
  -o ../../soomac_decision_response_dual_v2_9.tar \
  soomac/decision-response-dual:v2.9
```

## 결과 선택

각 단계의 `best_adapter`는 epoch별 validation loss가 가장 낮은 체크포인트이다. 실제 회귀 평가에서 다른 epoch가 더 좋으면 해당 `epoch_N_adapter`를 사용할 수 있다.

## 재개

`.lab-monitor/resume.json`의 `global_step`은 두 단계 전체 누적값이다. Response의 로컬 `checkpoint-57`은 Decision 978 step을 더해 global step 1,035로 게시한다.

Decision 완료 후 Response 시작 전에 실패하면 Decision 완료 표시를 확인하고 Response만 새로 시작한다. Response 체크포인트가 게시된 뒤 실패하면 해당 Response 체크포인트부터 optimizer/scheduler/RNG 상태를 복원한다.
