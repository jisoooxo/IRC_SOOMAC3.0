# 원격 Decision Unsloth 전환 가능성 — 읽기 전용 평가

## 최신 품질 우선 판단 — 기존 작업 유지

사용자 최신 지시는 **실행 성공·학습 정확성·validation/기능·속도/ETA·미검증 위험**을 분리하는 것이다. 2026-11-01 21:46 KST 로컬 Docker read-only inspect에서 candidate v3가 여전히 `sha256:bf465ad8543b8a280f8322980a879fb46cba09d2356ee8d49581c082a95c2d3a`, entrypoint `/app/run_training.sh`, CMD `train`임을 재확인했다. 이는 이미지 존재/구조 검증이며 원격 GPU 학습·품질 합격이 아니다. 현재 TAR는 배포 승인 대상이 아니라 품질 검증 대기 후보이다.

기존 원격 상태의 근거는 아래18:46 스냅샷뿐이다. 현재 진행률·드라이버·남은 시간의 live 조회는 하지 못했으므로77시간을 최신 ETA로 반복하지 않는다. 실제 기존 작업을 중단하거나 새 요청을 제출하지 않았다. 후보의 Dockerfile은 요청 GPU 예산44,000,000,000B와 allocator 외1GiB 예약을 두고 서버 byte/fraction/실물 예산 중 최솟값으로 제한한다. 이는 GPU VRAM이며 **CPU RAM16GiB와 별개**이다. 로컬 RSS peak나 낮은 GPU VRAM만으로 원격 RAM 적합성을 보증하지 않는다. 컨테이너별 peak RSS/cgroup `memory.peak`와 OOM event를 실제 시험에서 함께 확인해야 한다. 필요 시 새 작업에 RAM32~48GiB를 요청하는 선택지는 있으나 기존 작업 자원을 임의 변경하지 않는다.

기존 HF checkpoint는 config의 engine·패키지·스케줄·LoRA 상태와 같은 identity를 요구한다. 새 Unsloth 실행기에 그대로 full-state resume가 된다는 증거가 없고 현재 fail-closed 계약도 거절한다. adapter만 가져와 optimizer를 초기화하는 우회는 학습 정책 변경이므로 하지 않는다. 전환 손익은 보수적으로 **새 학습을0부터 시작하는 경우**로 계산한다.

```text
기존 유지 비용 = 현재 실제 남은 update수 × 기존 원격 warm 초/update + 남은 eval/save 시간
전환 비용 = 추가 검증·업로드·대기 + 1638 × 새 원격 warm 초/update + 새 eval/save 시간
전환 임계 초/update = (기존 유지 비용 − 추가 준비 비용 − 새 eval/save 시간) / 1638
```

예시일 뿐인 산술: 오래된77.5시간 잔여와 추가 준비2시간, eval/save 차이를0으로 가정하면 새0-start의 손익분기점은약166초/update다. 새 원격150초라면 총70.25시간(2시간+1638×150초),180초라면83.9시간이다. **이는 측정된 Unsloth 원격 속도나 ETA가 아니다.** 현재 원격 학습이 진행된 만큼 유지 쪽 비용은 더 낮아지고 전환 이득도 작아진다. 로컬 P의325→276초 비율을 다른 GPU의 D에 그대로 대입하지 않는다.

최종 전환 권고는 같은 원격 하드웨어에서 정확성·저장재개·기능 관문과 실제 warm 속도가 확인된 뒤에만 가능하다. 기존 작업은 유지하며, 추가 remote GPU 시험 제출 전 slot/VRAM/RAM/compute 충돌과 실행 권한을 다시 확인해야 한다. 로컬 candidate-image smoke는 그 자체로 원격 처리량 검증이 아니다.

Cause: 새 프레임워크는 기존 full-state 연속성을 자동 보장하지 않고 원격 실측도 아직 없다.
Effect: 현재는 Decision 교체를 권하지 않으며, 별도 이미지 준비 가능성과 실제 전환 채택을 구별한다.

참고: [Unsloth Gemma 4 학습 가이드](https://unsloth.ai/docs/models/gemma-4/train), [공식 Docker 이미지](https://hub.docker.com/r/unsloth/unsloth), [Unsloth 릴리스](https://github.com/unslothai/unsloth/releases), [봉인된 r5 Dockerfile](../training/learning_first_v43/training_release_20261101_r5/remote/Dockerfile), [연구실 이미지 규격 v2.9](</home/roma/Downloads/TRAINING_IMAGE_SPEC.ko_v2_9 (2).md>) → **하드웨어·모델 수준의 적용 가능성은 있다. 현재 작업에서 속도 개선과 기존 체크포인트 호환은 검증되지 않았다.**

## 1. 현재 실행과 적용 가능성

2026-11-01 18:46 KST의 읽기 전용 [작업 스냅샷](/tmp/soomac_accel_remote_compat_snapshot.json)에서 Decision 작업 `16e600dfc6cc41cabaf777cbe0cf89d5`는 `running`, 4/1638 optimizer step, 초기 표시 ETA 77:30:42였다. 이 수치는 **초기 4 step 누적 평균**이며 정상 상태의 완료 예측이 아니다. 서버 GPU는 RTX PRO 6000 Blackwell Max-Q Workstation Edition이고 총량은 97,887MiB(약 95.6GiB)이다. 그러나 작업은 shared 할당 `50,246,713,344`바이트(약 46.8GiB) 예산과 CPU 8개·RAM 16GiB 요청으로 실행 중이었다. 스냅샷의 GPU used/utilization과 시스템 RAM은 **호스트 전체 집계**이므로 이 컨테이너의 사용량으로 해석하지 않는다. 원격 NVIDIA 드라이버 버전과 실제 compute capability는 스냅샷에 없어 미확인이다.

[Unsloth 공식 Docker 안내](https://hub.docker.com/r/unsloth/unsloth)는 RTX PRO 6000을 Blackwell `sm_120` 지원 목록에 넣고, CUDA 12.8 경로에서 드라이버 570.26 이상을 요구한다. B200의 `sm_100`과 이 RTX PRO 계열의 `sm_120`은 다른 대상이다. [Gemma 4 학습 가이드](https://unsloth.ai/docs/models/gemma-4/train)는 **12B를 명시적으로 지원**한다고 적는다. 따라서 모델·GPU 계열에서 불가능하다는 근거는 없다. 다만 드라이버, 선택할 이미지 digest의 `sm_120` 바이너리, 실제 12B unified 모델 로딩과 긴 입력의 backward는 해당 원격 자원에서 아직 확인하지 않았다.

원인: 공식 지원 목록은 계열 호환성을 설명하지만 이 서버의 드라이버·할당 예산을 검사한 결과는 아니다.
결과: 새 패키지 후보를 만들 수 있으나 기존 작업을 중단하거나 ETA를 대체할 근거는 아직 없다.

## 2. 학습 코드·속도 계약

현재 봉인된 [공통 실행기](../training/learning_first_v43/training_release_20261101_r5/common/train_agent.py)는 `AutoProcessor`와 기존 Gemma 4 모델 경로, NF4/BF16, text decoder에만 LoRA(rank 16, alpha 32, dropout 0.05), 유효 배치 16, Decision 6 epochs를 사용한다. [공통 core](../training/learning_first_v43/training_release_20261101_r5/common/core.py)는 기존 chat template·정답 토큰 마스크·데이터 순서, `max_length`/history 계약을 반영한다. 별도 Unsloth runner는 이 입력·loss mask·가중치 대상 모듈·optimizer/스케줄러·seed·평가·저장 설정을 다시 구현하고 동등성을 검증해야 한다. **`pip install unsloth`만으로 기존 `train_agent.py`가 Unsloth 경로로 전환되지는 않는다.**

[Unsloth 공식 예시](https://unsloth.ai/docs/models/gemma-4/train)의 `FastModel.from_pretrained`·`get_peft_model`·`SFTTrainer`는 다른 예시 모델과 LoRA dropout 0, 다른 학습률·배치·마스크 처리를 보여준다. 이를 그대로 복사하면 현재 실험과 달라진다. [Unsloth 릴리스](https://github.com/unslothai/unsloth/releases)는 최근 LoRA dropout을 포함한 fused kernel과 Gemma 4 global attention kernel 개선을 보고하지만, **우리 0.05 dropout·unified 12B·512-wide global head·긴 sequence에 실제 선택되는 kernel과 이득은 미측정**이다. 공식 가이드의 약 1.5배 속도·약 60% VRAM 절감은 FA2 설정과의 공개 비교이지 현재 r5 runner와의 비교가 아니다. 현재 77.5시간을 1.5로 나눠 새 ETA라고 제시할 수 없다.

원인: 공개 벤치마크의 비교 대상과 현재 데이터·예산·학습 루프가 다르다.
결과: 동일 입력의 **완전한 유효 배치 16 optimizer update**를 양쪽에서 측정해야 ETA 단축을 판정할 수 있다.

## 3. 원격 이미지 전환에 필요한 묶음

1. **실행 기반:** digest 또는 고정 버전의 `unsloth/unsloth:core` 계열과 그 CUDA·PyTorch·Triton·GPU driver 호환을 확인한다. 공식 `latest`의 Studio/Jupyter 기본 진입점과 포트는 배치 학습 이미지에 맞지 않는다. [Docker 안내](https://hub.docker.com/r/unsloth/unsloth)의 `core`는 스크립트 실행 경로가 있다.
2. **학습 라이브러리:** 고정된 `unsloth`·`unsloth_zoo` 및 호환 `transformers`·`peft`·`trl`·`accelerate`·`bitsandbytes`·`safetensors` 조합을 한 이미지 안에서 확인한다. 기존 이미지에서 보고된 torch 2.12.1/cu130과 공식 Docker 이미지의 별도 스택을 동일하다고 가정하지 않는다.
3. **모델·데이터 연결:** 외부 읽기 전용 `MODEL_DIR` 가중치, `/app/assets/dataset`, 봉인된 `/app/assets/model-processor`, `RELEASE.json` pin, 원래 `lab-inputs.json`/`lab.training.inputs` 일치를 유지한다. 네트워크 다운로드나 새 tokenizer/chat template로 조용히 바꾸지 않는다.
4. **플랫폼 계약:** [원격 스크립트](../training/learning_first_v43/training_release_20261101_r5/remote/run_training.sh)처럼 UID 1000, `ENTRYPOINT`/`CMD train`, `lab.training.spec=2.9`, progress schema 1을 120초 안에 시작해 10초 heartbeat, shared GPU 예산 clamp/NVML 관측, `/checkpoints` 저장과 `output-dir-v1` 재개를 실제 코드로 제공한다. [규격서](</home/roma/Downloads/TRAINING_IMAGE_SPEC.ko_v2_9 (2).md>)상 라벨만으로 복원 성공은 증명되지 않는다.
5. **복구·품질 검증:** 새 이미지의 저장 5 step → 의도한 실패 → **그 이미지 자신의** adapter·optimizer·scheduler·RNG·step 복원 → 완료 영수증 검증, 그리고 같은 입력의 loss mask/gradient·출력 평가를 확인한다. 기존 r5/HF Trainer checkpoint를 다른 framework-native 상태로 이어 쓸 수 있다는 보장은 없다. 이식이 필요하면 별도 변환·검증 과제이다.

## 4. 판단과 현재 경계

**가능:** 공식 지원 범위상 RTX PRO 6000 Blackwell과 Gemma 4 12B에 대한 새로운 Unsloth 배치 학습 패키지의 설계·CPU 사전 검증.

**미검증:** 서버 드라이버와 선택한 이미지의 실제 조합, 46.8GiB shared 예산에서 긴 Decision 입력의 forward/backward, kernel 선택, steady-state 처리량, dropout 0.05 동작·수치 동등성, 원래 checkpoint와의 복원 호환성.

**다음 판단 기준:** 현재 Decision 작업의 보존 가능한 저장 상태를 확인하고, 별도 후보에서 동일 데이터·설정의 유한 GPU profile과 save/fail/resume를 통과한 뒤 기존 작업 대비 처리량·검증 손실·메모리를 비교한다. 이 문서는 원격 작업 중단, 이미지 설치, 새 TAR 업로드·실행을 승인하거나 수행한 기록이 아니다. 로컬 가상환경(venv)을 추가해도 서버 GPU가 늘거나 원격 실행 환경이 바뀌지 않는다.

## 용어 정리

- **SM(`sm_120`)**: RTX PRO Blackwell용 CUDA 계산 능력 대상이다. B200의 `sm_100`과 구분한다.
- **유효 배치(effective batch)**: 누적 후 optimizer update 한 번에 반영하는 표본 수이며 현재 16이다.
- **전체 상태 복원(full-state resume)**: adapter와 optimizer·scheduler·난수 상태·step을 같은 설정으로 복구하는 것이다.

(내가 의역해보자면, Unsloth는 검증할 가치가 있는 **새 실행 경로**이다. 이미 돌고 있는 Decision의 체크포인트를 그대로 넘길 수 있는 속도 스위치는 아니다.)

## CPU 전용 후속 확인: 별도 환경과 체크포인트 오프로드

2026-11-01에 프로젝트에서 gitignore되는 `training/learning_first_v43/training_release_20261101_r5/env`를 `uv venv --system-site-packages`로 만들고, 기존 `/tmp/soomac-training-accel-20261101/venv`의 **환경 전용 site-packages 오버레이만** 그대로 복제했다. 양쪽 오버레이의 `pip list --path ... --format=freeze` 출력 SHA-256은 `99d298f8ae51474758b40d88d1f38fbf787ede539de8594fdf06b046fc862035`로 같다. 새 환경의 `pip check`는 `No broken requirements found`이며 `unsloth`/`unsloth_zoo`는 둘 다 `2026.10.11`이다. 부모 인터프리터는 기존 `gemma4_env`의 Python 3.10.20이지만 패키지 설치·변경은 부모 환경에 하지 않았다. 시스템 site-packages가 **읽기 전용으로 마운트된 것은 아니므로**, 후속 명령도 반드시 이 환경의 `bin/python`을 명시하고 부모 환경의 `pip`/`uv pip`를 대상으로 쓰지 않아야 한다. 이번에는 네트워크 다운로드, CUDA 초기화, 모델 적재를 하지 않았다.

설치된 [`unsloth_zoo/gradient_checkpointing.py`](../training/learning_first_v43/training_release_20261101_r5/env/lib/python3.10/site-packages/unsloth_zoo/gradient_checkpointing.py)의 `UnslothCheckpointFunction`은 **해당 offloaded reentrant 경로가 선택될 때** 체크포인팅 입력 활성화를 CPU 버퍼로 내리고 역전파 시 GPU 스테이징 버퍼로 다시 읽는다(1532–1805, 2653–2661행). 다른 non-reentrant 오프로드 경로도 있으므로 새 러너가 어느 경로를 선택할지는 미확인이다. CPU 버퍼는 초기 200개×128 Ki 요소이며 BF16이면 **원시 저장량 약 50 MiB**이다(65–68, 1430–1492행). 이후 실제 입력 크기에 맞춰 버퍼를 키우고, pinned host 메모리 부족 시 pageable 메모리로 전환한다(143–195행). `UNSLOTH_GC_PINNED_MAX_GB` 또는 가용 RAM·local rank에 따라 pinned 한도를 정한다(320–344행). 이 한도는 **총 프로세스 RSS 한도가 아니다**. 원본 소스 SHA-256과 새 환경의 해당 파일 SHA-256은 모두 `4c42341ca141668c165852f144d58bea5ce2b58c5fcca28fa3f1b3c38d8853f2`이다.

봉인된 Gemma 4 설정의 hidden size 3840과 [데이터 매니페스트](</home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/releases/conversation_r5_20261101_build05_speed/dataset/MANIFEST.json>)의 Decision/Response 상한 9216 토큰을 가정하면 **BF16 체크포인팅 입력 하나**는 `9216×3840×2 = 70,778,880`바이트(67.5 MiB)이다. 48층 모두 같은 크기 입력을 별도 보관한다는 가상 계산은 원시 텐서만 약 3.16 GiB이다. Recommendation 5120 토큰이면 각각 37.5 MiB, 48개 약 1.76 GiB이다. 이는 **실측 RSS·pinned 할당량·성공 보장치가 아니다**. 마지막 층 생략, 실제 입력 길이, BF16/FP32 보관 선택, allocator 반올림·캐시, 그래프/토크나이저/데이터셋 및 다른 프로세스가 값을 바꾼다. 원격 16 GiB는 컨테이너 RAM *요청치*이며 실제 여유 RSS는 아직 측정하지 않았다. [봉인된 모델 설정](</home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/releases/conversation_r5_20261101_build05_speed/assets/model-processor/config.json>), [앞서 확인한 원격 스냅샷](/tmp/soomac_accel_remote_compat_snapshot.json)
