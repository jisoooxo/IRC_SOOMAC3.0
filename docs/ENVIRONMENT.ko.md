# IRC_SOOMAC3.0 실행 환경 명세서

참고: [README 수록 항목](README_CONTENT_PLAN.ko.md) → 개발 환경·변천사·전체 로직을 설명할 README의 구성안이다.

참고: [서버 실행 스크립트](../tools/vllm_server/start_vllm_server.sh), [모델 경로](../src/soomac_irc/soomac_irc/model_runtime.py), [ROS 패키지 설정](../src/soomac_irc/setup.py) → 이 명세의 실행 기준이다.

기준일은 **2026-11-25**, 소스는 `b80a8cb`(대회 최종본)이다. **실행 환경은 2026-11-05 실제 실행·평가에 사용한 기존 버전을 유지한다.** 새 환경이나 새 버전 도입을 요구하지 않는다. 이번 작성에서는 패키지 설치, 모델 적재, ROS 노드 실행, 로봇 구동을 수행하지 않았다.

### 기존 실행 기록과 버전 고정

버전의 기준 자료는 [HF/vLLM 실제 비교](2026-11-05_hf_vllm_compare_review.md), [운영 적용 결과](2026-11-05_d6_prompt_adoption.md), [TTS 기존 경로 유지 결론](2026-11-05_tts_vllm_trial_review.md), [서버·클라이언트 환경 기록](2026-11-01_training_package_build_worklog.md)의 환경 확인 항목이다. 당시 검증의 `training/learning_first_v43/final/goals/GL10/hf_vllm_compare_20261105/BEFORE.json`에는 네 환경의 전체 패키지 버전이 보존돼 있다.

| 기존 실행 환경 | 당시 설치 패키지 수 | 저장소에 옮긴 원본 버전 목록 |
|---|---:|---|
| `soomac_llm` | 148 | [soomac_llm.requirements.txt](environment/soomac_llm.requirements.txt) |
| `soomac_vllm_server` | 202 | [soomac_vllm_server.requirements.txt](environment/soomac_vllm_server.requirements.txt) |
| `gemma4_env` | 217 | [gemma4_env.requirements.txt](environment/gemma4_env.requirements.txt) |
| `cosyvoice3` | 229 | [cosyvoice3.requirements.txt](environment/cosyvoice3.requirements.txt) |

이번에 조회한 주요 패키지 버전은 당시 목록과 네 환경 모두 일치했다. 따라서 아래 표는 **기존 실행 버전의 요약**이며 11월 25일 조회는 보조 대조 자료이다. requirements 파일은 당시 설치 패키지 목록 그대로이며, Conda·ROS·시스템 라이브러리·로컬 wheel 출처까지 포함한 설치 잠금 파일은 아니다. 기존 환경을 재사용할 때에는 재설치하지 않는다.

## 1. 운영체제와 장비

| 항목 | 확인된 사양 또는 요구 사항 | 확인 범위 |
|---|---|---|
| 운영체제 | Ubuntu 22.04.5 LTS, Linux 환경 | 현재 `/etc/os-release` 확인 |
| ROS | ROS 2 Humble, `ament_python`, `colcon` | `/opt/ros/humble`·두 패키지 설정 확인; 전체 빌드 미실행 |
| GPU | 기존 측정 장비 NVIDIA RTX PRO 4500 Blackwell, 32623MiB | 2026-11-05 측정 기록 |
| NVIDIA 드라이버 | 기존 측정 버전 `580.178.04` | 현재 `nvidia-smi`는 드라이버 통신 실패, 현재 GPU 가용성 미확인 |
| RAM | 현재 총 약 62GiB, 명목 64GB급 | `free -h`; 최소 요구량은 별도 산정하지 않음 |
| 저장 장치 | 모델·음성 모델·어댑터·참조 이미지 DB·출력 WAV를 담을 공간 | 전체 자산의 최소 디스크 용량은 미산정 |
| 카메라 | RealSense 호환 RGB-D 카메라와 검증용 이미지 공급 경로 | 검출 코드의 컬러·깊이 입력 640×480, 30fps; 정확한 장치 모델 미확인 |
| 제어 장치 | 6축 Dynamixel 매니퓰레이터, 그리퍼, 레일, Arduino 기반 흡착 장치 | 제어 소스 기준; 현장 연결 미확인 |
| 입출력 장치 | PulseAudio 입력 마이크, ALSA/PortAudio 출력 스피커, UI용 브라우저 | 음성 코드·기존 운영 설정 기준 |

GPU 메모리 최소 요구량을 단일 수치로 보장하지 않는다. 현재 기본 설정은 32GB급 GPU에서의 과거 동시 부하 시험에도 여유 부족이 관측됐다. 상세 조건은 §4에 적는다.

## 2. Python 환경과 라이브러리

### 2.1 역할별 실행 환경

**모든 패키지를 하나의 가상환경에 합치지 않는다.** 기존 운영은 ROS/Python 3.10 노드와 Python 3.12 vLLM 서버를 별도 프로세스로 연결한다.

| 역할 | 현재 환경 / Python | 주요 패키지 버전 |
|---|---|---|
| LLM ROS 클라이언트 | `soomac_llm` / `3.10.20` | torch `2.12.1+cu130`, transformers `5.14.1`, peft `0.21.0`, accelerate `1.14.0`, bitsandbytes `0.50.2`, xgrammar `0.2.8`, langgraph `1.0.8`, jsonschema `4.26.0`, chromadb `1.5.9` |
| GPU 추론 서버 | `soomac_vllm_server` / `3.12.16` | vllm `0.32.0`, torch `2.13.0`, transformers `5.19.0`, bitsandbytes `0.50.2`, **vllm-bnb-plugin `0.0.3`**, xgrammar `0.2.8` |
| 최종 로컬 STT | `gemma4_env` / `3.10.20` | torch `2.12.1+cu130`, transformers `5.13.1`, numpy `2.2.6`; Silero VAD 로컬 코드·가중치 |
| TTS / UI | `cosyvoice3` / `3.10.19` | torch `2.10.0`, transformers `4.51.3`, numpy `1.26.4`, scipy `1.15.3`, sounddevice `0.5.5`, soundfile `0.12.1`, Flask `3.1.3`, Flask-SocketIO `5.6.1`; 외부 CosyVoice 소스 |
| 비전 / 제어 | ROS Python `3.10`; 실제 비전 PC 환경은 미확인 | numpy, scipy, ikpy, pyserial, dynamixel_sdk, OpenCV, pyrealsense2, ultralytics, sam2 |

버전은 기존 실행 기록과 2026-11-25 `importlib.metadata` 조회 결과를 대조했다. [조회 스냅샷](environment_metadata_2026-11-25.json)에 보조 환경 `camera_env`, `ros_env`도 기록했다. 이 두 환경의 존재는 최종 비전 PC에서 사용했다는 증거가 아니다.

**이 표와 스냅샷은 전체 의존성 잠금 파일(lockfile)이 아니다.** CUDA 확장 로딩, 라이브러리 호환성, 모델 추론 성공까지 확인한 값도 아니다. `+cu130`은 해당 PyTorch 빌드 표기이며 시스템 CUDA Toolkit 버전과 구분한다. 현재 Toolkit 버전은 확인하지 않았다.

### 2.2 ROS·운영체제 의존성

| 구분 | 필요한 구성 |
|---|---|
| ROS 메시지·Python | `rclpy`, `std_msgs`, `sensor_msgs`, `geometry_msgs`, ROS Python 3.10 바인딩 |
| 빌드 | `colcon`, `ament_python`, `setuptools`; 개발 검사 시 `pytest` |
| 오디오 시스템 | PulseAudio의 `pactl`, ALSA의 `arecord`·`aplay`, PortAudio·libsndfile |
| 하드웨어 | RealSense SDK 및 USB 접근 권한, Dynamixel SDK, 시리얼 장치 접근 권한 |
| 대체 STT | CLOVA 경로만 `grpcio`, `protobuf`, `pyaudio`, `webrtcvad`와 별도 인증·네트워크 필요; 최종 Qwen 경로에는 CLOVA 인증이 불필요 |

`soomac_irc/setup.py`는 `setuptools`, `jsonschema>=4.18`만 설치 의존성으로 선언한다. `package.xml`도 AI·음성·비전 의존성 전체를 담고 있지 않다. **`pip install .` 또는 `rosdep`만으로 전체 시스템 준비가 끝나지 않는다.**

학습을 다시 할 때 사용하는 Unsloth 환경·Dockerfile은 별도이다. 프로젝트 실행을 위해 학습 환경까지 설치할 필요는 없다.

## 3. 모델·외부 자산과 파일 경로

### 3.1 LLM / VLM

| 자산 | 코드가 사용하는 현재 경로 | 현재 파일 확인 |
|---|---|---|
| Gemma 4 12B 기반 모델 | `/home/roma/Desktop/sLLM/gemma-4-12B-it` | 디렉터리 존재 |
| Decision 학습 어댑터 | `models/runtime_adapters/decision_r5_unsloth_epoch6_step1638` | 디렉터리·가중치 존재 |
| Recommendation 학습 어댑터 | `/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/runs/r5_unsloth_local_authorized_20261101/recommendation/attempt-9726a651729b41a6bd73d0b2a0bea084/recommendation/checkpoint-234` | 디렉터리 존재 |
| Response 학습 어댑터 | `/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/runs/r5_unsloth_local_authorized_20261101/response/attempt-63872c5d7d584aab81aa73bbb6cc0155/response/checkpoint-150` | 디렉터리 존재 |
| vLLM 서빙용 어댑터 | `models/vllm_adapters/{decision,recommendation,response}/` | 세 디렉터리의 `adapter_config.json`·`adapter_model.safetensors` 존재 |

현재 선택 정본은 `model_runtime.py`의 직접 경로 상수이다. `config/llm_active_stack.json`은 정보용 사본이다. `models/MODEL_MANIFEST.md`의 10월 기록을 현재 어댑터 선택으로 사용하지 않는다.

기반 모델에는 가중치뿐 아니라 config·processor·tokenizer·chat template 파일도 필요하다. **서버와 클라이언트는 같은 기반 모델 자산을 사용해야 한다.** 코드에는 기반 모델의 배포 revision이 고정돼 있지 않아, 새 환경에서 같은 이름만 다운로드하면 동일 자산임을 보장할 수 없다.

서빙용 어댑터는 학습 원본을 그대로 복사한 파일이 아니다. `convert_adapters_for_vllm.py`가 Gemma의 `k_eq_v` 층에 필요한 LoRA 변환을 수행한다. 클라이언트는 각 서빙 디렉터리의 `CONVERSION.json`에서 원본 경로·기록된 가중치 해시를 대조한다. 새 어댑터를 준비할 때에는 서버가 꺼진 상태에서 변환하고 서버를 재시작해야 한다. 이번 문서 작성에서는 변환·교체하지 않았다.

### 3.2 음성·참조 이미지·비전

| 자산 | 현재 코드 경로 / 요구 내용 | 현재 확인 |
|---|---|---|
| Qwen3-ASR | `/home/roma/models/audio/stt/Qwen3-ASR-1.7B-hf` | 디렉터리 존재 |
| Silero VAD | `/home/roma/.cache/torch/hub/snakers4_silero-vad_master` | 디렉터리 존재 |
| CosyVoice 구현 | `/home/roma/CosyVoice_new`, 하위 `third_party/Matcha-TTS` | 루트 존재; 전체 의존성 구동 미검증 |
| CosyVoice3 모델 | `/home/roma/models/audio/tts/Fun-CosyVoice3-0.5B-2512` | 디렉터리 존재 |
| TTS 참조 음성 | `src/soomac_irc/soomac_irc/tts_prompts/sansa_natural.wav`, 코드 내 대응 전사문 | WAV 존재, Git 추적됨 |
| TTS 출력 | `/home/roma/ros2_ws/src/soomac_ai/src/tts_out` | 실행 시 생성·쓰기 권한 필요 |
| 참조 이미지 원본 | `/home/roma/ros2_ws/src/soomac_irc/vlm_data/mealkit_coco_v15` | 디렉터리 존재 |
| 참조 이미지 DB | `/home/roma/ros2_ws/src/soomac_irc/vlm_data/chroma_refs_v15`, Chroma 컬렉션 `mealkit_refs` | 디렉터리 존재; 컬렉션 내용·이미지 참조 유효성 미검증 |
| YOLO 가중치 | `/home/pc/irc_ws/irc_ws/src/irc_control_pkg/irc_control_pkg/best_6.pt` | 현재 PC에는 없음 |
| SAM2 가중치·설정 | `/home/pc/sam2/checkpoints/sam2.1_hiera_base_plus.pt`, `configs/sam2.1/sam2.1_hiera_b+.yaml` | 가중치 현재 PC에 없음; 외부 SAM2 설치 필요 |

대용량 모델과 어댑터 가중치는 `.gitignore` 대상이다. **현재 PC에 파일이 있어도 새 clone에 포함되는 것은 아니다.** 다른 PC에서는 외부 자산을 별도로 전달받고, 소스의 `/home/roma`·`/home/pc` 경로와 서버 경로를 실제 배치에 맞춰야 한다. 참조 DB와 원본 이미지도 함께 준비한다.

## 4. 서버·오디오·하드웨어 설정

### 4.1 vLLM 기본값

| 설정 | 기본값 | 동작 / 변경 시 주의점 |
|---|---|---|
| LLM 백엔드 | `LLM_BACKEND="vllm"` | 노드 파일의 상수이다. 같은 이름의 환경변수를 export해도 이 상수가 바뀌지 않는다. |
| 서버 주소 | `127.0.0.1:8000` | 서버는 `VLLM_HOST`·`VLLM_PORT`로 변경 가능하나, 클라이언트 `VLLM_BASE_URL`도 별도 일치시켜야 한다. |
| 모델 이름 | `gemma4-base` | VLM은 언어용 어댑터 없이 기반 모델을 호출한다. |
| 서빙 길이 | `VLLM_MAX_LEN=16384` | 입력+출력 합계 상한이다. 클라이언트 `SERVING_MAX_MODEL_LEN=16384`와 맞춰야 한다. |
| GPU 예약률 | `VLLM_GPU_MEM=0.58` | 서버의 GPU 메모리 사용 비율 설정이며 전체 STT/TTS 동시 메모리 보장이 아니다. |
| 배치 / 동시 요청 | `VLLM_MAX_BATCHED=16384`, `VLLM_MAX_SEQS=2` | 배치 토큰 기본값은 최대 길이를 따른다. |
| LoRA | rank 최대 `16`, 슬롯 `3` | 역할 이름은 `decision`, `recommendation`, `response`이다. |
| 양자화 / 계산형 | bitsandbytes NF4 / BF16 | HF 경로와 서빙 경로의 구현·라이브러리 차이가 있다. |
| 구조화 출력 / 캐싱 | XGrammar, prefix caching ON | JSON 형식 제약과 접두 KV 캐시 재사용을 사용한다. |
| 워밍업 / 배치 불변 모드 | `LLM_STARTUP_WARMUP=True`, `VLLM_BATCH_INVARIANT=1` | 시작 준비 호출을 수행한다. 서버 재기동 간 완전한 출력 동일성을 보장하지 않는다. |
| 멀티모달 / 오프라인 | 요청당 이미지 최대 `8`, 오디오 `0`; HF 오프라인 ON | 모든 모델·processor 자산을 실행 전에 로컬에 준비해야 한다. |

출력 예약은 Decision 1024, Recommendation 1024, Response 512토큰이다. 따라서 코드의 입력 상한은 각각 15360·15360·15872토큰이다. 실제 chat template으로 토큰화하고, 초과 시 오래된 대화부터 입력에서 제외한다. 현재 발화·상태만으로도 넘으면 `InputBudgetError`를 발생시킨다. 접두 캐시 적중이 이 길이 상한을 늘려 주지는 않는다.

이 문서는 프롬프트를 수정하지 않았다. 현재 서버의 실제 tokenizer·템플릿·입력 여유 재측정도 수행하지 않았으며, 위 수치는 코드 설정이다.

### 4.2 동시 GPU 메모리 측정의 한계

| 과거 시험 조건 | 결과 | 해석 |
|---|---|---|
| 16K / 0.58, Qwen STT·긴 TTS·LLM 동시 부하 | 관측 사용량 31.37GiB, 여유 0.49GiB에서 guard 중단 | 현재 기본값이다. 전체 부하 완주나 CUDA OOM 발생을 뜻하지 않는다. |
| 12K / 0.50, 같은 시험 묶음 | 관측 최고 28.97GiB, 최소 여유 2.89GiB, 완주 | 시험 후보이며 운영 기본값에 미적용이다. |

근거는 [동시 부하 측정](2026-11-05_stack_vram_12k_review.md)이다. **서버 환경변수만 12K로 낮추면 현재 16K 클라이언트의 시작 검사가 거부한다.** 설정 조합 변경에는 서버·클라이언트 일치와 실제 대화·이미지·음성 동시 부하 재확인이 필요하다. 최소 VRAM 또는 장시간 안정성 보장으로 일반화하지 않는다.

### 4.3 오디오와 시리얼

| 항목 | 현재 설정 / 요구 |
|---|---|
| STT 입력 | PulseAudio `pulse`, `SOOMAC_AUDIO_SOURCE`로 입력 소스 선택; 미지정 시 시작 시점 기본 소스 사용 |
| STT 샘플 | 16kHz·모노·PCM 16비트, VAD 32ms, 사전 버퍼 300ms, 무음 종료 0.8초 |
| TTS 출력 | 모델 24kHz → 출력 48kHz, `TTS_DEVICE` 미지정 시 `sysdefault`; 기존 운영 wrapper는 `soomac_speaker` 사용 |
| 오디오 로컬 설정 | 기존 `~/.asoundrc`, PulseAudio 프로필·공유 메모리 우회 설정은 Git 밖에 있음; 새 PC에서 해당 장치에 맞게 구성 |
| 모터·흡착 시리얼 | `/dev/ttyUSB0`·`1`은 모터 버스, Dynamixel Protocol 2.0·1,000,000baud; `/dev/ttyUSB2`는 Arduino·115200baud |

자세한 오디오 장치 복원은 [기존 오디오 문서](2026-10-25_audio_pulseaudio_shm_and_device_paths.md)를 따른다. 문서에 기록된 `PCH`·MATA 장치 이름을 새 PC에 그대로 적용할 수 있다고 가정하지 않는다.

ROS로 연결되는 모든 프로세스는 동일한 `ROS_DOMAIN_ID`와 서로 통신 가능한 DDS 설정을 사용해야 한다. 이 저장소의 단일 launch 파일에서 전체 설정을 통합하지는 않는다. UI는 기본 `0.0.0.0:5000`에서 서비스한다.

## 5. 준비와 실행 순서

**아래는 기존 경로 배치를 유지한 PC용 명령이다.** 새 PC는 §2~3의 환경·자산 배치를 먼저 마쳐야 한다. 소스·설정 검토에 근거한 명령이며, 이번 작성에서 실제 서버·노드를 실행한 결과는 아니다.

### 5.1 준비

1. 기존 PC에서는 준비된 역할별 Python 환경과 ROS Humble을 그대로 사용한다. 다른 PC로 옮길 때에는 위 기존 버전 목록과 외부 자산을 기준으로 복원한다. ROS 노드는 Python 3.10 환경에서 `rclpy`를 불러올 수 있어야 한다.
2. §3의 모델·어댑터·외부 소스·참조 이미지 DB를 준비한다. 서빙용 어댑터에는 `CONVERSION.json`도 필요하다.
3. 같은 ROS domain, 입력 마이크·출력 스피커, 시리얼 포트와 접근 권한을 설정한다.
4. ROS 실행 파일 등록이 필요한 경우 아래 빌드를 수행한다.
5. §5.2의 각 프로세스를 별도 터미널에서 시작한다. 실물 동작 전에는 §6의 누락 구현을 확보한다.

```bash
cd /home/roma/IRC_SOOMAC3.0
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select soomac_irc irc_control_pkg
source install/setup.bash
```

이 자리에서 살아있는 변수: 사용자 정의 변수 없음. `source`로 ROS·빌드 환경을 현재 셸에 정의한다. `colcon`이 설치돼 있어야 한다. 기존 `build/`·`install/`은 새 환경의 빌드 성공 근거로 사용하지 않는다.

### 5.2 AI·음성·UI 프로세스

1. **vLLM 서버**를 시작한다. 기존 운영 주석의 준비 시간은 약 150초이며 현재 실행 시간은 미측정이다.

```bash
bash /home/roma/IRC_SOOMAC3.0/tools/vllm_server/start_vllm_server.sh
```

이 자리에서 살아있는 변수: 사용자 정의 변수 없음. 서버 스크립트가 기본 경로·오프라인 환경·서빙 옵션을 정의한다.

별도 터미널에서 준비 상태를 확인한다.

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
```

이 자리에서 살아있는 변수: 사용자 정의 변수 없음. HTTP 200과 `gemma4-base`·세 역할 이름의 등록을 확인한다.

2. **LLM ROS 노드**를 시작한다.

```bash
source /opt/ros/humble/setup.bash
cd /home/roma/IRC_SOOMAC3.0/src/soomac_irc
/home/roma/miniconda3/envs/soomac_llm/bin/python -m soomac_irc.llm_langgraph_node
```

이 자리에서 살아있는 변수: 사용자 정의 변수 없음. 모듈 실행 위치를 명령 안에서 정의했다. 어댑터 검증·시작 워밍업과 VLM 참조 초기화 완료를 확인한다.

3. **Qwen STT**를 시작한다. 먼저 `pactl get-default-source`로 현재 기본 입력이 사용할 마이크인지 확인한다.

```bash
source /opt/ros/humble/setup.bash
/home/roma/miniconda3/envs/gemma4_env/bin/python /home/roma/IRC_SOOMAC3.0/src/soomac_irc/soomac_irc/stt_qwen3_asr_node.py
```

이 자리에서 살아있는 변수: 사용자 정의 변수 없음. 입력 소스를 따로 지정하지 않았으므로 노드가 PulseAudio 기본 소스를 선택한다. UI·LLM의 수음 허용 상태 전에는 발화가 전달되지 않을 수 있다.

4. **TTS**를 시작한다. 현재 장치가 `sysdefault`로 열리는 조건이며, 기존 운영의 `soomac_speaker` 별칭을 쓰려면 먼저 ALSA 설정을 복원해야 한다.

```bash
source /opt/ros/humble/setup.bash
/home/roma/miniconda3/envs/cosyvoice3/bin/python /home/roma/IRC_SOOMAC3.0/src/soomac_irc/soomac_irc/tts_node.py
```

이 자리에서 살아있는 변수: 사용자 정의 변수 없음. `TTS_DEVICE`의 기존 셸 설정이 있으면 그 장치가 사용된다. 출력 장치와 참조 음성 초기화 성공을 확인한다.

5. **UI**를 시작하고 브라우저에서 `http://127.0.0.1:5000`을 연다.

```bash
source /opt/ros/humble/setup.bash
/home/roma/miniconda3/envs/cosyvoice3/bin/python /home/roma/IRC_SOOMAC3.0/src/soomac_irc/soomac_irc/ui_node.py
```

이 자리에서 살아있는 변수: 사용자 정의 변수 없음. 템플릿·정적 자산은 소스 파일 기준으로 찾는다. UI 시작 버튼은 주문·제어 흐름의 시작 신호이므로, 실물 제어를 연결한 환경에서는 구동 준비 후 누른다.

### 5.3 로봇 제어 스택

| 구성 | 저장소에서 확인한 진입점 | 실행 준비 조건 |
|---|---|---|
| 기본 MAIN | `irc_control_pkg.main_irc` | 기본 제어·레일과의 연결 필요 |
| VLM 연동 MAIN | `irc_control_pkg.main_vlm` | 이 흐름과 짝이 맞는 제어 구현 필요; 기본 MAIN과 동시에 시작하지 않음 |
| 경로·모터·좌표 변환 | `point_pose_node`, `motor_motion_control_node`, `transform_node` | 실제 장치 연결·보정·시리얼 설정 필요 |
| 비전 | `detect_yolo_sam_obb_pub` | 별도 비전 환경·YOLO/SAM2 가중치·RGB-D 장치 필요 |
| 외부 레일·검증 카메라 | 레일 제어 프로세스, `/vision/overlay_image` 이미지 공급 프로세스 | 현재 저장소만으로 전체 실행 명령을 확정할 수 없음 |

**AI 노드 실행 성공과 로봇 포장 전체 실행 성공은 다른 완료 조건이다.** 실제 장치를 자동 구동하는 일괄 명령은 이 문서 작성 과정에서 실행하지 않았다.

## 6. 실행 전 보완해야 할 항목

| 위치 | 현재 확인한 조건 | 실행에 필요한 보완 |
|---|---|---|
| `irc_control_pkg/setup.py` | `point_pose_vlm_node`, `rail_node`가 등록돼 있으나 대응 `point_pose_vlm_node.py`, `rail_bridge.py`가 없음 | 현장 제어·레일 코드 확보와 사용할 MAIN 조합 확정 |
| VLM 이미지 입력 | 기본 입력은 `/vision/overlay_image`의 `CompressedImage`; 확인한 검출 소스에는 이 발행자가 없음 | 별도 이미지 공급 프로세스 확보 |
| 비전 가중치 경로 | `/home/pc/...` YOLO·SAM2 파일이 현재 PC에 없음 | 비전 PC의 모델·설정 전달과 경로 구성 |
| Python 패키지 배포 | `setup.py`가 전체 실행 의존성을 선언하지 않으며 TTS `tts_prompts/*.wav`도 `package_data`에 없음 | 현재 명령처럼 소스에서 실행하거나 패키지 배포용 자산·의존성 명세 보완 |
| 재현성·GPU | 전체 런타임 lockfile·기반 모델 revision 고정 없음, 현재 NVIDIA 조회 실패 | 환경별 완전한 내보내기·자산 identity 확보, 대상 장비의 실제 GPU·동시 부하 확인 |

토픽 이름의 문서상 차이를 README 작성 차단 항목으로 삼지 않는다. 위 표는 **없는 구현·자산·환경 검증**에 관한 실행 조건이다.

## 7. 검증 기록

이번에 확인한 항목은 Git 기준점·소스 진입점·현재 Python 패키지 메타데이터·외부 경로 존재·PPT 내용·정적 토픽 도식이다. 외부 디렉터리 존재를 모델 적재·DB 무결성 통과로 표시하지 않는다.

기존 회귀 테스트의 실행 예시는 다음과 같다. 모델 적재나 ROS 발행 없이 실행되는 계약 테스트 범위이다.

```bash
cd /home/roma/IRC_SOOMAC3.0
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/soomac_irc /home/roma/miniconda3/envs/gemma4_env/bin/python -m pytest -p no:cacheprovider src/soomac_irc/test/test_natural_multiturn_contract.py src/soomac_irc/test/test_section_question_events.py src/soomac_irc/test/test_evaluation_v43.py src/soomac_irc/test/test_scenario_context_evaluation.py -q
```

이 자리에서 살아있는 변수: `PYTHONDONTWRITEBYTECODE`·`PYTHONPATH`는 같은 명령에서 정의한다. 사용자 Python 변수·외부 셸 별칭은 사용하지 않는다. 이 명세 작성에서는 해당 회귀 테스트를 실행하지 않았으며, 문서 링크·명령 구문·패키지 진입점만 별도로 확인한다.

## 용어 정리

| 용어 | 의미 |
|---|---|
| 실행 환경(runtime environment) | 노드가 사용하는 Python·라이브러리·시스템 서비스·장치를 포함한 환경이다. |
| 잠금 파일(lockfile) | 재설치할 패키지·버전·배포 자산을 고정하는 파일이다. 이번 메타데이터 스냅샷은 이를 대체하지 않는다. |
| 모델 자산(model assets) | 가중치·설정·토크나이저·채팅 템플릿·전처리 설정을 함께 가리킨다. |
| 입력 예산(input budget) | 서버 길이 한도에서 최대 출력 예약을 제외하고 입력에 사용할 수 있는 토큰 수이다. |
| 프로세스 간 연결(inter-process communication) | 이 프로젝트에서는 ROS 토픽 또는 로컬 HTTP로 서로 다른 실행 환경을 연결하는 방식이다. |
