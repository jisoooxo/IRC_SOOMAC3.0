# 로컬 Response·Recommendation 속도 설정 연결

참고: [봉인된 r5 release](/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/releases/conversation_r5_20261101_build05_speed/RELEASE.json), [로컬 실행기](../training/learning_first_v43/training_release_20261101_r5/local/launch.py), [Response 설정](../training/learning_first_v43/training_release_20261101_r5/local/response_speed.json), [Recommendation 설정](../training/learning_first_v43/training_release_20261101_r5/local/recommendation_speed.json) → Decision production TAR 검증 뒤 Response와 Recommendation을 순서대로 실행하는 로컬 정책을 연결한다.

## 입력 계약

`local/launch.py`에 `--agent-settings-dir`를 주면 그 디렉터리의 `response_speed.json`, `recommendation_speed.json`을 **출력 생성 전에 함께 읽고 검증한다**. 각 파일은 schema 1, 해당 agent, 최신 MANIFEST SHA-256 `7850948287237c5ace9d1b0f56931419760fbce3f0afa02cea40681acb6ceffa`, 프로세스 요청 예산 32,000,000,000바이트, allocator 밖 예약량 **536,870,912바이트(최신 로컬 후보)**, `gradient_checkpointing=auto`, 유효 배치 16, 6 epochs, Response 학습률 5e-5 또는 Recommendation 학습률 1e-4, `requires_gpu_measurement` 상태를 기록한다. 요청 예산은 2GiB 초과·32,000,000,000바이트 이하, 예약량은 512MiB 이상·요청 예산 미만이어야 한다. 최초 후보는1GiB였으나 좁은 CLI 검수 후 로컬만512MiB 후보로 변경했다. 원격 봉인 build05의1GiB는 그대로이며, P/Q 적합성·속도는 아직 미측정이다.

실행기는 봉인된 bundle의 `RELEASE.json`이 지정한 `common/core.py`, `common/train_agent.py`, 완료 검증기 `common/checkpointing.py` 해시를 확인하고, 학습 코드의 epoch·학습률·유효 배치 값을 읽어 JSON과 대조한다. 완료 검증기 해시는 실제 import 직전에도 다시 확인한다. 이 값들은 공통 학습 코드의 고정 계약이다. JSON을 바꿔 epoch·학습률·유효 배치를 조정하는 경로는 없다. `--agent-settings-dir`와 전역 속도 override는 함께 사용할 수 없다. 설정 디렉터리 없이 실행하면 기존 전역 플래그와 pipeline identity가 유지된다.

원인: JSON은 속도 측정 후보이며 학습 파라미터를 변경하는 별도 runner 인자가 없다.
결과: 로컬 실행기는 봉인된 학습 계약과 일치하는 후보만 runner에 전달한다.

## 실행·재개

학습 순서는 **Decision production TAR 검증 → 로컬 Response → 로컬 Recommendation**이다. Decision TAR와 gate receipt의 생성·검증은 기존 실행 계약을 따른다. 로컬 실행기는 Response가 비정상 종료하거나 완료 영수증 검증에 실패하면 Recommendation을 시작하지 않는다. 각 agent 직전 GPU 점검과 runner의 `--budget-bytes`, `--non-allocator-reserve-bytes`, `--gradient-checkpointing`은 같은 agent JSON에서 온다. GPU 사용 가능 여부는 시작 시 다시 확인하며, JSON의 `requires_gpu_measurement`는 측정 완료를 주장하지 않는다.

Pipeline identity에는 정규화된 두 정책, 각 JSON 원본 바이트 SHA-256, 이 실행기의 SHA-256, 봉인 release·manifest·model·승인 방식이 들어간다. JSON의 공백만 바뀌어도 기존 output에서 재개를 거부한다. 값이 그대로인 재개는 기존 완료 영수증을 다시 검증하고, 검증된 Response는 건너뛴다. 공통 학습 코드는 커밋된 checkpoint의 adapter·optimizer·scheduler·RNG를 검증한 뒤 같은 설정으로 복원한다. 자동 무한 재시도는 없다.

`--agent-settings-dir` 실행에서 identity 최상위의 전역 옵션31e9/2GiB/on은 호환용 parser 기본값이다. **실제 정책은 `identity.agent_settings.<agent>.policy`와 child argv**를 읽는다. 현재 실제 P trainer PID914979의 인자는32e9/512MiB/auto이며, 이 기록을 읽기 쉽게 바꾼다는 이유로 실행 중 operator나 기존 identity를 수정하지 않는다.

이번 연결은 저장소의 `local/launch.py`를 실행하는 방식이다. 이 파일은 봉인 bundle의 `local/launch.py`와 해시가 다르며, 봉인 bundle이나 원래 RELEASE pin은 수정하지 않았다. 실행기의 실제 해시를 별도 pipeline identity로 기록한다. 따라서 새 실행기에 대해 **봉인 bundle 전체 해시가 통과했다고 주장하지 않는다**.

원인: 파일 내용이나 실행 코드가 달라진 상태로 기존 checkpoint를 이어 쓰면 같은 실험이라는 근거가 사라진다.
결과: 정책·코드 drift는 runner 시작 전에 차단되고, 검증된 완료 결과만 재사용된다.

## CPU 검증 경계

`test_agent_settings.py`는 실제 runner 인자, agent별 GPU 점검 예산, 원본 SHA-256·정규화 정책 identity, 변경된 JSON 재개 거부, Response 실패 시 중단, 봉인 코드 hash 및 설정 대조를 CPU mock으로 확인한다. 기존 `test_launch.py`의 gate·완료 검증·legacy 옵션도 함께 실행한다. 테스트는 GPU 학습이나 실제 속도 측정을 수행하지 않는다.

## 용어 정리

- **유효 배치(effective batch)**: 한 optimizer update에 누적되는 표본 수이다. 이 release에서는 16이다.
- **프로세스 예산(process budget)**: 학습 프로세스가 요청하는 GPU 메모리 상한이다. 실제 허용량은 서버·장치 한도에 따라 낮아질 수 있다.
- **재개 식별자(pipeline identity)**: output이 어느 release·설정·실행 코드에 속하는지 기록해 변경된 조건의 재개를 막는 값이다.

(내가 의역해보자면, JSON은 더 빠를 수 있는 선택지를 **GPU에서 시험할 조건**이지, 이미 빠르다고 판정한 결과가 아니다.)
