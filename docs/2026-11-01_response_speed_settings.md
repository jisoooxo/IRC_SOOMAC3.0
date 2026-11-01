# Response 로컬 학습 속도 설정

참고: [최신 데이터 MANIFEST](/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/candidates/compound_semantics_20261101_r2/dataset/MANIFEST.json), [이전 Response PROFILE](/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/runs/r5_local_response_reaudited_20261101_build04/response/PROFILE.json), [build05 Decision PROFILE](/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/runs/r5_package_test_20261101_build05_speed/gpu_verification/profile/PROFILE.json), [설정 JSON](../training/learning_first_v43/training_release_20261101_r5/local/response_speed.json) → 최신 데이터와 실제 프로파일에 묶인 로컬 속도 후보이며, Response GPU 선택 결과는 아직 없다.

## 입력과 길이

최신 `MANIFEST.json` SHA256은 `7850948287237c5ace9d1b0f56931419760fbce3f0afa02cea40681acb6ceffa`이다. Response `max_length`는 **9,216토큰**, train은 **1,664행**, validation은 **152행**이다. 현재 릴리스의 `core.encode_record`와 Gemma 4 processor로 CPU 인코딩한 실제 길이는 다음과 같다. 모델 가중치나 GPU는 로드하지 않았다.

| 분할 | route | 행수 | 최소 | 중앙값 | 95백분위 | 최대 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train | general | 141 | 1,507 | 1,721 | 1,802 | 1,839 |
| train | mixed | 58 | 3,023 | 3,069.5 | 3,258 | 3,259 |
| train | task | 1,465 | 7,361 | 7,565 | 7,838 | 8,430 |
| validation | general | 7 | 1,571 | 1,611 | 1,615 | 1,615 |
| validation | task | 145 | 7,398 | 7,534 | 7,699 | 8,158 |

원인: task 경로가 학습 행의 대부분이고, 최대 인코딩 길이는 현재 9,216토큰 상한 안에 있다.
결과: history·prompt·상한을 바꾸지 않고 메모리/속도 후보만 측정할 수 있다.

## 속도 후보와 측정 경계

`response_speed.json`은 프로세스 요청 예산을 **32,000,000,000바이트**, allocator 밖 예약량을 **536,870,912바이트(512MiB)**로 지정한다. 요청값 기준 allocator 한도는 **31,463,129,088바이트**로, 1GiB 예약 후보보다 536,870,912바이트 크다. **프로세스 전체 32,000,000,000바이트 한도는 그대로이다.** `Budget`은 GPU 총량, 외부 `GPU_MEMORY_BUDGET_BYTES` 또는 `GPU_MEMORY_FRACTION`이 더 낮으면 그 값으로 낮춘다. NVML 2초 표본과 PyTorch allocator 검사는 하드웨어의 물리 OOM을 보장하지 않는다.

`gradient_checkpointing=auto`는 OFF와 ON 각각에서 microbatch 1·2·4·8을 유한하게 측정한다. 각 후보는 스트레스 입력을 통과한 뒤 **effective batch 16을 채우는 누적 optimizer update**의 초당 sequence 수로 비교한다. 후보 OOM은 해당 모드의 더 큰 batch 측정을 멈추며, 통과한 후보 중 가장 빠른 것을 선택한다. OFF가 들어간다는 보장은 없다. ON만 통과하면 그중 가장 빠른 batch를 사용한다. 이 단일 참조 update는 긴 학습의 일정한 속도나 최종 완주 시간을 보장하지 않는다.

이전 build04 Response PROFILE은 checkpointing ON, 31,000,000,000바이트 프로세스 상한과 2GiB 예약에서 batch 1 통과, batch 2 OOM이었다. 당시 batch 2 OOM 메시지는 장치 여유 3.89GiB, 프로세스 사용 26.52GiB, 허용 26.87GiB였다. 그 프로파일의 `reference_longest` 시간은 단일 microbatch 측정이며 새 full effective-batch update 시간과 직접 비교할 수 없다. 더 높은 요청 예산과 낮은 예약량도 물리 GPU 여유 또는 다른 프로세스 점유를 늘리지는 않는다.

최신 build05 **Decision** PROFILE에서는 32,000,000,000바이트 프로세스 한도·1GiB 예약으로 ON batch 2가 1,010MiB 추가 할당에서 OOM이었다. 그 순간 보고된 프로세스 사용량은 28.74GiB, 물리 여유는 1.52GiB이다. 512MiB 예약이면 allocator 한도가 512MiB 늘어 그 할당을 시험할 여지는 있지만, 반올림된 사용량과 요청량 기준 프로세스 한도까지 남는 값은 약 **78MiB**뿐이다. 프로파일의 `observed_process_peak_bytes=31,204,573,184`는 다른 시점까지 포함한 2초 표본이며 그 OOM 순간의 여유를 입증하지 않는다. 미할당 예약 메모리의 재사용과 단편화가 결과를 바꿀 수 있어, 이 산술로 **반드시 통과한다거나 반드시 OOM이라는 결론을 낼 수 없다.** Decision 입력의 결과로 Response의 배치 적합성도 알 수 없다.

원인: 1GiB 예약에서 유한 프로파일이 메모리 경계에 닿았고, `Budget`은 512MiB 예약을 허용한다.
결과: 512MiB는 로컬에서 더 빠른 후보를 측정하기 위한 설정이며, 32GB 전체 프로세스 한도와 OOM 중단 정책은 유지된다. 원격 봉인 build05의 1GiB 설정은 바꾸지 않는다.

원인: 이전 프로파일은 ON의 batch 1만 통과했으므로 OFF/ON 속도 우열을 증명하지 못한다.
결과: JSON의 상태를 `requires_gpu_measurement`로 두고 GPU profile 및 본학습 로그로 최종 선택을 확인해야 한다.

## 학습과 복구 계약

effective batch **16**, **6 epochs**, Response learning rate **5e-5**와 기존 LoRA 구조·데이터·프롬프트는 그대로이다. train 1,664행 기준 계획 update는 `ceil(1664/16) × 6 = 624`이다. `core`는 본학습에서 첫 **5 step**에 full-state checkpoint를 저장하도록 하고 이후 50 step 간격과 마지막 step에 저장한다. 기록에는 adapter, optimizer, scheduler, RNG와 trainer state가 포함되어 검증 후 resume marker가 게시된다.

이전 취소된 12/624 Response 실행에는 커밋된 checkpoint와 resume marker가 없었다. 그 12 step은 복원할 수 없으며, 기존 출력·모델·데이터는 보존한다. 새 학습은 별도 출력에서 시작해야 한다. 프로파일은 버린 LoRA 후보로 측정하므로 본학습에서 다른 메모리 압력이나 OOM이 생길 수 있다. OOM 시 batch를 조용히 낮춰 재시작하지 않고, 저장된 동일 설정 checkpoint가 있을 때만 명시적으로 복구한다.

원인: `resume`은 커밋된 full-state checkpoint와 동일 설정을 요구한다.
결과: 새 실행의 첫 복구 가능 지점은 실제 5 step checkpoint가 검증되어 게시된 뒤이다.

## 용어 정리

- 유효 배치(effective batch): optimizer update 한 번에 누적되는 표본 수이며 여기서는 16이다.
- 마이크로배치(microbatch): 한 번의 forward/backward에 올리는 표본 수이다.
- 경사 체크포인팅(gradient checkpointing): 일부 활성값을 저장하지 않고 backward 때 다시 계산해 메모리를 줄이는 방식이다.
- 전체 상태 체크포인트(full-state checkpoint): adapter 외에 optimizer·scheduler·난수 상태를 포함해 학습을 복원하는 저장점이다.

(내가 의역해보자면, 이 설정은 빠른 쪽을 **측정할 기회**를 만든 것이며 빠른 쪽이 이미 확정됐다는 뜻은 아니다.)
