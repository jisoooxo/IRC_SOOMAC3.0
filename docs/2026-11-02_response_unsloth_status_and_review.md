# Response Unsloth 학습 진행·환경 검수·판단 기록

[로컬 본 학습 계획](2026-11-01_local_unsloth_production_plan.md) · [기존 작업 기록](2026-11-01_training_package_build_worklog.md) · [학습 실행기](../tools/unsloth_training/production.py) · [Golden Docker 담당 문서](2026-11-02_decision_unsloth_golden_candidate.md)
→ **학습은 유지하며, 전체 epoch의 loss 추이와 실제 시나리오 출력을 함께 보고 최종 checkpoint를 선택한다.**

작성 기준: **2026-11-02 11:24:38 KST**. 현재 로그를 읽은 시점의 스냅샷이며 자동 갱신 문서가 아니다. 작업 브랜치는 `llm_final_2`, HEAD는 `831109146f17a0e43da61da03d11943ee22dde1b`이다. 작업 트리에는 다른 담당자의 변경이 있으므로 HEAD만으로 현재 소스 전체를 식별하지 않는다.

## 1. 현재 Response 학습 상태

| 항목 | 관측 결과 |
| --- | --- |
| 진행 단계 | `training`, **215 / 624 optimizer step, 약 34.5%** |
| Epoch | 최근 logging 기준 **2.067 / 6** |
| 최신 train loss | **0.1740353**, 215step 기록; stdout 표시는 `0.174`이다. |
| 최신 gradient norm | **6.696**, 215step logging 값이다. |
| 최신 validation loss | **0.9426264**, 208step / epoch 2.0이다. |
| 현재 validation 최저 | **0.8207008**, checkpoint-150이다. |
| 마지막 저장본 | checkpoint-208; resume marker가 가리키는 경로이다. 이번 기록에서 전체 checkpoint 바이너리를 다시 해시하지 않았다. |
| 건너뛴 update | progress 기록상 **0**이다. |
| 로그 오류 검색 | 읽은 로그에서 `Traceback (most recent call last)`, `CUDA out of memory`, `LoRA gradient/loss gate failed`, `non-finite training loss` 일치가 없다. 모든 종류의 오류 부재를 증명하는 검사는 아니다. |

실행 출력 루트는 다음과 같다.

```text
/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/runs/r5_unsloth_local_authorized_20261101/response
```

학습 로그는 같은 부모의 `response.training.log`이다. 상태 근거는 출력 루트의 `.lab-monitor/progress.json`, `.lab-monitor/resume.json`, `INPUT_IDENTITY.json`과 checkpoint-208의 `trainer_state.json`이다.

현재 attempt 경로는 `attempt-63872c5d7d584aab81aa73bbb6cc0155/response`이다. 저장 영수증의 `best_model_checkpoint`는 이 경로 아래 **checkpoint-150**을 가리킨다. validation 최저라는 사실과 실제 시나리오에서 최선이라는 판정은 구분한다.

| 학습 계약 | 현재 identity 기록 |
| --- | --- |
| 데이터 | train 1,664행 / validation 152행; history 축소 없음 |
| 학습량 | 6epoch / 624step / batch 1 × accumulation 16 |
| 양자화·계산 | NF4 double quant / BF16 / native CE / `UNSLOTH_RETURN_LOGITS=1` |
| LoRA | r16 / alpha32 / dropout 0.05 / text decoder 대상 |
| Optimizer·scheduler | `paged_adamw_8bit` / cosine / learning rate 5e-5 / warmup ratio 0.03 |

원인: 기존 production 실행이 update와 주기적 검증을 계속 기록하고 있다.
결과: 학습 전진과 저장 상태는 확인되지만, 본 학습 완료·최종 모델 품질 합격은 아직 아니다.

## 2. Loss 추이와 사용자 판단

대표 train loss 이력은 다음과 같다. 초기 값은 같은 로그의 이전 조회에서 확인한 값이며, 현재 값은 작성 시 다시 읽었다. logging 구간의 평균이며 동일 고정 입력의 전후 비교는 아니다.

| Step | Train loss |
| ---: | ---: |
| 1 | 2.756 |
| 15 | 2.118 |
| 75 | 0.4075 |
| 210 | 0.1856 |
| 215 | 0.1740 |

checkpoint-208의 `trainer_state.json`에 보존된 validation 이력 전체는 다음과 같다.

| Step | Epoch | Validation loss |
| ---: | ---: | ---: |
| 50 | 0.4808 | 0.8667920 |
| 100 | 0.9615 | 0.8593374 |
| 104 | 1.0000 | 0.8582091 |
| **150** | **1.4423** | **0.8207008 — 현재 최저** |
| 200 | 1.9231 | 0.8739794 |
| 208 | 2.0000 | 0.9426264 |

**관측 사실은 train loss 감소와 최근 validation loss 두 차례 상승이다. 이 사실만으로 과적합을 확정하거나 학습 중단을 결정하지 않는다.**

사용자는 아직 6epoch 중 약 2epoch만 진행됐다는 점을 지적했고, 이후 epoch까지 관측한 뒤 판단하기로 했다. 이 대화에서 과적합 가능성을 먼저 강조했던 설명은 **반등 관찰과 원인 확정을 구분하는 표현으로 정정**한다. 모든 epoch를 완료하는 것 자체가 품질 합격을 뜻하지도 않는다.

또한 **validation loss와 실제 SOOMAC 시나리오 성능은 별도 평가 대상**이다. 정답 토큰 예측 오차가 낮은 checkpoint가 주문·수량·보존량·제외 조건·실제 완료 사실을 가장 적절히 말하는 checkpoint와 같다고 보장하지 않는다. (내가 이해한 현재 판단은, loss 추이를 끝까지 관찰하면서 실제 대화 출력으로 최종 후보를 고르는 것이다.)

원인: validation 반등의 원인은 미확정이며, 전체 학습과 실제 시나리오 평가가 아직 끝나지 않았다.
결과: **현재 학습을 유지하고, checkpoint-150·이후 epoch별 저장본·최종 결과를 비교 후보로 둔다.** 이 문서 작성으로 학습 정책이나 평가기를 변경하지 않는다.

## 3. ETA 변동의 원인과 해석

[진행 표시 코드](../training/learning_first_v43/training_release_20261101_r5/common/lab_progress.py)의 `_console_line()`은 `training_seconds`가 최신 기록보다 30초 이상 작은 가장 가까운 기록을 baseline으로 선택한다. 실제 확인한 봉인 소스 위치는 아래와 같다.

```text
/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/releases/conversation_r5_20261101_build05_speed/common/lab_progress.py
```

실행기의 `on_step_end()`에서 누적 학습 시간이 갱신된다. 현재 한 step은 30초보다 길기 때문에 `[LAB_PROGRESS]` ETA는 사실상 직전 한 step 속도에 크게 의존한다. heartbeat 출력만으로 새 step 처리 시간이 생기는 것은 아니다.

```text
추정 남은 시간 = 남은 step 수 × 관측 구간의 평균 step 시간
```

남은 step이 600개라면 step당 180초는 30시간, 190초는 31시간 40분이다. **한 step의 10초 차이가 ETA 약 1시간 40분 차이로 확대된다.** 입력 길이와 연산량 차이는 가능한 속도 변동 요인이지만 이번 변동의 원인별 기여는 측정하지 않았다.

작성 시점의 누적 학습 시간은 38,427.52초이며, 이를 완료한 215step으로 나누면 **평균 178.73초/step**이다. 남은 409step을 같은 평균으로 계산한 값은 **학습 step 계산만 약 20시간 18분**이다. 고정된 완료 시각이나 소요시간 상한이 아니다.

기존 validation 한 회의 기록은 약 423~444초이다. 위 학습 시간 기반 ETA에는 별도 validation·저장·최종 평가 시간이 모두 포함되어 있지 않다. tqdm의 표시값, `[LAB_PROGRESS]` 값, 누적 평균 추정치를 같은 계산으로 취급하지 않는다.

원인: 짧은 속도 관측 구간을 남은 전체 step에 확대 적용한다.
결과: **ETA 요동만으로 학습 이상을 판단하지 않는다.** 앞서 마지막 `ETA=` 줄만 보라고 했던 안내는 정정하며, 누적 평균과 별도 평가 비용을 함께 본다.

## 4. 앞서 수행한 Golden 환경 독립 검수

**이 절은 2026-11-02 00:51 KST까지 수행한 검수 이력이다. 현재 시각의 Docker 완료 상태로 읽지 않는다.** GPU 시험을 재실행하지 않고, 실제 PID·파일·기존 검증 기록을 읽었다.

| 증거 구분 | 확인한 내용과 한계 |
| --- | --- |
| 실제 PID 직접 관측 | PID 1405238의 실행 파일은 `gemma4_env/bin/python3.10`, 명령의 Python은 r5 venv 경로였다. systemd는 active/running, RAM 44GiB / swap 0이었다. |
| 실제 PID 직접 관측 | 시작 환경에 `PYTHONPATH`가 없었다. `/proc/environ`을 Python 코드가 이후 변경한 환경변수 전체의 증거로 취급하지 않았다. |
| 실제 PID 직접 관측 | 기반 환경의 Torch·Triton·CUDA 13·bitsandbytes·torchao·torchaudio·torchvision 확장이 매핑됐다. xFormers·외부 FlashAttention 확장 매핑은 없었다. |
| 별도 Python의 메타데이터·경로 해석 | Python 3.10.20 / r5 `include-system-site-packages=true` / 기반 gemma4_env를 확인했다. 무거운 패키지를 실제 import하지 않고 metadata·`find_spec`으로 경로를 해석했다. |
| 독립 파일 비교 | Golden은 r5 21개 + 기반 환경 113개 배포판이다. PID에 매핑된 site-packages 네이티브 파일 296개가 모두 목록에 포함됐고, 핵심 소스·바이너리 22개 약 41MB의 SHA가 Golden 기록과 같았다. |

핵심 조합은 Torch `2.12.1+cu130`, Transformers `5.13.1`, PEFT `0.19.1`, Triton `3.7.1`, bitsandbytes `0.49.2`, accelerate `1.14.0`, Unsloth와 Unsloth Zoo 각각 `2026.10.11`이다. Unsloth 두 패키지는 r5에서, 앞의 여섯 패키지는 gemma4_env에서 해석됐다.

| 알려진 위험 | 기존 증거에 따른 판정 |
| --- | --- |
| xFormers 0.0.35 | build 정보는 Torch 2.10.0+cu128이고 현재 조합과 불일치한다. 로그는 PyTorch attention fallback을 명시한다. 기존 warm benchmark profiler는 SDPA/CUTLASS를 기록했다. 경고만으로 현재 계산 오류라고 판정하지 않았다. |
| torchaudio·torchao·torchvision | 각각 2.11.0+cu130 / 0.17.0+cu130 / 0.27.1+cu130이다. 실제 확장 매핑은 확인했지만 모든 선택적 연산의 ABI를 검증한 것은 아니다. 현재 양자화는 bitsandbytes NF4 경로이다. |
| fla.utils | Unsloth Zoo의 vendored 코드가 Python 3.11을 권장하는 경고이다. 독립 flash-linear-attention 필수 패키지 누락으로 해석하지 않았다. |
| Native CE·LoRA | 기존 대표 입력 CE·full-state resume·표준 PEFT reload·vLLM staging 증거를 연결했다. 전체 학습·모든 입력의 수치 동등성이나 최종 기능 품질 합격으로 확대하지 않았다. |

warm benchmark·native CE·smoke·production은 핵심 패키지 버전과 model builder·release·모델 설정·tokenizer identity가 연결된다. smoke는 accumulation 1 / constant scheduler이고 production은 accumulation 16 / cosine이라는 의도된 차이가 있다. PEFT reload는 별도의 비패치 로딩 경로이며 vLLM staging은 학습 라이브러리 동일성 대신 adapter 호환성의 증거로 취급한다.

과거 임시 venv와 r5의 21개 배포판 RECORD 선언도 동일했다. 다만 **과거 각 실행 시점의 전체 import 경로·바이너리 해시가 모두 보존된 것은 아니므로, 전체 바이트 동일성을 소급 확정하지 않는다.**

이번 문서 작성 시 다시 읽은 현재 소스의 SHA256은 다음과 같다.

| 파일 | SHA256 |
| --- | --- |
| `tools/unsloth_training/production.py` | `86638bad3614a633b6fa023694f5c7b37ca840e9e37177be0eeff9d6a27e3208` |
| `tools/training_acceleration/benchmark.py` | `e89385b0e4257d3c0e7b7947a4a8c1bb7e67e8ce2e27873efd2afac8ed45394c` |

현재 identity의 release SHA는 `ff3285e46bff4f56c18327adb3b038a1010ab5e7482ca59729ab86c1f3a9459e`, 데이터 manifest SHA는 `7850948287237c5ace9d1b0f56931419760fbce3f0afa02cea40681acb6ceffa`이다. 생성 Unified forward SHA는 identity에 `a8f33a5684d9b3eb3a6867a843e5b33c7a27900075e0683971d2892a8eb2d987`로 기록되어 있고, 00:51 이전 파일 비교에서 일치를 확인했다.

원인: r5는 기반 환경을 상속하며 버전 문자열만으로 실제 출처·바이너리 동일성을 판단할 수 없다.
결과: 기존 검수 범위에서는 **현재 Response 학습을 중단할 확인된 계산 오류를 발견하지 못했다.** 이는 최종 품질 합격이나 모든 선택적 경로의 무결점 보증이 아니다.

## 5. Docker 검수의 당시 상태와 후속 범위

00:51 기준 Golden Docker는 휠 SHA256 잠금과 설치 파일 SHA256 검사를 모두 갖췄다. `pip install --no-deps --no-index --require-hashes`를 사용하며, xFormers의 같은 fallback을 보존하는 정책이었다. c1 빌드 로그에는 **134개 배포판·34,261개 설치 파일·차이 0개 PASS**가 있었고 c2도 해당 파일 검사 PASS 한 건을 확인했다.

검사는 RECORD에 등록된 해시 파일 대상이다. `.pyc`, 일부 설치 메타데이터, 추가 생성 파일·JIT 캐시까지 전부 동일하다는 뜻은 아니다. c1의 UID 1000 사용자 조회 오류는 담당 Dockerfile에 보고되어 있었고 c2에 사용자 등록이 추가됐다. 원래 CPU probe 컨테나는 제거되어 오류 로그를 독립적으로 재확인하지 못했다.

**당시 c2의 최종 빌드·CPU import/encode·실제 import 경로·GPU 검증은 완료 판정하지 않았다.** 이후 담당자의 진척은 [Golden Docker 담당 문서](2026-11-02_decision_unsloth_golden_candidate.md)에서 별도로 확인해야 한다. 이번 기록 요청에서 Docker 상태를 다시 검사하거나 승인 상태를 변경하지 않았다.

후속 판단은 다음 범위로 유지한다.

1. Response의 나머지 epoch와 주기적 validation을 관측한다. 두 번의 validation 반등만으로 중단하거나 학습 정책을 바꾸지 않는다.
2. 완료 후 validation best·후속 epoch·최종 결과를 기존 시나리오로 비교한다. 이번에 실제 추론이나 평가를 새로 실행하지 않았다.
3. Docker 제작과 학습 운영은 각 담당자의 범위이다. 이 문서의 과거 검수 결과를 원격 Decision 본 학습 투입 승인으로 사용하지 않는다.

원인: 환경 재현·학습 수치·실제 시나리오 성능·원격 실행 준비는 서로 다른 완료 조건이다.
결과: **이번 작업은 새 Markdown 기록만 작성하며, 학습·대기 서비스·원격 작업·가상환경·checkpoint·다른 담당자 파일을 변경하지 않는다.**
