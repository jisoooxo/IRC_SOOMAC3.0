# GC OFF 우선 실측 — Response 학습 시간 비교

[시험 코드](../tools/training_acceleration/memory_speed_trial.py) · [기준 벤치마크](../tools/training_acceleration/benchmark.py)
→ **예약 메모리를 적게 쓰는 설정이 아니라, 동일 학습의 optimizer update를 빨리 끝내는 설정을 고른다.**

## 1. 보존한 조건

폐기용 새 프로세스와 `/tmp/soomac-training-accel-20261101/`의 새 출력만 사용한다. production, 기존 checkpoint-5, 원본 데이터, 기존 환경, 원격 Decision 작업은 변경하지 않는다.

동일 Response 최장 행 8,430토큰을 update당 16회 사용한다. NF4 double quant/BF16, LoRA r16·alpha32·dropout0.05, native selected-logits loss(`UNSLOTH_RETURN_LOGITS=1`), gradient clipping1, 유효 배치16을 유지한다. 입력을 자르거나 history·정답 label을 바꾸지 않는다. 반복된 한 행의 처리량 시험이지 전체 데이터의 품질 검증은 아니다.

로컬 GPU는 RTX PRO 4500 Blackwell 32GB이다. 시험의 process budget32,000,000,000바이트에서512MiB를 제외한 allocator limit은31,463,129,088바이트이다. 이전 **12.9GB 예약량은 사용 실측이며 상한이 아니다.**

## 2. 실제 측정 결과

| 설정 | 실제 진행 | optimizer update 시간 | peak allocated / reserved |
| --- | --- | --- | --- |
| Unsloth GC ON, batch1×acc16 — 기존 기준 | 1 update 완료, cold 측정 | 276.3486초, 488.079 non-padding tokens/s | 12,297,832,448 / 12,855,541,760B |
| GC OFF, batch1×acc16 | microstep1 forward OOM | 없음: optimizer에 도달하지 못함 | 31,164,964,864 / 31,427,919,872B |
| native 부분 GC32/48, skip16, batch1×acc16 | microstep1 forward OOM | 없음 | 31,181,741,056 / 31,444,697,088B |
| native 부분 GC40/48, skip8, batch1×acc16 | **microstep1~3 forward/backward 진행 후 microstep4 forward OOM** | 없음: 누적16을 완료하지 못함 | 31,095,280,128 / 31,450,988,544B |
| Unsloth GC ON, batch2×acc8 | 1 update 완료, finite/nonzero LoRA gradient | 321.8207초, 419.115 non-padding tokens/s | 16,291,553,792 / 16,903,045,120B |
| native 전체 GC48/48, batch1×acc16 — smart GC와 구현 비교 | 1 update 및 별도 profiler 완료 | 281.3407초, 479.419 non-padding tokens/s | 15,184,823,808 / 15,644,753,920B |

각 결과는 순서대로 `response_unsloth_native1_smoke`, `response_gc_off_batch1_a1`, `response_gc_partial_skip16_batch1_a1`, `response_gc_partial_skip8_batch1_a1`, `response_gc_on_batch2_a1`, `response_gc_native_full_batch1_a1` 아래의 `RESULT.json`/로그에 남긴다. 실패 결과를 지우거나 성공으로 덮어쓰지 않는다.

Cause: GC를 끄면 재계산을 피하는 대신 backward에 필요한 활성값을 더 오래 저장한다. 부분 GC40/48도 gradient accumulation 중 필요한 메모리를 확보하지 못했다.
Effect: 이 입력·모델·32GB 조건에서는 GC OFF 및 시험한 두 부분 설정을 본 학습에 채택하지 않는다. 미시도 부분 설정 전체가 불가능하다는 뜻은 아니다.

## 3. GC 재활성화와 CPU offload 확인 범위

OFF 시험은 `FastModel.from_pretrained`와 `get_peft_model` 양쪽에 `use_gradient_checkpointing=False`, `TrainingArguments.gradient_checkpointing=False`를 기록했다. `model.train()` 이후에도 `model.is_gradient_checkpointing=False`, GC 속성이 있는49개 모듈의 활성 플래그0개였다. Trainer를 실행한 시험이 아니라 **276초 기준과 같은 수동 optimizer loop**이며, Trainer가 학습 도중 다시 켜지 않는다는 증거로 확대하지 않는다.

부분 GC32/48·40/48의 실제 활성 모듈은 각각33·41개였다. decoder 층32·40개 외에 상위 모듈 플래그1개를 포함한다. 두 시험은 `use_gradient_checkpointing=True`인 native 경로이며 CPU activation buffer0, offload generation0, nonreentrant offload counters0, 입력 embedding은cuda에 있었다. 이는 상태 관측이며 실제 모든 CUDA/CPU 이동을 증명하는 profiler 결과는 아니다. embedding weight offload와 activation offload는 별개로 기록한다.

완료된 smart GC batch2 시험에서는 CPU activation buffer200개/6,256,590,848바이트, offload generation8을 기록했다. nonreentrant counters는0이지만 **smart CPU activation offload는 실제 실행됐다.** 서로 다른 통계 범위를 혼동하지 않는다. 이 버퍼 용량을 전부 실사용·전송량으로 해석하지 않는다.

Unsloth smart GC(`"unsloth"`)와 native GC(`True`)는 별도 경로이다. 부분 GC 결과는 checkpoint coverage도 동시에 달라지므로 **offload만 껐을 때의 성능 차이로 해석하지 않는다.** 성공 후보는 실제 checkpoint 함수와 forward/backward trace를 추가 확인한다. 단순 클래스 이름 `CheckpointFunction`만으로 실행 경로를 판정하지 않는다.

**추가 검수 정정:** `use_gradient_checkpointing=True`는 smart offload뿐 아니라 `UnslothCheckpointFunction` 패치도 해제한다. 따라서48층 GC를 유지해도 이번 full native 비교는 **GC 구현과 offload가 함께 바뀌는 후보 비교**이며, offload 단독의 인과 효과를 분리한 실험은 아니다. 이전 진행 설명의 'CPU offload만 끈다'는 표현을 이 한계로 정정한다. 설치 소스 `unsloth/models/_utils.py:509` 및 `unsloth_zoo/gradient_checkpointing.py:2476~2503`을 root가 직접 확인했다. 라이브러리 내부 임계값을 임의 변경한 추가 실험은 시작하지 않는다.

full native 시험은 UPDATE_PASS, 실제 checkpoint decoder48개 assertion 통과, update 및 추가 profiler 뒤 모두 CPU buffer0/offload generation0이었다. 시간 밖 추가1회 forward/backward에서 실제 `CheckpointFunction`48회와 memory-efficient SDPA forward96회·backward48회를 확인했다. DtoD659회/DtoH1회가 관측돼 **CPU 전송이 전혀 없었다고 쓰지 않는다.** smart activation offload가 쓰이지 않은 경로의 증거이지, 전송 이벤트1건의 원인을 확정한 것은 아니다. 이 profiler는 원본 입력·가중치·전체 trace를 전송하지 않고 이벤트 이름/횟수만 보존했다.

## 4. loss·gradient 검증의 경계

위 세 실패 설정 모두 update 전 같은 입력의 eval native CE는0.7945173978805542, 기준과 차이0이었다. dropout0.05는 유지했다. **eval CE 일치가 학습 모드 gradient의 수치적 동등성을 뜻하지 않는다.** OOM 후보는 완전한 update·최종 gradient 검사에 도달하지 않았으므로 통과라고 보고하지 않는다. 성공 후보는 finite/nonzero LoRA gradient, 전체 누적 update, peak와 시간을 함께 기록한다.

smart GC batch2의 eval CE도 기준과 차이0이었다. 학습 loss 평균은 기존 batch1의0.8148882985115051 대비 batch2에서0.8166290521621704(차이+0.00174075365)였다. 학습 dropout을 켠 microbatch 형태가 다르므로 이 수치만으로 label 오류나 gradient 동등성을 판정하지 않는다. 정확한 training-mode gradient 수치 대조는 이 시험의 통과 항목이 아니다.

full native의 eval CE도 차이0, 학습 loss 평균0.8129087686538696(기존 smart batch1 대비−0.00197952986)이며 finite/nonzero LoRA gradient를 확인했다. 학습 loss 차이의 원인은 미확정이며 정확한 gradient 동등성을 주장하지 않는다.

시간에는 같은 수동 loop의 forward·backward·clip·optimizer를 포함한다. 최초 cold update와 warm update를 섞어 확정 배수를 만들지 않는다. 비교 기준276초 자체가 cold 1회였으므로, 새 cold 1회는 선별용이며 채택 전 warm 재측정이 필요하다.

## 5. 종료 기준과 다음 작업

부분 GC 임의 탐색은 두 설정에서 멈췄다. GC ON batch2×acc8은 cold 선별에서 기준보다16.45% 느렸고, full native batch1도1.81% 오래 걸려 추가 속도 개선은 확인되지 않았다. 작은 차이는 단1회의 cold 측정 잡음일 수 있으므로 두 batch1 설정의 warm 우열은 미확정이다. 현재 선택은 **기존 검증된 smart Unsloth GC ON batch1을 기준 후보로 유지**하는 것이다. production 전환·기존 checkpoint 이식은 하지 않았다. CPU offload만을 같은 checkpoint 구현 안에서 독립적으로 끄는 비교는 미실시이며, native 결과로 대신 통과 처리하지 않는다.

시험용 CPU4건 통과, git diff whitespace 검사 통과, GPU 시험 프로세스 종료 및 compute process0개를 확인했다. 본 학습은 이 비교 중 시작하지 않았다. 성능 개선을 찾았다는 이유 없는 설정 변경도 하지 않았다.

Codex CLI(gpt-6-sol/high) 읽기 전용 검수는 OFF OOM·동일 입력/유효배치 유지에 찬성했고, 실제 부분 층수 확인·CPU offload 증거 한계·eval과 training 동등성 구분에 조건부 지적을 남겼다. root가 결과의 활성 모듈33/41개, 실패 microstep1/4와 인용 코드를 직접 대조했다. 원본 검수: `/tmp/soomac_gc_speed_trial_review.out`.

추가 코드 검수(`/tmp/soomac_gc_native_review.out`)는48층 assertion·profiler 시간 분리·native 선별 실행에 조건부 동의했다. 'offload만 변경'이 아니라 checkpoint 구현도 바뀐다는 지적을 소스와 대조해 문서·결론에 반영했다. 단1회 cold 결과의 production 채택과 training gradient 동등성은 승인된 것으로 간주하지 않는다.
