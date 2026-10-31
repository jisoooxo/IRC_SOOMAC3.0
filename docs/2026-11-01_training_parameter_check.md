# 로컬·원격 학습 설정 점검 — 원격 VRAM 약44GB 기준

[작업 기록](2026-11-01_dataset_rebuild_worklog.md) · [현재 로컬 wrapper](../training/learning_first_v43/conversation_supplement_20261101_r1/train_r3.py) · [기존 batch profiler](../training/learning_first_v43/final/goals/GL10/docker_v5_20261028/batch44_v6/v6_profile.py)
→ batch1 고정을 그대로 재사용하지 말고, 원격 Decision과 로컬 Response/Recommendation의 메모리 예산·배치를 각각 측정해 정한다.

## 1. 이번 점검 범위와 실제 확인

2026-11-01. 사용자는 원격 사용 가능 VRAM을 약44GB로 지정했다. **설정·GPU 상태·기존 코드·로그·저장 인코딩 길이만 확인했다.** 새 모델 로드, optimizer 실행, GPU 학습 프로파일, 본 학습, Docker 제작/업로드, 원격 접속·실행, Runtime/prompt/VLM/가속 코드 수정은 하지 않았다. 기존 사용자 변경과 체크포인트도 보존했다.

| 항목 | 실제 확인 결과 |
| --- | --- |
| 로컬 GPU | 호스트 `nvidia-smi`: NVIDIA RTX PRO4500 Blackwell, 총32623MiB, 사용1348MiB, 여유30770MiB, driver580.178.04. 조회 시 compute 프로세스 목록은 비어 있음. 이 여유가 장시간 보장되거나 특정 batch가 들어간다는 뜻은 아님 |
| 샌드박스 조회 | 내부에서는 driver 통신 실패, 동일 조회를 호스트 권한으로 재실행하자 성공. 실제 드라이버 고장으로 진단하지 않음. 설치/서비스 재시작 없음 |
| 원격 | **사용자 지정 약44GB**를 요청 예산 기준으로 잡음. 실측된 프로세스 한도가 아니다. 원격 장치 모델·현재 점유·환경 변수 상한은 아직 실측하지 않음 |
| 학습 후보 | `candidates/compound_semantics_20261101_r2/dataset`, manifest SHA `7850948287237c5ace9d1b0f56931419760fbce3f0afa02cea40681acb6ceffa` |
| 실행 결속 | 최신 후보에 `TRAINING_APPROVAL.json`이 없고 manifest의 `user_training_authorized=false`/조립 당시 상태가 남아 있음. 사용자 승인이 없다는 뜻이 아니라 새 실행용 **manifest와 승인 영수증의 검증 계약**이 아직 결속되지 않았다는 뜻. 기존 r3 wrapper는 manifest=true도 요구하므로 영수증만 만들어서는 preflight를 통과하지 못함 |

## 2. 코드에서 확인한 문제와 속도 관련 요인

### 2.1 batch1 고정 + 배치 탐색 미연결

`train_r3.py:97-98`은 train/eval batch1, accumulation16을 덮어쓴다. `:124`는 profiler의 `measure_one(..., 1, ...)`만 호출한다. 같은 경로의 `v6_profile.py`에는 후보1/2/4/8 탐색과 처리량 선택 함수가 있지만 **현재 r3 wrapper가 그 탐색을 호출하지 않는다.** 따라서 VRAM이 더 있어도 현재 wrapper는 더 큰 batch를 시험하지 않는다. 이것은 코드로 확인한 제한이며 과거 전체 지연의 단일 원인이라고 단정하지 않는다.

Cause: 안전한 batch1 설정을 고정하면서 기존 자동 batch 탐색 경로를 재연결하지 않았다.

Effect: 새 runner에서는 배치·누적·프로파일 영수증을 연결해야 하며, 기존 wrapper를 그대로 돌리면 여전히 batch1이다. 이번에는 수정하지 않았다.

### 2.2 로컬 31GB와 원격 44GB 예산을 분리해야 한다

현재 wrapper `:111`은 `Budget(31_000_000_000)`을 하드코딩한다. budget 모듈은 이 값·실제 장치 용량·서버 환경 제한 중 최소를 쓰며 별도2GiB를 allocator 밖 여유로 둔다. 같은 wrapper를 원격에 그대로 옮기면 요청값이31GB라 사용자44GB 예산을 활용하지 못한다.

이번 기록에서 보수적인 44GB는 `44_000_000_000` bytes(약40.98GiB)의 **요청값**으로 명시한다. 실제 장치·서버 제한으로 줄어들지 않는 경우에만 기존 reserve 정책의 allocator 예산이 `41_852_516_352` bytes가 된다. `Budget`은 다른 프로세스의 점유를 자동 차감하지 않으므로 원격 실행 전에 실제 가용량/다른 작업 점유도 확인해 예산을 낮추거나 대기해야 한다. 로컬은 기존31GB 요청값과 실제 다른 점유를 따로 확인하며, 두 장소의 숫자를 한 상수로 합치지 않는다. 서버가 더 작은 제한을 주면 그 제한을 우회하지 않는다.

free VRAM이 있어도 allocator의 프로세스 상한을 넘으면 OOM이 날 수 있다는 것은 [PyTorch 메모리 상한 공식 설명](https://docs.pytorch.org/docs/2.12/generated/torch.cuda.memory.set_per_process_memory_fraction.html)과도 일치한다. 단, 실제 안전한 batch는 모델·역전파·optimizer 상태까지 측정해야 한다.

### 2.3 짧은 정답이어도 학습 입력은 길다

최신 `validation/TOKEN_MASK_ROWS.jsonl`의 train 길이를 다시 읽었다. 다음 수치는 system/history/user/target을 합친 **1개 학습 예시의 실제 인코딩 길이**다. 생성 정답만의 길이가 아니다.

| Agent / 장소 | train 행 | 길이 중앙값 / p95 / 최대 | 전체 인코딩 token 합 | target token 합 |
| --- | ---: | ---: | ---: | ---: |
| Decision / 원격 | 4360 | 8069 / 8264 / 8608 | 35,348,919 | 79,740 |
| Response / 로컬 | 1664 | 7546 / 7827 / 8430 | 11,535,187 | 43,842 |
| Recommendation / 로컬 | 623 | 3981 / 4134 / 4201 | 2,485,522 | 29,663 |

P의 답변이 짧아도 입력은 대체로7500token대다. D는 대체로8000token대다. 이 입력 전체를 모델이 읽으므로 정답 길이만 보고 학습이 수초에 끝나야 한다고 추정하면 안 된다. 그렇다고 history를 잘라 속도를 맞추지도 않는다. profiler는 train+validation의 최장 입력·가장 이른 target 시작·혼합 길이로 검증해야 한다. 기존 scored-position loss는 이미 불필요한 logits 생성을 줄이도록 구현돼 있으므로 이 최적화가 없다고 진단하지 않는다.

### 2.4 이미 사용 중인 설정과 아직 확인하지 못한 병목

기반 trainer는 NF4 4bit + BF16 compute, SDPA, text decoder LoRA r16/alpha32/dropout0.05, paged AdamW8bit, gradient checkpointing(`use_reentrant=True`), `use_cache=False`다. eval/save는50 optimizer-step마다 수행하고 eval_loss 기준 best를 고른다. 로컬 wrapper는 dataloader worker0을 지정한다. 이것들이 모두 버그라는 뜻은 아니며, optimizer/attention/길이/검증·저장 중 어디가 지배적인지는 새 실측 전에는 모른다.

과거 `response-b1acc16-resume-628bc4536cfa/full/training.log`에는 batch1×acc16, 관측 프로세스 peak20,840,448,000bytes, `native_microbatch_mean_then_accumulation_mean`, `bitwise_equivalent=false`, `global_token_weighting_equivalent=false`가 기록돼 있다. **과거의 이 peak 하나로 새 긴 입력이나 batch2의 안전성을 보장하지 않는다.** 전체 원격 지연을 이 로컬 로그 하나로 설명하지도 않는다.

### 2.5 추론 환경 이관을 학습 이관 완료로 혼동하지 않는다

패키지 metadata만 읽었으며 모델을 로드하지 않았다.

| 환경 | torch | transformers | peft | accelerate | bitsandbytes |
| --- | --- | --- | --- | --- | --- |
| gemma4_env | 2.12.1+cu130 | 5.13.1 | 0.19.1 | 1.14.0 | 0.49.2 |
| soomac_llm | 2.12.1+cu130 | 5.14.1 | 0.21.0 | 1.14.0 | 0.50.2 |

두 환경 모두 학습 관련 패키지는 있지만 동일 환경은 아니다. soomac_llm 추론 smoke 통과를 역전파·optimizer·저장·재개 성공으로 대신하지 않는다. 새 학습 환경은 짧은 독립 시험 뒤 결정해야 한다. prefix KV cache는 이 학습의 가속 수단으로 넣지 않으며 FP8도 재도입하지 않는다. 별도 Claude의 추론 가속 작업은 건드리지 않는다.

## 3. 다음에 측정할 설정 — 최종 채택값 아님

배치 비교는 먼저 단일 GPU effective batch16을 비교 기준으로 둔다. [Hugging Face Trainer 설명](https://huggingface.co/docs/transformers/main_classes/trainer)에 따른 관계는 `microbatch × accumulation × GPU수`다. 이는 수량 관계이지 서로 다른 microbatch의 수치·loss 가중치·학습 결과까지 완전히 같다는 보증이 아니다.

| 장소·Agent | microbatch 후보 | 대응 accumulation | 이번 채택 여부 |
| --- | --- | --- | --- |
| 로컬 Response | 1 → 2 → 4 | 16 → 8 → 4 | 미측정·미채택 |
| 로컬 Recommendation | 1 → 2 → 4 → 8 | 16 → 8 → 4 → 2 | 미측정·미채택 |
| 원격 Decision / 약44GB | 1 → 2 → 4 → 8 | 16 → 8 → 4 → 2 | 원격 실제 GPU에서만 결정. 특정 batch가 맞는다고 가정하지 않음 |

OOM이면 더 큰 후보로 진행하지 않고 해당 장치의 탐색을 끝낸다. 후보 수를 늘리거나 같은 OOM을 자동 반복하지 않는다. 생존 후보에서 동일 기준 입력을 사용해 warmup 제외 반복 시간을 비교하고, sample/s·실제 처리token/s·총 프로세스/allocator peak·optimizer 비용·eval/save 시간을 기록한다. 한 번의 짧은 통과만으로 batch를 확정하지 않는다. 실제 긴 입력 stress도 통과해야 한다.

LR의 과거 기준은 D/Q1e-4·P5e-5,6epoch다. 속도를 위해 LR을 임의로 올리거나 epoch를 근거 없이 자르지 않는다. checkpointing을 끄거나 attention/정밀도/라이브러리를 바꾸는 것은 배치 비교와 분리해 측정한다. low/normal/high·알레르기·commit·EOS/정답 mask와 history 무절삭이 보존돼야 한다. 길이 grouping 도입도 sampler/동적 padding/loss 정규화를 설치 버전에서 먼저 확인하며 처음부터 여러 변경을 섞지 않는다. best eval_loss 하나를 최종 시연 합격으로 쓰지 않고 저장 epoch별 후보를 실제 의미 회귀로 판정한다.

## 4. 결론과 미완료

**설정·정적 코드 점검은 완료, 실제 배치·속도 프로파일은 미실행이다.** 로컬 GPU는 읽기 조회 기준 정상/여유 상태였다. 원격44GB는 사용자 지정 요청 예산이며 원격 실측값이 아니다. 현재 runner의 batch1/예산31GB 고정과 최신 데이터용 실행 결속 미완료를 확인했다. 새 학습용 runner·승인 영수증뿐 아니라 **manifest의 승인 상태·해시·검증 계약까지** 정합시켜야 한다는 진단이다. 원 후보의 불변 manifest를 몰래 true로 바꾸거나 기존 검사를 삭제해 통과시키지 않는다. 기존 사용자 승인을 새 동결 실행용 릴리즈에 명시적으로 연결하는 방식으로 작업해야 한다. 이번에는 해당 코드를 수정하거나 새 릴리즈를 발행하지 않았다.

다음 한 작업은 **최신 manifest와 로컬/원격별 설정을 받는 학습 실행 경로를 정리하고, 유한 batch 프로파일을 연결하는 것**이다. 그 뒤 짧은 GPU 시험을 거쳐 실제 파라미터를 채택한다. 본 학습·원격 실행은 이번 점검에서 시작하지 않았다.

독립 Codex sol/high는 핵심 진단에 **조건부 찬성**했다(`/tmp/soomac_training_parameter_check_review.out`). root가 원본 `train_r3.py:33-44,97-124`와 `budget.py:48-54`를 읽어 확인했고, 「영수증만 추가하면 됨」으로 읽히지 않도록 manifest 검증 계약을 명시했다. 또44GB는 요청 예산이며 다른 프로세스 점유를 자동 차감하지 않는 점을 보강했다. 조건 반영 뒤 reviewer가 다시 검수했다고 주장하지 않는다. CPU에서 요청/장치/서버 제한 최솟값·2GiB reserve 산술·batch×누적16 관계도 확인했으며 GPU 초기화는 없었다.
