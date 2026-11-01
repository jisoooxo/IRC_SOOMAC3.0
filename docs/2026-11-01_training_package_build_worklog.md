# Decision 원격 패키지와 로컬 순차 학습 — 실제 작업 기록

## 최신 완료 — 첫 SDPA 실인자/FP32 기준 출력 분리

사용자 요청대로 전체 모델 반복을 멈추고 기존 도구에 `first-sdpa` 명령을 추가했다. A/B 각각 첫 layer0 low-level SDPA 호출 직후 중단한 GPU 진단이 exit0으로 완료됐다. 실제 low-level Q/K/V·bool mask·is_causal·scale·GQA·dtype/stride와 CUDA kernel을 기록했다. **B도 mask가 있었고 양쪽 is_causal=False**다. valid QKV bitwise 일치, 전체 valid bias 불일치0, 동일 유효 key 크기의 FP32 CPU reference는 표본7query×16head에서 bitwise 일치했다. 실제 BF16 CUTLASS 출력 A/B 상대L2는0.0290693%다. 최초 수치 차이는 mask 의미 누락보다 물리 길이/stride가 다른 BF16 실행으로 좁혀졌지만, 내부 CUDA 연산 및 큰 gradient 증폭 기전까지 확정하지 않았다.

상세 실인자는 [진단 §6](2026-11-01_padding_failure_diagnosis.md)에 있고, 원본은 `/tmp/soomac-training-accel-20261101/first_sdpa_a1/FIRST_SDPA.json`이다. 첫 검수 지적대로 SDPA 가로채기를 우회할 때도 layer0을 벗어나지 못하는 stop guard를 추가했다. 실제 완료한 GPU 사례는 이미 모두 가로채 중단했으며 가드 보완 뒤 GPU 재시험은 하지 않았다. CPU18건 통과, GPU compute process0개, production runner hash 불변이다. full model 완료0회·backward0회·optimizer0회이며 고속 커널/속도 시험도 없다.

최종 Codex gpt-6-sol/high는 조건부 동의·새 중대한 결함 없음이다. root가 실인자/JSON/코드를 직접 대조했다. 원인은 첫 호출의 BF16 수치 차이까지 좁혔으나 내부 연산/큰 gradient 증폭은 미확정이며, 추가 guard는 CPU-only 검증이라는 조건을 보존했다. 검수 로그 `/tmp/soomac_first_sdpa_result_review.out`.

Cause: 동일한 유효 QKV와 mask로도 다른 물리 길이의 BF16 SDPA 출력에 미세한 수치 차이가 관측됐다.
Effect: 전체 gradient 차이를 무시해 채택하지 않고, 추가 실험/본 학습 없이 기존 검증 padding8·smartGC·batch1×acc16·native CE 설정을 유지한다. 원격 Decision·checkpoint·데이터는 그대로다.

## 최신 완료 — padding 실패 원인 분리 / Attention 후보는 보류

[진단 결과](2026-11-01_padding_failure_diagnosis.md)에 전체48층 gradient norm/cosine/relativeL2를 기록했다. 같은 모델의 초기 LoRA·buffers·CPU/CUDA RNG를 매번 복원한 A05 두 번, A0 두 번, B0의 짧은 SDPA forward/backward를 수행했다. 첫 계측의 FP32 CPU cosine>1 결함을 수정해 FP64 분할 합산으로 한 번만 재확인했고, 잘못된 원본도 보존했다.

채택한 A2 결과: A/A p=.05의 loss차이0·gradient L2 6.7709%, A/A p=0의 loss차이0·gradient L2 1.1788%. A/B p=0은 loss차이0.00532645·gradient norm133.17015→30.26049·relativeL2 111.3597%·cos−0.414697로 자체 변동보다 훨씬 컸다. actual dropout p=0을 진단에만 적용하고0.05로 복원했다. 초기 state/RNG hash·target31개 값/위치가 같으며0층 QKV도 bitwise 같다. SDPA 출력에서 첫 관측 차이가 있었으나 모든 최하위 SDPA runtime 인자까지 대조한 것은 아니므로 정확한 CUDA/mask 처리 원인은 미확정이다. backward 자체 변동도 기전을 분리하지 않았다.

Codex 최종 검수는 조건부 동의/ON HOLD이며 한계를 문서에 반영했다. **dropout만으로는 설명되지 않지만 Attention 가능성을 기각하지 않고 속도 시험을 보류**한다. dedicated kernel·optimizer update·production 변경은 없고, 데이터·정상 checkpoint·원격 Decision도 그대로다. CPU12건 및 FP64 성질3건(stdout) 확인·GPU반환 완료.

Cause: dropout0에서도 A/B 차이가 남고, A 자기 반복의 gradient 변동도 있어 단일 gradient cosine만으로 원인을 확정할 수 없다.
Effect: 원인 후보를 attention 계산 경로와 backward 변동으로 좁혔으며, 필드/target 정렬 오류나 최적화 불가능으로 단정하지 않는다.

## 최신 종료 — padding2개 제거 CPU PASS / GPU 정확성 실패

사용자가 승인한 마지막 padding-only 검사까지 완료했다. [최종 보고서](2026-11-01_attention_kernel_result.md)의 최상단을 우선 읽는다. 원인은 `pad_to_multiple_of=8`과 left padding으로, 실제8,430토큰 앞에2개를 추가하던 것이다. 별도 입력에서 multiple 설정만 제거했고 tokens/labels/유효 position_ids2~8431·valid-token mask를 보존했다. CUDA 초기화 없는 CPU에서 Global HD512·sliding1024의 실제 설치 predicate를 모두 통과했다.

동일 초기 모델의 padded SDPA ↔ unpadded SDPA GPU 진단에서 native loss0.8244394064↔0.8191129565(차이0.00532645), LoRA gradient cosine−0.411088 및 allclose 실패를 확인했다. dropout 난수는 정확성 진단에서만 모듈 training=False로 통제했고 원래p=0.05는 변경하지 않았다. 수치 차이의 근본 원인은 미확정이며 mask 의미 오류로 단정하지 않는다. 사전 기준 실패 뒤 fast router·optimizer timing을 실행하지 않았고 update0회/timing=null이다. 실패 영수증을 보존하고 허용오차 완화·GPU 재시험·추가 Attention 탐색은 하지 않았다.

Codex CLI 조건부 동의로 설계·판정·중단을 확인했고, 시험용 budget.check와 중간 상태 표시 지적은 코드만 국소 보완했다. CPU12건·git diff whitespace 검사 통과, GPU compute process0개 반환 확인. 원격 Decision·기존 정상 checkpoint·원본 데이터·production collator/학습 설정은 그대로다.

Cause: 인위적 padding 제거는 CPU mask 조건을 해결했지만 GPU loss/gradient의 사전 수치 동등성 기준을 통과하지 못했다.
Effect: 간단히 지울 수 있는 padding이라는 이유만으로 학습 정책에 적용하지 않고 최적화 탐색을 종료했다.

## 최신 완료 — Attention 전용 커널 시험 / 정상 경로 후보 중단

[결과 보고서](2026-11-01_attention_kernel_result.md)에 사용자 지정 표와3가지 결론을 기록했다. 기존 `attention_ablation.py`를 재사용한 A/B 실제 native training CUDA profiler에서 Global8층·sliding40층 모두 CUTLASS memory-efficient SDPA였다. A는 현재 compile, B는 공식 `UNSLOTH_COMPILE_DISABLE=1` fresh process였으나 전용 HD512/FA2/banded 커널이 활성화되지 않았다. Global 실제 mask 진단 hi 감소1회로 interval 조건 불합격, sliding은 순수 band와1,025행 불일치였다. sealed collator의8,430토큰은 왼쪽2개 padding을 포함해8,432가 된다. padding/mask/window/dtype/softcap/dropout 의미는 수정하지 않았다.

사용자 종료 기준에 따라 B가 실제 다른 커널로 dispatch하지 않는 시점에서 중단했다. **warm optimizer A/B·분리 FWD/BWD 시간·정확한 output/loss/LoRA-gradient 동등성은 미실행**이며 성공으로 세지 않는다. 기존276.35초/488tok/s/예약12.86GB는 이전 cold1회 update의 역사적 수치이다. profiler 시간/peak로 이를 덮어쓰지 않았다. 오래 걸리는 counter-only timing은 실제 CUDA 증명 없이는 실행되지 않도록 시험 도구의 게이트를 막았다. GC/배치 재시험·새 GPU 도구·FlashAttention 설치·새 CUDA 개발·production 변경은 하지 않았다.

CPU11건 및 합성 mask4조건 통과, 실제GPU 두 프로세스 종료·compute process0 확인. Codex CLI 최종 검수는 조건부 찬성으로 CUDA 증거·mask 조건·후보 중단에 동의했고, 수치 검증 미실행 표현을 정정했다. root가 결과와 소스를 직접 대조했다. 원격 Decision/기존 checkpoint/학습 데이터/기존 환경은 보존했다.

Cause: 설치된 고속 라우터가 현재 mask 계약을 지원하지 않아 정상 설정만으로 활성화하지 못했다.
Effect: 확인하지 않은 배수를 만들지 않고 추가 최적화 탐색을 종료했으며, 기존 검증된 Unsloth 기준 후보를 유지한다.

## 최신 완료 — GC 속도 선별 / 추가 가속 이득 없음

[GC 비교 결과](2026-11-01_gc_speed_trial.md): 기준 smart GC ON batch1은276.3486초/update, GC OFF 및 부분32/48·40/48은OOM, smart GC ON batch2는321.8207초, full native GC48/48 batch1은281.3407초였다. 모든 성공 후보의8430토큰×유효16·native eval CE 차이0을 확인했다. native 성공 후보의 실제 checkpoint48회/SDPA forward96회·backward48회, CPU buffer0/generation0을 별도 시간 밖 profiler와 상태로 대조했다. 새 시험 CPU4건·whitespace 검사 통과, GPU compute process0개로 반환을 확인했다.

**채택 판단:** cold1회 선별에서 더 빠른 설정은 발견하지 못했으므로 기존 검증된 smart GC ON batch1을 기준 후보로 유지한다. warm 순위·정확한 training gradient 동등성은 미검증이며, 본 학습을 시작하거나 기존 production/checkpoint/원격 작업을 변경하지 않았다. native는 offload뿐 아니라 checkpoint 구현도 바뀌므로 offload 단독 실험으로 세지 않는다. 이전 문단의 'CPU offload만 분리' 표현은 이 검수 정정이 우선한다.

Cause: GC OFF는 메모리를 초과했고, 완료한 추가 후보들도 기준보다 짧은 update 시간을 보이지 않았다.
Effect: 더 빠르다는 근거 없는 설정을 production에 적용하지 않고, 원래 기준 후보와 데이터/정상 checkpoint를 보존했다.

## 최신 실측 — GC OFF·부분 GC 실패 / batch2 완료 / offload 분리 중

현재 결과·보존 경계·원본 경로는 [GC 속도 비교](2026-11-01_gc_speed_trial.md)를 우선 읽는다. GC OFF는 microstep1 OOM, 부분 GC32/48은 microstep1 OOM, 부분 GC40/48은 **microstep4 OOM**이다. GC ON batch2×acc8은321.8207초/update로 완료했으나 기존 batch1의276.3486초보다 cold 비교에서 느렸다. CPU smart activation offload buffer200개/6,256,590,848bytes, 실제 generation8을 확인했다. 같은48층 GC와 batch1을 유지한 native GC/offload OFF 시험을 새 프로세스에서 측정 중이다. 아래의 이전 'skip8 실행 중' 문단은 당시 상태이며 현재 결과를 대신하지 않는다.

현재까지 production·정상 checkpoint·원본 데이터·기존 환경·원격 Decision을 변경하지 않았다. 새 시험 코드 CPU4건 통과, 이전 Codex 검수 지적의 decoder 층수 assertion과 별도 runtime profiler를 추가했다. 새 코드 검수의 첫 자동 승인 심사는 안전성 거절이 아닌 시간 초과였으며, 허용된1회 재시도로 Codex 읽기 전용 검수가 실제 시작됐다.

Cause: 부분 체크포인팅의 실패 지점과 실제 offload 통계를 구분해야 속도 비교를 잘못 해석하지 않는다.
Effect: 임의의 부분 설정 탐색을 중단하고, batch/CPU offload를 분리한 완료 update만 채택 근거로 삼는다.

## 최신 우선 — GC OFF 최우선 실제 시험 / 제한된 부분 체크포인팅 비교

사용자의 최신 목표는 VRAM 최소화가 아니라 실제 학습 완료시간 단축이다. 기존 `benchmark.py`의 GC ON·batch1 기준과 원래 production/checkpoint는 수정하지 않고, 새 `tools/training_acceleration/memory_speed_trial.py`에서만 loader/PEFT 인자를 프로세스 로컬로 바꾼다. 실제 Trainer 학습이 아닌 기존276초 기준과 같은 manual optimizer loop이며 `TrainingArguments.gradient_checkpointing=False`도 명시한다. 원래 `gemma4_env`에는 설치/변경하지 않는다.

1. **GC OFF / batch1×acc16: OOM.** 두 loader 호출에False를 전달했고,49개 checkpoint-capable module flag가 모두False, `model.is_gradient_checkpointing=False`, bound checkpoint 함수도0개였다. 같은8430토큰 native CE는0.79451739788로 기준과 차이0이었다. 첫 학습 forward에서 추가66MiB 할당에 실패했으며 optimizer update를 하지 않았다. peak allocated31,164,964,864bytes/reserved31,427,919,872bytes이다. `/tmp/soomac-training-accel-20261101/response_gc_off_batch1_a1/RESULT.json`에 실패를 보존한다. 12.9GB는 이전 예약 사용량이지 상한이 아니며 이 시험도32e9 process/allocator외512MiB 예산이었다.
2. **native partial GC / 32층 checkpoint·16층 skip / CPU activation offload 없음: OOM.** module flag로32개 decoder checkpoint와16개 skip을 확인했다. CPU activation buffer0, offload generation0, input embeddingCUDA, block swap 없음, plain `CheckpointFunction`이었다. 같은 native eval loss 차이는0. 첫 학습 forward에서248MiB 추가할당에 실패했고 optimizer update는 없다. peak allocated31,181,741,056bytes/reserved31,444,697,088bytes이다. `/response_gc_partial_skip16_batch1_a1/RESULT.json`을 보존한다.
3. 유한한 마지막 partial 후보는 native GC40층·skip8층이다. 여기도 OOM이면 partial 개수를 계속 탐색하지 않고, 별도 GC ON batch2×acc8 비교로 넘어간다. 데이터·loss·LoRA dropout0.05·유효배치16·history는 동일하게 유지한다. 성공한 후보에서만 실제 optimizer update 시간/tokens/s를 보고하고 warm 반복을 검토한다.

Cause: GC를 완전히 끈 실제 forward가 기존32GB 예산을 넘었으므로 필요한 일부 층만 재계산하는 타협을 제한적으로 확인한다.
Effect: 실패를 성공으로 기록하거나 전체 GC ON으로 몰래 복귀하지 않는다. 기준 ON1 후보를 보존하고 추가실험으로 production 재개를 무기한 미루지 않는다.

## 최신 우선 — 19:28 KST / native CE 선택·RSS 확인·Attention 준비 검수

첫 fused0 평가의 loss는0.79728025198, 같은 모델/입력의 materialized native1 CE는0.79451739788로 차이0.002762854가 정한 atol/rtol1e-3 기준을 넘었다. `/tmp/soomac-training-accel-20261101/response_unsloth_fused_loss2/FAILURE.json`을 실패 그대로 보존한다. CPU 소스 교차검수에서는 explicit shift_labels가 전달되고 재시프트되지 않아 명백한 한 토큰 정렬 오류는 보이지 않았으나, 이 수치 차이의 실제 원인은 미확정이다. fused FP32 GEMM/CE와 기존 BF16 logits 경로 차이는 가설이다. 허용오차를 완화하지 않고 production 후보는 `UNSLOTH_RETURN_LOGITS=1`로 정했다.

그 경로의 cold1 update는276.3486초,488.079nonpad tok/s, reserved peak12,855,541,760bytes로 exit0이다. baseline324.9987초보다 이 한 번의 시간은 약15% 짧지만 warm 측정·층별 경로·품질 합격은 아니다. 실제 원본 결과는 `/tmp/soomac-training-accel-20261101/response_unsloth_native1_smoke/RESULT.json`이다. native CE 재대조 `/response_unsloth_native_loss3/RESULT.json`은 차이0·PASS이며 model/data/processor/packages/builder fingerprint를 기록했다. 최고 프로세스 RSS는25,820,037,120bytes였다. 이는 호스트 RAM 전체 수치가 아니며, 로컬 단일 진단의 loading high-water이므로 원격16GiB cgroup에서 반드시 실패/성공한다고 단정할 수 없다. 원격 후보 이미지에서 cgroup·RSS를 실제 확인해야 한다.

Claude는 신규 `attention_ablation.py`·독립 테스트·계획을 작성 중이다. root가 CPU 테스트8개를 실행해 통과했다. 실행 전 코드 검수에서 기존 HD512 토글 캐시와 mask/compile 관측 교란 가능성 등 추가 점검이 필요하여 GPU 적용 검증으로 세지 않는다. 사용자 요구의 warm 동일조건 비교·실제 layer별 kernel·LoRA gradient 동등성은 여전히 미완료다. 기존 raw trace는 head256 SDPA forward80회와 head512 forward16회를 기록한다(40/8층의 forward+recompute와 양립). 정확한 layer index/실제 router 원인 확정은 별도 실험이다.

기존 환경·원격 Decision·checkpoint는 보존했다. 공식 Unsloth Docker base pull은 별도 CPU/network 작업으로 진행 중이며, 최종 TAR은 아직 없다. root GPU는 다음 Recommendation native1 폐기용 update를 단독 실행한다. production 본 학습은 아직 시작하지 않았다.

## 최신 우선 — 19:19 KST / Claude에 층별 attention 실험 준비 위임

사용자가 Gemma4 attention의 **Transformers SDPA / Unsloth SDPA / FlexAttention**을 local40층·global8층으로 나눠 확인하도록 Claude를 지정했다. 실제 `/usr/bin/claude -p`를 별도 실행했고 출력은 `/tmp/soomac_claude_attention_review_20261101.out`이다. 허용 도구는 코드 읽기·새 폐기용 파일 작성·공식 문서 확인이며 GPU 실행 권한은 주지 않았다. root 실험이 끝나기 전에 GPU를 겹쳐 쓰지 않는다. 소유 산출물은 `attention_ablation.py`와 그 독립 테스트·계획 문서이며 기존 production/runtime는 변경하지 않는다.

요구 검증: 같은 8k 입력/LoRA/유효배치16의 warmup 이후 처리량·VRAM, 층별 실제 CUDA/Triton 경로·마스크, attention output/loss/LoRA gradient 수치차이를 분리한다. 단순 `attn_implementation=flex_attention` 변경이나 라이브러리 설치만으로 합격하지 않는다. [Unsloth PR13869](https://github.com/unslothai/unsloth/pull/13869)의 head512 라우터는 조건부 SDPA 대체이며 소스 등록과 실제 활성화는 별개다.

실제 첫 Unsloth smoke는 exit0이다. 최장8430토큰×16의 cold update는312.3241초,431.859nonpad tok/s, allocator reserved peak12,834,570,240bytes이다. baseline324.9987초와 비교하면 단일 cold update의 약4% 개선이며, warm 속도·장기 ETA 개선을 확정할 수 없다. profiler에는 memory-efficient SDPA/CUTLASS와 주변 Triton fused 연산이 함께 보인다. **이 목록만으로 head512 전용 커널 또는 local40층 경로를 확정하지 않는다.** 원본 결과는 수정하지 않고 보존한다.

새 영속 격리 환경 `training/learning_first_v43/training_release_20261101_r5/env/bin/python`을 만들었다. gemma4_env는 읽기 전용 상속이며 신규 설치·변경하지 않았다. 임시 env와 전용21패키지 freeze SHA가 `99d298f8ae51474758b40d88d1f38fbf787ede539de8594fdf06b046fc862035`로 같고 pip check가 통과했다. 이 영속 환경에서 same-model native1 CE/fused0 loss 대조를 별도 실행 중이다. production 본 학습과 원격 작업은 그대로 보존한다.

Cause: Unsloth 자체 설치의 첫 실측은 메모리 절감이 크지만 기대한 학습 속도 개선을 아직 입증하지 못했다.
Effect: 실제 층별 attention 경로를 Claude가 조사하고, root는 loss·복구 검증과 별도 TAR 준비를 이어간다. 검증 전 production attention을 바꾸지 않는다.

## 최신 상태 — 2026-11-01 19:12 KST / Unsloth GPU 실험 실행 중

**사용자 요청: 기존 baseline은 반복하지 않고 Unsloth부터 실행한다. 별도 Unsloth Decision TAR를 제작하고, 기존 원격 Decision 작업은 중단·변경하지 않는다. 학습용 prefix sharing은 보류한다.**

1. 기존 로컬 Response→Recommendation launcher/trainer는 승인받아 SIGINT로 종료했다. 마지막 로그는 6step, 검증 가능한 복귀 지점은 checkpoint-5이다. full-state 7파일 SHA 검증 PASS, 데이터·원본 환경·정상 checkpoint·기존 TAR 삭제 없음.
2. 별도 `/tmp/soomac-training-accel-20261101/venv`에 Unsloth 2026.10.11을 설치했다. 기존 `gemma4_env`는 읽기 전용으로 참조하며 패키지를 변경하지 않았다. 이 환경은 임시 시험 환경이며 장기 학습 전 영속 경로와 패키지 잠금을 확정한다.
3. 기존 방식의 최장 Response 8,430토큰×16 시험은 exit 0이다. 실제 324.9987초/update, 415.017 non-padding tok/s, allocator peak reserved 23,899,144,192bytes이다. profiler는 memory-efficient SDPA/CUTLASS BF16 forward/backward를 확인했다. math fallback, sliding-window 타일 생략, residual dtype은 확정하지 않았다. 이전 약330초 측정과 함께 baseline으로 사용하고 추가 반복하지 않는다.
4. 실제 Unsloth 모델의 같은 입력에서 selected-logits/native CE 대조가 통과했다. model loss와 native CE는 모두 0.7945173978805542, 차이 0이다. `UNSLOTH_RETURN_LOGITS=1`인 평가 모드의 1행 검사일 뿐, 기본 fused loss·실제 역전파·저장/재개 전체 통과는 아니다. 이어서 기본 logits 설정으로 실제 누적16 역전파·optimizer update 및 별도 attention profiler 시험을 실행했다. 로그: `/tmp/soomac-training-accel-20261101/response_unsloth_smoke1.log`. 본 학습은 아직 시작하지 않았다.
5. 병렬 작업자는 두 명이다. `response_speed`는 공통 D/P/Q 학습 실행기·strict resume·진행 계측을, `recommendation_speed`는 공식 Unsloth 기반 별도 원격 이미지·TAR 패키지를 담당한다. GPU는 root만 순차 사용한다. 공식 베이스 지원과 CPU 테스트 통과를 실제 원격 학습 성공으로 보고하지 않는다.

### 현재 승인·미완료 경계

- 기존 환경과 최신 `compound_semantics_20261101_r2/dataset` 및 sealed build05를 보존한다. NF4 double quant/BF16, max_length D/P=9216·Q=5120, LoRA r16/alpha32/dropout0.05, 유효배치16, history와 label mask는 변경하지 않는다.
- 기존 checkpoint-5의 optimizer를 다른 프레임워크에 무검증 이식하지 않는다. Unsloth 본 학습은 새 출력 폴더에서 시작하며 native 저장→의도적 실패→재개→PEFT 재로딩을 검증한다.
- 실제 attention kernel과 프로세스 RSS를 기록한다. 현재 xformers는 Torch ABI 불일치 경고가 있으며 전체 Unsloth attention 경로를 경고 한 줄로 판정하지 않는다. 원격 RAM16GiB와 activation CPU offload 여부는 소스·RSS로 확인한다.
- 원격 전용 Unsloth TAR는 새 파일명으로 제작한다. 현재 원격 Decision 작업 중단·수정·업로드·새 GPU 실험 제출은 하지 않는다. 공식 이미지 기반 준비만으로 spec2.9 GPU 저장/실패/재개 시험을 통과했다고 하지 않는다.
- 진행 순서: Unsloth 실제 처리량/메모리/loss·gradient 확인 → 실행기 GPU 저장/재개 검증 → 로컬 P→Q 순차 본 학습 및 별도 Decision 이미지 검증·TAR 내보내기. prefix 실험은 제외한다.

Cause: 기존 baseline 증거가 확보됐으므로 반복 측정 대신 실제 Unsloth 호환성과 처리량을 확인해야 한다.
Effect: baseline 재실행 없이 Unsloth GPU 시험으로 넘어갔으며, 실제 결과가 나오기 전 가속 배수·새 ETA·TAR 완성을 주장하지 않는다.

관련 문서: [시험 계획](2026-11-01_training_acceleration_trial.md), [환경 검증](2026-11-01_unsloth_environment_check.md), [벤치마크](2026-11-01_training_acceleration_benchmark.md), [원격 호환성](2026-11-01_remote_unsloth_feasibility.md), [원격 패키지](2026-11-01_remote_unsloth_package_plan.md), [실제 상태](training_acceleration_20261101/STATE.json).

## 전환 판단 — 18:42 KST, 아직 중단·삭제·설치하지 않음

사용자가 가속 시험을 위해 초반 학습을 멈추는 편이 유리한지 **판단부터** 요청했다. 로컬 Response 실제5/624step(0.80%), 누적 학습 계산1281.49초(약21분), 표시 ETA46:28:28을 확인했다. native checkpoint5는 `attempt-6b570c7c5d8a49e8b9a64bbe79b439e6/response/checkpoint-5`다. `verify_checkpoint(..., step=5)`를 CPU 읽기 전용으로 실행하여 필수7개 full-state 파일의 실제 SHA 및 설정/step 영수증을 검증했고 exit0이다. 디렉토리 파일 합은428,565,546bytes(약409MiB)다. 실제 GPU 재로딩 시험을 한 것은 아니다.

판단안은 로컬 순차 launcher/자식까지 정상 중단하여 Q 자동 시작을 막고, **현재 출력·정상 checkpoint·기존 환경·최신 데이터는 보존**한 채 별도 환경에서 제한된 Unsloth/attention 비교를 진행하는 것이다. 실측20% 이상 처리량 개선과 동일 입력·라벨/정상 저장·재개·PEFT 로딩을 전환 조건 후보로 제안한다.20%는 운영상 제안 기준이지 논문의 보장 수치가 아니다. 준비·시험60~120분은 예산이며 설치·컴파일 실패 시 연장될 수 있어 무한 호환 수정 대신 중간 판단이 필요하다.

시간 예시: 남은46시간·다시 계산할 완료분0.5시간·시험2시간으로 가정하면, 같은 계산 처리량1.2배일 때 새 완료까지 약40시간45분으로 약5시간15분 이득이다.1.5배면33시간으로 약13시간 이득이다. 둘 다 실제 ETA가 아닌 전환 손익 산술이며 초기 ETA 자체도 흔들린다. native optimizer 상태를 다른 프레임워크에 그대로 재개할 수 있다고 가정하지 않는다.

원인: 초기 진행분에 비해 남은 계산량이 커 제한된 가속 시험 비용을 회수할 가능성이 있다.
결과: 로컬 시험 우선 전환은 합리적이나 아직 가속 효과는 미검증이다. 이번 판단 요청만으로 학습 중단·checkpoint 삭제·환경 설치·원격 Decision 변경은 하지 않았다. 디스크 checkpoint 삭제가 GPU 연산을 빠르게 만든다고 주장하지 않으며, 정리는 나중에 정확한 폐기 대상만 확정한다.

## 가속 대안 조사 — Unsloth / attention / Liger / Axolotl, 실행 변경 없음

별도 원격 GPU가 없다는 사용자 설명에 따라 가속 도구와 병렬 GPU 확보를 구분했다. Unsloth는 로컬·원격 GPU에서 사용하는 프레임워크이며 GPU를 추가 제공하는 것과 다르다. 같은 물리 GPU의 Decision·다른 학습 동시 실행으로 전체 완료가 빨라진다고 보장하지 않는다. 현재 학습·환경·TAR는 변경하지 않았고 새 GPU 시험도 하지 않았다.

- [Unsloth 공식 Gemma4 학습 안내](https://unsloth.ai/docs/models/gemma-4/train)는12B 지원 및 FA2 대비 약1.5배라는 자체 수치를 제시한다. 우리 SDPA·8k·rank16/dropout0.05의 실측이 아니며46시간→약31시간은 **동일 연산이 실제1.5배 빨라질 때의 조건부 산술**일 뿐 ETA가 아니다. 초기 컴파일·환경 전환·검증·재학습 비용도 포함해야 한다.
- 실제 로컬 config의 model_type은 `gemma4_unified`, text는 `gemma4_unified_text`다. head_dim256/global_head_dim512, KV-shared layer0, attention_k_eq_v=true를 확인했다. 일반 Gemma4 다른 크기의 지원이나 벤치마크를 그대로 적용하지 않는다.
- [PyTorch2.12 SDPA](https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.scaled_dot_product_attention.html)는 입력에 따라 fused/math를 선택한다. 실제 kernel 미측정이므로 FlashAttention 미사용·math fallback을 확정하지 않는다. [Axolotl Gemma4 안내](https://docs.axolotl.ai/docs/models/gemma4.html)는 global head512에 FA2/FA4 단독 적용의 제한과 hybrid 경로를 설명한다. **우리 실제 head512와 연관된 확인 후보**이지 현재 병목 확정이 아니다. [FlashAttention 공식 README](https://github.com/Dao-AILab/flash-attention)의 Blackwell 예시는 B200을 포함하며 RTX PRO4500의 정확한 SM/학습 backward 호환은 별도 확인한다.
- [Liger 공식 지원표](https://github.com/linkedin/Liger-Kernel)는 Gemma4 text/multimodal을 나열하지만 정확한 unified 클래스의 실제 patch 경로를 확인해야 한다. 현재 `common/loss.py`는 이미 정답 위치 logits만 계산하므로 일반 full-logits 벤치마크와 추가 절감 폭이 다를 수 있다. [Axolotl 지원표](https://docs.axolotl.ai/docs/support-matrix.html)는 unified VL을 포함한다. 다만 [fused LoRA kernels](https://docs.axolotl.ai/docs/lora_optims.html)의 문서상 dropout 불가 조건과 현재0.05는 충돌한다. dropout을0으로 바꾸는 것은 승인 없는 시스템 최적화가 아니라 학습 정책 변경이다.

추천 검증 순서는 현재 checkpoint 저장 검증→사용자 승인 후 학습 일시 중단→격리 환경 Unsloth 비교→미달일 때만 다른 대안1개이다. 같은 model/data/tokenizer/input_ids/labels/history/LoRA·유효 배치를 보존하고 warmup 뒤3~5 실제 누적 optimizer update의 처리량/peak와 저장·재개·기존 PEFT 로딩을 확인한다. 잡음이 크면 제한된 재측정만 한다. 다른 프레임워크로 optimizer 상태를 그대로 복원할 수 있다고 가정하지 않는다. 준비·비교 시험은60~120분 작업 예산 후보이며 설치/컴파일/호환 실패에 따라 연장될 수 있다. 각 도구의 홍보 배수는 곱하지 않는다.

원인: 지금은 로컬 P의 긴 순차 실행과 Q 대기 문제가 있고, 하드웨어를 늘리지 않는 개선은 실제 연산 경로를 바꿔 비교해야 한다.
결과: Unsloth는 우선 실험할 가치가 있으나 현재 데이터/품질/시간 개선을 검증한 것은 아니다. 이번 조사는 설치·GPU 점유·중단 승인을 대신하지 않는다.

## ETA 관측 — 18:30 KST, 초기 값이며 완료 보장 아님

로컬 Response는 실제3/624step이며 최신 LAB ETA39:45:05, 직전2step ETA45:02:15다. 초기40~45시간 수준의 임시 전망이며 검증/저장 비용과 배치별 길이로 달라질 수 있다. Recommendation은 여전히 pending/실측 전이므로 두 로컬 학습의 합산 ETA를 확정하지 않는다. 원격 Decision은 profile 이후 target1638이 잡혔으나 본 학습0step이다. ON1 profile186.3669초/update를 단순 곱하면84.80시간이며 이는 최장 입력 시험의 계산량 환산일 뿐 실제 본 학습 ETA가 아니다. 실제 속도 향상이 확인됐다고 보고하지 않는다. 학습·설정은 변경하지 않았다.

## 최신 관측 — 18:27 KST 원격 loading 표시 / 실제 배치 시험 진행

사용자가 build05를 업로드해 시작한 원격 작업 `16e600dfc6cc41cabaf777cbe0cf89d5`의 GET snapshot을 읽기 전용으로 대조했다. 작업·설정·TAR·데이터는 변경하지 않았다. 관측 원본은 `/tmp/soomac_build05_remote_snapshot_20261101_0925.json`, `/tmp/soomac_build05_remote_snapshot_current.json`, `/tmp/soomac_build05_remote_snapshot_latest.json`이다. `/tmp` 자료는 영구 보존 경로가 아니므로 핵심 수치는 아래에 남긴다.

| 실제 원격 후보 | 상태 | 유효 배치16의 갱신 시간 | PyTorch 최대 예약 메모리 |
| --- | --- | --- | --- |
| GC ON / batch1 / accumulation16 | PASS | 186.3669초 | 23,519,559,680bytes ≈21.90GiB |
| GC ON / batch2 / accumulation8 | PASS | 192.1128초 | 37,302,042,624bytes ≈34.74GiB |
| GC ON / batch4 | OOM, 해당 후보만 중단 | 완료 측정 없음 | 오류 순간 process44.50GiB / allocator 허용45.49GiB / 추가 요청1.97GiB |

배치2의 처리량은0.083284seq/s, 배치1은0.085852seq/s다. 이는 최장 입력을 반복한 제한된 한 표본이며 장기 속도·최종 ETA 보장이 아니다. 18:27 snapshot에서 `target_step=1638`로 전환하여 profile 이후 코드에 도달했지만, 아직 본 학습 첫 optimizer update를 확인하지 않았다. GC OFF 후보의 원격 상세 로그는 이 tail에서 확보하지 못했으며 성공/실패를 추측하지 않는다.

`Progress` 기본 phase는 loading이고 `measure_profile()`은 후보를 시험하는 동안 이 값을 변경하지 않는다. 모델 재로딩이6초에 끝난 뒤에도 후보의 forward/backward·누적 갱신이 loading으로 보인다. heartbeat 갱신 자체는 trainer 진척 증거가 아니지만, 이번에는 배치2 PASS→배치4 OOM→target1638이라는 새 결과로 실제 진행을 확인했다. 전역 GPU100%·VRAM 표시를 작업 전용 사용량으로 해석하지 않는다. 후보별 PyTorch peak와 전체 장치 순간 사용량도 서로 다른 지표다.

원인: 프로파일 단계의 상세 진행 표시가 없어 장시간 연산도 단순 loading으로 출력된다.
결과: 사용자 화면만으로 멈춤을 판정하지 않고 PROFILE_RESULT와 실제 본 학습 step으로 판단한다. 이번 조사에서 실행 중 코드/이미지는 수정하지 않았다.

읽기 전용 Codex CLI `gpt-6-sol/high` 교차검수는 조건부 찬성이다(`/tmp/soomac_remote_loading_review.out`). 지적대로 heartbeat·전역 GPU만으로 개별 작업 진척을 단정하지 않았다. 검수자가 읽은 snapshot은 ON1까지였고, root가 이후 새 snapshot의 ON2 PASS·ON4 OOM·target1638을 직접 대조했다. 인용 코드의 후보 순회·phase 미변경·NF4/BF16·LoRA 설정도 직접 확인했다. 추가 snapshot 18:28 KST에서도 본 학습 첫 optimizer update는 아직 미확인이다.

NF4+LoRA는 공식 PEFT의 QLoRA 경로이며 가중치 저장4bit와 BF16 계산을 구분한다. 공개 예시와의 비교:

| 출처 / 대상 | 공개 설정 | 우리 설정과 다른 점 |
| --- | --- | --- |
| [HF bitsandbytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes), Llama13B / T4 16GB | sequence1024 / microbatch1 / accumulation4 / nested4bit | 우리 D 입력은 대체로 약8천 토큰, 유효 배치16이다. 공개 사례는 메모리 적합 예시이며 속도 기준이 아니다 |
| [HF PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization), Mistral7B 예시 | NF4 / double quant / BF16 compute / rank16 / alpha8 / dropout0.05 / attention4투영 | 우리 rank16 / alpha32 / dropout0.05 / attention4+MLP3투영이다. 공개 예시가 우리 최적 설정이라는 뜻은 아니다 |

가중치 양자화는 활성값·gradient·LoRA·optimizer·임시 버퍼 전체를4bit로 만들지 않는다. 23GB 관측만으로 설정 오류라고 할 수 없고, 반대로 효율이 최적이라고 확정할 수도 없다. 최장 입력 처리 및 실제 kernel/연산 비중 측정 없이 NF4 자체·RAM 상한을 속도 병목으로 확정하지 않는다.

로컬 P는 별도 profile에서 OFF1 OOM / ON1 PASS / ON2 OOM으로 ON1·accumulation16을 골랐으며 실제2/624step까지 확인했다. 아직 첫 full-state 저장5step은 미확인이고 Q는 P 완료 뒤 순차 대기다. 로컬 ON1의 최장 입력 갱신330.0587초는 profile 값, 실제 초기 두 step은 약250~260초/update로 구분한다. allocator `expandable_segments` 제안은 좁은 CLI 검수에서 조건부/현재 학습 중단 비권고였으며 적용하지 않았다. 현재 Goal 도구 표시는 paused로 관측되었고 이번 읽기 전용 진단에서 상태를 임의 변경하지 않았다.

## 최신 완료 — build05 학습용 TAR / 로컬 순차 본 학습 진입

**2026-11-01 18:05 KST: [Decision build05 TAR](2026-11-01_decision_build05_speed.md) 완성 및 실제 Docker load exit0.** 파일은 `/home/roma/Downloads/soomac-decision-r5-compound-semantics-20261101-build05-speed.tar`,6,998,482,944bytes, SHA `493acef5df21cf613eeee8be68da8535244980a062a31d961397fccf1d5b1298`다. 실제57layer/OCI/config/전체 SHA 검증과 load 뒤 image ID 일치를 확인했다. UID1000:1000/CMD train/spec2.9는 그대로다. 사용자 서버 업로드·원격 본 학습은 이 작업자가 시작하지 않았다. 기존 TAR는 보존했다.

실제 GPU 저장/실패/복원은 최종 exit0이고 원래 체크포인트 불변·네 native state 일치를 확인했다. 독립 CLI 결과는 TAR 내보내기에 조건부 찬성이다. 원격 속도/장기 완주를 뜻하지 않는다. 검수 모델의 자기 식별 문구는 실행 헤더 `gpt-6-sol/high`와 달라 근거로 쓰지 않았다.

로컬 실행기·두 정책을 `runs/r5_package_test_20261101_build05_speed/local_operator/`에 읽기 전용으로 보존했다. 실행기 SHA는 `40c97f22ba7286401efcc31c233d1df1057ae9806fc89a0ddd2505b6b69735ac`이며 P/Q JSON SHA는 아래 최신 기록과 같다. 봉인 release나 원래 source를 수정한 것이 아니다. TAR 검증 뒤 새 로컬 출력 `runs/r5_local_speed_20261101_build05`에 Response→Recommendation 순차 실행을 요청했다. 실제 optimizer 진행·첫 저장은 다음 관측으로 확인하며, 시작 명령을 보낸 사실만으로 합격 처리하지 않는다.

18:07 KST 실제 launcher PID914967/trainer PID914979와 live exec session49154를 확인했다. `pipeline_state.json`은 running/P started/Q pending이며 실제 child argv는 `--budget-bytes 32000000000 --non-allocator-reserve-bytes 536870912 --gradient-checkpointing auto --batch auto --resume auto`다. CPU preflight는 최신 P train1664/validation152·정확한 release/manifest SHA로 통과했고 현재 full-input profile 준비다. 아직 실제 optimizer step이나 checkpoint 저장을 확인하지 않았다.

**state 읽기 주의:** `identity` 최상위의31e9/2GiB/on은 settings-dir 없이 쓰는 과거 전역 옵션 기본값을 호환상 기록한 필드다. 이번 실제 실행 정책은 `identity.agent_settings.response.policy` 및 `recommendation.policy`와 실제 child argv다. 최상위 옛 기본값만 보고 설정이 반영되지 않았다고 판단하거나 실행 중인 학습을 다시 시작하지 않는다.

원인: 원격은 사용자 업로드가 필요하고, 로컬 두 학습은 GPU를 동시에 쓰면 OOM 위험이 있다.
결과: 원격용 파일은 제공 완료이며 로컬은 독립 profile·P 완료 검증 후 Q의 순서를 유지한다. Q pending/미측정이면 전체 Goal은 active로 유지한다.

## 최신 실제 진행 — P/Q 후보 연결 완료 / Decision step5 저장·의도적 실패 통과

완료 조건을 원래 Goal 문장과 다시 대조했다. P/Q 각각의 설정 실측·실제 optimizer 진행·첫 full-state 저장이 필요하다. 따라서 이후 P만 시작하고 Q가 pending이면 자동 순차 실행 준비는 성과로 보고하되 전체 Goal을 complete로 처리하지 않는다. Q 시작은 기존대로 P 완료 검증 후이며, 사용자에게 임의로 동시 학습을 권하지 않는다.

사용자 지시대로 Recommendation·Response를 별도 Codex sol/high 에이전트가 조정했고, 통합 작업자가 실제 argv·순차·설정 drift 차단을 연결했다. 두 로컬 정책의 최신 요청은32,000,000,000bytes/allocator 외512MiB/GC auto다.1GiB→512MiB는 실제 D OOM에 대한 좁은 CLI 조건부 검수를 거친 **로컬 시험 후보**이며 GPU 통과·속도 향상을 뜻하지 않는다. 원격 봉인 이미지의1GiB는 변경하지 않았다. 새 JSON SHA는 P `86ce9152b1a866ef511bb05100a97ea226667f75a28618c7ded74aa26afb3de3`, Q `3b096a82945b390941ed45321acd34ffa463ceba0eec69c4a94d57e5c02baa5f`다. 최종 관련 CPU42건이0.236초에 통과했다. 이 테스트의 train/complete 로그는 mock이며 실제 본 학습이 아니다. 이전 전체 P 인코딩 결과는 재검사하지 않고 보존했다.

진행 중 Decision 실제 GPU 시험은5step full-state checkpoint를 게시하고 정확한 `Injected resume test failure after global step 5`로 exit1했다. 이는 의도된 시험 성공이며 새 학습 오류가 아니다. marker는 `gpu_verification/smoke/.lab-monitor/resume.json`, 실제 checkpoint는 `attempt-2c10679513cb449da411fc5eb26012bc/decision/checkpoint-5`다. 이어서 같은 설정으로5→10step native 재개 시험을 진행 중이다. **최종 복원·GPU 결과 교차검수·TAR 내보내기·로컬 본 학습은 아직 미완료**다. 기존 출력·봉인 release·TAR는 변경/삭제하지 않았다.

원인: 설정만 쓰고 실제 인자·첫 저장을 확인하지 않으면 실행과 문서가 어긋날 수 있다.
결과: P/Q 설정은 실제 인자 검사까지 끝났고, Decision 복구 시험은 검증된5step에서 실제 재개 중이다. Goal은 active로 유지하며 완료로 표시하지 않는다.

## 최신 우선 — build05 이미지 완성 / 실제 GPU 검증 진행 / P·Q 에이전트 분담

속도 코드 CPU 검증은 common34·local29·remote8·root37건이 통과했다. 독립 Codex CLI 재검수는 확인된 blocker 없음/실제 GPU 시험 필요 판정이었다. 마지막으로 최장 정답 범위 stress를 모든 배치에 적용하고 해당 CPU 회귀도 통과시켰다. 새 봉인 release `conversation_r5_20261101_build05_speed` SHA `ff3285e46bff4f56c18327adb3b038a1010ab5e7482ca59729ab86c1f3a9459e`, 실제 image `soomac/decision-r5:20261101-build05-speed` ID `sha256:422b9b440f92e04ae441e56418bac6aa5b2c769bdc658fb1cf4c9c0596ea9de4`다. 아직 TAR 내보내기·새 로컬 본 학습 시작을 완료하지 않았다.

실제 GPU 검증은 `runs/r5_package_test_20261101_build05_speed/gpu_verification`에 별도 기록 중이다. 로컬32,000,000,000bytes 예산에서 OFF/ON·배치 유한 비교 후 저장5→실패→재개10→실패→최종 복원을 수행한다. 원격46.8GiB에서의 속도 실측으로 혼동하지 않는다. 기존 공식 외부 template/tokenizer_config overlay도 그대로 사용하여 이전 drift 회귀를 포함한다.

사용자 추가 요청으로 Codex sol/high 작업자 둘에게 각각 `recommendation_speed.json`·`response_speed.json`과 독립 CPU 검사/근거 문서를 맡겼다. Recommendation4건 CPU 통과·CLI 조건부 찬성이 도착했으며, 실제 GPU 측정은 하지 않았다. 세 번째 통합 작업자는 `--agent-settings-dir` 연결·실제 argv·resume identity 검사를 담당한다. 설정 파일이 존재한다는 사실만으로 실제 학습에 반영됐다고 하지 않는다. 이 로컬 외부 launcher 통합 때문에 이미 봉인한 원격 release를 수정하거나 기존 GPU 시험을 중복 실행하지 않는다.

원인: 모델별 설정을 문서/JSON에만 적으면 실행기는 기존 기본값으로 돌 수 있다.
결과: 실제 인자와 identity까지 연결해 검증하며, 원격 TAR 완성 전 P/Q 본 학습은 시작하지 않는다.

P 설정 작업자도 완료했다. 최신 Response train1,664/validation152행을 실제 processor로 CPU 인코딩했고 두 테스트가75.900초에 통과했다. 두 작업자 CLI 판정 모두 GPU 실측/실제 argv 연결 전 조건부 찬성이며, 새 속도나 품질 합격을 주장하지 않는다. Recommendation은 CPU4건 통과다. 통합 작업자는 별도로 실행 인자와 strict identity를 검수한다.

진행 중 GPU profile의 최초 관측은 **로컬32e9 예산 Decision / checkpointing OFF / microbatch1 OOM**이다. 같은 조합을 반복하지 않고 ON 모드로 넘어갔다. 이것은 P/Q나 원격46.8GiB에서 OFF가 불가능하다는 증거가 아니다. profile 전체·저장/복원 시험이 끝난 뒤 실제 로그를 독립 검수하고 최종 결과로 기록한다.

P/Q 외부 launcher 통합과 최종 관련 CPU41건 검사가 완료됐다. CLI 지적 중 검증기 pin import 전 검사·bounded JSON read를 국소 보완했고, 의도된 외부 launcher와 봉인본의 해시 차이는 원격 release 재봉인 사유가 아니므로 문서에 분리했다. 최종 delta 검수는 새 설정 경로의 확인된 회귀 없음이다. 원격 이미지/봉인 release는 그대로다. 실제 `--agent-settings-dir` CPU preflight를 새 로컬 출력 `runs/r5_local_speed_20261101_build05/.preflight`에서 실행했고 Response·Recommendation 모두 exit0, GPU 미로드/본 학습 미시작이다.

Decision 폐기 profile은 exit0으로 끝났다. 로컬32e9/1GiB reserve에서는 OFF1 OOM, ON1 PASS, ON2 OOM으로 유한 종료했고 선택은 ON1이다. ON1 최장 입력16개 누적 update 표본은325.0245초이며 이는 로컬 Decision 최장 입력의 단일 표본이지 P/Q·원격·평균 본학습 ETA가 아니다. 이후 실제 저장/실패/복원을 계속 진행한다. 아직 TAR ready는 아니다.

## 최신 우선 — 속도 우선 Goal / build05 TAR 먼저

사용자 요청으로17:06 KST 기존 로컬 launcher764128/trainer764137을 정확한 argv·부모/자식·SIGTERM 처리 확인 후 중단했다.12/624step이며 resume marker는 없었다. GPU compute 목록이 비었고 메모리가24GB→약1GB로 반환됐다. P/Q 자동 연속 실행은 취소 상태로 기록했고 로그·출력은 삭제하지 않았다. 원격 중단은 사용자가 담당한다.

새 Goal은 [속도 우선 계획](2026-11-01_speed_first_training_plan.md)의 **Decision TAR 제작/검증 먼저 → 로컬 P→Q 본 학습**이다. 새 source 후보는 checkpointing 양쪽 경로와 profiler를 같은 switch에 연결하고, OFF/ON 유한 배치 탐색·서버 VRAM 예산 반영·reserve1GiB 후보·strict resume identity·5step 첫 저장을 추가했다. build04 bundle/TAR와 data/model/prompt/history/6epoch/LR는 수정하지 않았다.

첫 Codex sol/high 검수의 optimizer 빈도·검사 비용 지적을 실제 코드로 확인해, timed reference를 유효 배치16 누적 update로 바꾸고 finite/nonzero 검사를 계측 밖으로 분리했다. 반면 budget CPU 테스트 산술 오류 지적은 reviewer가 total과 serverbytes 인자를 바꿔 읽은 것이었다. 원 함수 시그니처와 실제 통과한 테스트로 제외했으며 정상 테스트를 틀린 기대값으로 바꾸지 않았다. 교정 코드 재검수는 진행 중이다. 새 GPU 속도/복원·image/TAR·로컬 본 학습은 아직 미완료다.

## 최신 우선 — 2026-11-01 16:59 KST 속도 진단

사용자가 올린 build04 원격 작업 `8b07d54d930b4a5992de1b9075bd1f87`을 GET API로 실제 확인했다. RTX PRO 6000 Blackwell Max-Q에서 running,8/1638step이며6→7→8step은 약176초/step이었다. 실제 TAR 상위 layer의 학습/예산/프로파일/실행 파일을 읽어 source·봉인 bundle과 byte 동일함을 확인했다. 서버 예산46.80GiB에 비해 이미지의 자체 process 한도40.98GiB, allocator 한도38.98GiB가 적용되는 정책도 CPU 산술로 확인했다. 이는 이전44GB 승인 보존이며 실제 배치 탈락 원인으로 아직 확정하지 않았다.

화면 RAM7.9GiB와 GPU 사용량은 작업별이 아니라 host/device 전체 집계임을 frontend로 확인·정정했다. RAM16GiB 부족, attention kernel, 현재 microbatch는 아직 미확정이다. 초기10step gradient 검사 비용과 검사 제거 후 속도도 분리해야 한다. Codex CLI sol/high는 조건부 판정이고 root가 근거 코드를 대조했다. 전체 초기 로그는 웹 열람 제한, 연결 browser 부재로 확보하지 못했으며 우회하지 않았다. **학습/설정/TAR/data/prompt/adapter/ROS는 변경하지 않았다.** 상세 증거와 미완료는 [속도 점검](2026-11-01_training_speed_tar_audit.md)에 기록했다. 아래 로컬 초기 준비·원격 미확인 문구는 작성 당시 이력이다.

## 최종 우선 결과 — build04 완료 / 로컬 새 순차 실행

사용자 승인대로 기존 build03의52개 레이어를 그대로 유지하고 수정 레이어를 추가한 build04를 완성했다. 파일 `/home/roma/Downloads/soomac-decision-r5-compound-semantics-20261101-build04.tar`, 6,998,478,336bytes, SHA `def26c665ea75dc0342126950607eb5cca81a7e217f2b44a27db192942d69bac`다. 실제 공식 HF template/tokenizer_config overlay로 원본 오류를 재현하고, 수정 이미지 preflight/실제 GPU profile·save5/fail→resume10/fail→최종 native restore를 통과했다. CPU90건, 원래 체크포인트 불변, 네 native state digest 일치, 최종57layer/OCI/TAR SHA, 실제 Docker load까지 exit0이다. 기존 build03 TAR는 보존했다. 원격 서버 실행·새 모델 품질은 아직 미확인이다.

실제 GPU 결과만 Codex CLI로 마지막 교차검수한 판정도 범위 한정 찬성이다. 완료 receipt는 `runs/r5_package_test_20261101_build04/DECISION_READY.json`(16:09:50 KST)이다. [현재 결과·전 과정](2026-11-01_decision_build04_processor_fix.md)을 따른다. 아래 build03 사용 안내·미완료·NVFP4 필수대기는 모두 이전 시점 이력이다.

Response 두 독립 검수자의 총1,992행 검수에서 확증 새 FIX/HOLD0이며, 애매한 표현을 기계적으로 삭제하지 않았다. 최신 원본 r2 데이터와 model pins는 그대로다. P/Q CPU preflight 통과 뒤 사용자 변경 승인에 따라 NVFP4 영수증 대신 명시적 승인 플래그로 로컬 Response→Recommendation 순차 launcher를 시작했다. 외부 source launcher28개 모의검사·Codex 조건부 찬성·root 실제 코드 대조를 완료했으며, GPU lock·free/compute·native checkpoint·P 완료 검증 후 Q의 안전 경계는 유지한다.

로컬 출력은 `/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/runs/r5_local_response_reaudited_20261101_build04`, session86950/PID764128, 실제 상태 running/P started/Q pending이다. 현재 입력/배치 준비이며 optimizer step1 성공은 아직 확인하지 않았다. 옛 NVFP4 waiter PID679304는 정확한 명령·자식0을 확인하고 종료했으며 원래 출력은 보존했다. 새 영수증을 거짓 NVFP4 종료 신호로 만들지 않았다. 원격 업로드/실행·운영 adapter 교체·ROS·VLM·가속 코드·commit/push는 하지 않았다.

Cause: 원격 기본 캐시와 인코딩 원본의 메타데이터를 같은 것으로 가정한 포장 계약이 실패 원인이었다.

Effect: 원래 인코더를 내장하고 외부 가중치 검증을 유지한 수정 TAR를 제공하며, 재검수한 최신 데이터의 로컬 두 학습은 GPU를 공유하지 않고 순차 진행할 수 있다.

## 최신 우선 — build03 원격 실패 / NVFP4 대기 해제

**후속 승인·실행:** 사용자가 기존 TAR 기반 수정을 승인했다. 기존 build03 이미지를 FROM으로 삼은 build04를 제작하고 있으며 [수정 기록](2026-11-01_decision_build04_processor_fix.md)에 릴리즈/CPU/실제 오류 재현을 남겼다. 데이터 byte와 모델 pin은 동일하다. 최종 CPU88건+하위44건 통과, r2 image build exit0이다. 공식 template/config readonly overlay로 원본 오류를 재현하고 수정 이미지 실제 GPU 저장/복원을 시험 중이다. 아직 최종 TAR/GPU PASS는 아니다. Response 독립 검수는 두 담당996행씩 총1,992행을 마쳤고 확증 새 FIX/HOLD0이다. 선택적 명료화는 삭제 근거로 쓰지 않았으며 실제 모델 시나리오 합격으로 확대하지 않는다.

사용자 원격 실행이 `base model drift: chat_template.jinja`로 학습 전에 중단됐다. 공식 기본 commit707f0a3b…를 실제 읽어 비교한 결과 template과 tokenizer_config가 로컬 pin과 다르고, config/processor_config/tokenizer.json은 같으며 weight LFS SHA도 일치했다. 실제 원격 캐시 전체 SHA는 미제공이라 같은 공식 파일이 선택됐다고 단정하지 않는다. [조사·수정 후보](2026-11-01_build03_model_metadata_failure.md)에 실제 해시와 경로 덮어쓰기 위험을 기록했다. 로컬 GPU 시험을 원격 입력 일치까지 확대했던 검수 누락이다. 이전의 ‘공식 기본 캐시를 사용하면 된다’는 안내를 철회한다. build03/데이터/영수증은 보존하며 새 TAR는 아직 제작하지 않았다.

Codex CLI 읽기 전용 교차검수 exit0(`/tmp/soomac_build03_template_review.out`), 별도 TAR 검수 에이전트도 시작했다. Response 두 에이전트는 같은 최신 r2의 정렬된 실제 행을 중복 없이 전반/후반으로 나눠 의미 검수 중이다. 근거 있는 FIX/HOLD만 root가 대조하며 기존 해결 건을 새 결함/성과로 세지 않는다.

사용자 최신 지시는 **NVFP4 대기 철회 → Response 추가 의미검수/국소 교정·제외 → Response→Recommendation 순차 학습**이다. 기존 NVFP4 조건을 충족했다고 가장하지 않으며 실제 GPU guard·동시 학습 금지·원본 보존은 유지한다. 이 변경 이후 본 학습은 아직 시작하지 않았다. 아래 build03 완료/대기 조건은 당시의 역사적 기록이다.

Cause: 선언한 외부 모델 버전과 실제 인코딩에 쓴 로컬 파일의 호환성 비교가 제작 검수에서 빠졌다.

Effect: 단순 재시도·해시 무시 대신 원래 인코딩을 유지하는 입력 복구안을 독립 검수하고, 로컬 데이터 검수는 병렬로 계속한다.

## 최신 완료 결과 — build03 / 2026-11-01 14:34 KST

사용자 제작 승인에 따른 수정 TAR는 `/home/roma/Downloads/soomac-decision-r5-compound-semantics-20261101-build03.tar`,6,991,625,216bytes, SHA256 `4ded2e17fd4064e1c84e003ae76c9320fc522ed352ddd0bb15a69268df929e05`이다. finalizer17984/Docker load45071 모두 exit0이다. 실제 assets22파일/UID1000 preflight/CPU81건/새 GPU profile+save5→fail→native resume10→fail→final native restore/52layer/TAR identity 검증을 통과했다. 실제 서버 업로드/접수·원격 학습·시나리오 추론은 하지 않았다. build02의 경로 위반과 잘못된 준비 완료 표현은 철회한 상태로 보존한다.

build03의 새 Decision receipt는14:33:29 KST에 작성됐다. 새 local waiting session12238/PID679304도 실제 running·waiting으로 확인했으며 P/Q pending, 자식 프로세스0, GPU compute0이다. 옛 waiting13153/PID582218은 exit143으로 종료했다. 새 gate는 build03 아래 `LOCAL_READY_AFTER_NVFP4_RETEST.json`이며 미존재다. 옛 Claude OOM marker로 신호를 만들지 않았다. 약20:34까지6시간 대기하며 본 학습은 아직0회다. Goal의 외부 NVFP4 의존성은 완료로 표시하지 않았다. [현재 인계 문서](2026-11-01_nvfp4_to_training_handoff.md)는 build03 기준으로 정리했으며 이전 고유 관측은 이 worklog 이력에 보존했다.

다음 사용자 작업은 기존 실패 업로드를 취소하고 새 build03 파일로 업로드하는 것이다. 서버가 확인한 내장 데이터(`/app/assets/dataset`)를 선택하고, 기본모델은 선언된 Google707f0a3b… 캐시를 사용한다. root가 대신 원격 전송/학습을 시작하거나 기존 전송을 취소하지 않았다. 아래 제작 중/blocked/미완료 표시는 해당 시점 이력이며 이 최신 절을 우선한다.

최종 git status에는 다른 작업자의 Decision/Recommendation/Response model·vllm_client/input_budget·관련 테스트 변경도 추가로 보였다. 이 TAR 교정에서 만든 변경이 아니며 수정/되돌리기/stage하지 않았다. 이 작업의 GPU 시험도 끝났고 새 대기 프로세스는 모델을 점유하지 않는다. build03 판정은 봉인된 학습 패키지와 해당 실제 시험에 대한 것이며 병렬 Runtime 변경이나 서버/모델 품질 전체의 합격으로 확대하지 않는다.

## 사용자 제작 승인 후 build03 교정 진행

사용자 `제작 ㄱㄱ` 승인으로 새 build03를 제작 중이다. [별도 실제 수정 기록](2026-11-01_decision_build03_assets_fix.md)을 우선한다. 새 RELEASE SHA는 `e213d6188fc71739051c4d04f8acc419bc5727fc326d12c0f46c2707aeb4284c`, 새 image ID는 `b312009fef020df9390e44820457b2859180bc448dbdf5a4a98ebb42ff402647`이다. 실제 assets22파일 hash/일반파일/링크 검사, UID1000 entrypoint preflight, CPU81건을 통과했다. 새 GPU 시험은 session97525/별도build03 output에 진행 중이며 새 최종 TAR는 아직 게시하지 않았다. 임시 내보내기는 session14813에서 병렬 진행한다. 기존 waiting launcher는 실제 exit143으로 종료해 옛 TAR 영수증으로 시작하지 않게 했고, 다른 사용자의 작업은 종료하지 않았다. build02를 덮어쓰거나 삭제하지 않는다.

## 최우선 정정 — 서버 업로드 규격 위반 발견 / build02 사용 보류

사용자 업로드 화면에서 `dataset: 내장 입력은 /app/assets/ 아래에 배치하세요`가 확인됐다. 실제 spec v2.9의287/291/479행은 내장 입력을 `/app/assets/` 아래로 제한하지만 build02의 remote/lab-inputs.json은 `/app/dataset`을 선언했다. Dockerfile과 기본 DATASET_DIR도 이 잘못된 경로를 사용한다. **root의 패키징·규격 검수 누락이며, 앞선 ‘바로 올리면 된다’ 및 서버 사용 준비 완료 주장을 정정한다.** CPU 검사는 잘못된 경로를 정상으로 assert했고 finalizer는 이 규격 제한을 검사하지 않았다. 실제 Docker load/해시/optimizer 복원 통과는 사실이나 **서버 업로드 규격 통과를 의미하지 않는다.**

기존 TAR·영수증은 삭제/덮어쓰기하지 않으며 역사적 검사 증거로만 보존한다. 이번 사용자 메시지는 오류 진단이므로 현재 코드/이미지 재빌드는 아직 적용하지 않았다. 최소 수정 범위는 내장 실제 데이터 위치·입력 라벨·기본 DATASET_DIR·해당 규격 검사/테스트의 일치이며 학습 데이터 내용·모델·학습 의미론을 바꿀 문제가 아니다. 새 수정 TAR의 서버 준비 상태는 실제 새 검증 전에는 완료로 기록하지 않는다. 새 gate 생성/본 학습/원격 조작도 하지 않았다. 아래 완료 표시는 이 결함을 발견하기 전 기록이다.

Codex sol/high 읽기 전용 교차검수도 규격 위반에 찬성했다(`/tmp/soomac_r5_embedded_path_review.out`). 지적된 spec479/493, remote label/Dockerfile/entrypoint/test 기대값을 root가 직접 대조했다. reviewer는 TAR 실물을 읽지 않았으므로 root가 이어서 **실제 Downloads TAR의 manifest가 가리키는 config blob**을 읽었고 `dataset.embedded.path=/app/dataset`, `DATASET_DIR=/app/dataset`을 확인했다. 따라서 소스만의 가설이 아니라 제작된 실제 TAR에서도 같은 오류다. 업로드된 원격 바이트 SHA는 미조회이며, 서버 재업로드/변경은 하지 않았다.

**최신 Goal 상태(13:21 KST): `blocked` — 도구 반환으로 확인했다.** 미완료 외부 조건은 새 Claude NVFP4 재시험 종료 확인이다. Decision 제작/검증과 대기 launcher 구현은 완료돼 있으며 본 학습은 미시작이다. 기존 live 대기 프로세스는 중단하지 않았다. 아래13:12의 active 표시는 당시 이력이다.

## 최신 실제 상태 — 2026-11-01 13:12 KST

**Decision production TAR 완성, 실제 Docker load 통과, 로컬은 새 NVFP4 재시험 완료 신호 대기다.**

최종 파일 `/home/roma/Downloads/soomac-decision-r5-compound-semantics-20261101-build02.tar`, 크기6,991,230,976bytes, SHA256 `ab897fca674fdc9084e4a14bb53f9db7d49cae6a1c3a961fa7c59560e8035e08`. 외부 finalizer session80706 exit0, 실제 `docker image load` session16260 exit0. load 뒤 image Id `fe9967bd7c1f35d91c040f50533f38407a4037e93fedfca97264e74eb813e20f`, linux/amd64, UID1000:1000, CMD train 그대로다. 게시 성공 뒤 sibling partial hardlink만 정리했으며 검증된 최종 TAR 내용은 보존됐다. 원격 업로드/학습 실행은 하지 않았다.

실제 `runs/r5_package_test_20261101_build02/DECISION_READY.json`은13:11:32 KST에 생성돼 GPU/source/release/TAR/manifest/config/native manifest 해시를 기록한다. source manifest와 동결 RELEASE는 변하지 않았다. 이번 helper 교정은 외부 게시 도구만이다.

통합 CPU 검사 **74건 exit0**: 공통17+로컬21+원격6+연결4+exporter26. `/tmp/soomac_r5_all_cpu_tests.out`에 실제 출력이 있다. 첫 검사 명령은 시스템 Python의 torch 부재와 remote import path 누락으로2 import error가 났다. 설치/코드 변경 없이 기존 gemma4_env Python과 해당 PYTHONPATH로 교정해74건을 재실행했다. 모의 테스트의 train 로그를 실제 학습으로 세지 않는다.

대기 프로세스는 실제 exec session13153/PID582218, `pipeline_state.json status=waiting`, P/Q pending, 자식 프로세스0이다. 새 gate `LOCAL_READY_AFTER_NVFP4_RETEST.json`은 미존재이며 root가 새 NVFP4 종료를 창작하지 않았다.6시간 bounded wait이므로 약19:12 KST까지 신호가 없으면 종료하며 자동 재시도하지 않는다. 실제 NVFP4 재시험·로컬 본 학습은 아직 시작됐다고 확인하지 못했다. 관측 GPU compute의 다른 소유 RustDesk PID564330은 변경하지 않았다. 학습 시작은 새 신호뿐 아니라 compute/여유 검사도 통과해야 한다.

남은 한 단계: Claude가 새 재시험을 실제 종료하고 GPU를 반환한 뒤 [인계 문서](2026-11-01_nvfp4_to_training_handoff.md)의 helper로 새 gate를 명시적으로 게시한다. Goal은 이 외부 조건을 기다리는 active이며 완료로 표시하지 않는다. 아래 기록은 시간순 과거 상태이며 이 최신 절을 우선한다.

### 후속 관측 — 대기/담당자 연결 확인

### 13:20 KST — 외부 재시험 의존성의 blocked 감사

직전 제작/게시 턴은 progress였고, 이후 live session13153을 poll한 턴들은 verified wait였다. 같은 미충족 조건(새 Claude NVFP4 재시험 완료 신호 부재)은13:14/13:15/13:18/13:20 관측과 세 번 이상의 연속 Goal 턴에서 유지됐다. 단순히 느리거나 상태 파일만 보고 종료/정지를 단정한 것이 아니다. 현재 session13153은 다시 실제 running으로 확인됐으며 P/Q pending, 새 gate 없음이다.

추가 확인으로 이 PC의 NVFP4/vLLM/컴파일 작업은 실행 중이 아니고 유일한 시험 종료 기록은 과거 RAM OOM abort임을 확인했다. Orca에서 현재 연결된 담당 터미널도 찾지 못했다. 이 작업자의 승인 범위는 NVFP4 재실행/가속 변경이 아니라 **Claude 재시험 종료 뒤 학습**이므로 그 선행 작업을 대신 실행하거나 옛 abort를 성공 신호로 만들어 진행할 수 없다. 새 제작/검증을 반복해도 외부 조건은 해결되지 않는다. 따라서 Goal은 완료가 아닌 외부 의존성 blocked로 전환할 조건을 충족한다. 이는 이미 살아 있는 대기 launcher를 종료한다는 뜻이 아니다. 실제 대기는 약19:12 KST까지 유지되며 유효한 새 신호+GPU 여유가 오면 기존 승인대로 순차 학습한다.

사용자/담당자가 할 다음 한 단계: NVFP4 담당 Claude에서 재시험을 실제 재개하고, 종료·GPU 반환 뒤 인계 문서의 새 gate를 게시한다. 재시험 완료 전 본 학습은 계속 금지다. 원격 업로드/실행·운영 adapter 변경·VLM·commit/push는 하지 않았다.

사용자 `확인 한번 해보쇼` 요청 뒤 실제 호스트를 확인했다. GPU compute 목록은 RustDesk뿐이며 NVFP4/vLLM/nvcc/cicc/ninja 프로세스는 관측되지 않았다. Python은 로컬 waiting launcher PID582218만 보였다. localhost:8000 health는 connection refused(HTTP000)였다. 따라서 **이 PC에서 NVFP4 재시험이 현재 실행 중이라고 볼 근거가 없고, 확인된 실행은 중단된 옛 시험뿐**이다. 외부 PC/다른 포트에서의 별도 시험까지 없다고 일반화하지 않는다.

이번에는 실제 옛 marker 경로를 찾았다: `training/learning_first_v43/gpu_handoff/CLAUDE_NVFP4_TEST.done`. 12:55:38 KST 기록이지만 내용은08:23의 `aborted_system_oom`, `NVFP4 시험은 미완료`이다. 새 시험 완료가 아니다. 기존 scratchpad는 `/tmp/claude-1000/-home-roma-IRC-SOOMAC3-0/f10905d6-bf59-4085-a889-d658537908bd/scratchpad/`이며 `nv_redhat_w4a4/server.log`에 Ninja build failed/Killed/08:23:41 NCCL 종료 기록을 확인했다. 스크립트·가속·VLM 파일은 읽기만 했고 실행/수정하지 않았다. RAM available 약54GiB이며 이는 시험 완료 증거가 아니다. 새 gate를 만들지 않고 대기한다.

session13153을 다시 poll해 실제 running 응답을 받았고 새 gate 부재와 P/Q pending을 확인했다. 이전 Goal 턴은 live handle을 확인한 **verified wait**이며 launcher를 중복 실행하지 않았다. Orca CLI의 새 runtime에서 현재 repo terminal list는 totalCount0이었다. worktree ps에 남은 Claude3개는 과거 done 데이터검수/계획검수 기록이며 NVFP4 담당으로 승격하지 않았다. Downloads의 무관한 Claude에도 보내지 않았다. 따라서 **담당자에게 인계 전달 완료 또는 재시험 진행 중이라고 주장하지 않는다.** 실제 담당의 재시험 상태·보고서 경로만 비동기 질문으로 요청했다. 별도 NVFP4 실행/가속 코드 변경/옛 abort 신호 채택/학습 시작은 없다.

## 최신 사용자 결정 / 보존된 TAR 상태 (2026-11-01)

사용자는 `NVFP4 재시험이 끝날 때까지 학습 대기`를 명시했다. 기존 Claude의 `aborted_system_oom` 파일은 완료 신호로 채택하지 않는다. 새 재시험 종료·GPU 반환 전 로컬 본 학습은 실행하지 않는다. Decision TAR 완료 작업은 계속한다.

첫 TAR 내보내기는 완료됐으나 외부 finalizer가 잘못된 ID 가정으로 검증을 중지했다. `.tar.partial` 6.6GiB는 원본 그대로 보존했고 Decision 성공 영수증은 없다. 실제 Docker inspect의 ID `fe9967...`는 **OCI image index digest**, TAR Config SHA는 `258d7f...`이다. 실제 `index.json → fe9967 index blob → 9ad3a5 native linux/amd64 manifest → 258d7f config` 연결을 읽어 확인했다. image index에는 Docker attestation 메타데이터도 있고 이는 두 번째 실행 이미지가 아니다. 비교를 느슨하게 하지 않고 이 descriptor hash chain을 검증하도록 exporter만 교정한다. 이미지/학습 코드/GPU checkpoint는 바꾸지 않는다.

추가 Codex 후속 전달 중 Orca 종료가 확인됐다. 지침대로 앱을 다시 열었으며 runtime은 ready로 복구됐다. 옛 handle로 재전송하거나 편집자를 중복 생성하지 않는다. 마지막 국소 교정 뒤 독립 재검수는 아직 받았다고 주장하지 않는다.

국소 OCI 교정 뒤 exporter의 CPU 검사는 **23건 통과**다(이전13건 포함, 합산 중복 없음). legacy Config ID와 OCI index ID 양쪽, linux/amd64 단일 실행 manifest, 정상 attestation, root/size/config/layer/attestation 오류, explicit sibling partial 복구를 검사했다. 실제 보존 TAR는 읽기 전용 전체 레이어 검증 중이다. 새 Codex sol/high 읽기 전용 검수는 `/tmp/soomac_r5_tar_oci_review.out`에 진행 중이며 결과를 아직 승인으로 쓰지 않는다. 이전 NVFP4 abort와 혼동하지 않도록 신규 gate 이름을 `LOCAL_READY_AFTER_NVFP4_RETEST.json`으로 분리했다. 새 완료 신호·본 학습·대기 프로세스는 아직 없다.

후속 실제 결과: 보존 TAR의 **51개 ordered layer 참조·52개 TAR member·OCI index/native/config/attestation 연결과 압축 전후 layer 해시가 exit0으로 통과**했다. manifest SHA는 `eea548a7d92775eb1919141eb3650992613e2b57e1df5783e288ce660540a452`, config SHA는 `258d7fdcd2f22ca5435589f9d1fbf82b46e0538f602f162aa086383fd96c1a43`이다.

Codex 검수는 **조건부 찬성**이었다. 조건은 실제 `finalize_tar.py`의 최종 TAR 링크 뒤 receipt 게시 실패 시 재실행이 막히는 순서였다. root가 지적 행을 확인해 receipt 게시 성공 전 partial hardlink를 보존하고, explicit recovery에서 기존 최종 TAR와 같은 inode인 경우만 재검증/receipt 게시하도록 고쳤다. 무관한 기존 TAR는 계속 거부한다. 중간 JSON 임시는 고유 파일명을 써 실패가 다음 재시도를 막지 않는다. 동시 변경 추측은 실제 증거가 없지만 검증/해시 전후 stat 변화 차단을 추가했다. 관련3 CPU 검사 추가로 **현재26건 통과**다(이전23건 포함). 모델 지정을 확인할 수 없다는 reviewer의 문구와 달리 실제 CLI header는 gpt-6-sol/high로 확인했다. 마지막3건 추가 뒤 독립 재검수 완료로 확대하지 않는다.

외부 finalizer의 실제 게시를 시작했다. 같은 보존 partial을 사용하며 재내보내기·이미지 재빌드·GPU 재시험·본 학습은 하지 않는다. Decision 영수증은 모든 실제 검증이 끝난 뒤에만 생성된다.

[현재 승인·계획](2026-11-01_training_package_build_plan.md) · [데이터 작업 기록](2026-11-01_dataset_rebuild_worklog.md)
→ 과거 학습/검수 결과와 이번 새 패키지의 실제 검증을 구분한다.

## 2026-11-01 — 구현·CPU 검수 진행

최신 원본은 `compound_semantics_20261101_r2/dataset`, manifest SHA256 `7850948287237c5ace9d1b0f56931419760fbce3f0afa02cea40681acb6ceffa`이다. 사용자가 원격 대상을 Decision으로 확정했으므로 선택을 다시 비교하는 작업은 하지 않는다. 기존 사용자/Claude 변경·VLM·추론 가속·운영 adapter를 수정하지 않았다.

1. 새 `training/learning_first_v43/training_release_20261101_r5/common/`에 기존 trainer·loss·budget·progress의 별도 복사본과 릴리즈/체크포인트 검증·단일 Agent 실행기를 작성했다. 원본 history 절삭을 명시적 오류로 바꾸고, 최종 smoke step 저장·매 epoch 보존·서로 다른 best/current 재개를 처리한다. 새 코드와 기존 Runtime을 혼동하지 않는다.
2. 원본 manifest는 false/pending 그대로 보존하고, 사용자 조건부 승인·학습 코드·base 전체 가중치·실제 encoding 영수증을 새 RELEASE에 결속하는 조립기를 작성했다. 첫 `releases/conversation_r5_20261101_build01`은 **중간 조립 증거**이며 이후 교정이 포함된 최종 패키지가 아니다. 덮어쓰지 않고 보존한다. 최신 source 파일을 바꾼 것이 아니다.
3. 공통 CPU 17건과 로컬 fake-executor 18건을 root가 실행해 통과했다. optimizer/스케줄러 누락·손상·설정/step 불일치·저장 실패 후 이전 marker·다른 best 복제·entrypoint drift·RNG tensor/list 형식을 포함한다. 실제 CUDA optimizer 재개 시험은 아직 미실행이다.
4. 독립 Codex CLI는 첫 sandbox에서 read-only state DB로 시작하지 못했고, 같은 검수를 호스트 권한으로 실행해 결과를 `/tmp/soomac_r5_training_common_review.out`에 받았다. 지적 중 remote 실행 코드 pin 누락·최종 step shortcut을 코드로 확인해 보완했다. RNG는 native 단일 GPU tensor/분산 list가 다름을 직접 source로 확인하고 양쪽 형식 검사로 보완했다. 마지막 교정 뒤 재검수와 GPU 검증은 남았다. 첫 검수 판정을 최종 승인으로 쓰지 않는다.
5. 호스트 GPU는 RTX PRO 4500 Blackwell, 총32623MiB이다. Docker NVIDIA runtime/CDI 선언과 Toolkit 실행 파일은 없었다. 시스템 설치/서비스 재시작 없이 임시 UID1000 컨테이너에 GPU 장치와 host driver `.so`를 read-only 연결하여 CUDA availability=True·tensor1=1.0·NVML process 조회를 실제 확인했다. 이미지에 host driver를 넣지 않는다. 이 선행 검사를 실제 모델 학습/복원 합격으로 세지 않는다.

기존 dependency 이미지 `edf75b660f97ddcc0ca1b9e48006a0ee3a50ffdb308a94b93804fe3e3d1cf086`에서 직접 CPU import한 버전은 torch2.8.0+cu128/transformers5.13.0/peft0.21.0/accelerate1.14.0/bitsandbytes0.50.2/safetensors0.8.0이다. 로컬 추론 환경 `soomac_llm`으로 학습이 자동 전환됐다고 쓰지 않는다. 새 common의 실제 버전 호환성은 다음 통합 시험 대상이다.

**다음:** remote 편집 결과를 동결해 새 릴리즈에 포함하고 이미지 기본 ENTRYPOINT/CMD로 preflight→짧은 optimizer 저장/실패/복원→TAR 검증을 수행한다. Claude NVFP4 시험 완료 영수증은 아직 없으며, 본 학습·대기 launcher는 실행하지 않았다. Goal active이다.

## 2026-11-01 — 통합 동결 build02

공통17·로컬21·원격6·워크플로 연결4 = **CPU48건 통과**다. 로컬21에는 유효한 smoke 완료를 train에서 건너뛰거나 채택하지 않는 검사와 미래 시각 영수증 거부를 추가했다. 원격의 기본 CMD는 train이며, smoke의 기본 실패 주입은0이다. 이전 모델/소스와 체크포인트를 자동 채택하지 않는다.

새 불변 릴리즈는 `/home/roma/SOOMAC_TRAINING_WORKSPACE/upgrade_v4/learning_first_v43_llm_final/releases/conversation_r5_20261101_build02`, RELEASE SHA256 `6b036af72d96c98362fbe225c8ab6beebc6cdd8a9c632edf6d946cce9c289cbc`이다. remote shell/입력 JSON/Dockerfile도 code_files에 포함했다. 최초 build01은 중간 조립 증거로 보존하며 최종 학습 소스로 쓰지 않는다.

실제 이미지 ID는 `sha256:fe9967bd7c1f35d91c040f50533f38407a4037e93fedfca97264e74eb813e20f`, amd64/linux·UID1000:1000·ENTRYPOINT `/app/remote/run_training.sh`·CMD `["train"]`이다. UID1000·read-only root/model·신규 출력으로 actual entrypoint preflight가 D train4360/val309에서 exit0였고, 같은 이미지 공통17 CPU 시험도 exit0이다. 이는 17개 새 사례 추가가 아니라 동일 사례의 이미지 환경 재검증이다.

실제 GPU 시험은 host exec session46394에서 `/.../runs/r5_package_test_20261101_build02/gpu_verification/`에 시작했다. 첫 profile 컨테이너 `2b47d552d0cc`를 실제 running으로 확인했다. 프로파일은 train+validation 전체를 실제 encoder로 처리한 뒤 유한 batch를 검사한다. local test byte budget32000000000이며 원격44GB 성능을 측정했다고 주장하지 않는다. 실제 모델/optimizer/복원 합격은 아직 확인 전이다. 각 phase COMMAND/EXIT/log를 남기고 예상된 실패 주입 외의 오류면 멈추며 자동 재시도하지 않는다.

## 2026-11-01 — 실제 profile 완료 / 복원 smoke 진행

profile은 exit0이다. 배치1의 mixed/max-target-span/reference 세 optimizer 시험에서 loss/gradient가 유한하고 0이 아닌 것을 확인했다. 예약 peak23559405568bytes(약21.94GiB), allocated peak22523206656bytes이며 cold step17.97~19.86초다. 배치2는 허용 allocator29852516352bytes에서 추가506MiB 할당에 실패(OOM)했다. 유한 검색은 즉시 중단돼 배치4/8을 시도하지 않았고, 측정해 통과한 배치1을 선택했다. 이는 폐기용 profile이지 본 학습 실패/기존 checkpoint 손실이 아니다. 원격 TAR의 배치는 여전히auto이며 서버의 실제44GB/상한 아래에서 별도 측정한다. 이 local profile을 원격 배치1 강제로 적용하지 않는다.

같은 driver session46394가 `GPU_TEST_EXIT profile code=0` 뒤 `GPU_TEST_START save5_fail`을 출력했다. 실제 모델 smoke10 전체 상태 시험은 진행 중이다. `save5_fail.log`의 loading은 같은 전체 SFT 토큰화이며 무한 재시도가 아니다. 모델·저장/복원 합격, TAR 완성, Claude 종료, 본 학습 시작은 아직 주장하지 않는다.

후속 실제 관측: train5step의 유한 loss·gradient norm과 평가를 마친 뒤 전체 파일과 `CHECKPOINT_COMMIT.json`, `.lab-monitor/resume.json` step5가 저장됐다. `Injected resume test failure after global step 5`로 exit1을 냈고 driver가 이를 확인했다. host의 sealed `resume_marker()`로도 step5/파일 hash 검증이 통과했다. 첫 원본 경로는 `smoke/attempt-4352aebdd42d4651b459ad6e692171a7/decision/checkpoint-5`이다. 이후 같은 driver가 `resume5_save10_fail`을 시작했다. 아직 native 복원 4종 equality와 최종 저장/복원은 진행 중이다.

GPU가 도는 동안 기존 local editor를 재사용해 root-level TAR exporter/검증기와 synthetic CPU 시험만 작성하도록 했다. 동결된 common/local/remote와 RELEASE/image는 수정하지 않는다. exporter는 실제 GPU 합격·image/release identity·최종 smoke 영수증을 확인하기 전에는 TAR/Decision 성공 영수증을 만들지 않아야 한다. synthetic TAR 검사를 실제 산출물 합격으로 세지 않는다.

실제 로컬 queue 대상 경로의 `.preflight/response`와 `.preflight/recommendation`도 기존 `gemma4_env` Python에서 각각 exit0이었다. train pipeline state나 GPU 학습 프로세스는 만들지 않았다. 추가 read-only Codex 통합 검수는 `remote/INTEGRATION_REVIEW.md`에서 **조건부 수용·새 구체 결함 없음**이며, 조건은 root의 실제 GPU/최종 TAR 검증이다. root가 파일·줄과 최초 지적 교정을 대조했고, 이를 GPU 통과로 확대하지 않는다.

## 2026-11-01 — 실제 GPU 복원 최종 통과

driver session46394는 정상 exit0이다. 실제 단계별 종료는 profile0 → step5 저장/의도적 실패1 → native resume5→10 저장/의도적 실패1 → native resume10 최종 완료0이다. `GPU_VERIFICATION.json`은 `PROFILE_AND_SAVE_FAIL_RESUME_PASS`이며 source/image/release는 동결 build02와 같다. 검증기 조건을 읽고 실제 로그·영수증에도 대조했다. 5→6step부터 재개됐으며 최종10step 저장본의 adapter/optimizer/scheduler/RNG도 네 SHA가 각각 저장본과 완전히 일치한다. 첫/두 번째 원본 checkpoint tree hash는 불변이다. smoke optimizer 업데이트는 총10번이고 폐기 profile의 통과 업데이트는3번이다. 배치2 profile OOM은 더 큰 후보를 중단시킨 한 번의 유한 시험이다. 본 학습/원격 실행/adapter 배포는0이다.

최종 smoke checkpoint는 `smoke/attempt-2f758741f98a4cb888b9d2a4b7aa0d28/decision/checkpoint-10`이다. host `nvidia-smi` compute app 목록과 해당 image의 `docker ps`는 시험 종료 후 모두 비었다. 여기서 GPU가 반환됐다는 것과 **Claude NVFP4가 끝났다는 것은 다르다**. Claude 종료 신호는 아직 없다.

발견한 비차단 표시 한계도 남긴다. resume5의 첫 stdout progress rate는 초기 loading history의 global0을 비교 기준으로 잡아 순간적으로0.31step/s/ETA14초를 표시했고, 다음 step부터0.05로 정상화됐다. 실제 `global_step=6`, `start_step=5`, `run=1`, 누적 sample·training time과 native 복원은 맞다. 권위 있는 JSON의 필드/체크포인트가 바뀐 것은 아니며, 선택적 콘솔 ETA를 학습 속도나 재개 정확성의 근거로 쓰지 않는다. 이 표시 교정은 **이번 동결 코드에 미적용**이다. 전체 표시가 무오류라고 보고하지 않는다.

최종 exporter9 synthetic CPU 검사는 root도 실행해 통과했다. exporter는 실제 GPU 결과와 seal 검증을 먼저 확인하고 정상 production TAR만 게시해야 한다. 현재 최종 독립 exporter 검수/실제 내보내기·로드 재검증·Claude 인계·로컬 대기 arming이 남았다. source manifest/운영 Runtime/가속·VLM·기존 이미지·기존 adapter는 바꾸지 않았다.

`workflow_receipt.py`는 root가 실제 Decision 검증 후 작성할 영수증과 Claude의 명시적 시험 종료·GPU 반환 보고서를 연결한다. GPU가 빈 것을 보고 종료를 창작하지 않는다. root Decision 영수증·완료 신호·본 학습은 **아직 생성/실행하지 않았다**. helper를 실행하는 것은 운영자의 명시적 종료 확인이며 자동 감지 결과가 아니다.

로컬/원격 편집 세션 두 개는 각 CPU 보고 후 idle을 확인했다. 일반 Orca 편집 세션으로서 통합 재검수에 재사용할 수 있게 유지한다. Run/Dispatch를 생성하거나 worker_done을 가장하지 않았다. 새 Docker tag `soomac/decision-r5:20261101-build02`의 빌드를 시작하며 실제 모델 optimizer/복원·최종 TAR 검증은 여전히 다음 관문이다.

Cause: 파일/모의 검사 합격만으로 실제 저장 형식과 native 복원을 보장할 수 없다.

Effect: CPU 합격과 실제 GPU·컨테이너·최종 TAR 관문을 분리하고 미완료 항목을 유지한다.
## 최신 추가 — GC OFF·부분 GC·batch2 속도 우선 시험

상세 결과는 [GC 속도 비교](2026-11-01_gc_speed_trial.md)에 기록했다. GC OFF는 microstep1 forward OOM, native 부분 GC32/48도 microstep1 OOM, native 부분 GC40/48은 microstep1~3 forward/backward 이후 **microstep4 forward OOM**이었다. 세 후보 모두 optimizer update를 완료하지 못해 속도 수치가 없다. 전부 첫 forward 실패라고 한 설명은 정정한다.

GC ON batch2×accumulation8은 321.8207초/update로 완료했다. 기존276.3486초/update보다 약16.5% 오래 걸렸으므로 속도 후보로 채택하지 않는다. 두 수치 모두 cold1회이며 warm 확정값은 아니다. 다음 full native GC 시험은48층 GC와 batch1을 유지하고 smart activation offload를 끄는 별도 비교이다. 부분 GC 결과로 offload 단독 효과를 추정하지 않는다. production·checkpoint·원본 데이터·기존 환경·원격 Decision은 변경하지 않았다.

Cause: GC OFF는 보관할 활성값이 늘고, 부분 GC도 누적 도중 메모리 확보에 실패했다.
Effect: 실패 후보의 가상 속도를 성과로 세지 않고, 완료한 optimizer update 시간으로 설정을 선택한다.

