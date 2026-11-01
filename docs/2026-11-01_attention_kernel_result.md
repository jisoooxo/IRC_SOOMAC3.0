# Gemma4 12B Attention 전용 커널 — 제한된 GPU 시험 결과

## 최신 결론 — 원인 분리 후 ON HOLD, 최적화 가능성 기각 아님

[원인 분리 결과와 전체48층 표](2026-11-01_padding_failure_diagnosis.md)를 우선 읽는다. 같은 초기 상태/RNG로 A를 반복하면 loss/logits는 같지만 gradient는 dropout0에서도 상대 L2 1.1788%의 자체 변동이 있었다. dropout0 A/B 차이는 loss0.00532645·gradient 상대 L2 111.3597%로 남아 dropout만의 원인으로 설명되지 않는다. 유효 position/target 정렬은 같고, 0층 QKV도 bitwise 같지만 그 SDPA 출력에서 첫 측정 차이가 발생했다. 단, 최하위 SDPA is_causal/scale/GQA 실인자를 모두 별도 검증한 것은 아니므로 특정 CUDA 연산·mask 경로를 근본 원인으로 확정하지 않는다.

따라서 후보는 **보류(ON HOLD)**이다. Attention 최적화 전반을 기각하지 않는다. 이번 원인 분리에서는 dedicated kernel·optimizer update·speed timing·production 변경을 하지 않았다. 사전 중단 결과의 역사적 기록은 아래에 보존한다.

## 최신 사용자 지시 — 원인만 분리, 가능성 최종 기각 아님

아래 수치 실패는 Attention 최적화 전체의 불가능 증거가 아니다. 사용자가 기존 SDPA만으로 A 반복·진단 전용 dropout0 A/B의 원인을 추가 분리하도록 승인했다. `padding-diagnose` 단일 GPU 프로세스는 각 회차 초기 LoRA·모델 버퍼·CPU/CUDA RNG를 복원하고 hash로 확인한다. 실제 dropout p=0은 진단 내부에서만 적용하고 finally에서0.05를 복원한다. selected-logits target 값의 SHA/개수/유효 position_ids, GPU 실제 mask·layer0/5 QKV, 전체 및48층별 gradient norm/cosine/relativeL2를 기록한다.

이 과정은 optimizer update0회이며 전용 커널·production·원격 Decision·기존 checkpoint 변경이 아니다. 원본 경로는 `/tmp/soomac-training-accel-20261101/padding_diagnose_a1/`이다. 자기 A 반복이 불안정하면 padding 원인이라고 단정하지 않는다.

## 최종 후속: padding2개 제거 — CPU 통과 / GPU 정확성 기준 실패로 종료

**인위적 padding 제거 자체는 간단했지만, 이번 GPU 수치 동등성 기준을 통과하지 못했다. Production에는 적용하지 않고 추가 Attention 탐색을 종료했다.**

| 확인 | 결과 |
| --- | --- |
| padding2개 원인 | `padding_side=left` + `pad_to_multiple_of=8`,8430→8432 |
| 실제 tokens·labels·유효 position_ids·mask 의미 | CPU에서 보존 확인, positions2~8431 |
| CPU Global HD512 / sliding1024 mask gate | 양쪽 PASS |
| GPU native loss | padded0.8244394064 → unpadded0.8191129565, 차이0.00532645 > 사전 기준0.001 |
| GPU LoRA gradient | finite/nonzero이지만 allclose 실패, cosine−0.411088 |

attention layer0/5 출력과 logits도 사전 allclose 기준을 통과하지 못했다. label/hook 치환이 확인된 것은 아니며, **이 수치 차이의 근본 원인은 미확정**이다. 부동소수점 실행 경로 차이인지 별도 구현 차이인지 이번 짧은 검사에서는 더 분리하지 않았다. CPU mask 의미 일치만으로 GPU loss/gradient 동등성을 보장한다고 하지 않는다.

요청한 중단 기준을 따라 **unpadded fast router 비교·실제 고속 kernel 활성화·optimizer update timing을 실행하지 않았다.** optimizer update0회, timing=null이다. tolerance 완화·추가 설정 탐색·데이터/정답/history 변경·기존 체크포인트 변경은 없다. 최종 원본은 `/tmp/soomac-training-accel-20261101/padding_gpu_a1/PADDING_GPU.json`, 상태 `PADDING_CORRECTNESS_FAILED_STOP_NO_TIMING`이다. 기본 unpadded position_ids0~8429 경로를 시험한 결과로 일반화하지 않는다.

Codex CLI 최종 검수는 조건부 동의이며(`/tmp/soomac_padding_gpu_review.out`), CPU 보존·GPU 비교 설계·실패 뒤 중단 판정을 확인했다. root가 인용 코드/메타데이터를 직접 대조했다. 지적된 시험 도구의 budget.check 누락과 중간 JSON 상태 표시를 국소 보완했지만, 실패 원본 JSON을 덮어쓰거나 GPU를 재시험하지 않았다. 원래 중간 `PADDING_EQUIVALENCE.json`의 status=FAILED는 최종 판정 필드로 쓰지 않고 최종 `PADDING_GPU.json`을 우선한다. 이번 실제 NVML process peak13,613,662,208바이트는32GB 예산 아래였다. CPU12건 통과, GPU compute process0개로 반환을 확인했다.

Cause: padding만 제거한 SDPA 대조부터 사전 loss·gradient 수치 기준을 통과하지 못했으며 정확한 원인은 미확정이다.
Effect: 고속 커널의 성능·학습 동등성 합격을 주장하지 않고, 기존 padded 설정과 production을 그대로 유지했다.

아래 결과는 **기존 padded 입력**의 역사적 결론이다. 사용자가 마지막으로 실제 토큰은 보존하고 `pad_to_multiple_of=8`의 인위적 padding2개만 제거하는 별도 검사를 승인했다. 기존 production collator를 바꾼 것이 아니다.

원인은 sealed processor의 `padding_side=left`와 실험/학습 collator의 `pad_to_multiple_of=8`이다.8,430은8의 배수가 아니어서8,432가 되고, 앞의2개 input padding과 labels−100이 추가된다. `padding_cpu`에서는 같은 collator에 `pad_to_multiple_of=None`만 전달했다. 실제8,430개 input_ids/labels가 원본 valid 부분과 정확히 같고, 원래 default arange8432의 valid position_ids **2~8431**도 보존했다. 두 attention mask의 유효 토큰 부분은 정확히 동일하다.

CUDA 초기화 없이 실제 설치된 predicate 함수를 CPU에서 확인했다. Global은 새 mask=None의 causal 조건, sliding은 순수 window1024 band 조건을 모두 통과했다. source hash·CPU 영수증은 `/tmp/soomac-training-accel-20261101/padding_cpu_a1/PADDING_CPU.json`이다. 이는 GPU kernel 실행·loss·gradient 통과를 뜻하지 않는다.

GPU 계획은 동일 모델/초기 LoRA의 padded forced-SDPA → unpadded forced-SDPA → unpadded normal router 순서였다. 실제로 두 번째 단계에서 중단했다. 모든 실제 토큰·labels·position_ids·native loss/GC 설정을 유지했다. 정확성 진단만 LoRA Dropout 모듈의 training flag를 잠시False로 하여 난수 영향을 분리하고, 원래p=0.05는 변경하지 않으며 finally에서 training flag를 복원했다. 사전에 loss abs/rel1e-3, output/logits/gradient allclose atol1e-4·rtol1e-3를 정했고 실패 후 완화하지 않았다. 실제 성능 시험은 하지 않았다.

GPU 증거 위치: `/tmp/soomac-training-accel-20261101/padding_gpu_a1/`. 원격 Decision·기존 checkpoint·production 학습은 변경하지 않는다.

[기존 계획](2026-11-01_claude_gemma4_attention_plan.md) · [재사용한 시험 코드](../tools/training_acceleration/attention_ablation.py) · [기존 GC 측정](2026-11-01_gc_speed_trial.md)
→ **현재 실제 mask를 보존하면 설치된 전용 라우터가 선택되지 않는다. 새 커널 개발로 확대하지 않고 후보를 중단한다.**

## 1. 먼저 보는 결론

| 항목 | 기존 Unsloth A | 전용 Attention 활성화 후보 B |
| --- | --- | --- |
| Global 커널 | CUTLASS memory-efficient SDPA, BF16 | 같은 CUTLASS SDPA로 fallback |
| Sliding 커널 | CUTLASS memory-efficient SDPA, BF16 | 같은 CUTLASS SDPA로 fallback |
| Update 시간 | 기존276.3486초/update, **cold1회** | 전용 커널 미활성화로 optimizer A/B 미실시 |
| Tokens/s | 기존488.079 non-padding tokens/s | 미측정 |
| Peak VRAM | 기존 optimizer update allocated12.30GB / reserved12.86GB | 동등한 optimizer update peak 미측정 |
| Loss/gradient 검증 | 기존 native CE eval1행 검증·finite/nonzero update gradient | 전용 커널 비교는 미실시, 정확성 합격 아님 |

1. **적용 가능한가?** 설치된 HD512/sliding 정상 라우터는 측정한8,430토큰 입력의 실제 mask 조건을 지원하지 않았다. 모든 모델·입력에서 전용 커널이 불가능하다는 뜻은 아니다.
2. **몇 배 빨라졌는가?** 확인한 가속 배수가 없다. 같은 SDPA fallback을 전용 커널 성공으로 세거나1.0배를 실측값으로 만들지 않는다.
3. **지금 production에 채택할 가치가 있는가?** 이번 후보는 채택하지 않는다. 기존 검증된 Unsloth smart GC·batch1 기준을 보존한다. production·checkpoint·데이터·원격 Decision은 변경하지 않았다.

## 2. 실제 시험과 CUDA 증거

A는 기존 compile 설정이고 B는 새 프로세스에서 공식 `UNSLOTH_COMPILE_DISABLE=1`을 사용했다. 이 설정은 Unsloth rewritten forward를 유지하면서 compile을 끄는 정상 경로이다. 후보는 이것 하나로 제한했다. 학습 설정은 동일 NF4/BF16·LoRA r16/alpha32/dropout0.05·smart GC·native selected-logits CE이며 동일 encoded 입력 SHA는 `bd6d064b685caf3c19d1dbcc8b9247449d66006306541c77a75e1337c6786b8f`이다. GPU process 예산32,000,000,000바이트/allocator 외512MiB를 보존했다. GC OFF·부분 GC·batch2·native GC를 재시험하지 않았다.

실제 `model.train()` forward/backward를 **ALL_ATTENTION_FUNCTIONS wrapper 없이 먼저** CUDA profiler로 측정했다. CPU attention operator의 query shape와 profiler의 **직접 연결 CUDA kernel**을 함께 보존했다. 카운터는 보조 자료일 뿐이다.

| 유형 | 실제 query shape | A 연결 FWD/BWD | B 연결 FWD/BWD | 실행 CUDA kernel |
| --- | --- | --- | --- | --- |
| Sliding40층 | `[1, 8432, 16, 256]` | 120 / 40 | 80 / 40 | `fmha_cutlassF_bf16_aligned_32x128_gmem_sm80`, `fmha_cutlassB_bf16_aligned_128x64_k65536_sm80` |
| Global8층 | `[1, 8432, 16, 512]` | 24 / 8 | 16 / 8 | 같은 FWD/BWD kernel 계열 |

A의 profiler는 최초 compile 과정도 포함하므로 FWD120/24를 추가120개/24개의 모델 층이나 정상 update의 고정 실행 횟수로 해석하지 않는다. B에서는 forward+GC 재계산 때문에 FWD80/16, BWD40/8로 관측됐다. 두 프로파일 모두 HD512 `_lh_fwd`/`_lh_bwd` CUDA 실행을 관측하지 못했다. FA2=False/Xformers=None 또는 `LARGE_HEAD_ENGAGED=0`만으로 내린 판정이 아니다.

원본:

- A: `/tmp/soomac-training-accel-20261101/attention_current_a1/NATIVE_PROFILE.json`, `CHECK.json`, `.log`.
- B: `/tmp/soomac-training-accel-20261101/attention_compile_off_b1/NATIVE_PROFILE.json`, `CHECK.json`, `.log`, `native_attention_trace.json`.
- B Chrome trace SHA: `16119bcb6f66c9660d573bc696f9d03f8681b3f4a70ebe9cc91059b6c2a9d255`.

## 3. 정확한 fallback 조건

**Global:** `large_head_attention.py:634~689`는 각 query가 non-empty 연속 key 범위를 가지며 lo/hi 양쪽이 행에 따라 감소하지 않는 mask만 허용한다. 실제 입력의 별도 mask 진단은 빈 행0·불연속 행0이지만 **hi 감소1회**였고, 원본 `intervals_from_mask`는None을 반환했다. 따라서 compile을 꺼도 정상 HD512 router가 실행할 수 있는 interval mask가 아니다. compile ON에서는 추가로 `large_head_attention.py:904~929`의 compiled+명시적 mask 조건도 fallback을 일으킨다.

**Sliding:** `unsloth_zoo/temporary_patches/gemma4_flash_sliding.py:90~188`는 기존 fast path이고, `gemma4_banded_attention.py:75~117`의 순수 causal/window band 검사를 통과해야 한다. 실제 mask에서 **1,025개 query 행이 그 band와 달랐다.** compile 중에는 이 검사 자체도 거절된다. compile을 꺼도 mask 조건은 그대로 실패하여 FA2/banded 경로가 아닌 기존 SDPA로 갔다.

CPU에서 원래 sealed processor·collator를 재현해 실제8,430 non-padding 토큰이 **왼쪽2개 padding**으로8,432가 되는 것을 확인했다. mask scalar 진단은 동일 입력의 별도 intrusive eval probe이며 production profiler와 구분한다. Gemma4의 mask 생성은 `modeling_gemma4_unified.py:654~667`에서 동일 config/input attention_mask/position을 사용한다. mask의 empty/band/interval 조건을 바꾸거나 padding2개를 제거해 통과시키지 않았다.

Cause: 현재 mask는 설치된 HD512의 단조 interval 조건과 sliding의 순수 band 조건을 만족하지 않는다.
Effect: 공식 compile OFF 설정으로도 전용 경로가 활성화되지 않아 이 후보를 중단했다.

## 4. 측정하지 않은 것을 통과로 쓰지 않음

B에서 다른 Attention 커널이 실행되지 않았으므로 **warm optimizer A/B, forward 시간·backward+GC 시간 분리, tokens/s·samples/s, update peak 비교를 수행하지 않았다.** 276.35초를 이번 warm 실측으로 고쳐 쓰지 않았다. profiler 소요 시간을 update 시간으로 환산하지 않았다.

전용 커널의 attention output/eval loss/LoRA gradient 수치 동등성도 합격 처리하지 않았다. 직접 HD512 layer probe는 mask qualification 단계에서 `SDPA_ONLY_MASK_NOT_INTERVAL`로 끝났다. 원본 CHECK에는 별도 `NOT_RUN` 판정 필드나 `numeric_equivalence` 결과가 없으며, **그 비교를 실행하지 않았다는 뜻**이다. 각 native training backward가 끝났다는 사실은 정확한 gradient 동등성 증거가 아니다. dropout0.05·원본 history·label mask·dtype/softcap/window 의미는 바꾸지 않았다.

프로파일·probe 전체의 A peak allocated11,950,501,888/reserved12,488,540,160바이트는 **optimizer update peak가 아니며** 표의 기존12.86GB와 섞지 않는다. B 프로파일의 RSS25,817,837,568바이트도 원격 RAM 필요량이나 update peak로 일반화하지 않는다.

## 5. 종료와 검수

기존 하네스의 counter-only timing gate는 CUDA 증명 없이 장시간 timing을 허가하지 않도록 막았다. 새 중복 GPU 도구를 만들지 않았다. 기존 CPU10건 및 추가 CLI 회귀1건, 합성 mask4조건(빈 행/구멍/정상 causal/window) 검사를 수행했다. 합성 mask 검사는 CUDA를 초기화하지 않았다.

계획의 Codex CLI(gpt-6-sol/high) 검수는 정상 compile OFF 후보에 조건부 동의했고, 실제 shape→CUDA 연결·카운터 한계·속도/정확성 미검증 구분을 지적했다. 실제 결과 최종 검수도 **조건부 찬성**이며 `/tmp/soomac_attention_result_review.out`에 남겼다. CUDA 연결·mask 거절·후보 중단에 동의했고, CHECK에 없는 `NOT_RUN` 필드가 있다고 표현하지 말라는 지적을 반영했다. root가 인용 코드·CHECK 결과를 직접 다시 대조했다. 학습 원문·가중치·gradient 배열·자격증명을 보내지 않고 코드/설치 소스와 결과 메타데이터만 사용했다. 시험 뒤 GPU compute process0개를 확인했다.

추가 설정 탐색·FlashAttention 설치/빌드·Flex 실험·새 CUDA/Triton kernel 개발·본 학습 시작은 하지 않았다. 이번 결론은 **현재 데이터·정책을 그대로 쓰면 설치된 전용 커널의 간단 활성화로 얻을 추가 가속은 확인하지 못했다**는 것이다.
