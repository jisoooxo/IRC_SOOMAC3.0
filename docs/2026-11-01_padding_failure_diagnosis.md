# Padding 실패 원인 분리 — SDPA 진단 결과

[기존 Attention 기록](2026-11-01_attention_kernel_result.md) · [시험 코드](../tools/training_acceleration/attention_ablation.py)
→ **dropout만의 문제가 아니다. 초기 상태·위치·target·유효 mask가 같아도 SDPA 출력에서 첫 수치 차이를 관측했다. 전용 커널 가능성은 기각하지 않고 속도 시험을 보류한다.**

**최신 상태 — 첫 실제 SDPA 호출 분리까지 완료:** layer0에서는 A/B 모두 bool sliding mask가 있고 `is_causal=False`다. 유효 Q/K/V와 유효 토큰 간 mask 전체가 같으며, 같은 유효 key 배열 크기로 계산한 FP32 기준 출력은 표본 query에서 bitwise 일치했다. 최초 차이는 BF16 CUTLASS 실행의 수치 차이로 좁혀졌다. **후단 gradient 큰 차이의 정확한 증폭 기전과 안전한 해결책까지 밝혀진 것은 아니다.** 자세한 실인자·수치는 §6이다. 전용 커널/속도 시험은 계속 보류한다.

## 1. 핵심 결과

| 비교 | loss 차이 | gradient norm A → B | gradient 상대 L2 | cosine |
| --- | --- | --- | --- | --- |
| A 반복, dropout0.05 | 0 | 132.472987 → 141.160330 | 6.7709% | 0.999867 |
| A 반복, 진단 dropout0 | 0 | 133.170150 → 133.339849 | 1.1788% | 0.999931 |
| A/B, 진단 dropout0 | 0.00532645 | 133.170150 → 30.260494 | 111.3597% | −0.414697 |

A는 padding2개, B는 padding0개이며 모두 같은 SDPA 경로다. 전용 HD512/sliding router를 끄고 진단했고 optimizer update0회, timing=null이다. A를 반복해도 backward 결과가 완전히 고정되지 않았으므로 과거 한 번의 gradient cosine만으로 padding 변경을 단독 원인으로 확정할 수 없다. 다만 dropout0의 A/B 차이는 이번 A/A 자체 변동보다 훨씬 크며 loss/logits 차이도 남아 **dropout만으로 설명되지 않는다.**

Cause: dropout0·동일 초기 상태로 통제한 뒤에도 loss와 gradient 차이가 재현됐다.
Effect: 전용 커널 속도 시험은 보류한다. Attention 최적화 전체를 최종 기각하지 않는다.

## 2. 무엇을 동일하게 통제했는가

동일 모델 객체·초기 LoRA 값·모델 buffers를 사례마다 복원했다. 매 사례의 LoRA hash, buffer hash, CPU/CUDA RNG hash, dropout p/training 상태를 기록했다. 같은 확률의 비교끼리 시작 상태와 RNG가 정확히 같았다. LoRA hash는 forward/backward 전후에 같았고, frozen parameter의 version도 유지됐다. **모든 사후 buffer/RNG 상태가 불변이었다고 주장하지 않는다.**

실제 input8,430개·labels·history·native selected-logits CE·NF4/BF16·smartGC·학습 정책은 유지했다. B의 position_ids는0부터 다시 시작하지 않고 기존 유효 위치 **2~8431**을 명시했다. 정답31개의 target 값 SHA가 같고 예측 위치에 대응하는 absolute position_ids8400~8430도 같았다. 물리적 배열 index는 padding2개만큼 달라지지만, 같은 실제 토큰의 같은 다음 토큰을 채점한다.

진단의 LoRA dropout은 named LoRA Dropout 모듈의 **실제 p를0으로 설정**했다. 이전 training=False만 사용한 진단과 구별한다. 진단 후 모든 p를0.05로 복원했고 production 설정은 수정하지 않았다. 이전 원본 데이터·정상 checkpoint·원격 Decision·학습 환경도 변경하지 않았다.

## 3. 처음 어디서 달라지는가

| 확인 위치 | dropout0 A/B 결과 |
| --- | --- |
| 초기 가중치·버퍼·CPU/CUDA RNG | 같은 hash |
| selected-logits target 값 및 유효 position_ids | 일치 |
| 실제 CUDA mask의 유효 토큰 관계, 0층·5층 | 일치 |
| 0층 SDPA 직전 Q/K/V | bitwise 일치, max absolute0 |
| **0층 SDPA 직후 출력** | **첫 관측 차이**, relative L2 0.027384%, max absolute0.0625 |
| 5층 Q/K/V | 이미 차이가 있음 |
| 최종 유효 target logits | relative L2 1.44681%, max absolute1.5 |
| 최종 loss/LoRA gradient | 위 표처럼 차이가 남음 |

따라서 관측한 범위에서는 labels/target index/absolute position_ids 이동보다 **첫 SDPA 호출 전후에서 관측한 수치 차이**가 앞선다. 작은 SDPA 출력 차이가 후단에서 커지는 양상이다. 다만 B의 global mask=None은 비교 시 causal 삼각 mask로 해석했으며, 실제 최하위 SDPA 호출의 is_causal/scale/GQA 등 모든 인자를 별도로 기록·대조한 실험은 아니다. 설치된 Transformers SDPA 소스는 mask=None일 때 query길이>1 및 module.is_causal 조건으로 causal을 정하지만, **그 소스 해석과 runtime 실인자 검증을 같다고 쓰지 않는다.** 따라서 mask 전달 경로를 완전히 배제하거나 SDPA 내부의 특정 연산을 근본 원인으로 확정하지 않는다. backward 자체 변동도 SDPA backward와 smartGC 등의 기여를 따로 분리하지 않았다. 여기서 의미론적 mask 오류나 기존 학습 무효를 주장하지 않는다.

## 4. 계측 검증과 한계

첫 진단 `padding_diagnose_a1`의 FP32 CPU cosine에는1을 넘는 값이 있어 해당 cosine/norm 요약을 채택하지 않았다. 실패 원본을 그대로 보존했고, FP64 분할 합산으로 고친 뒤 **단 한 번** 전체 짧은 진단을 재확인했다.16M CPU random 벡터의 자기 비교 및2배 벡터 비교에서 cosine1·relativeL2 0/1·norm 배율2의3개 성질을 확인했다. CUDA는 이 CPU 검사에서 초기화되지 않았다.

채택한 원본은 `/tmp/soomac-training-accel-20261101/padding_diagnose_a2_fp64/PADDING_DIAGNOSIS.json`이다. 같은 폴더의 사례5개 JSON에는 loss·초기 상태 hash·target 정렬이 있고, 최종 JSON에는 아래48층 gradient 차이와 실제 mask/QKV/SDPA 출력 차이가 있다. 원문 토큰·가중치·gradient 배열은 기록/전송하지 않았다. hook/wrapper는 진단 전용이며 production profiler나 속도 측정이라고 주장하지 않는다. 엄밀한 deterministic CUDA 알고리즘을 강제한 실험도 아니므로 A/A 변동의 정확한 기전을 확정하지 않는다.

기존 하네스/CPU12건 통과, FP64 reduction 성질3건은 실행 stdout으로 확인했다. 별도 저장 영수증은 없으므로 이를 독립 검수자가 재실행한 시험이라고 쓰지 않는다. Codex 읽기 전용 최종 검수(`/tmp/soomac_padding_diagnosis_final_review.out`)는 **조건부 동의/ON HOLD**이다. 통계·초기 상태·수치 차이의 관측에는 동의했고, 실제 SDPA is_causal 등의 runtime 인자를 완전히 대조하지 않아 정확한 원인 판정은 조건부라고 지적했다. root가 인용 코드와 결과를 직접 열어 대조하고 이 한계를 반영했다. GPU compute process0개를 확인했다.

## 5. 전체48층 gradient 차이 — 진단 dropout0 A/B

각 층의 모든 LoRA parameter를 고정 순서로 묶어 계산했다. None gradient는 해당 parameter shape의0으로 넣어 비교 순서를 보존했다. 모든 수치는 FP64 분할 합산이다.

| decoder 층 | A norm | B norm | relative L2 | cosine |
| --- | --- | --- | --- | --- |
| 0 | 36.966218 | 9.370540 | 110.789% | -0.321840 |
| 1 | 26.934107 | 6.558707 | 111.821% | -0.392384 |
| 2 | 14.986727 | 3.680944 | 112.519% | -0.418811 |
| 3 | 12.553494 | 3.292323 | 113.957% | -0.438178 |
| 4 | 17.718879 | 4.435254 | 114.628% | -0.501974 |
| 5 | 21.431888 | 5.466633 | 116.647% | -0.579443 |
| 6 | 51.058378 | 13.182894 | 115.980% | -0.539273 |
| 7 | 63.846229 | 14.245028 | 114.092% | -0.564538 |
| 8 | 31.753516 | 8.436472 | 116.286% | -0.530057 |
| 9 | 27.883364 | 6.709360 | 112.409% | -0.427404 |
| 10 | 27.895358 | 7.740987 | 116.579% | -0.508221 |
| 11 | 33.649343 | 7.656745 | 110.667% | -0.380013 |
| 12 | 29.184443 | 5.054142 | 103.186% | -0.100295 |
| 13 | 30.151569 | 4.470978 | 102.327% | -0.084591 |
| 14 | 12.509658 | 1.515350 | 101.891% | -0.097027 |
| 15 | 13.057582 | 1.351291 | 102.136% | -0.156841 |
| 16 | 18.480083 | 1.780252 | 103.144% | -0.283328 |
| 17 | 19.694598 | 1.888716 | 103.452% | -0.318222 |
| 18 | 24.099318 | 2.408623 | 101.903% | -0.142255 |
| 19 | 24.599265 | 1.861797 | 101.480% | -0.159175 |
| 20 | 8.379036 | 0.737404 | 100.828% | -0.050436 |
| 21 | 1.762075 | 0.872339 | 90.505% | 0.430215 |
| 22 | 0.811702 | 0.831524 | 26.179% | 0.966842 |
| 23 | 0.812522 | 0.841830 | 20.684% | 0.979980 |
| 24 | 1.101037 | 1.131835 | 20.677% | 0.979585 |
| 25 | 1.745747 | 1.786837 | 17.389% | 0.985500 |
| 26 | 1.709856 | 1.739624 | 15.077% | 0.988978 |
| 27 | 1.221028 | 1.254440 | 14.150% | 0.990620 |
| 28 | 1.716371 | 1.764686 | 13.005% | 0.992161 |
| 29 | 1.154399 | 1.188742 | 13.467% | 0.991624 |
| 30 | 1.244871 | 1.271634 | 14.044% | 0.990573 |
| 31 | 1.416547 | 1.432638 | 12.989% | 0.991723 |
| 32 | 1.004854 | 1.023410 | 12.597% | 0.992377 |
| 33 | 1.240644 | 1.275764 | 13.778% | 0.991160 |
| 34 | 0.490924 | 0.506624 | 15.229% | 0.989259 |
| 35 | 0.894249 | 0.936167 | 16.176% | 0.988553 |
| 36 | 0.717792 | 0.743568 | 15.583% | 0.988902 |
| 37 | 0.677288 | 0.695867 | 14.955% | 0.989482 |
| 38 | 0.570909 | 0.594014 | 15.564% | 0.989146 |
| 39 | 0.616176 | 0.626787 | 11.705% | 0.993412 |
| 40 | 0.633442 | 0.641554 | 13.554% | 0.991012 |
| 41 | 2.058121 | 2.118224 | 9.802% | 0.995747 |
| 42 | 0.557929 | 0.571003 | 12.866% | 0.992182 |
| 43 | 0.676686 | 0.693239 | 12.291% | 0.992919 |
| 44 | 0.517654 | 0.529197 | 12.468% | 0.992640 |
| 45 | 0.537810 | 0.551298 | 12.329% | 0.992893 |
| 46 | 0.821415 | 0.840046 | 10.004% | 0.995358 |
| 47 | 1.089734 | 1.122638 | 9.815% | 0.995767 |

고속 커널·장시간 optimizer 반복·본 학습은 이번 원인 분리에서 실행하지 않았다. 원격 Decision과 기존 checkpoint는 그대로다.

## 6. 첫 실제 SDPA 호출만 분리한 추가 진단

### 실제로 실행한 범위

`attention_ablation.py first-sdpa`를 기존 도구에 추가했다. A/B 각각 layer0의 `torch.nn.functional.scaled_dot_product_attention` 실인자를 가로채 **원래 함수를 그대로 호출**하고, 출력 직후 전용 `BaseException`으로 모델 실행을 종료했다. router의 일반 `except Exception` 재시도에 삼켜지지 않는다. 이후 층·최종 logits·loss·backward·optimizer는 실행하지 않았다. 최초 모델 진입은 두 번이지만 완주한 전체 모델 forward는0회다. 기존 loader·NF4·smartGC를 재사용했으며 별도 진단에서만 LoRA dropout0, 종료 시0.05를 복원했다. 전용 HD512/sliding router는 모두 OFF다.

실제 원본은 `/tmp/soomac-training-accel-20261101/first_sdpa_a1/FIRST_SDPA.json`이며 SHA256은 `27bbeedfe159d9f1595674e15e3d2cc74125d6fe3337689c5e2e0a150fc02de7`이다. 같은 폴더의 `A_padding2.json`, `B_padding0.json`에는 Q/K/V·mask shape/stride/dtype/hash와 실제 low-level 인자가 있다. 원문 토큰·QKV·가중치 배열은 메모리에만 두고 파일에 저장하지 않았다. GPU profiler는 첫 원본 SDPA 호출 내부에만 적용했다. 속도 비교용 계측이 아니다.

### low-level 실인자 대조

| 항목 | A: padding2 | B: padding0 |
| --- | --- | --- |
| decoder layer / head_dim | 0 / 256 (sliding) | 0 / 256 (sliding) |
| 실제 Q/K/V shape | `[1,16,8432,256]` | `[1,16,8430,256]` |
| Q/K/V dtype | BF16 | BF16 |
| Q stride | `[34537472,256,4096,1]` | `[34529280,256,4096,1]` |
| K/V stride | `[34537472,2158592,256,1]` | `[34529280,2158080,256,1]` |
| 실제 mask | bool `[1,1,8432,8432]` | bool `[1,1,8430,8430]` |
| **실제 `is_causal`** | **False** | **False** |
| 실제 `scale` / attention dropout | 1.0 / 0.0 | 1.0 / 0.0 |
| 실제 `enable_gqa` | False | False |
| 실제 유효 position_ids | 2~8431 | 2~8431 |
| 실제 SDPA operator | `_scaled_dot_product_efficient_attention` | 동일 |
| 실제 attention CUDA kernel | `fmha_cutlassF_bf16_aligned_32x128_gmem_sm80` | 동일 |

**이번 layer0의 B mask는 None이 아니다.** 이전에 등장한 None은 global layer5의 mask 설명이며, 첫 sliding SDPA 호출과 혼동하면 안 된다. `is_causal=False`여도 명시적인 bool mask가 미래 토큰과1024창 밖을 차단한다. GQA=False는 그룹 정보를 잃었다는 뜻이 아니다. Transformers가 원래8개 KV head를16개로 반복한 후 low-level SDPA에 전달한 결과이며, 실제 캡처 shape로 확인했다. 커널명의 `sm80` 문자열은 실제 GPU가 Ampere라는 증거가 아니다. 장치는 RTX PRO4500 Blackwell이며 profiler에서 관측한 커널 심볼을 그대로 기록했다.

### FP32 기준 출력과 실제 BF16 출력

유효8,430개 토큰의 실제 Q/K/V는 A/B bitwise 일치했다. **전체 유효 query×key mask 관계를256개 행씩 비교한 불일치는0개**다. FP32 기준은 CUDA/TF32를 사용하지 않는 CPU matmul→정확한 mask bias/scale→softmax→V matmul이다. 메모리 제한을 위해 query는 실제 상대 위치 `[0,1,31,1023,1024,4215,8429]`의7개, head는16개 모두, key는 각 호출 전체를 사용했다. None을 causal로 간주하지 않으며 bool True=허용, float mask는 유한 bias까지 그대로 보존하는 함수를 사용했다.

| 비교 | 상대 L2 | 최대 절대 차이 |
| --- | --- | --- |
| FP32 A(8432 key) 대 B(8430 key) | `4.0653e-8` | `9.5367e-7` |
| **FP32 A/B 모두 동일 유효8430 key 크기로 정렬** | **0, bitwise 일치** | **0** |
| 실제 BF16 A/B, 같은7 query | **0.0290693%** | **0.03125** |
| 실제 A 대 자체 FP32 기준 | 0.140624% | 0.0302382 |
| 실제 B 대 자체 FP32 기준 | 0.140553% | 0.0302372 |

FP32 full-key 대조의 약1e-6 차이도 key 배열 크기를 동일하게 맞추면 사라진다. 표본별 허용 key 수는 A/B 모두 `[1,2,32,1024,1024,1024,1024]`이고, 표본 query에서 padding key 허용은0개다. 표본 FP32를 전체 query 검증이라고 확대하지 않는다. 수치 reduction은 FP64이며 gradient를 재계산하지 않았다.

### 결론과 중단 경계

**첫 layer0 호출의 차이는 causal 누락/유효 mask 의미 변경이 아니라, 같은 논리 연산을 다른 물리 길이·stride로 실행할 때 발생한 BF16 CUTLASS 수치 차이로 좁혀졌다.** 커널 내부의 어느 누산/타일/반올림 연산이 달라졌는지까지는 규명하지 않았다. 또한 이 작은 차이가 전체 모델에서 큰 gradient 차이로 이어지는 정확한 기전이나 안전한 해결책은 이번 첫 호출 진단 범위 밖이다.

따라서 과거 gradient cosine−0.4147·상대L2 111.36%를 무시하거나 허용오차를 늘려 통과시키지 않는다. **padding 제거를 production에 적용하지 않고, 전용 Triton/전체 update 속도 시험은 계속 보류한다.** 이 결과로 Attention 최적화 전체가 불가능하다고 최종 기각하지도 않는다. 학습을 진행할 경우 현재 검증된 **기존 padding8·smartGC ON·batch1×accumulation16·native CE·LoRA dropout0.05** 구성을 유지하는 것이 이번 진단과 양립하는 선택이다. 본 학습 재개는 이번 요청으로 실행하지 않았다.

CPU 검사18건(기존12+실제 mask reference/중단 가드6)이 통과했다. 최초 Codex 검수는 가로채기를 우회할 때 전체 forward까지 진행할 수 있다고 지적했다. 실제 GPU 두 사례는 모두 첫 호출을 가로채 sentinel로 종료했으므로 해당 관측 자체는 유효하다. 이후 코드에는 layer0 attention이 정상 반환하면 강제로 실패하는 별도 가드를 추가하고 다음 layer에 도달하지 않는 CPU 검사를 붙였다. **이 가드 추가 뒤 GPU를 반복하지 않았다.** GPU 프로세스 exit0, 종료 후 compute process0개다. production runner SHA256 `86638bad3614a633b6fa023694f5c7b37ca840e9e37177be0eeff9d6a27e3208`은 그대로다. 데이터·checkpoint·원격 Decision·기존 환경·운영 코드에는 수정이 없다.

최종 Codex(gpt-6-sol/high) 읽기 전용 교차검수는 **조건부 동의·새 중대한 결함 없음**이다. `/tmp/soomac_first_sdpa_result_review.out`에서 실제 인자·FP32/BF16 수치·보류 판단을 확인했고, root도 인용 코드와 JSON을 직접 대조했다. 조건은 정확히 **실제 GPU 캡처 완료, 추가 가드는 CPU에서만 검증**, 내부 CUDA 연산/gradient 증폭 원인 미확정이라는 범위다. 검수자가 GPU/테스트를 독립 재실행한 것으로 보고하지 않는다.

Cause: 같은 유효 QKV와 같은 명시적 mask로도 BF16 SDPA는 물리적 배열 길이/배치에 따라 첫 출력의 미세한 수치 차이를 보였다.
Effect: 첫 호출의 mask 의미 오류를 분리해 배제했지만 전체 gradient 정확성 관문은 미통과이므로, 추가 속도 시험 대신 원래 검증 설정을 유지한다.

