# Decision Unsloth candidate v3 — 원격 TAR 검증

[Claude 검증 지시](/tmp/claude-1000/-home-roma-IRC-SOOMAC3-0/08db5239-b8f8-450f-b4da-3f231477494d/scratchpad/codex_brief_v3_validation.md) · [제작 코드](../tools/unsloth_remote/README.md)
→ **기존 build05-speed 원격 학습은 유지한다. 새 Unsloth TAR는 학습 계산·저장재개·adapter 호환성을 실제 확인한 뒤에만 제출용으로 게시한다.**

## 1. 현재 범위와 이미지

2026-11-01 22:26 KST. 상태: **세 경로 GPU 실행 완료, 학습 계산 동등성 불합격으로 candidate v3 제출 중단, 최종 TAR 미생성**. 원격 작업의 실시간 상태를 조회하거나 중단/수정하지 않았다. Production 시작·기존 checkpoint 삭제/이식·운영 adapter 경로 교체·기존 환경 변경은 하지 않았다.

- 후보: `soomac/decision-unsloth:candidate-20261101-v3`
- 실제 이미지 ID: `sha256:bf465ad8543b8a280f8322980a879fb46cba09d2356ee8d49581c082a95c2d3a`
- 봉인 release: `conversation_r5_20261101_build05_speed`
- 데이터 manifest: `7850948287237c5ace9d1b0f56931419760fbce3f0afa02cea40681acb6ceffa`
- 데이터/학습 정책: D train4360/validation309, NF4 double quant/BF16, max9216, text LoRA r16/alpha32/dropout.05, batch1×acc16, 6epoch. 원본/history/labels 보존.

| 환경 | torch | transformers | Unsloth | PEFT / vLLM |
| --- | --- | --- | --- | --- |
| 실제 candidate v3 | 2.11.0+cu128 | 5.17.0 | 2026.10.10 | PEFT0.21.2 / 내장 vLLM0.26.0 |
| 현재 별도 vLLM 서버 env | 2.13.0 | 5.19.0 | 사용하지 않음 | vLLM0.32.0, PEFT 미설치 |

서버 env 버전은 `importlib.metadata`로 확인했다. 학습 이미지의 vLLM0.26.0을 실제 서빙 버전으로 혼동하지 않는다. 서로 다른 env의 버전을 억지로 같게 바꾸지 않는다. 호환성은 새 PEFT adapter의 설정·tensor key/rank와 실제 서버 로딩으로 판정한다. 아직 새 adapter 로딩 합격은 없다.

## 2. 사용자 시나리오 — 봉인된 실제 train 행 대조

[재현용 검사](../tools/unsloth_remote/audit_scenario.py)는 manifest 및 D/Q/P train SHA, 아래 필드·variation의 동일 정답·canonical compound의 후속 주문/제한과 앞 발화 포함 여부를 assert한다. 최신 영수증: `/tmp/soomac-sealed-scenario-audit-v3-20261101.json`. 전체 데이터의 모든 history 의미가 자동 검수됐다는 주장은 아니다.

| 발화/단계 | 실제 데이터의 정답 | 증거 ID |
| --- | --- | --- |
| 얇은면 보통+느끼한 소스 추천 | D 면 normal + recommendation request. Q는 면이 아니라 크림만 제안. P는 면 선택과 크림 후보를 함께 안내 | `DA31-CORE-separate-T01-D/Q/P` |
| 크림으로 선택할게 바로 담아 | 크림 확정+commit=true, 직전 pending 후보=크림 | `DA31-CORE-compound-T02-D` |
| 양파·버섯 둘 다 low → 버섯만 normal로 바꿔 바로 담아 | 앞 주문 두 재료 low, patch는 버섯 normal만+commit=true. 양파 low 보존 | `DA31-CORE-compound-T04-P`, `T05-D` |
| 게살 알레르기+소시지만 많이 바로 담아줘 | 게살 allergy 추가, 게살 none·소시지 high+commit=true | `DA31-CORE-compound-T07-D` |
| 페퍼론치노 제외+치즈 low+바로 담아 | 치즈 low·페퍼론치노 none+commit=true, 앞 게살 allergy 보존 | `DA31-CORE-compound-T09-D` |

해당 `DA31-CORE` 계열은 **D26/Q1/P10행**이다. separate11 + compound5/polite5/directive5의 D26행이며, 세 복합 variant 각각의 즉시 실행 4행 모두 commit=true이다. 후반 턴은 앞선 실제 선택·변경·제한과 실행/단계 안내가 history에 누적된다. 토큰·target 또는 source를 수정하지 않았다.

Variation 비교에서 감사 코드의 잘못된 가정도 정정했다. episode마다 이벤트 턴 번호가 달라 T05와 T06 등을 숫자로 묶으면 안 된다. 또 directive의 “그렇게 선택하고 바로 담도록 해.”는 크림을 다시 직접 적는 order patch가 아니라 **실제 pending 크림 후보에 confirmation=accept+commit=true**를 주는 정답이다. 두 raw JSON이 같아야 한다는 검사를 쓰지 않고, 이 후보/수락 필드를 독립 확인한 뒤 나머지 정답의 정확한 개수·내용을 대조했다. 원본 데이터는 바꾸지 않았다. 이 정정 뒤 최신 v3 영수증이 통과했으며 최초 두 감사 코드 실패를 데이터 실패로 세지 않는다.

**P의 실행 시작 안내가 별도 학습 행에 없는 것은 의도된 제외다.** [보강 명세](2026-11-01_conversation_supplement_spec.md)의 기존 deterministic 시작 경로를 유지한다는 정책에 따른다. P 학습 행이 10개라는 것을 복합 실행 D 정답 누락으로 세지 않으며, P가 모델로 실행 시작 발화를 학습했다고 주장하지도 않는다. 데이터에 존재한다는 증거와 새 adapter가 실제 시나리오에 성공한다는 증거는 별개다.

## 3. 실제 후보 이미지 계산 대조

RTX PRO4500 Blackwell32GB에서 단일 GPU 프로세스, 컨테이너 RAM32GiB/swap32GiB, CPU8, shm4GiB로 수행한다. NVIDIA Toolkit/daemon은 변경하지 않았다. GPU 장치·현재 host driver를 임시 read-only mount로 연결한다. **host driver는 이미지/TAR에 복사하지 않는다.** 이 로컬32GB 검사를 원격44GB 실측으로 쓰지 않는다.

경로별 새 프로세스에서 같은 Decision8608토큰(마스킹8593+정답15, 추가 padding0), 동일 입력/mask/선택 target hash, 동일 초기 LoRA를 대조했다. 초기328개 lora_B는 전부0이다. dropout0은 진단 전용이며 끝나면 .05로 복원했다. GC flag와 실제 CheckpointFunction.forward48회, backward 완료 gradient 전체를 로컬에 보존했다. native/HF는 CheckpointFunction, smart는 UnslothCheckpointFunction이 관측됐다. autograd의 별도 thread에서 실행되는 backward는 main-thread Python profiler에 안 잡힐 수 있으므로 호출을 관측했다고 창작하지 않는다.

**관측 한계:** compiled 경로의 fused dropout Python callback은 잡히지 않았다.328개 module.p=0 및 복원은 확인했지만, 이를 compiled fused 확률의 직접 관측 합격이라고 쓰지 않는다. 비교기의 `runtime_control_issues`도 이를 별도 미확인으로 남긴다. 아래 **forward loss/logits 불일치는 이 callback 부재와 독립적인 수치 실패**이며, 초기 LoRA B=0이라 LoRA dropout만의 영향으로 설명할 수 없다.

| 지표 / 경로 | HF/PEFT reentrant | Unsloth native GC | Unsloth smart GC |
| --- | --- | --- | --- |
| GPU forward/backward 실행 | 2회 완료 | 2회 완료 | 2회 완료 |
| loss A / A2 | 3.917093 / 3.917093 | 4.086672 / 4.086672 | 4.086672 / 4.086672 |
| 같은 경로 gradient 상대L2 | 1.362% | 2.259% | 1.759% |
| HF 대비 gradient cosine, 4쌍 | 기준 | −0.0824~−0.0797 | −0.0781~−0.0748 |
| HF 대비 gradient 상대L2, 4쌍 | 기준 | 130.44~130.97% | 129.75~130.42% |
| HF 대비 logits 상대L2 | 기준 | 2.656% | 2.656% |
| 층별 기준 초과 | 기준 | 48/48층 | 48/48층 |
| peak reserved VRAM | 21.81GiB | 13.92GiB | 11.20GiB |
| process peak CPU RSS | 23.38GiB | 24.02GiB | 24.08GiB |
| 판정 | 대조 기준 실행 완료 | 동등성 불합격 | 동등성 불합격 |

서로 다른 loss의 절대 차이는0.1695787907이고, 각 경로 안 loss/logits A/A 차이는0이다. loss normalization의 native CE 재계산 오차는 약1e-6 이하여서 이 차이를 CE 스칼라 정규화만의 문제로 설명하지 않는다. 동일 초기 LoRA 파일 SHA는 `76907a2c8a52ecb5ed92fd7850fb01d75b5f3c27f2fe5d239e7cf78b61585ddf`이다.

영수증: `/tmp/soomac-candidate-v3-audit-runtime-20261101/GC_EQUIVALENCE.json`, status=`GC_EQUIVALENCE_BLOCKED`, selected=null. 원시 gradient/logits는 이 경로에만 보존했고 외부 검수로 보내지 않았다. 전 경로 exit0 후 Docker/GPU compute 목록이 비었음을 확인했다. **optimizer update 속도 시험·새 adapter export·vLLM 로딩·full-state 재개·final TAR는 이번 후보에서 진행하지 않는다.**

HF/native/smart의 loss·logits·전체 및48층 LoRA gradient norm/cosine/relativeL2를 같은 경로 A/A 변동과 비교했다. 허용오차를 늘리지 않았다. **최종 원인은 아직 모른다.** 같은 환경에서 두 Unsloth GC 모드의 forward loss/logits가 서로 같으므로, smart GC 단독 원인으로 결론내리지 않는다. 이 한 장문 반례는 candidate v3의 기준 동등성 합격을 차단하지만, Unsloth 전체가 학습 불가능하다거나 기존 원격 학습이 잘못됐다는 뜻은 아니다.

기존 host 환경 Response warm 측정은 233.358/231.685/231.647초, 평균232.230초/update이다(8430×16, warmup2/timed3). 이것은 다른 torch/transformers/Unsloth 환경의 속도 증거이며 **candidate v3 Decision 속도·학습 정확성 증거로 전용하지 않는다**.

## 4. 중단 결과와 이후 재개 조건

1. **Claude brief의 ‘둘 다 어긋나면 중단·보고’ 조건에 도달했다.** 현재 candidate v3의 제출용 TAR 생성은 중단한다. 다음 국소 조사는 HF↔Unsloth의 forward 차이이며, 데이터 전수검수·GC/batch/padding/Attention 전체 행렬을 다시 반복하지 않는다. 새 수정 후보가 통과하기 전에는 기존 원격 Decision을 교체하지 않는다.
2. 통과 경로에서 dropout.05, batch1×acc16 warmup1+timed2 update, VRAM/RSS, adapter export를 측정한다. 기존186.37초와도 동일 환경 warm 대조가 아니므로 단정하지 않는다.
3. 새 adapter의 r/alpha/target_modules/use_rslora/use_dora/modules_to_save 및 safetensors key를 운영본과 대조한다. 기존 vLLM 변환기의 OUT을 새 staging에만 지정하고 글로벌8층 K→V16 tensor 복제를 확인한다. 운영 폴더 swap/main은 호출하지 않는다. 실제 vLLM0.32 새 adapter 로딩은 아직 미완료다.
4. 동일 이미지/기본 entrypoint에서 짧은 저장→의도적 실패→출력 복제→full-state 재개, 원본 checkpoint 해시 보존을 검사한다. 이 네 관문을 통과하기 전 final PASS 영수증을 만들지 않는다.
5. 통과 시 final 이미지는 검증 candidate의 layer extension으로만 만들고, 최종 이미지 재검사 후 새 이름 `docker image save` TAR+SHA를 게시한다. 기존 build05-speed TAR와 실행 중인 원격 작업은 그대로 둔다.

원격 새 시험의 RAM16GiB는 이 진단의 RSS보다 작다. 최소32GiB 요청을 다음 통합 시험의 기준으로 잡되, 전체 데이터 인코딩·실제 entrypoint 학습의 RAM 보장은 아직 하지 않는다. 현재 원격 자원 설정은 수정하지 않았다.

## 5. 검증 코드 국소 보강·재사용

`validate_candidate.py`는 GPU lock 획득 후 신규 출력 폴더를 만들고 actual GC/zero module dropout/restoration을 기록한다. `local_candidate_driver.py --root`로 이전 실패 증거를 덮어쓰지 않는 새 시도를 선택한다. `compare_candidate.py`는 누락된 runtime 증거를 합격으로 바꾸지 않고 수치 실패도 보존하며, 전체48층 FP64 대조 및 원시 증거 SHA를 결속한다.

Codex의 timing 게이트 결속 누락 지적을 반영해 `candidate_proof.py`에18개 증거 파일 SHA·입력·초기값·원래 report identity 결속을 추가했다. 후속 독립 검수는 **STOP 판정에 동의**했지만, SHA 결속만으로 proof의 수치 판정을 재계산하지는 못한다고 지적했다. 따라서 **현재 `verify_timing_proof`는 무조건 차단한다.** 실패 영수증의 status만 고쳐 timing에 들어갈 수 없다. 완전한 raw FP64 수치 재계산 관문을 통합·검수하기 전 이 잠금을 해제하지 않는다. 결속 helper는 CPU bookkeeping용일 뿐 실행 승인 함수가 아니다.

같은 검수의 native checkpoint 이름 부분문자열 지적은 정확히 `CheckpointFunction.forward`를 요구하도록 교정했고, smart trace가 native로 통과하지 못하는 CPU 회귀를 추가했다. 후속 history/게살 제한 자동 단언도 보강해 최신 별도 영수증에 재검사했다. **현재 CPU21건 및 py_compile/diff check 통과**, 마지막 국소 교정 뒤 GPU 재실행은 하지 않았다. STOP 검수 로그: `/tmp/soomac_candidate_v3_final_gate_review.out` (CLI header gpt-6-sol/high 확인).

GPU 프로브3개는 source SHA `abb2eb106d5a27a4ab998dd8b735a476a2654b0602542ff97b65136dae577c36`인 동일 버전으로 실행됐다. 이후 timing-only 잠금·CPU predicate 변경을 GPU에 재검증했다고 확대하지 않는다. 이전 `/tmp/soomac-candidate-v3-audit-20261101`은 탐색 기록으로 보존하며, 현재 판정의 권위 있는 경로는 `...audit-runtime-20261101`이다. 두 묶음을 섞거나 이미 끝난 GPU 행렬을 반복하지 않는다.

Cause: 실행 가능한 Unsloth와 기존 학습 계산이 동등한 Unsloth는 같은 조건이 아니다.
Effect: 빨리 도는 이미지라는 이유만으로 잘못 학습될 수 있는 TAR를 원격에 제출하지 않는다.
