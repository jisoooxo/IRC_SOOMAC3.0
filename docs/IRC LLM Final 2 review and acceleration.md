# IRC LLM Final 2 — 문제 대조·few-shot 검수·추론 가속

**2026-11-01 최신 후속:** D/P prompt·대조 예시 v2 적용과 CPU834건/하위6건, 기존 adapter20회 A/B 뒤 [전체17턴 누적 계측](2026-11-01_llm_scenario_regression.md)을 완료했다. 사용자 요청대로 실패해도 후속 발화를 모두 입력했으며 소스null·실행0·실제section=noodle 유지의 연쇄 영향을 기록했다. 자연스러운 크림 제안 자체는 유지할 요구다. [전체 출력 독립 검수](2026-11-01_scenario_output_review.md)와 계측을 보고 사용자가 수락을 결정하며 추가 코드/학습 변경을 자동 시작하지 않는다. 아래 조사 당시 예시23개·prompt미수정 등의 문구는 그때의 근거이며 최신 D18/Q4/Task9/General1/Mixed1=33개와 구분한다.

[데이터 감사 결과](../training/learning_first_v43/final/goals/GL10/data_reaudit_20261031/RESULT.md) · [잔여 계획](<IRC LLM Final remaining work.md>) · [최신 로그 분석](../training/learning_first_v43/final/goals/GL10/dataset_supplement_plan_20261031/COMMIT_QUANTITY_LATENCY_213138.ko.md)
→ **데이터를 정제한 것과 현재 모델의 대화·실행·속도가 개선된 것은 구분한다.**

작성일: 2026-11-01 KST. 최신 상태: **조사 뒤 사용자 승인으로 pending Runtime 두 분기 수정·LLM 테스트812건 통과, GPU 벤치마크 미실행**. [수정 상세와 검증 기록](2026-11-01_llm_pending_runtime_fix.md)을 함께 읽는다. 이 문서를 먼저 생성한 뒤 사용자 지정 `llm_final_2` 브랜치를 만들었다. 이후 사용자가 직접 commit/push했다. 확인 시점 local HEAD와 `origin/llm_final_2`는 `3798a0e012c1e9165c9c78694b9bfcf98794469b`로 같았다. 이 일치는 로컬 원격추적 ref 기준이며 별도 원격 서버 재조회는 하지 않았다. 현재 문서와 코드의 후속 수정은 그 커밋 이후 미커밋 변경이다.

## 1. 작업 기준과 보존

출발 브랜치 `llm_final`, 기준 커밋 `8cd8bef6ff904b5cb620696b1ff15e0b50b6ce7c`. 기존 미커밋 변경이 있으므로 전부 보존한다. 브랜치 생성은 그 변경을 커밋·백업하는 작업이 아니다. 원본 branch commit, 학습 결과, adapter, ROS topic, history, busy STT, 사용자 임시 고정 안내를 유지한다. VLM 프롬프트·판정·관련 사용자 변경은 수정하지 않는다.

최초 승인 범위는 문서 작성, 새 브랜치 생성, 실제 코드·로그·정제 산출물 읽기와 공식 가속 자료 조사·검수였다. 이후 사용자가 Python 수정과 관련 로직 모순 교정을 승인하여 §2의 두 Runtime 분기를 고쳤다. prompt 변경·라이브러리 설치·GPU 모델 실험·학습·운영교체·ROS·commit/push는 이번에 하지 않았다.

브랜치 생성 직전/직후 `git diff --binary` SHA는 `f5591e232c5cc6aacd22f80afa6e97f591b2acb1db930f614c6dae6e66f448e3`, short status SHA는 `3da71b3e57b83be86672c6ad09aad7ec10a89a3601a697af0dfc71f61e4f7e81`로 동일했다. 기준 `llm_final` SHA도 전환 중 불변이었다. 이후의 사용자 commit은 별도 이력이다.

`git check-ignore`로 `training/learning_first_v43/final/goals/GL10/data_reaudit_20261031/RESULT.md`가 ignore 대상임을 확인했다. 저장소 branch push가 이 로컬 감사/학습 자료 전체를 보관한다는 뜻은 아니다. 기존 의도된 학습 작업장 분리는 유지하며 이번에 ignore를 해제하거나 큰 산출물을 Git에 넣지 않았다. 로컬 링크3개는 존재를 확인했고 문서 whitespace 검사는 통과했다.

## 2. 사용자가 요구한 기능과 로그 문제 대조

목표는 자연스러운 직원형 대화이다. 앞선 여러 사용자 발화와 최신 state를 함께 해석하고, 선택·변경·취향·제약·추천 수락·즉시 실행을 동시에 반영한다. 현재 물리 실행 가능 범위와 미래 추천 범위를 혼동하지 않는다. 실제 완료 전 완료 발화, 없는 pending 확인 질문, 언급하지 않은 기존량 변경은 허용하지 않는다.

검토 대상: 크림 선택+바로 담아, 버섯 normal 변경+양파 low 보존+바로 담아, 게살 allergy+소시지 high+바로 담아, 페퍼론치노 제외+치즈 low+바로 담아. 일반 선택과 부정·인용·조건부 실행은 별도 대조한다.

### 문제 → 근거 → 개선 상태

| 사용자 체감 문제 | 실제 최초 실패 지점 | 이번에 한 것 | 아직 안 한 것 |
| --- | --- | --- | --- |
| “선택하고 바로 담아”인데 다시 묻거나 안 담음 | 21:03 T2 D가 commit만 내고 크림 후보 수락/선택을 누락. 21:31 T4는 버섯 normal만 내고 commit 누락 | 후보 수락+commit, 주문 변경+commit을 같은 턴에 갖춘 연결 시연을 작성·검수 | 새 모델 학습·누적 GPU 회귀. Python 단어 trigger로 commit 보충하지 않음 |
| 양을 바꿨는데 “이미 보통”이라고 말함 | 21:03 T6 state는 low→normal로 바뀌었지만 P가 변경 시점을 잘못 설명. 별도17:39 T11에는 실제 high를 normal이라고 말한 값 오류도 있음 | 값 오류와 변경 사실 오류를 분리. low/normal/high 전후 대조 및 언급하지 않은 양파량 보존 검수 | P 모델이 실제로 정확히 설명하는지. 새 before field를 무조건 추가하지 않음 |
| allergy 누락·없는 실행 확인·불필요 추천 | 20:01 T6 D가 allergy를 preference로 기록. 21:31 T8 pending=null인데 P가 실행 질문을 만듦 | allergy+명시 주문+commit, 후보/확정/실행/완료 구분, 정상 일반 선택의 확인 질문을 오류로 오인하지 않도록 검수 | 현 모델 오류 제거의 실증. 최신 모든 STT 오인식·warning 변형을 무오류라고 보장하지 않음 |
| 매운 걸 못 먹는데 면만 추천, 현재 단계에만 갇힘 | 이전 Q 출력의 소스 누락과 scope 제한을 분리해서 조사. 현재 Q prompt는 남은 코스/초기 소스 고려 규칙이 있으나 과거 모델 실패는 남음 | 현재 Q 입력 builder 차이3행을 후보에서 이관. 느끼한 소스→크림 및 미래 범위 대조 보강 | 새 통합본 학습·실제 전체코스 추천 성공, 상단 범위 토글 구현 여부를 별도 계약으로 확정 |
| 모델이 이해해도 코드 때문에 막히거나 선택이 사라짐 | 수정 전 `llm_langgraph.py:907~932` reject 조기 반환이 별도 현재 commit을 막음. `:471~474` preselected 실행 거절이 주문까지 제거 | **승인 후 두 분기 수정.** 실행 유예 시 확정 주문 보존, 추천/제한 철회 거절+독립 commit은 기존 안전 검사로 전달. 새 반례24건과 기존 회귀 통과 | 실제 모델 누적 회귀는 아직. 위임 추천 실행 계약HOLD12는 미해결. 과거 줄 번호는 수정 전 기준 |

필수 시연의 기대 정답은 모두 새 데이터에 들어갔다. 다만 임시 단계 고정 안내는 현재 Q/P를 호출하지 않으므로 “아까 매운 걸 못 드신다고…” 같은 자발적 모델 단계 안내가 복원됐다고 말할 수 없다. 섹션과 무관하게 대화·후보를 다루는 것과 이미 담은 재료를 물리적으로 다시 빼는 것은 다르다. 후자는 기존 안전 검증을 유지한다.

Cause: D의 의미 누락, Q의 후보 문제, P의 사실 설명 오류, Python의 두 계약 결함이 함께 존재한다.

Effect: “전부 데이터 탓” 또는 “Python만 풀면 해결”로 묶지 않고 고칠 위치와 검증을 나눴다.

## 3. 데이터·계약·프롬프트·few-shot 상태

직전 감사는 원본7784행의 장부 연결/불변 의미 판정 재사용/영향 검수 및 새10대화104행의 독립 의미 검수이다. 비활성 고유93행·원본대체108행·계약HOLD12행이 저장됐으며 새 학습본 조립/학습은 아니다.

현재 `agent_prompts.py`는 `render_examples()`를 실제 system에 붙이고 D/Q/P wrapper는 해당 system을 사용한다. 예시 수는 **D11/Q4/Task6/General1/Mixed1 = 총23개**다. Python이 예시를 선택해 정해진 답을 보내는 방식이 아니라 모델이 system 안의 예시를 읽는다. 실제 few-shot input/output와 user history·state를 읽어 최신 정책과 대조했다. 이번 문서 검수는 독립 모델의 새 교차검수가 아니라 주담당의 읽기 전용 코드·자료 대조다.

### 무엇을 검수했고 무엇을 실제로 만들었나

원본7,784행은 구조·해시·장부 연결을 확인하고 이전 불변 의미 판정은 재사용했다. 전행을 매번 새로 읽었다는 뜻이 아니다. D 영향19행, Q 새47행 및 새 수량정책 영향행, P 영향90행을 구분했다. 새10대화88기록/104개 D/Q/P 호출은 독립 의미검수, 실제 graph/builder 정답 callback 연결, 학습 mask/종료토큰/8192길이 검사를 했다. duplicate11개는 alias로만 남겼고 고유93개만 보강 후보로 저장했다. 원본대체108개는 D commit누락4 + Q수량/scope104이며 D계약HOLD12는 따로 격리했다. 원본·운영모델은 바꾸지 않았다.

이는 **좋은 정답과 연결된 학습 후보를 만든 결과**다. 모델이 그 정답을 생성했거나 새 가중치가 학습됐다는 결과가 아니다. 가상 통합 뷰의 exact 입력 충돌0은 의미적으로 비슷한 시연 간 누수0이나 정량 편향 합격을 뜻하지 않는다. 수치 편향 판정은 사용자 지시대로 보류했다.

### 현재 prompt/few-shot의 국소 검수 결과

| 항목 | 확인된 내용 | 판정과 필요한 보완 |
| --- | --- | --- |
| D의 주요 규칙 | 복합 제한+추천, 못 먹어=cannot_eat, 기존량 보존, 새 선택normal, 실제 history 기반 나머지, 명시 즉시 실행 규칙이 존재 | “필드 규칙을 아예 안 넣었다”는 진단은 아님. 실제 모델 준수는 별도. 미수락 후보량/기존량을 덮지 않는 예시는 유지 |
| D few-shot | D04는6개 history message를 참조한 취소+선택+commit. N02는 기존 주문 즉시 실행. D09는 제한+추천 | 최신 allergy+주문량+commit, 추천 후보 수락+변경+commit과 부정/인용의 짝 예시는 더 직접적으로 보완해야 함. 새93행이 자동으로 few-shot이 된 것은 아님 |
| Q few-shot 수량 | R01의 소시지high에는 명시적 소시지량 근거가 없음. R06의 치즈low도 치즈 선호만 있고 양 근거는 없음 | **최신 “근거 없는 새 추천량만normal” 정책과 정합성 보완 필요.** R02 양파low도 담백함을 특정 수량 근거로 허용할지 명시하고 정렬해야 함. 기존 면low는 다른 재료량의 근거가 아님 |
| Q system과 P system | Q는 남은코스·초기소스 고려, P는 없는 실행pending 질문 금지·실제 변경/완료 구분·근거 있는 권유 규칙이 있음 | 최신 Q 제안량 정책은 system에 직접 명문화되지 않음. “이번에 새로 바꿈” vs “원래 같은 값 유지” 대조를 P 예시에도 추가/교체해야 함 |
| 길이·예시 오검수 방지 | 전체 system 문자 수 D15614/Q9314/Task15978. N01 “버섯 넣어줘”의 high는 기존 확정 버섯high를 보존한 정상 예시 | history를 자르거나 모든 양을normal로 덮지 않는다. 전체 JSON의 중복 구조만 최소화하는 후보를 실제 serializer/회귀로 검증한다. 문자가 곧 토큰 수는 아님 |

현재 few-shot에는2~6개의 선행 message가 있는 예시가 있다. 이것만으로 사용자가 요구한 여러 단계의 복합 멀티턴을 완벽히 학습했다고 할 수 없다. 새93행에는 전체 누적 흐름과 기억 취향 대조가 있지만 아직 미학습/비활성이다. **“few-shot은 있다. 그러나 최신 정책까지 완전히 정렬되지는 않았다”가 현재 결론**이다.

위 예시 수정 시 하나의 정답만 바꾸지 않는다. system 규칙·few-shot·신규/기존 학습 후보·builder·평가기 oracle을 같은 계약 버전에 결속하고, 기존 승인 의미가 바뀐 행만 재개한다. 추천을 항상 크림/소시지로 고정하거나 광범위 추천에 현재 섹션만 쓰는 편향을 다시 만들지 않는다. 소스 크림은 이 제한 없는 느끼한 소스 시연의 답이지 모든 비매운 요청의 유일 정답이 아니다.

## 4. 실제 추론 설정과 속도 근거

호스트 읽기로 확인한 GPU는 NVIDIA RTX PRO 4500 Blackwell, VRAM32623MiB, compute capability12.0(SM120), driver580.178.04이다. Gemma4Unified12B config는 local head_dim256/global_head_dim512와 sliding/full attention 혼합이다. B200 SM100용 성능·지원표를 그대로 적용하지 않는다.

현재 경로는 Transformers+PEFT 공유 Base/3개 LoRA, NF4 저장·BF16 계산, `attn_implementation="sdpa"`, 생성 내부 KV cache 사용, greedy/비thinking/XGrammar다. 코드에서 요청 간 prefix KV 재사용이나 compile/static cache 설정은 확인되지 않았다. SDPA가 실제 어떤 GPU 커널을 쓰는지는 프로파일 전 미확인이다.

환경 metadata: torch2.12.1+cu130, transformers5.13.1, peft0.19.1, bitsandbytes0.49.2, triton3.7.1. flash-attn/flash-attn-4/FlashInfer/vLLM/TensorRT-LLM 배포 패키지는 해당 gemma4_env에서 미설치이다. 이를 근거로 PyTorch 내장 fused attention도 없다고 단정하지 않는다.

최초 sandbox의 nvidia-smi 실패는 호스트 읽기로 해소했다. 드라이버 고장으로 오진하거나 설치/재시작하지 않았다. 버전 metadata는 gemma4_env의 설치본 기준이며 실행 중 프로세스가 다른 환경을 사용했는지는 기동 identity와 함께 확인해야 한다.

### 최신 실제 로그가 말하는 속도

19:58~21:31의8세션45사용자턴 분석 근거를 재사용했다. 아래는 STT 시작부터 TTS 완료까지가 아니라 **계측된 사용자 턴 처리 시간**이다.

| 호출 경로 | 개수 | 중앙값 |
| --- | ---: | ---: |
| D만 호출, 실행 시작 고정 안내 | 18 | 4.23초 |
| D→P | 22 | 9.26초 |
| D→Q→P | 5 | 15.26초 |

입력은 D6209~6847/Q3832~3835/P5839~6923토큰, 출력은 D10~47/Q56~58/P20~35토큰이었다. 짧은 응답도 긴 입력을 다시 처리한다. wrapper 시간에는 토큰화·전송·adapter 전환·generate·decode/검증이 모두 들어가므로 아직 “GPU attention이 몇% 병목”이라고 단정할 수 없다.

“버섯만 보통으로 바꾸고 바로 담아”에서 commit누락→재질문→추가실행요청의 실제 두 턴 처리 합은13.131초였다. 의도를 한 번에 맞추면 불필요한 P 호출/사용자 한 턴을 줄일 수 있다. **수정 후 몇 초인지는 미측정**이다. 이45턴에서 Decision repair 재호출은0이므로 이번 지연을 repair반복 탓으로 설명하지 않는다. max_new_tokens1024/512는 상한이며 실제 출력이 이미 수십 토큰이므로 상한만 낮추면 비례해서 빨라진다는 주장은 틀리다.

## 5. 가속 대안과 후속 검증

**사용자 추가 확정(2026-11-01): FP8은 가속 후보에서 제외한다.** 이전 시험에서 더 느렸다는 사용자 보고를 이유로 유지하며 이번에 다시 측정하지 않았다. 순서는 Python 문제 분리·수정 → 1번 정확한 복합 의도/계약·prompt 정렬 → adapter별 prefix cache 검토다. 최초 브리핑 뒤 승인된 Python 두 분기는 수정했고, prompt·가속 구현이나 학습은 시작하지 않았다.

공식 자료를 확인해 다음을 비교한다: 실제 prefill/decode/adapter 전환 계측, prompt 중복 정리와 adapter별 prefix cache, SDPA/FlashAttention/FlashInfer의 SM120·head512 지원, vLLM Gemma4Unified+LoRA/구조화 출력 호환, BF16/NVFP4와 기존 NF4의 품질·VRAM·가속 차이. 예상 배속을 실제 측정값처럼 쓰지 않는다. 최신 라이브러리 설치만으로 성공을 주장하지 않는다.

최종 문서는 문제→근거→이번 개선→아직 미해결→추천 다음 단계와 성공/실패 기준을 연결한다. 실제 환경 변경은 별도 승인 후 수행한다.

### 공식 자료 대조에 따른 가속 우선순위

| 순위·대안 | 왜 후보인가 | 적용 경계·판정 |
| --- | --- | --- |
| 1. 정확한 복합 실행 + prompt 중복 정리 | 불필요한 P호출/한턴을 없애고, 현재 긴 system·예시의 중복 전처리를 줄이는 후보 | history/state/안전 규칙을 날리지 않음. 동결된23예시 중 영향 예시만 교정. prompt 축약은 실제 품질 A/B 전 가설 |
| 2. adapter별 prefix cache + 전용 서빙 | 같은 Agent의 system/few-shot prefix를 매번 계산하지 않는 후보. vLLM에는 Gemma4Unified 및 상속된 LoRA 경로가 문서화되어 있음 | [모델 구현](https://docs.vllm.ai/en/stable/api/vllm/model_executor/models/gemma4_unified/)과 [prefix 설계](https://docs.vllm.ai/en/latest/design/prefix_caching/) 확인. D/Q/P는 다른 adapter이므로 KV를 서로 섞지 않음. prefix key에 LoRA identity·토큰·모델revision 필요. 실제3adapter 로드/구조화JSON/템플릿 지원은 별도 smoke 통과가 필요 |
| 3. 현재 SDPA 커널 확인 → FlashAttention/FlashInfer 후보 | 현재 SDPA도 이미 fused attention을 자동 선택할 수 있음. 설치된 별도 flash-attn이 없다고 attention이 전부 느린 구현이라는 뜻은 아님 | [PyTorch2.12 SDPA](https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.scaled_dot_product_attention.html). GPU trace로 실제 kernel/형상 확인. [FA4 공식 코드](https://github.com/Dao-AILab/flash-attention)는 SM120 경로도 있지만 **B200 SM100과 다른 구현/공유메모리 제약**이 있음. local256/global512·sliding/full·GQA·mask별 검증 없이 강제하지 않음 |
| 4. BF16/NVFP4 비교, FP8 제외 | NF4는 메모리 절약 경로이지 Blackwell native FP4 tensor 연산을 쓴다는 증거가 아님. 현재는 NF4+BF16계산. NVFP4 전용 GEMM은 별도 후보 | [NVIDIA NVFP4](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)와 [ModelOpt 서빙](https://docs.vllm.ai/en/latest/features/quantization/modelopt/) 확인. NF4→NVFP4는 이름/flag 교체가 아니라 원본에서 별도 양자화·교정·export가 필요. Gemma4Unified12B+SM120+3LoRA의 정확한 조합은 미시험. FP4 fallback이면 기대한 native가속이 아닐 수 있음 |
| 5. compile/static cache/CUDA graph 또는 더 큰 하드웨어 | 커널 launch·Python overhead가 큰 것으로 실측되면 후보. 32GB BF16은12B가중치만 대략24GB(십진)이고 KV/LoRA/작업공간/디스플레이 여유가 별도라 사전 peak 측정 필요 | PEFT adapter변경·가변입력·XGrammar와의 graph break/재컴파일을 확인.3Base복제·3프로세스병렬은 현재 VRAM에서 기본 권장 아님.96GB GPU가 무조건 빠르다는 결론도 아님. 장치명/대역폭/실할당/동일입력으로 비교해야 함 |

**FlashAttention을 우선 “설치해서 ON” 하는 것보다, 현재 커널 측정 + adapter별 prefix 캐시 검토가 먼저다.** [SM120 공식 forward 코드](https://raw.githubusercontent.com/Dao-AILab/flash-attention/main/flash_attn/cute/flash_fwd_sm120.py)는 SM80계열 MMA를 재사용하고 shared-memory 조건을 별도로 검사한다. “Blackwell 최적화”라는 같은 이름 아래에서도 SM100/B200의 발표 배속을 RTX PRO4500에 옮길 수 없다. global head512가 모든 backend에서 지원된다고도, 전부 불가능하다고도 미리 단정하지 않는다.

NVFP4는 attention만이 아니라 Linear/GEMM 가속 선택이다. attention backend 변경과 정밀도 변경은 처음에는 각각 따로 시험한다. 기존 LoRA를 다른 양자화 Base에 붙였을 때의 commit/수량/allergy 품질도 다시 확인한다. 세 adapter를 각각 Base에 merge하면 각각 다른 가중치가 되므로, 현재 공유Base/3LoRA 구조와 VRAM 요구가 달라진다. 기존 학습 결과는 보존하고 변환 산출물은 별도 후보에 만든다. 지금은 변환하지 않았다.

### adapter별 prefix cache의 정확한 뜻

현재 생성 함수의 `use_cache=True`는 한 번의 생성 안에서 이전 토큰 계산을 재사용한다. 호출이 끝난 뒤 다음 턴에서 동일한 system/few-shot의 KV를 찾아 재사용하는 prefix cache가 구현됐다는 뜻은 아니다. 현재 wrapper는 전체 messages를 다시 만들어 `generate()`에 전달하고, 호출 간 `past_key_values` 보존/주입은 없다.

prefix cache는 **이전 답을 재사용하는 응답 캐시가 아니라 같은 입력 앞부분의 계산 결과를 재사용하는 것**이다. D의 긴 규칙·few-shot을 첫 D호출에서 읽어 둔 뒤 같은 D adapter의 다음 호출에서 앞부분이 일치하면 그만큼 재계산을 줄인다. 새 사용자 말/history/state는 계속 읽는다. 앞부분이 달라진 이후는 새로 계산하므로 이전 주문 답을 그대로 재생하지 않는다. 캐시는 VRAM을 사용하며 용량/eviction에 따라 hit 여부가 달라진다.

```text
Decision:       동일 D 가중치·D system/few-shot prefix → D 전용 KV 재사용
Recommendation: 동일 Q 가중치·Q system/few-shot prefix → Q 전용 KV 재사용
Response:       동일 P 가중치·같은 route의 system prefix → 해당 prefix KV 재사용
```

일반 LoRA에서는 가중치가 달라지면 같은 문자라도 K/V가 달라질 수 있으므로 D의 KV를 Q/P에 그대로 넘기지 않는다. Response의 task/general/mixed system이 다르면 이들도 동일 prefix라고 가정하지 않는다. 모델revision/양자화/adapter 내용이나 실제 prefix가 바뀌면 캐시를 분리/무효화한다. [vLLM 설계](https://docs.vllm.ai/en/latest/design/prefix_caching/)도 LoRA identity를 cache hash의 일부로 다룬다. 같은 Base를3번 복제해야 하는 최적화는 아니다. 재학습이 선행 조건도 아니다.

Cause: 매 턴의 긴 고정 입력을 반복 계산하는 비용이 발생할 수 있다. 실제 prefill 비중은 아직 미측정이다.

Effect: history를 삭제하지 않고 같은 부분의 재계산을 줄이는 후보가 생긴다. 의미 오류를 고치거나 출력 decode 시간을 전부 없애는 기능은 아니다.

### FlashAttention/NVFP4 때문에 LoRA를 다시 만들어야 하나

| 변경 | 기존 LoRA의 처리 | 필수 확인 |
| --- | --- | --- |
| FlashAttention으로 attention 구현만 변경 | **원칙적으로 재학습 불필요.** 같은 Base/층/adapter 구조라면 기존 LoRA를 유지하는 경로로 시험 | kernel의 SM120·head256/512·mask 지원, 수치 차이와 실제 누적 회귀. attention kernel은 LoRA 학습 목적을 새로 정의하는 것이 아님 |
| 기존 NF4 Base → 별도 NVFP4 Base | **NVFP4 전용 LoRA를 무조건 새로 학습하는 것은 아님.** 같은 원본 모델 구조의 adapter를 지원하는 엔진이면 기존 adapter 재사용을 먼저 검증 | 그 엔진이 해당 NVFP4 Linear와 LoRA를 함께 지원하는지, 모듈이름/shape/rank/scaling·원본revision·tokenizer 일치, 새 양자화 오차에 따른 품질 변화 |
| NVFP4+동적 LoRA 조합이 미지원 | 새 LoRA를 학습한다고 엔진 미지원이 해결되지는 않음 | 현재 NF4 경로 유지 또는 다른 지원 엔진 검토. 각 adapter를 원본Base에 merge후 각각 양자화하는 별도 배포 방식은 가능성 검토 대상이나,3개 Base/저장량·전환 비용이 달라 기본 권장 경로가 아님 |

LoRA는 원래 가중치에 저랭크 변경량을 추가하는 방식이라는 [PEFT 설명](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora)에 근거한다. 위 “재학습 불필요/재사용 가능”은 구조와 지원 엔진이 맞을 때의 원칙이지 Gemma4Unified12B+SM120+현재3adapter 조합의 실제 로드/성능 인증이 아니다. NVFP4 checkpoint 포맷 지원과 동적 LoRA 지원의 조합도 각각 확인해야 한다. 공식 문서만으로 이 정확한 조합의 지원/품질은 확정하지 못했다.

**의미 실패를 고치기 위한 재학습과 가속 backend를 바꾸기 위한 변환/검증은 별개**다. 지금 모델의 복합 의도·allergy·P 사실 오류 때문에 새 학습이 필요할 수는 있지만, FlashAttention이라는 이유만으로 다시 학습해야 하는 것은 아니다. 기존 NF4 weight를 이름만 NVFP4로 바꾸거나 무검증 재양자화하지 않는다.

### 다음 구현 묶음 — 중복 감사 없이 순차 진행

1. **계약·prompt 정렬:** 최신 제안량 정책 명문화, R01/R02/R06 국소 판정/교정, 직접적인 복합 수락+commit 및 P 변경사실 대조 few-shot 후보. 기존량보존 N01 같은 정상 예시는 유지한다. code/prompt와 새 학습본의 계약을 함께 동결한다.
2. **Runtime2건 수정 완료 / 위임실행 계약 후속:** 사용자 승인 후 CPU 반례 before17실패/7통과→최소 수정→LLM13파일812통과/하위6통과. [수정 기록](2026-11-01_llm_pending_runtime_fix.md) 참조. 계약HOLD12는 미해결이며 말이 모호하다는 이유로 없애지 않는다. 실제 모델 성공·전체 계약 완료와 혼동하지 않는다.
3. **같은 고정 스택의 계측:** 동일 입력/가중치/양자화/템플릿/grammar로 tokenize·입력전송·adapter전환·prefill(TTFT)·decode·검증 및 end-to-end를 나눈다. CPU 벽시계와 필요한 GPU event/sync 경계를 구분한다. 기기전력/클럭/온도/VRAM/OOM도 기록한다.
4. **성능 후보를 한 변수씩 비교:** 같은10대화의 실제 모델 close-loop, history/state counterfactual와 부정/인용 대조를 캐시 cold/warm으로 비교한다. 원시토큰/의미 필드/실행계획 동일성, p50/p95, peakVRAM, 로드/워밍업을 기록한다. 최소 반복 횟수는 실험 전 고정하고 표본 부족 p95를 과신하지 않는다. GPU모델시험/설치 승인은 별도다.
5. **데이터 조립·필요 학습·최종 재검증:** 비활성93행+108대체/12HOLD를 기존split에 결속하고 정책으로 달라진 few-shot과 의미 충돌을 검사한다. 필요한 Agent만 학습한다. 이전 alignment139행은 자동병합하지 않는다. 새 모델의 실제 누적 시연과 unseen변형/안전/언어 검증을 마친 뒤 운영 채택한다.

실험 통과 조건은 **의도된 주문·양·제약·수락·commit·실제 실행허가·발화가 맞고, 기존 안전 회귀가 나빠지지 않으며, 같은 조건의 지연이 줄어드는 것**이다. 새로 틀린 수량·무승인 실행·제약 누락·허위 완료가 나오면 속도가 빨라도 채택하지 않는다. 정확성 시험에서 막힌 이후 gold를 주입한 것은 후속 Agent 진단이며 전체 자연 성공으로 세지 않는다. 실제 로그분석45턴이나 CPUgold10대화를 새 GPU 모델 성적으로 바꾸지 않는다.

### 이번 조사에서 확정하지 않은 것

실제 kernel 이름, prefill/decode별 시간 비율, 각 가속 후보의 배속, BF16 최대길이 fit 여부, NVFP4의 해당 모델+adapter 지원/품질, 현재 모델의 전체 자연시연 성공은 **미측정**이다. FP8은사용자지시로후속실험에서제외했다. LLM Runtime 두 분기만 승인 후 수정했으며 가중치·라이브러리는 바꾸지 않았다. 최신 source코드의 지원 흔적은 설치된 버전/실제 지원조합의 실행 증명이 아니다. TensorRT-LLM/FlashInfer는 호환 조사·엔진 후보이지 이PC에서 이미 작동하는 구성이라고 보고하지 않는다.

### 용어 정리

- 사전 계산(prefill): system·few-shot·대화/state 입력을 읽어 첫 출력에 필요한 KV를 만드는 단계이다. (내가 의역해보자면 말을 하기 전에 긴 주문서를 다시 읽는 비용이다.)
- 토큰 생성(decode): 그 뒤 답을 한 토큰씩 만드는 단계이다. 출력 상한을 낮추는 것과 입력 읽기 비용을 줄이는 것은 다르다.
- 접두 캐시(prefix cache): 같은 모델/adapter와 같은 입력 앞부분의 계산 재사용이다. 생성 중 `use_cache=True`와 요청 간 자동 재사용은 다르다.
- NF4 / NVFP4: 서로 다른4bit 표현·실행 경로이다. BF16은16bit 실수, FP8은8bit 실수 계열이다. 작은 비트수만으로 실제 빠른 kernel 사용을 보장하지 않는다.
- 정답 주입(gold replay) / 실제 누적 추론(closed-loop): 전자는 올바른 입력/필드를 넣어 코드 연결을 시험하고, 후자는 모델이 자신의 이전 결과/history를 이어받아 전체 대화를 실제로 푸는 시험이다.

Cause: 정책·few-shot 정렬과 실제 입력 재계산/서빙 비용이 아직 최종 운영 구성에 반영되지 않았다.

Effect: `llm_final_2`에서 Python 두 결함의 수정과 모델·가속 미완료를 구분했다. 다음 한 작업은 §5의1번 계약·prompt 정렬 범위를 확정하는 것이다.
