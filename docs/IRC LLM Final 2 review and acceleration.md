# IRC LLM Final 2 — 문제 대조·few-shot 검수·추론 가속

[데이터 감사 결과](../training/learning_first_v43/final/goals/GL10/data_reaudit_20261031/RESULT.md) · [잔여 계획](<IRC LLM Final remaining work.md>) · [최신 로그 분석](../training/learning_first_v43/final/goals/GL10/dataset_supplement_plan_20261031/COMMIT_QUANTITY_LATENCY_213138.ko.md)
→ **데이터를 정제한 것과 현재 모델의 대화·실행·속도가 개선된 것은 구분한다.**

작성일: 2026-11-01 KST. 상태: 조사·검수 진행 중. 이 문서를 먼저 생성한 뒤 사용자 지정 `llm_final_2` 브랜치를 만든다.

## 1. 작업 기준과 보존

출발 브랜치 `llm_final`, 기준 커밋 `8cd8bef6ff904b5cb620696b1ff15e0b50b6ce7c`. 기존 미커밋 변경이 있으므로 전부 보존한다. 브랜치 생성은 그 변경을 커밋·백업하는 작업이 아니다. 원본 branch commit, 학습 결과, adapter, ROS topic, history, busy STT, 사용자 임시 고정 안내를 유지한다. VLM 프롬프트·판정·관련 사용자 변경은 수정하지 않는다.

현재 승인 범위는 이 문서 작성, 새 브랜치 생성, 실제 코드·로그·정제 산출물 읽기와 공식 가속 자료 조사·검수이다. Runtime/prompt 변경·라이브러리 설치·GPU 모델 실험·학습·운영교체·ROS·commit/push는 이번에 하지 않는다.

## 2. 사용자가 요구한 기능과 로그 문제 대조

목표는 자연스러운 직원형 대화이다. 앞선 여러 사용자 발화와 최신 state를 함께 해석하고, 선택·변경·취향·제약·추천 수락·즉시 실행을 동시에 반영한다. 현재 물리 실행 가능 범위와 미래 추천 범위를 혼동하지 않는다. 실제 완료 전 완료 발화, 없는 pending 확인 질문, 언급하지 않은 기존량 변경은 허용하지 않는다.

검토 대상: 크림 선택+바로 담아, 버섯 normal 변경+양파 low 보존+바로 담아, 게살 allergy+소시지 high+바로 담아, 페퍼론치노 제외+치즈 low+바로 담아. 일반 선택과 부정·인용·조건부 실행은 별도 대조한다.

## 3. 데이터·계약·프롬프트·few-shot 상태

직전 감사는 원본7784행의 장부 연결/불변 의미 판정 재사용/영향 검수 및 새10대화104행의 독립 의미 검수이다. 비활성 고유93행·원본대체108행·계약HOLD12행이 저장됐으며 새 학습본 조립/학습은 아니다.

현재 `agent_prompts.py`는 `render_examples()`를 실제 system에 붙이고 D/Q/P wrapper는 해당 system을 사용한다. 예시 수는 D11/Q4/Task6/General1/Mixed1이다. 예시 존재와 최신 정책 일치는 별도로 검수한다. **추천량 normal 최신 정책과 Q 예시 R01/R02/R06의 근거 없는 high/low 가능성을 전체 입력 기준으로 대조해야 한다.** 아직 수정하지 않았다.

## 4. 실제 추론 설정과 속도 근거

호스트 읽기로 확인한 GPU는 NVIDIA RTX PRO 4500 Blackwell, VRAM32623MiB, compute capability12.0(SM120), driver580.178.04이다. Gemma4Unified12B config는 local head_dim256/global_head_dim512와 sliding/full attention 혼합이다. B200 SM100용 성능·지원표를 그대로 적용하지 않는다.

현재 경로는 Transformers+PEFT 공유 Base/3개 LoRA, NF4 저장·BF16 계산, `attn_implementation="sdpa"`, 생성 내부 KV cache 사용, greedy/비thinking/XGrammar다. 코드에서 요청 간 prefix KV 재사용이나 compile/static cache 설정은 확인되지 않았다. SDPA가 실제 어떤 GPU 커널을 쓰는지는 프로파일 전 미확인이다.

환경 metadata: torch2.12.1+cu130, transformers5.13.1, peft0.19.1, bitsandbytes0.49.2, triton3.7.1. flash-attn/flash-attn-4/FlashInfer/vLLM/TensorRT-LLM 배포 패키지는 해당 gemma4_env에서 미설치이다. 이를 근거로 PyTorch 내장 fused attention도 없다고 단정하지 않는다.

## 5. 가속 대안과 후속 검증

공식 자료를 확인해 다음을 비교한다: 실제 prefill/decode/adapter 전환 계측, prompt 중복 정리와 adapter별 prefix cache, SDPA/FlashAttention/FlashInfer의 SM120·head512 지원, vLLM Gemma4Unified+LoRA/구조화 출력 호환, BF16/FP8/NVFP4와 기존 NF4의 품질·VRAM·가속 차이. 예상 배속을 실제 측정값처럼 쓰지 않는다. 최신 라이브러리 설치만으로 성공을 주장하지 않는다.

최종 문서는 문제→근거→이번 개선→아직 미해결→추천 다음 단계와 성공/실패 기준을 연결한다. 실제 환경 변경은 별도 승인 후 수행한다.
