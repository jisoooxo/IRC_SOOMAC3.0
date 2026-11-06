# 최신 프롬프트·D/Q/P HF/vLLM 실측 비교 — 2026-11-05

**현행 HF 캐시ON → 현행 vLLM 캐시ON/invariantON의 동일입력 warm 합계는 34.36초 → 14.76초(관측 비율2.33배)이다. 36턴은 둘 다 통과했다. 7조건 중6조건이36턴을 완주했고 HF 캐시OFF는 메모리 안전 기준으로18턴에서 중단했다.**

근거: [원시 결과·SUMMARY.json](../training/learning_first_v43/final/goals/GL10/hf_vllm_compare_20261105/SUMMARY.json) · [계획](2026-11-05_hf_vllm_compare_plan.md) · [실제 답변 전체](../training/learning_first_v43/final/goals/GL10/hf_vllm_compare_20261105/OUTPUTS.md).

## 동일 입력 반복 속도

D5·Q2·P5의 12개 입력을 조건별 3회 실행했다. 아래는 2·3회차의 역할별 응답 완료 시간 중앙값이다(D10/Q4/P10 표본). 첫 토큰 시간(TTFT)은 측정하지 않았다. HF는 시스템 접두를, vLLM은 적중 가능한 전체 입력 블록을 재사용하므로 캐시 정책 자체도 비교에 포함한다. 이 숫자를 새 질문마다 보장되는 지연으로 해석하지 않는다.

| 조건 | D 초 | 추천 초 | Response 초 | warm 24회 합계 초 | 출력 JSON/종료 |
|---|---:|---:|---:|---:|---|
| HF 접두 캐시 OFF | 5.86 | 3.78 | 5.89 | 115.92 | 36/36 JSON, 36/36 stop |
| HF 접두 캐시 ON (기존 FP32 LoRA) | 0.77 | 2.21 | 1.67 | 34.36 | 36/36 JSON, 36/36 stop |
| HF 접두 캐시 ON (BF16 LoRA 대조) | 0.73 | 2.16 | 1.71 | 33.47 | 36/36 JSON, 36/36 stop |
| vLLM 16K/.58 접두 OFF·invariant ON | 2.67 | 1.87 | 2.86 | 55.60 | 36/36 JSON, 36/36 stop |
| vLLM 16K/.58 접두 ON·invariant ON (현행) | 0.31 | 0.94 | 0.59 | 14.76 | 36/36 JSON, 36/36 stop |
| vLLM 16K/.58 접두 ON·invariant OFF | 0.28 | 0.77 | 0.48 | 11.86 | 36/36 JSON, 36/36 stop |
| vLLM 12K/.50 접두 ON·invariant ON | 0.34 | 0.94 | 0.66 | 16.22 | 36/36 JSON, 36/36 stop |

## 실제 5개 시나리오·36턴 및 VRAM

| 조건 | 상태·계약 통과 | 전체 LLM 시간 초 | GPU peak GiB | 최소 여유 GiB |
|---|---:|---:|---:|---:|
| HF 접두 캐시 OFF | 18/18 관측 / 예정36 | 미완료 | 31.16 | 0.70 |
| HF 접두 캐시 ON (기존 FP32 LoRA) | 36/36 관측 / 예정36 | 103.16 | 25.96 | 5.90 |
| HF 접두 캐시 ON (BF16 LoRA 대조) | 36/36 관측 / 예정36 | 96.49 | 24.40 | 7.46 |
| vLLM 16K/.58 접두 OFF·invariant ON | 36/36 관측 / 예정36 | 170.15 | 30.06 | 1.80 |
| vLLM 16K/.58 접두 ON·invariant ON (현행) | 36/36 관측 / 예정36 | 49.14 | 29.55 | 2.31 |
| vLLM 16K/.58 접두 ON·invariant OFF | 36/36 관측 / 예정36 | 40.88 | 29.54 | 2.32 |
| vLLM 12K/.50 접두 ON·invariant ON | 36/36 관측 / 예정36 | 49.58 | 27.55 | 4.30 |

Qwen STT와 기존 HF TTS를 각 조건에 함께 적재하고 각각 저장 음성·문장으로 1회 추론 후 상주시켰다. 위 시간은 **LLM graph 처리 시간**이며 STT 입력·TTS 재생·실물 로봇 동작 시간은 제외한다. VLM 이미지 추론도 이번에는 실행하지 않았다. 따라서 실제 전체 시연 시간이나 VLM 동시 부하 안전성을 새로 보장하는 수치가 아니다.

시나리오의 발화·기대 상태는 같지만 실제 생성된 답변·history가 달라질 수 있다. HF는 replay 캐시를 제거하고 새 callable로 시작했고 vLLM은 replay 캐시와 공통 접두 워밍업을 유지했다. 이 표의 전체 시간을 동일 계산량의 엔진 속도비로 해석하지 않는다. 순서가 고정된 단회 closed-loop이며 시나리오 계약 통과와 자유 발화 전체 정확도는 다르다.

## 첫 순회 및 출력 동일성

첫 순회도 앞선 요청의 공유 접두가 적중할 수 있으므로 모든 요청이 완전 cold라는 뜻은 아니다. P95 등 상세 통계와 출력 토큰 수는 SUMMARY.json에 보존했다.

| 조건 | D 첫 순회 중앙값 초 | 추천 첫 순회 중앙값 초 | Response 첫 순회 중앙값 초 | 3회 사이 JSON이 바뀐 입력 ID |
|---|---:|---:|---:|---|
| HF 접두 캐시 OFF | 5.74 | 3.75 | 5.78 | [] |
| HF 접두 캐시 ON (기존 FP32 LoRA) | 0.78 | 2.97 | 1.68 | [] |
| HF 접두 캐시 ON (BF16 LoRA 대조) | 0.73 | 2.81 | 1.74 | [] |
| vLLM 16K/.58 접두 OFF·invariant ON | 2.64 | 1.85 | 2.84 | [] |
| vLLM 16K/.58 접두 ON·invariant ON (현행) | 2.51 | 1.84 | 1.05 | [] |
| vLLM 16K/.58 접두 ON·invariant OFF | 2.59 | 1.73 | 0.86 | [16, 48, 49] |
| vLLM 12K/.50 접두 ON·invariant ON | 2.54 | 1.85 | 1.03 | [] |

| 비교 기준 → 대상 | 같은 입력 warm 표본 | JSON 내용 일치 | 시간 합계 비율(기준/대상) |
|---|---:|---:|---:|
| HF 접두 캐시 OFF → vLLM 16K/.58 접두 ON·invariant ON (현행) | 24 | 22/24 | 7.854 |
| HF 접두 캐시 ON (기존 FP32 LoRA) → vLLM 16K/.58 접두 ON·invariant ON (현행) | 24 | 22/24 | 2.328 |
| HF 접두 캐시 ON (BF16 LoRA 대조) → vLLM 16K/.58 접두 ON·invariant ON (현행) | 24 | 20/24 | 2.268 |
| vLLM 16K/.58 접두 ON·invariant ON (현행) → vLLM 16K/.58 접두 ON·invariant OFF | 24 | 18/24 | 1.244 |
| vLLM 16K/.58 접두 ON·invariant ON (현행) → vLLM 12K/.50 접두 ON·invariant ON | 24 | 18/24 | 0.910 |

현행 HF 캐시ON과 vLLM의 warm 출력 차이는 동일 Response 입력ID56의 “보통 양으로 선택했어요”와 “보통 양으로 골랐어요” 표현이다(각2회). 면종류·수량·크림추천은 같다. JSON 내용 일치는 의미 정답 검사가 아니다. 출력 길이·정밀도·커널·버전·캐시 정책 차이가 남아 있으므로 합계 비율은 해당 배포 조건의 관측값이다. 소수 표본 P95는 서비스 SLA 추정이 아니다.

## 확인된 실패와 실제 답변

invariant OFF의 반복 입력ID16(크림 추천 뒤 “그걸로 담아줘”)은 `commit` 유무가 변했다. 기존 `_resolve_pending`을 CPU에서 적용해도 false→true→false 차이가 남았다. 입력ID49(치즈 실행 확인 뒤 같은 발화)는 JSON이 달라도 모두 실행 수락/commit=true로 처리됐다. ID48은 문장에 “치즈”를 한 번 더 쓰는 차이다. 따라서 JSON 변동 전체를 실행 오류로 세지 않는다. 정규화 증거는 `DECISION_VARIATION_EFFECT.json`이며 새 GPU추론이나 실물 실행 결과가 아니다. 36턴 closed-loop는 OFF도 통과했지만, 이 반복 안정성 관측을 근거로 **invariant ON 유지**를 권한다. 내부 수치 변동 원인은 UNKNOWN이다.

- HF 접두 캐시 OFF: Resource guard stopped owned scopes. 완료된 replay와 부분 시나리오만 결과로 사용한다.

HF 캐시OFF는 GPU 여유<1GiB가 4개 연속 표본에서 관측돼 시험 소유 scope를 중단했다. OOM으로 단정하지 않는다. 첫 vLLM 캐시OFF 시도는 6개 replay 이후 기록이 끊겼고 종료 원인은 UNKNOWN이다. 부분 기록을 보존했으며 새 `_r2` 디렉터리 결과만 완주 조건으로 쓴다.

## 동일성·측정 범위

모델은 로컬 Gemma4 12B, adapter는 D epoch6/step1638, Q epoch6/step234, P best/step150이다. 세 adapter를 병합하지 않고 활성 adapter를 바꿔 실행했다. 원본656개 텐서×3와 vLLM 변환본의 원본 대응 값, 추가16개 tied K/V 복사 텐서×3를 CPU에서 대조했다. 프롬프트·tokenizer·입력 token ID·grammar·greedy·출력 한도를 고정했다. 실제 adapter 경로·SHA는 BEFORE.json에 있다.

기존 HF는 NF4/BF16 본체에 PEFT가 LoRA1968개 텐서를 FP32로 올린다. vLLM은 LoRA auto가 모델 BF16을 따른다. HF BF16 대조는 시험 프로세스 내 LoRA dtype만 바꿨으며 원본 체크포인트를 수정하지 않았다. HF torch2.12.1+cu130/transformers5.14.1, vLLM torch2.13.0/transformers5.19.0/vLLM0.32.0으로 라이브러리도 다르다. 따라서 완전히 같은 연산 구현을 비교한 실험은 아니다.

장비는 NVIDIA RTX PRO 4500 Blackwell(32623MiB), 드라이버580.178.04이다. NVML50ms 표본의 장치 전체 사용량이다. RustDesk 등 외부 점유도 포함하며 첫 HF 캐시OFF와 이후 조건의 외부 부하가 같지는 않다. CUDA allocator 단독 peak나 순간 최대치 전부를 포착했다는 뜻은 아니다. 각 조건의 실제 최대 표본 간격은 SUMMARY.json에 남겼다.

보존 검증은 source40개·모델JSON·입력·원본/변환adapter 파일 SHA와 환경별 배포 이름/버전 목록을 비교한다. base weight는 크기·mtime 비교이며 전체 SHA 검사가 아니다. 패키지 설치 파일 전부의 binary hash 검사도 아니다. 최종 검증·GPU 반환은 AFTER.json을 근거로 별도 판정한다.

## 재개 및 보존 예외

첫 HF BF16 대조 준비에서 wetext의 ModelScope 다운로드/HTTP403 재시도가 관측돼 소유 scope를 정리했다. 보조 FST 캐시 일부는 실제 갱신됐다. 원래 HF 오프라인 변수는 ModelScope를 막지 못했다. 두 마지막 재시험은 시험 프로세스에서 wetext snapshot 조회만 기존 로컬 디렉터리로 고정했다. 원래 helper는 text_frontend=False로 합성하며 frontend.py:134~135에서 정규화를 건너뛴다. 운영 helper·모델 설정은 바꾸지 않았다. 첫 BF16의 부분 replay3회와 첫 vLLM OFF의6회는 최종 조건 통계에서 제외한다. 외부 보조 캐시 불변을 주장하지 않는다.

## 최종 판정

1. **현행 vLLM 접두 캐시ON·invariantON 유지 권고.** D/Q/P의 warm 중앙값은0.31/0.94/0.59초이며36턴 LLM처리49.14초이다. 이 시간에 음성 재생·로봇 동작은 포함되지 않는다.
2. **12K/.50은 메모리 여유를 우선하는 후보.** 36턴49.58초·36/36통과, 전체peak27.55GiB/최소여유4.30GiB이다. 현행16K/.58은49.14초·29.55GiB/2.31GiB이다. 이번 시나리오 추가history삭제0이며 더 긴 대화·VLM동시부하까지 보장하지 않는다. 운영설정은 자동교체하지 않았다.
3. **정밀도 차이만으로 속도차이를 설명할 수 없다.** HF LoRABF16 대조도 warm33.47초·36턴96.49초/36통과였다. 설치버전·커널·캐시정책 차이는 여전히 남는다.
4. **보존·반환 검증 통과.** source40/모델JSON5/입력2/원본·변환adapter SHA 동일, baseweight크기·mtime 동일,4환경배포이름·버전목록 동일, 각실행의adapter경로가초기값과일치한다. 시험scope0·18001포트해제·소유GPU프로세스0이며 RustDesk311MiB만남았다. 현재8000포트도비어있다. GPU모델을계속켜둔상태가아니다.
5. **검수 오류와 예외를 보존했다.** 처음보존검사에잘못추가한 `all_gates_pass=True` 단정은실패했다. 현재소스는manifestresolver가아닌직접경로를쓰며,manifest의false값은GitHEAD와같다. 검수조건만수정해최종VERIFYexit0을얻었고 `AFTER_FIRST_ASSERTION_FAILURE.json`에원래결과를남겼다. 보조FST캐시다운로드예외는위절대로남는다. 모든환경·모든입력·실물통합전체합격을선언하지않는다.

최종 검증 근거: `AFTER.json`, `VERIFY.log`. 원인: 최신 프롬프트·세 adapter·동일 frozen입력을 사용하고 실제 생성history를 이어 평가했다. 결과: 운영소스나adapter를교체하지않고배포조건별속도·메모리·출력변동을비교할근거를확보했다.


## 교차검수 범위와 원본 근거

Codex CLI(gpt-6-sol/high) 방법 검수와 최종 증거 검수는 **조건부** 판정이다. 속도 비교 범위·6완주/1안전중단·invariantON 유지 해석을 지지했다. 재개 이력과 .50 값의 원본 로그가 최종 검수 입력에서 빠졌다는 지적은 남겨 둔다. 이 항목을 독립 교차검수 완료라고 표시하지 않고 아래 직접 확인한 자료로 구분한다. 새GPU실험은 요구되지 않았다.

- 12K의 .50: [CONFIG.json 6행](../training/learning_first_v43/final/goals/GL10/hf_vllm_compare_20261105/vllm_12k/CONFIG.json), 실제 argv는 같은 디렉터리 LAUNCH.json.
- ModelScope403: [첫 HF BF16 TTS 로그](../training/learning_first_v43/final/goals/GL10/hf_vllm_compare_20261105/hf_cache_lora_bf16/tts.log), 정리는 EXTERNAL_STOP.json. 첫 vLLM replay6행과 첫HF BF16 replay3행은 각 원본 REPLAY.jsonl에 보존했다.
- 최초 잘못된 검수단정: AFTER_FIRST_ASSERTION_FAILURE.json. 수정된 검수의 실제exec종료코드0은 VERIFY_EXECUTION.json에 주담당 관측으로 기록했다.
- [최종 교차검수 로그](../training/learning_first_v43/final/goals/GL10/hf_vllm_compare_20261105/FINAL_REVIEW.log). 검수에 제공한 자료는 FINAL_REVIEW_PROMPT.md에 그대로 있다.
