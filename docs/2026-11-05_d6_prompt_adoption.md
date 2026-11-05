# D epoch6·전체 보강 프롬프트 운영 적용 완료

**최신 VRAM 검증(2026-11-05): 현재16K/.58은 Qwen STT·TTS·LLM 동시 부하에서31.37GiB/여유0.49GiB로 guard 중단했다. 임시12K/.50은28.97GiB/여유2.89GiB로 완주, 필수17/17·추가history trim0이다.** [측정 보고서](2026-11-05_stack_vram_12k_review.md). 운영 기본값은 아직16K/.58이며12K는 미적용 후보이다. 측정노드/GPU는 모두 반환했다. 마이크/외부 이미지 토픽 수신은 미검증이며 저장 입력으로 실제 모델 연산을 측정했다.

**3/3 완료. 사용자 승인에 따라 운영 경로를 D epoch6로 교체하고, 평가한 보강 문구 전체를 적용했다.** Q epoch6/P best150과 고정 실행 안내, VLM 실패 이유 ON은 유지했다. 실제 로봇/VLM 정책·이미지·재시도 경로는 수정하지 않았다.

## 1. 적용된 구성

| 항목 | 운영 선택 |
|---|---|
|Decision|r5 Unsloth epoch6 / step1638|
|Recommendation|r5 Unsloth epoch6 / step234, 기존 선택 유지|
|Response|r5 Unsloth best / step150, 기존 선택 유지|
|Decision prompt|16K 시험의 patched.txt와 문자열이 완전히 같은 전체 보강 문구|
|서빙|NF4/BF16, 입력+출력16,384, max batched16,384, GPU 할당률.58, batch invariant1|

D 원본 운영 경로는 `models/runtime_adapters/decision_r5_unsloth_epoch6_step1638`이다. 서버는 `models/vllm_adapters/decision` → `models/vllm_adapters/releases/d6_unsloth_20261105/decision`을 읽는다. 이전 Decision 디렉터리는 `.rollback-decision-20261105`로 보존했다. 원본과 검증된 vLLM 변환본을 그대로 복사했으며 가중치 병합은 없다. 원본/변환본 SHA는 각각 기존 평가의 d13c36fb…/effe67e1…과 일치한다. CONVERSION.json의 원본 경로만 새 운영 복사본에 연결했다.

`model_runtime.py`의 직접 지정 경로가 정본이며 `config/llm_active_stack.json`은 이를 반영하는 정보용 사본이다. active=true는 사용자 승인 선택을 뜻하고 all_gates_pass=false는 잔여 오류/실물 미검증을 그대로 나타낸다. 복구용 resolver는 전체 gate를 요구하는 기존 동작을 유지한다.

## 2. 실제 검증

관련 CPU 검사100개 통과. 초기 프롬프트 테스트의 예시-마지막 조건은 보강 문구가 뒤에 붙는 새 계약에 맞게 갱신했고, 예시/보강 문구가 실제 history에 섞이지 않는 검사는 보존했다. 초기 실패 로그도 남겼다.

운영 경로 그대로 새 vLLM 서버를 띄웠다. 평가용 prompt/system override나 입력 예산 override 없이 운영 모듈의 기본값을 사용했다. 서버 변환본과 원본 source SHA, 등록 경로를 실제 load_vllm_backend 검사로 대조했다. source-package는 기존 build 심볼릭링크가 실제 src를 가리키므로 colcon 재빌드는 필요하지 않았다.

| 검증 | 결과 |
|---|---|
|필수 compound6 + split11|17/17 계약 통과|
|“그걸로 하고 면은 바로 담아줘”|크림 선택, 얇은면 normal 실행 정상|
|누적 멀티턴 관측|13대화82/82턴, 실패 뒤에도 실제 상태 유지|
|주문·단계·계획·실행 검사|80/82|
|route·후보·취향 등을 포함한 전체 계약|77/82|

146 HTTP 요청 모두 입력 JSON 값/자료형과 raw 출력이 trace와 일치했다. 입력+예약출력 최대10,091, 추가history trim0, 모든 응답 finish_reason=stop이었다. 원래의 최근20메시지 입력 창은 유지한다. 예시나 정답 Decision을 주입하지 않았다. 완료/단계 전환은 실제 실행 허용 뒤 적용한 명시적 오프라인 fixture이며 실물 동작 시험은 아니다.

## 3. 남은 차이와 제한

`accepted_short_then_explicit_execution`의 “그걸로 담아줘”는 이번에는 면을 즉시 실행했다. 다음 “그럼 면은 지금 바로 담아줘”는 중복 실행하지 않았다. 재질문을 기대한 동결 대본과 이2턴이 달랐고, 4턴부터 기대 단계와 다시 맞았다. 실제 수량은 얇은면 normal이다. 결과를82/82로 고쳐 쓰지 않았다. 시연에서는 “그걸로 하고 면은 바로 담아줘”처럼 실행 대상을 명시하는 문장을 권고한다.

나머지3건은 기존에 남아 있던 인사/역할/추천 이유의route 차이이다. 보강으로 모든 일반대화·완곡한 섭취 제한이 해결됐다는 주장은 하지 않는다. 이전 동일 요청의 서버 재시작 간 출력 변동도 미해결이다. 이번 실제 결과는 재시작 후의 보장 성공률이 아니다.

16K/.58의 단독 서버는 검증했으나 STT/TTS가 GPU에 함께 상주하는 실물 스택은 이번에 실행하지 않았다. 이전10K/.45의 동시 메모리 측정을 새16K 설정의 보장으로 사용하지 않는다.

## 4. 검증기와 보호 범위

최초 운영 검증은 model_runtime import가 CUDA를 초기화해 검사기의 “초기화 없음” assertion에서 멈췄다. 단계별 확인에서 클라이언트 모델은 CPU VllmModel, torch GPU할당0바이트였다. 서버/어댑터 로딩 오류로 분류하지 않았다. 검사기를 실제 클라이언트 타입·CPU device·GPU tensor할당0 검사로 교정하고 다시 실행했다. 첫 로그와 소스는 VALIDATION_ATTEMPT_01, 원인 확인은 CLIENT_INIT_DIAGNOSIS.json에 보존했다.

Q/P의 운영 경로와 바이너리 SHA, llm_langgraph_node.py와 VLM 관련 보호 파일 해시는 불변이다. `LLM_BACKEND=vllm`, `EXECUTION_REPLY_MODE=fixed`, `ENABLE_VLM=True`, `VLM_JUDGE_REASON_SPEAK=True`를 유지한다. 명확한fail에서 기존 VLM의 판단 근거를 고정 정책 안내 앞에 붙이는 동작도 그대로다. 새 VLM 추론으로 이 기능을 재검증한 것은 아니다.

## 5. 현재 실행 상태와 복구

운영 설정은 적용됐고 **검증용 서버는 종료했다.** 소유한 검증unit2개inactive, GPU compute목록 비어 있음,8000/18001 포트 반환 확인. 다음 기본 서버 실행부터 D6/보강/16K가 사용된다. 실행 명령은 저장소 루트에서 `bash tools/vllm_server/start_vllm_server.sh`이다.

복구 설정은 `before/`, 이전 변환본은 `models/vllm_adapters/.rollback-decision-20261105`에 있다. 원본 checkpoint·기존 환경·원격 작업·VLM 파일은 수정하지 않았다. 이전 평가의 source 해시와 raw는 그대로 보존했고, 이번 사용자 승인 운영 변경은 새 ADOPTION_FILE_AUDIT/SOURCE_AFTER로 구분한다.

Cause: 새 D6와 보강 문구가 평가 경로에만 있어 기본 실행은 구 Decision을 읽었다.

Effect: 기본 운영 경로와 서버·클라이언트 예산을 함께 바꿔 새 조합이 실제로 로딩되는 것을 확인했다. 알려진 의미 오류와 재현성 한계는 유지한다.

[실제 전체 답변·Decision·계획](../training/learning_first_v43/final/goals/GL10/d6_prompt_adoption_20261105/OUTPUTS.md) · [파일·보호 범위 감사](../training/learning_first_v43/final/goals/GL10/d6_prompt_adoption_20261105/ADOPTION_FILE_AUDIT.json) · [종료 확인](../training/learning_first_v43/final/goals/GL10/d6_prompt_adoption_20261105/RESOURCE_CHECK.json)
