# 2026-11-05 D/Q/P 시작 워밍업 적용

사용자 승인: “ㄱㄱ여 가능함?” — 시작 시 D/Q/P 준비, 출력 폐기, 입력 준비 전 완료를 적용했다. **3/3 완료.**

1. [x] 경계 확인: vLLM HTTP 서버의 `Application startup complete`와 LLM 입력 준비는 별개다. 모델·어댑터 확인 이후, live state·publisher·subscriber·worker 생성 전에 동기 실행한다.
2. [x] 구현: `LLM_STARTUP_WARMUP=True`. 실제 system prompt·토크나이저·채팅 템플릿·역할별 JSON grammar를 사용한다. 빈 사용자 입력/A/B가 공유하는 접두만 각 adapter에 보내고 1토큰을 생성해 버린다. 사용자 예문·graph·history·TTS·로봇 실행으로 연결하지 않는다. HTTP 오류/비정상 응답은 시작을 중단하며 준비 완료를 표시하지 않는다.
3. [x] 검증: 새 CPU 14건 및 기존 관련 47건·6 subtests 통과. 기존 어댑터 테스트 2건은 기존 코드/manifest와 가정이 맞지 않아 실패하며 원본 그대로 기록했다. 실제 vLLM 3회 워밍업+후속 2턴 4회 추론 통과, 전체 7회. 임시 scope/GPU/포트 반환, worklog 기록.

## 최종 동작

`load_vllm_backend` → 공통 접두 D/Q/P(task) 준비 → 생산용 호출 함수/graph/state 초기화 → publisher/subscriber/worker 시작 → **LLM 입력 준비 완료**.

워밍업은 HTTP completion `max_tokens=1`이므로 `finish_reason=length`는 의도한 정상 종료이다. 출력 JSON 완결이나 업무 판단으로 해석하지 않는다. 실제 대화 4회는 전부 `stop`이었다. 각 HTTP 요청은 기존 120초 timeout이며 자동 retry는 없다. CPU 준비를 포함한 전체 시작 시간 hard deadline은 아니다. 어댑터는 HTTP `model`에 요청별 명시한다.

기존 16K/.58, D6/Q6/Pbest150, 고정 실행 안내, VLM/STT/TTS 정책은 유지했다. HF backend·호출 함수 주입 시험에서는 자동 워밍업을 건너뛴다. `LLM_STARTUP_WARMUP=False`이면 명시적으로 생략한다. 설치된 egg-link → build → source symlink를 확인했으므로 다음 LLM 노드 실행부터 반영된다.

## 실제 검증 결과

| 항목 | 관측값 |
|---|---:|
| 공통 접두 워밍업 준비 | 5.79초 |
| 후속 첫 복합 요청 D→Q→P | 2.839초 |
| 후속 명시적 선택+즉시 실행 D | 0.471초 |
| 첫 대화 D/Q/P 재사용 토큰 | 8,320 / 3,456 / 6,976 |
| live history | 실제 2턴의 user/assistant 4개만 존재 |

사용자 “면은 얇은면으로 보통 양으로 주고, 소스는 느끼한 걸로 추천해줘.” → “얇은면은 보통 양으로 골랐어요. 소스는 크림을 제안드릴게요. 크림으로 고를까요?”

사용자 “크림으로 선택할게. 바로 담아.” → “얇은면을 보통 양으로 담을게요.”

첫 턴은 면/보통 양 확정·크림 후보·미실행, 둘째 턴은 크림 확정·면 실행 승인으로 검사했다. 실제 ROS 토픽/로봇 명령·음성은 실행하지 않았다. 현재 측정은 LLM graph 시간이며 STT/TTS 포함 첫 소리 시간은 아니다.

## 첫 시도 및 이전 성능 해석 정정

처음에는 임시 완전 예문으로 D/Q/P를 준비했다. 8.45초 걸렸고 다음 다른 질문의 세 역할 캐시 재사용은 모두 0이었다. 첫 대화 7.353초, 둘째 2.504초로 상태·출력 검사는 통과했으나 속도 개선 근거가 없었다. 이 구현은 미채택하고 증거 폴더의 `startup_warmup_attempt1.py`에 보존했다.

첫 검증 스크립트는 마지막 `not torch.cuda.is_initialized()` 가정에서도 실패했다. 생산 경로의 클라이언트는 CUDA 초기화가 관측됐으며, 초기화 지점은 미분리다. 두 번째 시험에서는 이를 실제 측정값으로 기록했다: `client_cuda_initialized=true`, 클라이언트 Torch allocated=0. 추가 HF 모델을 로딩했다는 증거는 없다.

이전 약 5.7초 첫 음성 출력 요청 단축은 워밍업과 첫 요청이 같은 시험 결과였다. 임의 질문에 그대로 일반화한 설명은 정정한다. 최종 접두 방식은 예문과 다른 실제 입력에서도 cache hit와 첫 LLM 2.839초를 확인했다. 두 시도에서 첫 응답 문구는 동일했지만 별도 서버 실행의 단회 비교이며, 약 4.51초 차이 전체를 엄밀한 인과 효과나 항상 보장되는 수치로 쓰지 않는다.

vLLM 설치 소스의 sliding-window cache lookup은 연속 캐시 블록을 요구한다. 완전 예문에서 다른 사용자 입력으로 바뀔 때 부분 접두를 재사용하지 못하는 현상과 부합하지만, 첫 시도 미사용의 유일한 원인이라고 확정하지 않았다. 캐시 eviction·다른 route(P general/mixed)·다른 실행 부하에서는 효과가 달라질 수 있다.

## 검수와 증거

Codex CLI gpt-6-sol/high에 일반 설계만 전달해 **조건부 찬성**을 받았다. 프로젝트 코드/로그의 독립 감사와 구분한다. 요구한 접두 경계·실제 요청의 동일 token ID·요청별 adapter·timeout·격리·readiness를 로컬 검사했다. 빈 입력/A/B 경계 보강 후에도 최종 token ID가 GPU 검증본과 완전히 같음을 CPU에서 확인했다.

CPU 결과: `61 passed, 6 subtests passed, 2 failed`. 실패는 `test_adapter_routing.py`의 미채택 manifest 및 literal-only 경로 가정 2개다. 해당 테스트·model_runtime·manifest 세 파일이 HEAD와 바이트 동일함을 기록했다. 전체 테스트 성공으로 표시하지 않는다.

[증거 폴더](../training/learning_first_v43/final/goals/GL10/startup_warmup_adoption_20261105/)의 `prefix_r2/RESULT.json`, `TURNS.json`, `REQUESTS.json`, `CLEANUP.json`, `FINAL_PREFIX_IDENTITY.json`, `CPU_TESTS_FINAL.log`, `BASELINE_TEST_EVIDENCE.json`을 참조한다. 기존 VLM prompt·학습 어댑터 선택 코드·서버 launcher·입력 한도·LLM prompt 불변 확인. 서버 상시 실행이나 실물 ROS 검증 완료를 뜻하지 않는다.
