# IRC LLM Final — 현재 작업 범위

## 사용자 고정 지침 (2026-10-26)

- **이 작업의 담당 범위는 LLM이다.** Decision / Recommendation / Response, LLM prompt·state·멀티턴·입출력 계약·평가기·관련 테스트와 인계 문서를 진행한다.
- **VLM 프롬프트는 그대로 둔다.** `src/soomac_irc/soomac_irc/vlm_prompts.py`, `src/soomac_irc/test/test_vlm_prompts.py`, `docs/2026-10-26_vlm_sausage_judgment_review.md`의 기존/병렬 변경은 사용자 또는 다른 작업자 소유다. 수정·정리·되돌리기·재생성하지 않는다. VLM 판정/재시도 정책·이미지 처리·카메라 경로도 이번 LLM 작업에서 변경하지 않는다.
- 공유 `llm_langgraph_node.py`는 LLM 호출·이벤트 입력·대화 상태 경로만 수정한다. LLM에 이미 존재하는 VLM/실제 완료 증거를 읽기 전용으로 전달할 수는 있지만 VLM 결과나 실제 완료 사실을 새로 만들거나 판정 정책을 바꾸지 않는다.
- 병렬 VLM 수정이 전체 파일 해시 검사에 잡히면 그 원래 결과와 변경 파일을 기록한다. 별도로 명시한 LLM 의존 파일의 불변 여부를 확인한다. 전체 검사를 통과했다고 고쳐 쓰거나 VLM 파일을 되돌려 해시를 맞추지 않는다.
- 이 범위는 후속 Codex 검수·병렬 작업자에게도 적용한다. 이후 모델은 사용자 지정 `gpt-6-sol / high`를 사용한다. 기존 history·GL04 안전 계약·ROS topic·busy STT를 보존하며 학습·adapter 교체·실물 ROS·원격 변경·commit/push는 별도 승인 없이 하지 않는다.

작업 순서와 완료 조건은 `docs/IRC LLM Final goals.md`, `docs/IRC LLM Final handoff.md`, `docs/IRC LLM Final plan.md`를 따른다. GL05 진행과 GL06 준비만 병렬이며, GL06 최종 통합/완료는 GL05 계약 동결 뒤이다.
