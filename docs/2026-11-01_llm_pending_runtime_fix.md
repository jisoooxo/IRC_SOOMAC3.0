# LLM pending 거절·주문 보존·독립 실행 수정

[원인 조사와 가속 계획](<IRC LLM Final 2 review and acceleration.md>) · [수정 코드](../src/soomac_irc/soomac_irc/llm_langgraph.py) · [새 회귀 테스트](../src/soomac_irc/test/test_pending_rejection_regression.py)
→ **실행 유예는 주문 취소가 아니며, 추천 거절은 기존 주문의 별도 실행 요청을 취소하지 않는다.**

작성일: 2026-11-01 KST. 브랜치 `llm_final_2`, 기준 HEAD `3798a0e012c1e9165c9c78694b9bfcf98794469b`. 사용자 Python 수정 승인 뒤의 미커밋 구현 기록이다. 실제 모델 추론·학습·ROS 발행·commit/push는 하지 않았다.

## 1. 확인한 원인과 최소 수정

| 위치 | 원인 | 적용한 수정 |
| --- | --- | --- |
| `_resolve_pending`, execution 거절 | `source=preselected`일 때 이미 확정된 주문을 후보와 함께 지웠다 | pending만 닫고 주문은 보존한다. 명시적 `order` 취소 `none`은 기존 검증·적용 경로로 처리한다 |
| `finalize_task`, 거절 조기 반환 | 추천/제한 철회 거절을 현재 주문 실행 거절과 동일하게 취급하여 별도 `commit=true`를 검사 전에 버렸다 | 추천/제한 철회 거절과 함께 온 commit은 기존 최종 실행 검사로 계속 진행한다. 실행 허가를 강제하지 않는다 |
| 같은 실행 확인의 `reject+commit` | 같은 대상을 거절하면서 실행하는 필드 조합은 그대로 실행할 근거가 없다 | 유효한 현재 주문 재지정이 없다면 기존대로 실행하지 않는다. 현재 주문을 명시적으로 교체한 복합 발화의 기존 실행 경로는 보존한다 |

`_remove_patch_from_order`는 잘못된 preselected 삭제 분기만 쓰던 함수였다. 저장소의 참조를 확인하고 호출부와 함께 제거했다. 새 state/schema field, 자연어 키워드 판정, 고정 응답, 추가 repair 재시도는 넣지 않았다.

Cause: 기존 코드가 미수락 추천 후보, 이미 확정한 주문, 실행 동의를 같은 거절 분기로 묶었다.

Effect: 모델의 구조화된 판단이 올바르면 주문·기존량과 독립 실행 의도를 Python이 더 이상 잃지 않는다.

## 2. 바뀐 동작과 유지한 안전 경계

| 입력 의미 / 필드 | 기대 결과 |
| --- | --- |
| 소시지 high 실행 대기 → “아직 담지 마”, confirmation=reject | 소시지 high 유지, pending=null, execute=false |
| 위 턴에 게살 allergy 추가 → 다음 턴 “이제 바로 담아”, commit=true | 실제 앞 턴 session/history를 이어받고 allergy와 소시지 high 유지, 안전 검사 통과 시 execute=true |
| “추천은 말고 기존 주문 바로 담아”, recommendation reject+commit | 미수락 후보는 버림. 현재/미래 후보 모두 기존 확정량 low/normal/high 보존, 기존 주문 실행 검사로 진행 |
| “알레르기 해제는 하지 말고 선택한 소시지 바로 담아”, restriction removal reject+commit | 강한 제한 유지. 안전한 현재 주문만 실행 검사. 다른 강한 제한 해제 확인이 남아 있으면 차단 |
| 실행 거절+명시적 소시지 none / 같은 실행 reject+commit만 있음 | 전자는 실제 주문에서 소시지 제거, 후자는 주문 보존하되 실행하지 않음 |

필수 주문 누락·빈 optional section·active task·task queue·알레르기 충돌·이미 지난 물리 단계 변경·미지원 메뉴는 모두 execute=false를 유지하고 각 기존 사유로 남는다. 추천을 거절했다는 이유로 이 검증을 건너뛰지 않는다. 실행 수락/일반 선택/후보 선택 구분, 미지정량 기존 값 보존, ROS topic, busy STT, history, 사용자 임시 단계 고정 안내, VLM 코드는 변경하지 않았다.

## 3. 검증 결과와 기존 테스트 정정

1. 수정 전 새 회귀 24건: **17 실패 / 7 통과**. 실패에는 실행 의도 소실, preselected 주문 삭제, 최종 안전 검사를 거치기 전에 일반 거절 사유로 끝나는 문제가 포함됐다. 17개의 별도 근본 원인이라는 뜻은 아니다.
2. 수정 후 관련 5개 파일: **412 통과**. 새 24건, 누적 session 전달, 일반 거절, 명시적 취소, 거절+현재 주문 교체+commit을 포함한다.
3. 범위를 넓힌 LLM 13개 파일: **812 통과 / 하위 검사 6 통과**, exit 0. SWIG 사용 중단 경고 2종 및 sandbox NVML 초기화 경고가 있었으며 GPU 모델은 실행하지 않았다.
4. 기존 기대값 2곳 정정: `test_natural_multiturn_contract.py`의 preselected 실행 거절=주문 삭제를 주문 보존으로 교정. `test_graph_role_refactor.py`의 추천 거절+기존 주문 commit 조합만 새 기대값에 맞췄다. execution 거절 및 deferred 제한 확인 차단은 유지한다.

두 테스트를 없애거나 실패 검사를 약화해 통과시킨 것이 아니다. 명시적 none 취소, 순수 거절의 비실행, 변경 없는 동일 실행 reject+commit의 비실행, 추천 미요청 시 Q 미호출과 입력 객체 불변을 새 독립 테스트에서 검증한다. 테스트 mock Response의 `ok`는 자연스러운 언어/사실성 합격 판정이 아니다.

추가 사유 검사 중 allergy fixture의 기대 사유 1건을 바로잡았다. 기존 확정 주문과 강한 제한의 충돌은 `enforce_restrictions`가 실행 전 재료를 제거하므로, 이 fixture의 최종 사유는 `restriction_conflict`가 아니라 `empty_section_commit`이다. 실제 코드의 `changed_order_keys`/`requested_conflicts` 경로를 읽어 확인했고 재료 제거·제한 유지도 함께 assert했다. 이 차이를 모델 실패로 세거나 Runtime을 바꾸지 않았다.

실행 환경: `/home/roma/miniconda3/envs/gemma4_env/bin/python`, `PYTHONPATH=src/soomac_irc`.

재현 명령:

```bash
PYTHONPATH=src/soomac_irc /home/roma/miniconda3/envs/gemma4_env/bin/python -m pytest -q \
  src/soomac_irc/test/test_adapter_routing.py \
  src/soomac_irc/test/test_evaluation_v43.py \
  src/soomac_irc/test/test_evaluator_oracle_gl02.py \
  src/soomac_irc/test/test_gl04_state_contract.py \
  src/soomac_irc/test/test_gl05_fast_path.py \
  src/soomac_irc/test/test_gl05_response_contract.py \
  src/soomac_irc/test/test_gl06_prompt_contract.py \
  src/soomac_irc/test/test_graph_role_refactor.py \
  src/soomac_irc/test/test_natural_multiturn_contract.py \
  src/soomac_irc/test/test_pending_rejection_regression.py \
  src/soomac_irc/test/test_recommendation_global_scope.py \
  src/soomac_irc/test/test_scenario_context_evaluation.py \
  src/soomac_irc/test/test_section_question_events.py
```

이 명령은 Python 모듈을 실행하며 별도 snippet 변수를 요구하지 않는다. `SOOMAC_GRAPH_BASELINE`은 설정하지 않았다. 옛 baseline과의 완전 동등성은 의도적으로 바뀐 두 계약에 대해 주장하지 않는다.

## 4. 남은 범위와 후속 검증

**모델이 commit/알레르기/선택 자체를 누락하거나 Response가 양·변경 사실을 잘못 말하는 문제는 이번 수정으로 해결됐다고 하지 않는다.** 올바른 Decision을 주입한 코드 회귀이며, 실제 최신 best 모델의 자연어 이해/닫힌 루프 시연/속도는 미측정이다. prompt·few-shot·학습 원본·비활성 정제 후보·adapter 경로를 바꾸지 않았다. 위임 추천 실행 계약 HOLD12도 해제하지 않았다.

이전 데이터 감사의 Runtime hash는 그때의 불변 코드에 대한 역사적 증거다. 이번 승인된 LLM 코드 수정 이후에도 동일하다고 주장하거나 옛 manifest/RESULT를 수정해 통과시키지 않는다. 다음 데이터 조립/계약 재생성 시 **이 두 동작과 연결된 영향 행만** 새 Runtime에 대조하고, 이미 끝난 전수감사를 반복하지 않는다.

후속 한 단계는 기존 계획의 **D/Q/P prompt·few-shot 최신 수량/복합 commit 계약 정렬**이다. 추천 범위 토글, 불완전 일반 주문의 next_prompt 보완, 실제 GPU 모델 누적 회귀, prefix cache/FlashAttention/NVFP4 실험은 여전히 별도 미완료이며 FP8 제외 정책을 유지한다.

작업 중 `.claude/settings.json`에 외부 변경이 추가됐다. 이 작업에서 해당 파일은 수정·되돌리기·stage하지 않았다. VLM 파일도 수정하지 않았다.

후속 기록: [실제 출력·prompt/few-shot 대조 Worklog](2026-11-01_llm_prompt_alignment_worklog.md). 실패 당시 실제 system과 현재 prompt의 일치, 복합 의도/P 변경 사실 예시 공백을 확인했다. 이 후속 대조에서는 prompt를 수정하지 않았다.
