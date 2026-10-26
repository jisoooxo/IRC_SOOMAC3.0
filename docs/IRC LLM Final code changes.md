# IRC LLM Final — 코드 수정 명세

[최종 plan](<IRC LLM Final plan.md>) · [Goal 순서](<IRC LLM Final goals.md>) · [인계](<IRC LLM Final handoff.md>)
→ **문서 작성은 수정 완료가 아니다. GL02 정답 기준을 확정한 뒤 GL03에서 평가기 코드를 실제 수정하고 테스트한다.**

기준: 2026-10-26, `llm_final`, HEAD `d3d6e40b641adf546491c3860c9ccd60c79323f0`.

최신 구현 우선: **GL05 응답·이벤트·fallback 계약, GL06 prompt/few-shot·serializer·토큰 검증 범위 완료.** `agent_prompts.py`·`prompt_examples.py`의독립시연은모델입력이며Python이발화문구로응답을선택/반환하는고정응답분기가아니다. Decision입력cap은추가history손실을막는실측최소16384,출력상한1024/1024/512·20messages/overflow방식은유지했다. 반복오류두건과권한충돌회귀를포함전체612pass8subtests·최종Codex찬성. 근거는 GL05/RESULT와GL06/RESULT·manifest이다.

후속 필수 코드/학습 계약: `training/learning_first_v43/train_in_container.py:27`의MAX_LENGTH4096은새prompt에맞지않는다. 지금파일/원격학습을수정하지않았으며실제32행CPUencoding에서24행실패를확인했다. GL07~10새messages생성/GL13승인전전체token/trainer길이·mask·packing/preflight를교정/검증해야한다. 격리16384후보에서는32행오류/trim0·최종target만labels임을확인했지만전체release확정값이아니다. 실제생성의두의미오류는새model평가필수차단항목으로유지하며해결됐다고간주하지않는다.
현재 구현 상태: §1 E01~E10은 GL03 구현(미커밋). GL04의 R01/R02/R06/R07 Runtime 범위는 구현·검증·최종 Codex 찬성으로 완료했다. 최신 증거는 `final/goals/GL04/RESULT.md`와 manifest이다. R07의 데이터 정제·GL05/06 전체·새학습·모델성능 채택은 미완료이다.
경로는 저장소 루트 기준이다. 줄 번호는 위 HEAD의 탐색점이며 수정 후 재확인한다.

## 1. 이번 묶음에서 실제 수정할 코드 — GL03

선행 조건: GL02의 독립 oracle, 정상/오답 대조, HOLD 사유 확정. 허용 파일은 `src/soomac_irc/soomac_irc/evaluation_v43.py`, 관련 평가 테스트, 필요한 오프라인 평가 fixture/결과이다. Runtime·prompt·학습 원본은 이번 묶음에서 바꾸지 않는다.

| ID / 위치 | 확인된 문제와 수정 요구 | 반드시 확인할 대조 |
|---|---|---|
| E01 `_claims_execution_started`:43 | 문장 끝 질문이 앞의 실행 선언까지 무효화한다. 실행 주장과 질문의 범위를 분리한다 | 미승인 “담기 시작할게요. 더 넣을까요?” FAIL / “담아드릴까요?”는 실행 주장 아님 / 승인된 시작 PASS |
| E02 `grade_recommendation_prediction`:400 | 제한 검사가 field 이름을 보아 `sauce=크림`의 값을 놓친다. 구조·지원 값·실제 재료·강한 제한·유효 scope를 함께 검증한다 | 유제품 제한+크림/치즈 FAIL / 허용된 오일 PASS / 알레르기 재료를 제외한다는 설명은 위험 추천과 구별 |
| E03 `_claims_completion`:49, `_response_fact_checks`:483, `grade_response_prediction`:541 | 아무 항목 하나의 완료를 다른 재료 완료 근거로 쓰면 안 된다. 항목·단계·관측 결과를 맞추고 정책상 진행을 실물 완료와 구분한다 | 면만 완료+“크림 소스 완료됐어요” FAIL / 확인된 면 완료 PASS / 부정·질문·잘못된 과거 발화 인용은 별도 oracle |
| E04 `_response_fact_checks`:483 | 수량의 마지막 단어에 의존해 부사 “보통”을 주문 수량으로 오인한다. 병렬 재료·변경 전후·현재 확정 수량 대조를 보강한다 | low+“양파 조금 골랐고 보통 이 정도면 괜찮아요” PASS / low+“양파 많이 골랐어요” FAIL / 두 재료 공동 수량 오류 FAIL |
| E05 `_grade_global_invariants`:263 | `execute`이면 plan 1개를 무조건 요구한다. 허용된 optional skip, 빈 commit, 실제 재료 실행, 자동 lid plan의 관측 경계를 분리한다 | 허용 skip의 재료 plan 0 / 빈 commit 임의 실행 FAIL / 일반 주문 plan 0 / 유효 direct commit 재료 plan 1 |

### 1.1 누락하면 안 되는 추가 평가기 수정

| ID / 위치 | 구현 요구 | 검증 기준 |
|---|---|---|
| E06 `_response_fact_checks`, transition pending 검사 | pending.type만 보고 실제 확인 대상이 있다고 판정하지 않는다. 대상·수량·현재 주문의 정합성을 확인한다 | 빈 주문/빈 대상의 “기존 주문 담을까요?” 차단. 정상 pending 질문은 유지. 불완전 fixture를 모델 오류와 분리 |
| E07 회상/철회 검사 | user 발화와 assistant 추측, 부정과 긍정, 최신 철회와 과거 취향을 구분한다 | “좋아하지 않아”를 좋아함으로 회상하면 FAIL. 철회 과정을 설명하는 정상 문장은 통과. history 삭제 금지 |
| E08 changed-item / policy issue 검사 | 필드명 자체나 내부 `low` 문자열을 자연어에 강제하지 않는다 | “오일로 바꿨어요” 정상 허용. 물리적으로 변경 불가한 양파 설명에 `low`를 요구하지 않음 |
| E09 proposal 구조 검사와 평가 호출부 | 잘못된 nested type/미지원 메뉴·수량을 예외 종료 대신 명시적 실패로 반환한다. 기존 Grade 소비자 호환성을 검사한다 | 오류 입력이 전체 평가를 중단하지 않음. 정상 입력·기존 결과 직렬화 회귀 없음 |
| E10 평가 범위 표시 | 규칙 통과를 전체 의미·자연스러움 합격으로 승격하지 않는다. 자동 검사와 독립 의미 판정, 미해결 HOLD를 분리한다 | 정답 verdict를 그대로 읽어 자동 의미평가 성공으로 보고하지 않음. 판정 근거·입력/출력 identity 기록 |

원인: 평가기의 거짓 통과와 거짓 탈락이 모두 학습 데이터 선별·모델 채택 판단을 왜곡한다.
결과: 각 수정은 정상 사례를 보존하면서 오류 사례만 잡았다는 대조 증거를 요구한다.

## 2. 후속 묶음에서 실제 수정할 코드 — GL04~GL06

**2026-10-26 사용자 지시로 GL04 착수·브리핑 승인.** R01/R02/R06/R07을 우선 범위로 진행하며 GL05/06 전체는 아직 착수하지 않았다. 아래는 기존 감사 근거를 인계한 수정 대상이다. 미확정 원인은 재현으로 좁힌 뒤 최소 diff로 수정한다.

| ID / 위치 | 필요한 수정과 보존 조건 | 완료 증거 |
|---|---|---|
| R01 `agent_contract.py`:79,98 / `llm_policy.py`:267 / prompt serializer | 수량 schema와 domain enum을 일치시킨다. patch의 취소 `none`과 persisted order 상태를 구별한다. 자연어→수량 Python 키워드 변환은 추가하지 않는다 | “많이”의 raw 문자열 때문에 주문이 반복 거절되는 계약 재현과 교정. 정상 low/normal/high·취소·null 회귀 |
| R02 `llm_langgraph.py` Decision→state/policy 경로 | 복합 추천+제약 저장, commit+추천의 조기 반환, clarification과 execution pending 우선순위를 점검·수정한다 | B1 선호만으로 실제 주문 생성 금지, B2 모호한 답변의 오래된 pending 오실행 금지, 유효 직접 실행과 성공 멀티턴 유지 |
| R03 `llm_langgraph.py`:81,722,941~951 / `agent_prompts.py` | 추천 이유를 현재 단계 고정 설명으로 덮는 경로와 10/11 이후 추가된 네 고정 발화 경로를 모델 Response 계약으로 교정한다 | 실행 확인 거절·제한 철회 거절·추천 제안·추천 수락의 호출 증거. 10/11 실행 시작/불가 fallback은 기준 대조 후 유지 |
| R04 `llm_policy.py`:116 / `llm_langgraph_node.py` 단계 이벤트 | 추천 가능 범위는 이미 현재+미래를 포함한다. 원인을 scope 필드 하나로 단정하지 말고 raw 추천→검증→최종 발화 전체를 대조한다. 이벤트 Response 입력에는 가짜 user를 만들지 않는다 | broad 요청은 남은 코스, narrow 요청은 대상만. 미래 주문은 현재 실행하지 않음. 단계 도착 시 선호/확정 주문을 구별해 질문 |
| R05 `agent_prompts.py` 및 실제 입력 생성·출력 설정 | Agent별 엄격한 계약과 session/multiturn few-shot을 추가한다. user의 최신 뜻·검증된 state·물리 사실의 우선순위를 명시하고 토큰 실측한다 | history 역할/순서/기존 성공 보존, 조사·존댓말·부정·ASR 모호성 대조, 근거 있는 자연스러운 이유+다음 권유. 과도한 단답·장문 일괄 강제 금지 |
| R06 `llm_langgraph.py`:48-75 매운맛 변환 / `domain.py` RESTRICTION_CATEGORY_ITEMS | 사용자 결정 D3(plan §8.0): “못 먹어”는 강한 제한이다. `매운맛 cannot_eat`을 선호 `맵지 않게`로 바꾸는 변환을 교정하고, 매운맛 제한이 페퍼론치노 추천·plan을 막게 한다. dislike는 소프트로 유지 | P1-1·P1-5: 페퍼론치노 제안·plan 0, 치즈 제안. 기존 강한 제한 철회 확인(F10)·dislike 회귀 |
| R07 Decision prompt·데이터·`agent_contract.py` 수량 계약 | 사용자 정정 D2: 수량 생략 시 기존 확정량 유지. 이번에 수락/참조한 유효 후보의 수량도 보존하며 둘 다 없는 새 항목에만 normal 기본값. “나머지” 대상·실행 의도는 history/state/직전 질문/pending으로 판단. Python 키워드 변환 금지 | 같은 P1-4 발화의 확정 high/low 유지·미확정 새 항목 normal·후보 high 수락·새 명시 low·참조 모호성 대조. 실행 수락과 일반 선택을 따로 평가하고 normal=commit으로 연결하지 않음 |

강한 제한·물리 불가·중복 실행·ROS/VLM 검증, 기존 topic, busy STT 차단, 자동 lid/sauce 제어는 보존한다. “모델이 자연스럽게 말하도록” 바꾸는 것과 안전장치를 없애는 것은 다른 작업이다.

## 3. 구현 순서와 완료 조건

1. **GL02:** `GL01/INPUT_MANIFEST.json`과 기존 감사에서 정상/오답/HOLD oracle을 동결한다. 현재 Runtime 출력을 무조건 정답으로 쓰지 않는다.
2. **GL03 재현:** E01~E10별 실패 재현 테스트를 먼저 추가한다. 기존 감사 산출물을 덮어쓰는 재현 스크립트는 그대로 실행하지 않는다.
3. **GL03 구현:** evaluator 최소 diff를 적용한다. 새 자연어 Runtime 규칙은 추가하지 않는다. 평가의 의미적 한계는 숨기지 않는다.
4. **GL03 검증:** 기존 네 테스트 파일과 새 대조 테스트를 실행하고 명령·exit code·분모·개별 미해결 항목·독립 검수 결과를 저장한다. 예상한 실패가 해결되지 않으면 완료 처리하지 않는다.
5. **인계:** 변경 파일/함수·before/after·미해결 HOLD·회귀 결과를 이 문서와 Goal checkpoint에 반영하고 BATCH01 종료. BATCH02는 승인 후 진행한다.

기존 baseline 테스트 파일: `test_natural_multiturn_contract.py`, `test_section_question_events.py`, `test_evaluation_v43.py`, `test_scenario_context_evaluation.py` (`src/soomac_irc/test/`). 실행 환경은 GL01/WORKLOG.md의 검증된 명령을 사용한다.

## 4. 상태·증거 기록

| 구분 | 현재 상태 | 증거/다음 기록 |
|---|---|---|
| GL01 현황 동결 | COMPLETE | `final/goals/GL01/RESULT.md`, manifest, baseline, CODEX_REVIEW |
| GL02 oracle | COMPLETE | ORACLE.md r1 FROZEN·독립 검수 9건 반영. 동결본은 보존하고 사용자 정정 D2는 현 plan/후속 계약으로 인계 |
| E01~E10 | IMPLEMENTED (GL03, Codex 보완 포함, 미커밋) | `final/goals/GL03/CODEX_FOLLOWUP_RESULT.md`. bool/null/pattern/targets·정답bool 추가검증, 전체384 passed·Codex찬성. Response 문구 검사는 검토 표시 |
| R01/R02/R06/R07 | GL04 Runtime 범위 COMPLETE, R07 데이터는 후속 | 수량 enum·기존량 보존 prompt·강한 제한·복합 pending/commit·명확화 권한. GL04/RESULT·manifest·AUTHORITY_REVIEW(찬성). raw모델실패는별도미해결 |
| R03/R04/R05 | NOT_IMPLEMENTED | 후속 GL05/06 선행·실행 범위 확인 |

### 4.1 GL04의 최소 추가 계약 — 질문 목적과 실행 권한

실제 기존 어댑터 GPU run2/run3에서 직전 명확화 history를 전달했는데도 `응`이 옛 execution pending의 accept로 출력되어 실행 허용되는 실패가 반복됐다. next_prompt 우선순위만 바꾸는 방법과 최소 대조 예시만 추가하는 방법으로는 다음 턴 권한이 보존되지 않았다. 실패한 예시-only 후보는 채택하지 않았다.

새 SessionState/Decision 최상위 필드 없이 기존 execution pending에 선택적 `question_type=clarification`을 기록한다. 대상·기존량·source·history는 유지한다. 이 상태의 대상 없는 confirmation은 기존 1회 Decision repair로 돌리며, 재차 구조가 맞지 않으면 clarify로 종료한다. Python은 발화 키워드를 검사하지 않는다. 명시적 commit은 정상 검증 뒤 실행 가능하며, 유효한 현재 주문 정정은 표식을 해소하되 실행 승인을 만들지 않는다. 품질 감사기의 optional pending 필드와 next_prompt 계약도 맞춘다.

여러 강한 제한 철회는 기존 restriction_removal pending의 선택적 deferred 목록으로 보존하고 한 번에 하나씩 확인한다. 미수락 미래 추천은 독립된 현재 주문 실행에 의해 소실되지 않는다. 단, extra 이후 lid/sauce 자동 구간은 대화 기회가 없으므로 미수락 소스 후보를 마지막 대화 구간에서 먼저 해소한다. 기존 자동 제어 자체는 변경하지 않는다.

동일값을 반복한 추천은 이미 확정된 현재 주문의 실행을 막지 않는다. 독립 검수의 “현재+미래 혼합이면 무조건 차단” 지적은 사용자 계약보다 강한 방어로 재판정·철회했다(FOLLOWUP_REVIEW.md). **실제로 변경되는 현재 후보는 미수락 상태에서 차단**한다. 추천 이유/자연스러운 발화 복원은 GL05에 남는다.

## 용어 정리

Oracle은 독립적으로 정한 정답 기준, HOLD는 근거 부족으로 합격/오류를 확정하지 않은 상태, Runtime은 로봇 대화·상태·실행을 처리하는 코드이다. (내가 의역해보자면, 채점부터 바로잡고 실제 대화 코드를 고쳐야 하며 문서가 생겼다고 고친 것은 아니다.)
