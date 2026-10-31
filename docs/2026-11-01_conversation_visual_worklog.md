# 실제 대화와 보강 데이터 — 시각화 작업 기록

[실제 17턴 계측](2026-11-01_scenario_measurements.json) · [시나리오 결과](2026-11-01_llm_scenario_regression.md) · [기존 데이터 감사](../training/learning_first_v43/final/goals/GL10/data_reaudit_20261031/RESULT.md)
→ **실제 모델 발화와 저장된 필드를 나란히 보고, 미학습 보강 데이터와 구분한다.**

## 1. 사용자 지시와 소유 범위

2026-11-01 KST, `llm_final_2`. 최신 지시: **속도는 Claude 담당이고, 이번 작업은 대화에만 집중한다.** 접두 캐시·속도 최적화·`model_runtime.py`·D/Q/P 추론 래퍼는 이번 작업에서 변경하지 않는다. 학습·추가 GPU 추론·ROS·VLM·adapter 경로 변경·commit/push도 하지 않는다.

기존 실제 계측 JSON은 읽기 전용이다. 원본 17턴은 이전 prompt/adapter identity에 대한 관측이며, 동시에 진행되는 Claude의 속도 코드 변경 뒤 재측정한 결과가 아니다. 실행 중 `decision_model.py`, `recommendation_model.py`, `response_model.py`의 외부 변경이 추가로 관측됐으며 보존한다.

## 2. 병렬 작업과 산출물

Orca Run `run_5dcc02059ec1`, 전용 coordinator `term_5204c04f-7c3e-441b-abe8-d5fae9f747b5`를 재사용했다. 이전 작업자는 완료·release 상태를 확인한 뒤, 독립 작업 세 개를 시작했다. 세 launch receipt 모두 실제 `gpt-6-sol/high`, `turnStart=observed`다.

| 작업 | Task / Dispatch | 허용된 산출물 |
| --- | --- | --- |
| 대화 HTML | `task_aa4945f94635` / `ctx_b6a44eb88b5e` | `2026-11-01_conversation_view.html`, 필요 시 전용 빌더 |
| 보강 데이터 출처·필드 추출 | `task_8893c7065c0c` / `ctx_901f8a99f3d4` | `2026-11-01_scenario_supplement_view.json`, 출처 MD |
| 대화 필드 독립 대조 | `task_6fe0c5cae63f` / `ctx_72631d567a46` | `2026-11-01_conversation_visual_review.md` |

작업자별 파일 소유권을 분리했다. 데이터 담당자는 기존 감사·판정 장부를 재사용하며 완료된 전수 검수를 반복하지 않는다. HTML은 실제 발화·Decision·Q 호출 여부·주문 전후·pending·실행 계획을 표시한다. 보강 데이터는 별도 미학습/비활성 영역에 출처와 함께 표시한다.

## 3. 해석상 지켜야 할 경계

1. 자연스럽게 크림을 추천하는 말 자체는 사용자 의도다. 이를 금지하는 것이 목표가 아니라 후보·선택·응답의 연결을 맞추는 것이 목표다.
2. `commit=true`는 실행 의도이고 `policy.execute=true`와 같지 않다. 필수 주문 누락·안전 검증·물리 상태는 별도로 확인한다.
3. 실제 17턴은 전부 면 단계에 머물렀다. 후반 육류·토핑 발화를 정상 육류·토핑 단계에서 독립 검증했다고 표현하지 않는다.
4. 실제 최종 소스는 `null`이다. “크림으로 선택했어요”라는 응답을 실제 선택으로 표시하지 않는다. 양파 low 유지·버섯 normal 변경·게살 allergy 저장은 관측된 성공으로 구분한다.
5. 보강 후보 93행·원본 대체 108행·HOLD 12행은 기존 감사 산출물이다. 이번에 새로 학습했거나 현재 adapter가 배웠다고 하지 않는다. prompt few-shot도 학습 데이터와 구분한다.

Cause: 발화가 자연스러워도 주문·추천 후보·실행 허가가 연결되지 않으면 다음 수락과 즉시 실행이 어긋난다.

Effect: 실제 실패 위치와 준비된 보강 데이터의 차이를 사람이 확인할 수 있으며, 이번 시각화 자체를 해결·재학습 완료로 오인하지 않는다.

## 4. 진행 상태

세 작업자 실제 시작 확인 완료. Task 정본의 제목·spec·assignee와 각 Dispatch를 대조해 표를 기록했다(병렬 launch의 반환 순서를 작업 순서로 가정하지 않는다). 아래는 마감 검증 기록이다.

원본 계측 JSON SHA-256: `a05f4122cc6bdf4757c6c528a4198e1026062db1ca315f70e5e349ceb0ed36fb`. 독립 집계는 compound6/split11, D17/Q0/P17, commit6/execute0, pending 전후 모두 null, 실제 section 모두 noodle, 소스 전행 null이다.

### 산출물 및 원본 대조

- [대화 화면](2026-11-01_conversation_view.html), [재생성 빌더](build_conversation_view.py): 실제 17턴과 보강 13턴을 서로 다른 영역에 포함했다. `measurement-data`/`supplement-data`의 내장 JSON을 각 원본과 비교해 정확히 일치함을 주담당도 확인했다. 외부 script 0개, 계측 원본 해시 불변이다.
- [보강 예시](2026-11-01_scenario_supplement_view.json), [출처](2026-11-01_scenario_supplement_provenance.md): 기존 비활성 93행의 대표 13턴·Agent target 22개를 선별했다. 신규 데이터 생성이나 학습 편입이 아니다. 주담당은 3개 원본 파일 해시와 13개 history/pending 전후 연결을 확인했다.
- 데이터 작업자를 같은 터미널에서 후속 Task `task_044617bca8dd` / Dispatch `ctx_2c7add7faff7`로 재사용해 문서 상대 링크 5개만 정정했다. 13/13턴 after_state·history, 22/22 target의 representative 원본 일치 검증 결과를 수락했다. 원래 및 후속 완료 메시지를 처리했고 후속 작업자를 release했다.
- [독립 대화 검수](2026-11-01_conversation_visual_review.md)는 실제 17턴·코드·HTML 내장 자료를 대조했다. 내부 모델 원인은 아직 미상이며 재학습 필수라고 단정하지 않는다. 완료 메시지를 수락하고 `ctx_72631d567a46`을 release했다.

### 화면 확인

Orca의 기존 사용자 서버 탭은 보존하고 새 페이지 `c6b8e1b8-f9fe-4477-8ccb-2fdf2b428418`에서 로컬 파일을 열었다. C2 선택 시 실제 크림 미확정·commit true·missing_order·execute false가 함께 보였다. 복합 6개/분리 11개 턴 선택과 “모든 대화 보기”의 17개 카드, 보강 13개 표시를 확인했다. 1247px 화면에서 가로 넘침 없음, 브라우저 console 메시지 0개였다.

초기 화면의 추천 action `none`을 재료 제외와 같은 “제외”로 표시한 오류와 보강 state가 JSON에만 들어 있던 표시를 HTML 담당자에게 국소 보완하도록 전달했다. 최종 재검증·완료는 아래에 추가한다.

### 최종 화면 검증

국소 표시 수정 후 브라우저에서 17개 턴을 각각 실제 선택해 `user_text`와 `reply`가 원본과 17/17 정확히 일치함을 확인했다. “모든 대화 보기” 카드 17개, 보강 주문 전후 표 13개, 추천 `none`의 “추천 요청 없음” 표시, 브라우저 console 메시지 0개, 1247px 화면 가로 넘침 없음이 확인됐다. 보강 첫 턴의 D 추천 요청·Q 크림 후보·P 정답·pending 생성과 소스 미확정이 구분되어 표시된다. 작은 화면의 실제 브라우저 실측은 하지 않았으므로 모바일 검증 완료라고 하지 않는다.

최종 HTML 188,662bytes의 두 내장 JSON도 다시 원본과 정확히 일치했다. 보강 target은 모델 답변이 아니며 보강 후보를 현재 학습본으로 연결하지 않았다. 이번 작업의 파일 변경은 대화 화면·빌더·출처/검수 JSON·문서뿐이다. Claude의 병렬 속도 수정과 기존 사용자 변경은 보존했다.

다음 사용자 확인 한 단계: [대화 화면](2026-11-01_conversation_view.html)에서 **C2**를 선택해 “크림으로 선택했어요”와 실제 `소스=없음`을 대조한다. 이 작업은 실패를 시각화한 것이며 해당 모델 오류를 고쳤다는 선언이 아니다.

### 종료

HTML 담당자의 마지막 `worker_done`을 수락하고 `ctx_b6a44eb88b5e`를 release했다. 이번 세 작업과 출처 링크 후속 작업의 완료·인계 기록을 처리했고, Run의 `reclaimable` 조회 결과는 0개다. 작업 3/3 완료. 속도 최적화는 Claude 소유로 유지하며, 이번 작업에서는 모델 추론·프롬프트 수정·새 학습을 추가하지 않았다. `git diff --check` 통과.
