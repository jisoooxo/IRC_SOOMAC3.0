# 단계 전환 고정 안내 ON/OFF와 보강 명세 국소 검수

[운영 노드](../src/soomac_irc/soomac_irc/llm_langgraph_node.py) · [토글 회귀](../src/soomac_irc/test/test_section_response_toggle.py) · [대화 명세](2026-11-01_conversation_supplement_spec.md) · [보강93행 검사](../training/learning_first_v43/section_response_toggle_20261101/SUPPLEMENT_CHECK.json)
→ **상단 `hard_cording_response`로 야채·육류·추가재료 단계 안내를 선택한다. 기본True는 기존 고정 안내이며, False는 기존 모델 event 경로이다.**

2026-11-01, `llm_final_2`. 사용자 승인 뒤의 미커밋 구현이다. 새 GPU 추론·학습·adapter 교체·실물 ROS·원격 변경·commit/push는 하지 않았다. cache/환경 migration과 VLM 파일·판정 정책은 수정하지 않았다.

## 1. 실제 적용 범위

| 설정 | 단계 전환 안내 | 주문·추천·실행 상태 |
| --- | --- | --- |
| `hard_cording_response = True` (기본) | 기존 `_section_prompt()` 또는 미리 고른 주문의 확인 문장 | 기존 경로 유지, 단계 전환 Q/P 미호출 |
| `hard_cording_response = False` | 기존 `generate_section_event_response()`를 호출해 Q/P 모델 안내 | 유효한 사전 주문은 execution 확인으로 전달. 후보가 없으면 기존 조건부 Q가 제안할 수 있으나 주문 확정·실행 허가는 만들지 않음 |
| False에서 모델 오류 | 기존 안전 fallback | 잘못된 schema/빈 응답/예외를 처리하고 제한·주문 유지. 모델의 그럴듯하지만 의미가 틀린 문장을 자동 인증하는 기능은 아님 |

토글은 `llm_langgraph_node.py`의 import 다음 상단에 있다. 파일을 바꾼 뒤 **노드를 재시작해야** 적용된다. 실행 중인 Python 프로세스가 파일 편집만으로 즉시 전환되지는 않는다.

뚜껑/소스의 자동 작업·실행 시작 고정 안내, 초기 안내, 안전 fallback, 사용자 발화 턴의 원래 D/Q/P 경로는 이 토글로 일괄 변경하지 않는다. ROS topic·레일 이동·중복 전환 방지·busy STT 정책도 그대로다. 실제 단계 이동 event만 전달하고 모델 입력의 사용자 발화는null이다. 가짜 user 메시지를 만들거나 모델이 새로운 ROS 명령을 발행하게 하지 않는다.

Cause: 모델 event 응답 호출부가 주석 처리되어 있어 데이터나 모델이 좋아져도 해당 단계 안내는 항상 고정 문장이었다.

Effect: 현재 고정 안내를 기본값으로 보존하면서, 노드 재시작으로 모델 안내 경로를 선택할 수 있다.

## 2. 구현과 로그

주석 처리된 호출부를 최소 조건 분기로 복원했다. 기존 event helper·입력 builder·후보 검증·응답 fallback은 재사용한다. 새 자연어 keyword 판정·고정 문장·state/schema field·repair 재시도는 넣지 않았다.

`section_transition_response` 로그에 토글 값과 실제 `next_prompt`를 남긴다. 모델모드에서 호출 전 Response trace 개수를 보관하고, 호출 뒤 늘어난 trace만 이번 응답으로 기록한다. Q실패로 P가 호출되지 않은 경우 과거 응답 trace를 이번 응답인 것처럼 붙이지 않는다. 같은 사용자 턴의 Decision trace를 지우지 않는다. 기존 오류 로그와 history의 실제 발행 문장은 보존한다.

Cause: 기존 주석을 그대로 살리면 Q실패로 P가 미호출된 때에도 `traces[-1]`이 과거 응답일 수 있다.

Effect: 새 호출의 입력/출력만 로그에 연결하여 모델 안내와 고정 안내·fallback을 구분할 수 있다.

## 3. 테스트 결과

기존 운영 메서드를 AST로 분리해 실제 본문을 실행하고 모델/publisher만 mock으로 대체했다. ROS전송·GPU생성 문장 품질 시험은 아니다.

1. 기존 단계/Response 계약 테스트68건·하위6건 통과. 기본True에서 기존 경로가 유지된다.
2. 토글 신규17건을 더한 관련 묶음85건·하위6건 통과. False의 completed/skipped, 사전 주문, 안전 후보 필터, 모델 오류, trace 분리, 중복 전환, lid/sauce 보존, 실제 skip 호출 경로를 포함한다.
3. 확장 LLM15파일 **853건·하위6건 통과**, exit0. SWIG 사용중단 경고2종과 sandbox NVML초기화 경고가 있었다. VLM프롬프트 테스트는 범위에 넣지 않았다.

확장 검증 명령:

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
  src/soomac_irc/test/test_section_question_events.py \
  src/soomac_irc/test/test_section_response_toggle.py \
  src/soomac_irc/test/test_recommendation_quantity_prompt.py
```

## 4. 신규 보강 명세·감사 표현 제외와 데이터 검수 범위

사용자의 최신 지시는 감사 표현 예시를 추가하는 것이 아니라 **이번 신규 보강에서 제외하는 것**이다. 명세의 감사 단독 행을 제거하고 대화 종료·즉시 실행 수락을 감사 표현으로 추정하는 새 규칙도 만들지 않았다. 원본/history에서 해당 문자열을 검색해 지우는 방식은 사용하지 않는다.

기존 비활성 후보의 DA31 고유93행을 읽어 실제 현재 builder로 재구성하고 target schema와 대조했다. D59/Q4/P30 모두 입력필드·값과schema 일치, session원본불변이다. 이전 의미 판정/10대화gold연결 결과는 재사용하며 새 전수 의미93건 합격으로 보고하지 않는다. 모델모드 이벤트의 생성 문장·추가 Q호출 이후 닫힌루프 품질도 이번CPU검사로 인증하지 않는다.

이미 있는 DA31 감사 단독 예시는 D2/P2이며 `SUPPLEMENT_CHECK.json`의 `next_assembly_exclusions`에 정확 source·line·record SHA와 함께 기록했다. 대상은 `DA31-CORE-polite-T05-D/P`, `DA31-CORE-directive-T05-D/P`이다. **현재 후보 파일을 덮거나 4행을 실제 삭제한 것은 아니다.** 다음 버전 조립에서 학습 대상만 제외하고 뒤 턴에 남은 실제 history는 보존해야 한다. 기존 93행 중 이4행을 제외하면89행이지만, 이는 조립 전 선택 산식이며 새 후보 총행수를 바꿔 적은 결과가 아니다.

이번 명세에 새로 제안한 문맥들은 아직 정식 학습JSONL이 아니다. 주문/추천 분리·선택/실행 구분·수량보존·없는pending·과거assistant오답·반복질문을 명세에서 대조했고, 실제행 생성 뒤 schema/Graph전후/의미검수/중복/길이 검증은 남아 있다. 기존 후보의runtime hash를 맞추려고 과거검수결과를 덮거나 전수검수를 반복하지 않는다.

Cause: 감사 단독 사례는 현재 사용자 목표인 주문 흐름 보강에서 제외됐고, 새 문서 초안을 실제 검수된 학습행처럼 취급할 수 없다.

Effect: 제외 대상을 출처에 결속하고 원본을 보존하면서, 다음 학습 조립과 신규행 검수에 적용할 수 있다.

## 5. 남은 한 단계

기본True를 유지한 채, 대화 명세의 주문·추천·복합실행·실패복구 추가분을 실제 입력/정답으로 생성하고 다음 비활성 후보에 합칠 때 위4행 제외를 적용한다. False의 실제 모델 품질/반복 여부는 별도 GPU 누적회귀로 확인해야 한다. 이번 작업으로 재학습·실물 실행까지 승인된 것은 아니다.
