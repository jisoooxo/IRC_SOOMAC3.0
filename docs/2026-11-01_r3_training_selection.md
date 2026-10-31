# r3 교정본 채택과 신규 학습

[작업 기록](2026-11-01_dataset_rebuild_worklog.md) · [보강 명세](2026-11-01_conversation_supplement_spec.md)
→ 필요한 첫 실행 확인을 복원하고 복합 실행을 보강한다. 원본·기존 adapter를 덮어쓰지 않는다.

**역사적 r3 실행 기록:** 최신 후보와 지침은 [worklog의 최상위 후속 지침](2026-11-01_dataset_rebuild_worklog.md)을 따른다. 현재는 `compound_semantics_20261101_r2`가 최신 저장 후보이며 **Decision 원격 별도 / Response→Recommendation 로컬 순차**다. 아래 「P→Q→D 모두 로컬, batch1×acc16·6epoch 재사용」은 당시 설정이지 새 최종 학습 설정이 아니다. 새 runner/승인 영수증 결속과 파라미터 점검·실측은 미완료이며 기존 r3 큐를 그대로 재개하지 않는다.

## 채택 범위

최신 사용자 「교정 ㄱㄱ … 전체 데이터에서 뺄건 빼고 학습시키자」를 신규 학습 승인으로 기록한다. 운영 배포·ROS 실물·원격 변경·commit/push는 승인하지 않은 범위다. 현재 작업 브랜치는 `llm_final_2`이다.

최종 데이터는 외부 학습 작업공간의 `candidates/conversation_supplement_20261101_r3_a3/dataset`이다. 총 **8,230행 / train 6,566행**이며 train은 Decision 4,313 / Recommendation 617 / Response 1,636행이다. validation/test/challenge/holdout의 기존 분할과 원문은 보존한다.

1. 기존 HOLD13 제외·이전 교체108·계약 정답 교정68을 연결한 `base_filtered` 7,860행의 불변 판정을 재사용한다. 감사 단독4행은 이 기반에서 이미 제외됐으므로 다시 제외한 성과로 세지 않는다.
2. 새39대화/201턴을 실제 CPU Graph로 재생한다. P26개에 필요한 첫 실행 안내를 보완하고 그 다음 턴의 D/Q/P history를 함께 재생성한다. 새 초회 알레르기+주문+commit, 추천 수락+commit, 버섯만 변경+commit, 토핑 제외+commit을 보강한다. 일반 선택·상태질문·명시적 보류는 자동 실행하지 않는다.
3. 신규 HOLD2행(`NS01-A04-T04-P`, `NS01-D01-T02-P`)은 정답 질문 시점이 모호해 학습에서 격리한다. 실제 다음 턴 history는 삭제하지 않는다. 정확 중복6행은 alias 장부로 제외한다. 새 채택은 D197/Q15/P158, 총370행이다.
4. 학습은 기본 모델에서 새 LoRA를 시작한다. 이전 데이터 optimizer를 resume하지 않는다. P→Q→D 순차, batch1×accumulation16, NF4, 기존 learning rate/6epoch 설정과 text-decoder LoRA를 재사용한다. 새 출력 디렉터리만 사용하고 기존 adapter는 바꾸지 않는다.
5. GPU는 다른 Claude의 STT/TTS/vLLM 시험 소유다. 강제 종료하지 않는다. 학습 전 유휴 상태를 30초 간격3회 확인하고 실제 최장 train+validation 입력으로 역전파/optimizer 프로파일을 통과해야 본 학습을 시작한다. 대기는 각 단계 최대45분, 오류·OOM·기한 초과 시 중단하며 자동 재시도하지 않는다.

## 검증과 교차검수

전체8,230행의 builder·schema·정확 중복·동일 입력 상충 정답을 검사했다. 기반7,860행의 실제 tokenizer/template/EOS/mask 영수증을 전체 row hash와 현재 system에 다시 결속했다. 바뀌거나 새로 만든370행은 실제 trainer AST로 다시 인코딩하고 저장 후 row hash를 대조했다. 최대 길이는 D8616/Q4332/P8430이며 cap은 D9216/Q5120/P9216이다. history는 자르지 않는다. 이 cap은 아직 GPU 학습 성공 값이 아니다.

첫 독립 Codex 검수의 F07/F08 지적을 채택했다. 「처음에 얘기한」의 선행 대상에 거절한 재료도 있어 「방금 골라둔 재료」로 명확히 하고 a3를 생성했다. 과거 패키저 지적은 새 `package_r3.py`로 해소했으며 모든 NS01/NS03 행을 검사한다. 최종 Codex(`gpt-6-sol/high`)는 코드 검토상 차단 사유 없음으로 찬성했다. 검수 시 a3 summary를 확인하지 못한 한계는 root가 실제 summary/영수증으로 대조한다. 독립 검수는 모델 성공·GPU 성공 판정이 아니다. CLI 시작 때 sandbox 상태 경로 쓰기 실패는 escalated 동일 명령으로만 해소했고, MCP 진단 경고는 실제 최종 검토 판정과 구분한다.

초기 r3 안내문 생성에서 잘못 연결된 재료/양을 root 전후 필드 대조가 발견했다. 그 시도는 최종 학습본으로 채택하지 않았고 외부 `conversation_supplement_20261101_r3`에 그대로 보존한다. a2는 지시 대상 명확화 이전 시도다. 두 시도 모두 adapter 학습에 사용하지 않는다. 올바른 a3만 manifest hash로 선택한다.

새 회귀의 「보류+수량 변경이면 pending=null」 단언은 잘못된 검사 가정이었다. `llm_langgraph.py`의 `_resolve_pending` 실행 거절 뒤 `finalize_task` 9번 분기는 유효한 새 현재 주문 변경에 execution pending을 다시 만들 수 있다. pending은 실행 권한이 아니다. 검사만 실제 계약에 맞게 교정해 **execute=false, 정확한 pending 대상, 변경량·언급하지 않은 기존량 보존, 다음 턴 state 전달**을 함께 확인한다. Runtime을 검사에 맞춰 바꾸지 않았다.

## 한계와 완료 기준

전역 의역 누수 부재는 미입증이고 수치 편향 합격 기준은 사용자 보류 상태다. 기존 장부의 의미 검수와 이번 교정/보강 범위의 독립 검수를 재사용한 것으로, 전체8,230행의 새로운 LLM 의미 전수검수라고 하지 않는다. CPU gold 주입과 builder 성공은 실제 adapter 성능이 아니다. 학습 완료 후 세 adapter를 붙인 실제 누적 시나리오에서 최신 주문·수량·제약·commit·실행·응답을 대조해야 한다.

Cause: 첫 실행 안내 누락과 정상 복합 의도 예시 부족이 검수 후에도 남았고, 기존 학습 길이8192는 이번 모든 입력을 보존하지 못한다.

Effect: 제외/교정 근거·전후 필드·직렬화와 학습 선택을 결속한 새 버전으로 순차 학습할 수 있다. 실제 학습 시작/완료는 큐 `STATE.json`과 각 단계 로그/`training_complete.json`으로만 판정한다.
