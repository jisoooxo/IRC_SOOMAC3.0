# 대화 시각화용 보강 데이터 출처

[감사 결과](../training/learning_first_v43/final/goals/GL10/data_reaudit_20261031/RESULT.md) · [비활성 후보 장부](../training/learning_first_v43/final/goals/GL10/data_reaudit_20261031/candidate/natural_order_delta_r1/MANIFEST.json) · [replay 원본](../training/learning_first_v43/final/goals/GL10/data_reaudit_20261031/required_replay_r1/TRACES.jsonl)
→ **13턴의 D/Q/P 정답과 전후 state를 시각화용으로 추출했다. 학습 또는 실제 모델 성공 증거가 아니다.**

## 범위와 선택

- 원본: 10대화·88기록에서 작성한 Agent 정답 104개. 최종 고유 보강 후보는 **93행(D59/Q4/P30)**, alias 11개이다.
- 이 파일은 **대표 13턴, Agent 정답 22개**만 담는다. 네 사용자 흐름에서 추천 제안, 별도 수락, 복합 즉시 실행, 수량 변경, 알레르기, 제외를 골랐다.
- `examples` 한 행은 oracle replay의 실제 사용자 턴이다. `decision`, `recommendation`, `response`는 그 턴에서 호출된 Agent의 **정답 target**이다. 호출되지 않은 Agent는 `null`이다.
- `history`는 턴 직전 대화이고 `before_state`·`after_state`는 replay 원본 state이다. `pending`은 두 state의 pending을 그대로 가리킨다. `commit`은 trace의 Decision 해석값이고 `policy.execute`와 분리해 읽는다.
- `source`에는 Agent별 원래 ID, 고유 대표 ID, 보강 파일 경로와 파일 SHA-256, target SHA-256, ledger 상태를 남겼다. `KEEP_WITH_ALIAS`는 원래 replay 호출이 대표 고유 행에 연결됐다는 뜻이다.

## 상태 경계

- `candidate/natural_order_delta_r1`는 비활성 overlay다. **새 학습본 미조립, 학습 실행 없음, loader 연결 없음**이다. 각 행의 `training_inclusion`은 `NOT_TRAINED_OR_NOT_PROVEN`이다.
- 원본 대체 108행과 계약 HOLD 12행은 이 13턴의 보강 예시에 섞지 않았다.
- `prompt_contrast_examples.py`의 C01–C07/S10–S12는 정적 few-shot 대조이며 **prompt-only**이다. 이 JSON의 학습 후보 예시로 넣지 않았다.
- 별도 모델 실측 17턴은 여기 없다. 이 JSON의 `model_inference: false`와 `ROS: false`는 정답 주입 replay의 성격이다.
- 원본 상태 `STRUCTURE_ALIGNED_SEMANTICS_PENDING`는 replay 작성 당시 표기다. 최종 `KEEP`/alias 판정은 [LEDGER](../training/learning_first_v43/final/goals/GL10/data_reaudit_20261031/candidate/natural_order_delta_r1/LEDGER.jsonl)와 [RESULT](../training/learning_first_v43/final/goals/GL10/data_reaudit_20261031/RESULT.md)에서 재사용했다.

Cause: 사용자 발화 하나의 정답만 보면 다음 턴의 pending·수량·실행 여부가 보이지 않는다.

Effect: 같은 에피소드의 실제 history와 전후 state를 Agent 정답 및 해시 장부에 함께 연결했다.

복사 검증: 13/13턴의 `after_state`·`history`가 정본 `TRACES.jsonl`과 일치, 22/22개 D/Q/P target이 각 `representative_id` 원본 record와 일치, 5/5개 Markdown 출처 링크가 실제 파일에 연결됨.
