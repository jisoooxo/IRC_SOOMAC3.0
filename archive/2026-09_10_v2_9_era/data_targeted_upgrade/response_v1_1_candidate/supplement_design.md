# Response v1.1 Targeted Supplement Design

Response v1 train 898행은 그대로 두고 pending 이후 대화 진행 실패를 보강한다.

## Family allocation

| family | rows |
|---|---:|
| pending_general_question | 70 |
| pending_status_question | 35 |
| pending_unrelated_explanation | 25 |
| pending_preference_statement | 20 |
| pending_recommendation_context | 20 |
| no_pending_control | 20 |
| execution_completion_control | 10 |

## 원칙

- pending이 있으면 현재 질문에 답한 뒤 필요한 확인 질문을 이어간다.
- pending이 없거나 실행 중·완료 상태이면 불필요한 질문을 붙이지 않는다.
- structured truth 밖의 주문·실행·완료 사실을 만들지 않는다.
- 기존 Response v1을 덮어쓰지 않으며 학습을 시작하지 않는다.
