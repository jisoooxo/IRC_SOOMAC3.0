# Decision v3.2 Targeted Supplement Design

Decision v3.1 candidate 3,015행은 그대로 두고 실제 v3.1 failure family를 보강한다.

## Family allocation

| family | rows |
|---|---:|
| general_question_vs_task | 90 |
| pending_accept_reject_boundary | 70 |
| recommendation_confirmation_reject | 55 |
| recommendation_revise_preference | 45 |
| question_vs_mutation | 45 |
| description_impression_vs_mutation | 35 |
| reference_ambiguous_reference | 35 |
| compound_intent_spillover | 25 |
| field_category_amount | 20 |
| regression_preservation | 30 |

## 원칙

- Meaning = Model, Physical truth = Code 원칙을 유지한다.
- unsupported 주문은 target 문자열을 그대로 보존한다.
- validation/test/challenge/holdout와 v3.1 failure 원문은 복사하지 않는다.
- runtime semantic guard를 만들지 않으며 학습을 시작하지 않는다.
