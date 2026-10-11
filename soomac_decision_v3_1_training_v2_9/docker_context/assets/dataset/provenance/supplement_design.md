# Decision v3.1 Targeted Supplement Design

기존 Decision v3 train 2,595행은 변경하지 않고 corrected TRUE failure만 보강한다.

## Family allocation

| family | rows | 핵심 contrast |
|---|---:|---|
| pending confirmation vs order | 60 | 실행 취소 / 주문 삭제 / 실행 수락 |
| question vs mutation | 55 | entity 상태 질문 / 직접 주문 |
| description vs request | 45 | 감상 / 취향 요청 / grounded 요청 |
| preference vs grounding | 60 | preference only / strong grounding / dislike restriction |
| ambiguous reference | 45 | 단일 후보 resolve / 복수 후보 clarify / 명시 target 우선 |
| order field category | 55 | sauce / noodle type / noodle portion |
| noncanonical amount | 35 | 수치 양 clarify / canonical low-normal-high |
| unsupported preservation | 45 | 설명·지원 질문 / target 문자열 보존 주문 |
| general interlude return | 10 | task → general → 이전 task 수정 |
| compound spillover | 10 | 여러 명시 intent만 반영 |

## Exclusion

- CONTRACT_CORRECT, AMBIGUOUS_EVAL, downstream failure는 source failure로 사용하지 않는다.
- validation/test/challenge/holdout/corrected long-horizon 문장을 train으로 복사하지 않는다.
- Python keyword guard나 runtime semantic rule을 만들지 않는다.
- 이번 단계에서는 학습을 시작하지 않는다.
