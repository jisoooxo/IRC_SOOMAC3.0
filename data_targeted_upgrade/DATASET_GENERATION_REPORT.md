# Decision v3.2 + Response v1.1 Targeted Dataset Generation

- 참고: [Decision validation](decision_v3_2_candidate/validation_report.json)

  → Decision supplement의 schema, semantic, 중복, contamination 검사 결과이다.
- 참고: [Response validation](response_v1_1_candidate/validation_report.json)

  → Response supplement의 runtime lifecycle, 사실성, pending, next-prompt 검사 결과이다.

## Decision v3.2

```text
base rows: 3015
supplement rows: 450
candidate total: 3465

general/task: 90
pending: 70
recommendation: 100
preference: 45
reference: 35
unsupported: 65
compound: 25
rehearsal: 30

schema invalid: 0
duplicate: 0
frozen overlap: 0
semantic suspicious: 0
```

## Response v1.1

```text
base rows: 898
supplement rows: 200
candidate total: 1098

pending+general: 70
pending+status: 35
pending+unrelated: 25
no-pending control: 20
execution control: 10

fact failures: 0
pending failures: 0
next-prompt failures: 0
duplicate: 0
frozen overlap: 0
```

## 판정

```text
Decision v3.2: PASS
Response v1.1: PASS
training_started: false
```

**기존 dataset, runtime, prompt, Adapter는 수정하지 않았다.**

Cause: v3.1 평가에서 남은 route·pending·recommendation·over-mutation과 Response pending reminder 실패를 직접 보강했다.

Effect: 두 candidate는 기존 base를 보존한 채 별도 학습 입력으로 사용할 수 있다.

## 용어 정리

- **Targeted supplement**: 실제 실패 family만 보강한 추가 학습 데이터이다.
- **Minimal pair**: 표현은 비슷하지만 의미 경계가 다른 두 예시이다.
- **Frozen contamination**: 평가 문장이나 입력이 학습 데이터에 섞이는 문제이다.
- **Pending reminder**: 일반 질문에 답한 뒤 아직 남은 실행·추천 확인을 다시 묻는 응답이다.
- **Rehearsal**: 새 보강 때문에 기존 성능이 무너지지 않도록 이미 잘 되는 예시를 소량 유지하는 방법이다.
