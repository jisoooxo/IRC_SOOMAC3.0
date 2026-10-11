# Decision v3.1 Candidate Dataset

IRC_SOOMAC3.0 `natural_multiturn_v3` Decision Adapter용 targeted candidate dataset이다.

## 구성

- `train.jsonl`: 최종 v3.1 candidate train, 3,015 rows
  - 기존 Decision v3 train: 2,595 rows
  - targeted supplement: 420 rows
- `supplement_train.jsonl`: v3.1 targeted supplement 420 rows
- `supplement_provenance.jsonl`: supplement 생성/분류 provenance
- `supplement_design.md`: supplement family 설계
- `source_manifest.json`: 원본 경로, row count, SHA256, frozen evaluation reference
- `validation_report.json`: schema/duplicate/overlap 검증 결과
- `SHA256SUMS.txt`: 이 패키지 내부 파일 checksum

## 핵심 목적

Decision v3의 높은 recall을 유지하면서 over-mutation을 줄이기 위한 보강 데이터다.
주요 contrast는 pending confirmation vs order mutation, question vs mutation,
description vs request, preference vs grounding, ambiguous reference,
order field category, noncanonical amount, unsupported meaning preservation 등이다.

## 중요한 원칙

- `Meaning = Model`
- `Physical truth = Code`
- 명확한 unsupported 주문은 Decision이 의미를 보존하고, 지원 여부는 Policy가 판단한다.
- Python keyword/regex semantic guard를 학습 목표로 되돌리지 않는다.

## 평가 격리

Frozen validation/test/challenge/holdout 및 corrected long-horizon evaluation 데이터는
이 패키지에 포함하지 않았다. `source_manifest.json`의 reference를 사용해 동일 평가셋으로
Decision v3와 v3.1을 비교해야 한다.
