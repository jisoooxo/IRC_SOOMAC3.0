# Failure source audit

참고 기준: [`DECISION_RESPONSE_DATA_SALVAGE_AUDIT.md`](DECISION_RESPONSE_DATA_SALVAGE_AUDIT.md)

→ 한 turn의 최종 문장이 이상해도 앞 단계가 틀렸다면 Response hallucination으로 중복 집계하지 않는다.

## STT failure

대표 sample 26행 중 3행이다.

| sample | observed text | 판단 |
|---|---|---|
| `D02` | `계살 먹고 싶어` | 문맥상 게살이 명확하여 Decision이 복원 가능하다. |
| `D03` | `소시질한 계살 다 보통양으로 줘` | 소시지·게살 두 대상을 복원할 수 있는데 legacy가 clarify했다. |
| `D04` | `나 물커한 걸 싫어해` | 특정 재료가 아니라 물컹한 식감 preference로 보존한다. |

STT raw audio나 별도 ASR 원문/정규화 pair는 발견하지 못했다. runtime의 `user_text`는 STT 결과로 보이지만 원음 대조가 없으므로 과도한 교정 gold를 만들지 않는다.

## Decision failure

대표 sample에서 5행이며 Multiple과 중복된다.

| sample | legacy 문제 | v3 label |
|---|---|---|
| `D03` | 복원 가능한 두 재료를 clarify | 소시지 normal + 게살 normal |
| `D04` | dislike 문장을 preference remove로 반전 | `물컹한 식감 싫어` add |
| `D06` | 문맥상 라면 주문을 Python reference guard로 차단 | `noodle_type=라면` 의미 보존 |
| `D07` | unsupported 햄 삭제 의도를 clarify | `toppings.햄=none` |
| `D11` | 매운 취향을 recommendation criteria에만 넣음 | preference add + recommendation request |

## Policy/runtime failure

대표 26행에서는 0행이다. 별도 evidence에는 hardened routing의 `python_runtime` 29건과 full-cycle failure가 있다. 이들은 현재 runtime에서 재실행하기 전에는 v3 policy failure로 단정하지 않는다.

## Response failure

대표 sample에서 3행이며 모두 앞 단계 문제와 함께 나타났다.

| sample | 문제 |
|---|---|
| `D03` | Decision이 clarify인데 소시지와 게살을 반영한다고 말했다. |
| `D04` | 물컹한 식감을 매운맛 dislike로 바꾸어 말했다. |
| `D07` | 명확한 unsupported target을 알려주지 않고 다시 말해 달라고만 했다. |

Response 단독 failure는 structured state가 맞는 turn에서만 붙인다. 새 Response review sample 12행은 모두 current structured input에서 다시 쓴 gold이므로 기존 response failure count에 넣지 않는다.

## Ambiguous

대표 sample 1행이다.

```text
history: 떡볶이와 라면을 함께 설명함
user: 그거 줘
gold: {"route":"task","clarify":true}
```

두 후보가 남아 있으므로 특정 메뉴 mutation으로 만들지 않는다.

## Multiple

대표 sample 3행이다.

- `D03`: STT + Decision + Response
- `D04`: STT + Decision + Response
- `D07`: Decision + Response

## No failure

대표 sample 26행 중 19행이다. legacy label이 current contract와 완전히 같다는 뜻이 아니라, salvage할 사용자 의미나 structured fixture 자체에 failure가 없다는 뜻이다.

## 현재 집계

failure label은 복수 선택이므로 합계가 sample 수와 일치하지 않을 수 있다.

| label | count |
|---|---:|
| stt | 3 |
| decision | 5 |
| policy | 0 |
| response | 3 |
| ambiguous | 1 |
| multiple | 3 |
| no_failure | 19 |

Cause: 기존 평가는 최종 PASS/FAIL이나 issue class만 있어 단계별 원인을 안정적으로 분리하지 못했다.

Effect: 같은 turn을 Decision 재학습과 Response 재작성에 서로 다른 grade로 사용할 수 있다.
