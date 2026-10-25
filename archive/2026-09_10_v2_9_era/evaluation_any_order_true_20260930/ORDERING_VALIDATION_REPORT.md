# XGrammar any_order=True validation report

## Change scope

Only the JSON object key-order constraint was relaxed.

```diff
- compile_json_schema(DECISION_SCHEMA)
+ compile_json_schema(DECISION_SCHEMA, any_order=True)

- compile_json_schema(schema)
+ compile_json_schema(schema, any_order=True)
```

Changed files:

- `/home/roma/IRC_SOOMAC3.0/src/soomac_irc/soomac_irc/decision_model.py`
- `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/dataset/soomac_decision_adapter_v2_2_5_semantic_rebalance/scripts/eval_decision_adapter.py`

Schema, prompt, dataset, model, checkpoint, quantization and generation settings were unchanged. No LoRA training was performed.

## Official eight-suite result

| Suite | any_order=False | any_order=True | Improved | Regressed | Remaining failures |
|---|---:|---:|---:|---:|---:|
| latest_contract | 45/48 | 46/48 | 1 | 0 | 2 |
| hard_v221 | 14/16 | 16/16 | 2 | 0 | 0 |
| hard_v222 | 23/24 | 23/24 | 0 | 0 | 1 |
| hard_v223 | 44/48 | 45/48 | 1 | 0 | 3 |
| hard_v224 | 35/36 | 35/36 | 0 | 0 | 1 |
| hard_v225 | 74/96 | 95/96 | 21 | 0 | 1 |
| challenge | 42/64 | 64/64 | 22 | 0 | 0 |
| runtime_holdout | 41/45 | 44/45 | 3 | 0 | 1 |
| **Total** | **318/377** | **368/377** | **50** | **0** | **9** |

## Challenge compound metrics

```text
multi_intent_completeness=45/45=1.0000
confirmation_compound=16/16=1.0000
restriction_add compound=5/5=1.0000
challenge exact=64/64=1.0000
```

## Regression check

`regressed_cases.jsonl` has zero rows. No case changed from correct under `any_order=False` to incorrect under `any_order=True`.

## Result files

Each suite has `{suite}_predictions.jsonl`, `{suite}.log`, and `{suite}_semantic.log` in this directory.
