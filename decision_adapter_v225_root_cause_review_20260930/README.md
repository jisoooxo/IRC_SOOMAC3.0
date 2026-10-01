# Decision Adapter v2.2.5 root-cause review bundle

## Scope and safety

- Operational reference: `/home/roma/IRC_SOOMAC3.0`
- Actual trainer: `/home/roma/Downloads/SOOMAC_DECISION_DOCKER_V2_7_REBUILT/docker_context/train_in_container.py`
- This bundle contains no base-model weights and no LoRA `adapter_model.safetensors`.
- Build/install trees and large training logs are excluded.
- All copied originals were copied byte-for-byte and verified with SHA-256.
- **The actual operational source was not modified.** `git status --short` was empty before packaging.

## any_order=True temporary experiment

The original evaluator contains:

```python
compile_json_schema(schema)
```

XGrammar 0.2.5 defaults this to `any_order=False`. The temporary wrapper in
`temporary_any_order_eval/eval_anyorder_runner.py` loaded the evaluator source in memory and replaced exactly one call with:

```python
compile_json_schema(schema, any_order=True)
```

It did not write the change back to the evaluator or to IRC_SOOMAC3.0. Temporary predictions are isolated under `temporary_any_order_eval/`.

## Result summary

| Scope | any_order=False | temporary any_order=True |
|---|---:|---:|
| Existing 281 | 244/281 | 273/281 |
| New hard v225 96 | 74/96 | 95/96 |
| Challenge 64 | 42/64 | 64/64 |
| Runtime holdout 45 | 41/45 | 44/45 |
| All 377 | 318/377 | 368/377 |

## Remaining nine errors with any_order=True

| ID | Family | Expected | Predicted |
|---|---|---|---|
| `real_runtime_040` | `real_runtime_holdout` | `{"queries":[{"type":"order_field","target":"noodle_type"}]}` | `{"queries":[{"type":"robot_completed"}]}` |
| `v222h20010_01` | `hard_clarify` | `{"clarify":true}` | `{"recommendation":{"action":"cancel"}}` |
| `v223h3001_01` | `hard_clarify_dense` | `{"clarify":true}` | `{"order":{"toppings":{"페퍼론치노":"high"}}}` |
| `v223h3002_01` | `hard_clarify_dense` | `{"clarify":true}` | `{"order":{"toppings":{"치즈":"low"}}}` |
| `v223h3006_01` | `hard_clarify_dense` | `{"clarify":true}` | `{"order":{"toppings":{"치즈":"low"}}}` |
| `v224h2004_01` | `hard_near_neighbor_unsupported` | `{"clarify":true}` | `{"order":{"toppings":{"페퍼론치노":"low"}}}` |
| `v225h0068_01` | `hard_stt_surface` | `{"restrictions":[{"target":"버섯","reason":"dislike","action":"add"}]}` | `{"order":{"toppings":{"버섯":"low"}}}` |
| `v22g9038_01` | `confirmation_partial` | `{"order":{"toppings":{"치즈":"high"}},"commit":true,"confirmation":"reject"}` | `{"order":{"toppings":{"치즈":"high"}},"confirmation":"reject"}` |
| `v22g9039_01` | `confirmation_partial` | `{"order":{"toppings":{"치즈":"high"}},"commit":true,"confirmation":"reject"}` | `{"order":{"toppings":{"치즈":"high"}},"confirmation":"reject"}` |

The same records are available as JSONL at `temporary_any_order_eval/failed_9_cases.jsonl`.

## File roles

- `actual ...`: file was used directly by the trainer, evaluator, or current IRC runtime.
- `temporary ...`: output or wrapper created only for the any_order=True investigation.
- `analysis-generated ...`: inventory, summaries, README, or reproducer created for review.
- Every copied file's original absolute path is listed below and in `FILE_ORIGINS.tsv`.

## Complete origin map

| Bundle path | Original absolute path | Role |
|---|---|---|
| `dataset_contract/decision_schema.json` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/dataset/soomac_decision_adapter_v2_2_5_semantic_rebalance/contract/decision_schema.json` | actual v2.2.5 dataset/contract used by trainer |
| `dataset_contract/decision_system.txt` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/dataset/soomac_decision_adapter_v2_2_5_semantic_rebalance/contract/decision_system.txt` | actual v2.2.5 dataset/contract used by trainer |
| `dataset_contract/decision_train.jsonl` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/dataset/soomac_decision_adapter_v2_2_5_semantic_rebalance/records/decision_train.jsonl` | actual v2.2.5 dataset/contract used by trainer |
| `dataset_contract/decision_validation.jsonl` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/dataset/soomac_decision_adapter_v2_2_5_semantic_rebalance/records/decision_validation.jsonl` | actual v2.2.5 dataset/contract used by trainer |
| `dataset_contract/family_distribution_and_audit_summary.json` | `N/A (analysis-generated)` | analysis-generated exhaustive train/validation family and structural audit |
| `dataset_contract/manifest.json` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/dataset/soomac_decision_adapter_v2_2_5_semantic_rebalance/manifest.json` | actual v2.2.5 dataset/contract used by trainer |
| `evaluation_code/eval_decision_adapter.py` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/dataset/soomac_decision_adapter_v2_2_5_semantic_rebalance/scripts/eval_decision_adapter.py` | actual evaluation source |
| `evaluation_code/eval_semantic_metrics.py` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/dataset/soomac_decision_adapter_v2_2_5_semantic_rebalance/scripts/eval_semantic_metrics.py` | actual semantic metric source |
| `evaluation_data/challenge.jsonl` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/dataset/soomac_decision_adapter_v2_2_5_semantic_rebalance/sft/decision_challenge_sft.jsonl` | actual evaluation input |
| `evaluation_data/hard_v221.jsonl` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/dataset/soomac_decision_adapter_v2_2_5_semantic_rebalance/hard_eval_v221/hard_eval_v221_sft.jsonl` | actual evaluation input |
| `evaluation_data/hard_v222.jsonl` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/dataset/soomac_decision_adapter_v2_2_5_semantic_rebalance/hard_eval_v222/hard_eval_v222_sft.jsonl` | actual evaluation input |
| `evaluation_data/hard_v223.jsonl` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/dataset/soomac_decision_adapter_v2_2_5_semantic_rebalance/hard_eval_v223/hard_eval_v223_sft.jsonl` | actual evaluation input |
| `evaluation_data/hard_v224.jsonl` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/dataset/soomac_decision_adapter_v2_2_5_semantic_rebalance/hard_eval_v224/hard_eval_v224_sft.jsonl` | actual evaluation input |
| `evaluation_data/hard_v225.jsonl` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/dataset/soomac_decision_adapter_v2_2_5_semantic_rebalance/hard_eval_v225/hard_eval_v225_sft.jsonl` | actual evaluation input |
| `evaluation_data/latest_contract.jsonl` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/dataset/soomac_decision_adapter_v2_2_5_semantic_rebalance/latest_contract_eval/latest_contract_eval_sft.jsonl` | actual evaluation input |
| `evaluation_data/runtime_holdout.jsonl` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/dataset/soomac_decision_adapter_v2_2_5_semantic_rebalance/runtime_holdout/real_surface_current_schema_sft.jsonl` | actual evaluation input |
| `operational_code/__init__.py` | `/home/roma/IRC_SOOMAC3.0/src/soomac_irc/soomac_irc/__init__.py` | current IRC_SOOMAC3.0 operational source |
| `operational_code/agent_contract.py` | `/home/roma/IRC_SOOMAC3.0/src/soomac_irc/soomac_irc/agent_contract.py` | current IRC_SOOMAC3.0 operational source |
| `operational_code/agent_prompts.py` | `/home/roma/IRC_SOOMAC3.0/src/soomac_irc/soomac_irc/agent_prompts.py` | current IRC_SOOMAC3.0 operational source |
| `operational_code/decision_model.py` | `/home/roma/IRC_SOOMAC3.0/src/soomac_irc/soomac_irc/decision_model.py` | current IRC_SOOMAC3.0 operational source |
| `operational_code/decision_overrides.py` | `/home/roma/IRC_SOOMAC3.0/src/soomac_irc/soomac_irc/decision_overrides.py` | current IRC_SOOMAC3.0 operational source |
| `operational_code/domain.py` | `/home/roma/IRC_SOOMAC3.0/src/soomac_irc/soomac_irc/domain.py` | current IRC_SOOMAC3.0 operational source |
| `operational_code/llm_langgraph.py` | `/home/roma/IRC_SOOMAC3.0/src/soomac_irc/soomac_irc/llm_langgraph.py` | current IRC_SOOMAC3.0 operational source |
| `operational_code/llm_langgraph_node.py` | `/home/roma/IRC_SOOMAC3.0/src/soomac_irc/soomac_irc/llm_langgraph_node.py` | current IRC_SOOMAC3.0 operational source |
| `operational_code/llm_policy.py` | `/home/roma/IRC_SOOMAC3.0/src/soomac_irc/soomac_irc/llm_policy.py` | current IRC_SOOMAC3.0 operational source |
| `operational_code/model_runtime.py` | `/home/roma/IRC_SOOMAC3.0/src/soomac_irc/soomac_irc/model_runtime.py` | current IRC_SOOMAC3.0 operational source |
| `results_any_order_false/challenge/generation.log` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/evaluation/v225_challenge.log` | actual any_order=False evaluation log |
| `results_any_order_false/challenge/predictions.jsonl` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/evaluation/v225_challenge_predictions.jsonl` | actual any_order=False prediction output |
| `results_any_order_false/challenge/semantic_metrics.log` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/evaluation/v225_challenge_semantic.log` | actual semantic metric log |
| `results_any_order_false/challenge/summary.json` | `N/A (analysis-generated)` | analysis-generated summary of actual any_order=False predictions |
| `results_any_order_false/hard_v221/generation.log` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/evaluation/v225_hard_v221.log` | actual any_order=False evaluation log |
| `results_any_order_false/hard_v221/predictions.jsonl` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/evaluation/v225_hard_v221_predictions.jsonl` | actual any_order=False prediction output |
| `results_any_order_false/hard_v221/semantic_metrics.log` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/evaluation/v225_hard_v221_semantic.log` | actual semantic metric log |
| `results_any_order_false/hard_v221/summary.json` | `N/A (analysis-generated)` | analysis-generated summary of actual any_order=False predictions |
| `results_any_order_false/hard_v222/generation.log` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/evaluation/v225_hard_v222.log` | actual any_order=False evaluation log |
| `results_any_order_false/hard_v222/predictions.jsonl` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/evaluation/v225_hard_v222_predictions.jsonl` | actual any_order=False prediction output |
| `results_any_order_false/hard_v222/semantic_metrics.log` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/evaluation/v225_hard_v222_semantic.log` | actual semantic metric log |
| `results_any_order_false/hard_v222/summary.json` | `N/A (analysis-generated)` | analysis-generated summary of actual any_order=False predictions |
| `results_any_order_false/hard_v223/generation.log` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/evaluation/v225_hard_v223.log` | actual any_order=False evaluation log |
| `results_any_order_false/hard_v223/predictions.jsonl` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/evaluation/v225_hard_v223_predictions.jsonl` | actual any_order=False prediction output |
| `results_any_order_false/hard_v223/semantic_metrics.log` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/evaluation/v225_hard_v223_semantic.log` | actual semantic metric log |
| `results_any_order_false/hard_v223/summary.json` | `N/A (analysis-generated)` | analysis-generated summary of actual any_order=False predictions |
| `results_any_order_false/hard_v224/generation.log` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/evaluation/v225_hard_v224.log` | actual any_order=False evaluation log |
| `results_any_order_false/hard_v224/predictions.jsonl` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/evaluation/v225_hard_v224_predictions.jsonl` | actual any_order=False prediction output |
| `results_any_order_false/hard_v224/semantic_metrics.log` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/evaluation/v225_hard_v224_semantic.log` | actual semantic metric log |
| `results_any_order_false/hard_v224/summary.json` | `N/A (analysis-generated)` | analysis-generated summary of actual any_order=False predictions |
| `results_any_order_false/hard_v225/generation.log` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/evaluation/v225_hard_eval_v225.log` | actual any_order=False evaluation log |
| `results_any_order_false/hard_v225/predictions.jsonl` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/evaluation/v225_hard_eval_v225_predictions.jsonl` | actual any_order=False prediction output |
| `results_any_order_false/hard_v225/summary.json` | `N/A (analysis-generated)` | analysis-generated summary of actual any_order=False predictions |
| `results_any_order_false/latest_contract/generation.log` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/evaluation/v225_latest_contract.log` | actual any_order=False evaluation log |
| `results_any_order_false/latest_contract/predictions.jsonl` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/evaluation/v225_latest_contract_predictions.jsonl` | actual any_order=False prediction output |
| `results_any_order_false/latest_contract/semantic_metrics.log` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/evaluation/v225_latest_contract_semantic.log` | actual semantic metric log |
| `results_any_order_false/latest_contract/summary.json` | `N/A (analysis-generated)` | analysis-generated summary of actual any_order=False predictions |
| `results_any_order_false/runtime_holdout/generation.log` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/evaluation/v225_runtime_holdout.log` | actual any_order=False evaluation log |
| `results_any_order_false/runtime_holdout/predictions.jsonl` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/evaluation/v225_runtime_holdout_predictions.jsonl` | actual any_order=False prediction output |
| `results_any_order_false/runtime_holdout/semantic_metrics.log` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/evaluation/v225_runtime_holdout_semantic.log` | actual semantic metric log |
| `results_any_order_false/runtime_holdout/summary.json` | `N/A (analysis-generated)` | analysis-generated summary of actual any_order=False predictions |
| `temporary_any_order_eval/challenge/predictions.jsonl` | `/tmp/v225_anyorder_challenge.jsonl` | temporary any_order=True prediction output; not operational |
| `temporary_any_order_eval/challenge/summary.json` | `N/A (analysis-generated)` | analysis-generated summary of temporary any_order=True predictions |
| `temporary_any_order_eval/eval_anyorder_runner.py` | `/tmp/eval_anyorder_runner.py` | temporary in-memory evaluator wrapper; not operational |
| `temporary_any_order_eval/failed_9_cases.jsonl` | `N/A (analysis-generated)` | analysis-generated collection of the nine remaining any_order=True failures |
| `temporary_any_order_eval/hard_v221/predictions.jsonl` | `/tmp/v225_anyorder_hard_eval_v221_sft.jsonl` | temporary any_order=True prediction output; not operational |
| `temporary_any_order_eval/hard_v221/summary.json` | `N/A (analysis-generated)` | analysis-generated summary of temporary any_order=True predictions |
| `temporary_any_order_eval/hard_v222/predictions.jsonl` | `/tmp/v225_anyorder_hard_eval_v222_sft.jsonl` | temporary any_order=True prediction output; not operational |
| `temporary_any_order_eval/hard_v222/summary.json` | `N/A (analysis-generated)` | analysis-generated summary of temporary any_order=True predictions |
| `temporary_any_order_eval/hard_v223/predictions.jsonl` | `/tmp/v225_anyorder_hard_eval_v223_sft.jsonl` | temporary any_order=True prediction output; not operational |
| `temporary_any_order_eval/hard_v223/summary.json` | `N/A (analysis-generated)` | analysis-generated summary of temporary any_order=True predictions |
| `temporary_any_order_eval/hard_v224/predictions.jsonl` | `/tmp/v225_anyorder_hard_eval_v224_sft.jsonl` | temporary any_order=True prediction output; not operational |
| `temporary_any_order_eval/hard_v224/summary.json` | `N/A (analysis-generated)` | analysis-generated summary of temporary any_order=True predictions |
| `temporary_any_order_eval/hard_v225/predictions.jsonl` | `/tmp/v225_anyorder_hard225.jsonl` | temporary any_order=True prediction output; not operational |
| `temporary_any_order_eval/hard_v225/summary.json` | `N/A (analysis-generated)` | analysis-generated summary of temporary any_order=True predictions |
| `temporary_any_order_eval/latest_contract/predictions.jsonl` | `/tmp/v225_anyorder_latest_contract_eval_sft.jsonl` | temporary any_order=True prediction output; not operational |
| `temporary_any_order_eval/latest_contract/summary.json` | `N/A (analysis-generated)` | analysis-generated summary of temporary any_order=True predictions |
| `temporary_any_order_eval/runtime_holdout/predictions.jsonl` | `/tmp/v225_anyorder_real_surface_current_schema_sft.jsonl` | temporary any_order=True prediction output; not operational |
| `temporary_any_order_eval/runtime_holdout/summary.json` | `N/A (analysis-generated)` | analysis-generated summary of temporary any_order=True predictions |
| `training/results/adapter_config.json` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/checkpoints/gemma4_decision_nf4_qlora_v2_2_5_semantic_rebalance/adapter_config.json` | actual v2.2.5 training result/config |
| `training/results/all_results.json` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/checkpoints/gemma4_decision_nf4_qlora_v2_2_5_semantic_rebalance/all_results.json` | actual v2.2.5 training result/config |
| `training/results/checkpoint-396_trainer_state.json` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/checkpoints/gemma4_decision_nf4_qlora_v2_2_5_semantic_rebalance/checkpoint-396/trainer_state.json` | actual Trainer state; records best checkpoint and eval losses |
| `training/results/run_manifest.json` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/checkpoints/gemma4_decision_nf4_qlora_v2_2_5_semantic_rebalance/run_manifest.json` | actual v2.2.5 training result/config |
| `training/results/train_results.json` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/checkpoints/gemma4_decision_nf4_qlora_v2_2_5_semantic_rebalance/train_results.json` | actual v2.2.5 training result/config |
| `training/results/training_complete.json` | `/home/roma/Downloads/SOOMAC_DECISION_LOCAL_INT4/checkpoints/gemma4_decision_nf4_qlora_v2_2_5_semantic_rebalance/training_complete.json` | actual v2.2.5 training result/config |
| `training/train_in_container.py` | `/home/roma/Downloads/SOOMAC_DECISION_DOCKER_V2_7_REBUILT/docker_context/train_in_container.py` | actual Docker trainer source |
| `xgrammar_verification/compile_json_schema_callsites.txt` | `N/A (analysis-generated)` | analysis-generated callsite index |
| `xgrammar_verification/reproduce_any_order.py` | `N/A (analysis-generated)` | analysis-generated minimal XGrammar reproducer |
| `xgrammar_verification/xgrammar_version_and_signature.txt` | `N/A (analysis-generated)` | analysis-generated environment verification |
