# Dual Adapter Model Manifest

- Git HEAD at evaluation: `9332c069fa22bad8db49db27ba6ee13a12aac4ee`
- Base: `/home/roma/Desktop/sLLM/gemma-4-12B-it` (`nf4`)
- GPU: `NVIDIA RTX PRO 4500 Blackwell, 580.178.04, 32623 MiB`
- Torch / CUDA: `2.12.1+cu130` / `13.0`
- Transformers / PEFT: `5.13.1` / `0.19.1`

## Adapters

- Decision: `/home/roma/IRC_SOOMAC3.0/models/decision_adapter_v3_2_best_epoch4`
  - SHA256: `4ab6483b120a69b2aab2c9f9e361dbd3eb59880515be067be73ca74e4922d9ac`
  - Rollback: `/home/roma/IRC_SOOMAC3.0/models/decision_adapter_v3_1_best_epoch3`
- Response: `/home/roma/IRC_SOOMAC3.0/models/response_adapter_v1_best_epoch1`
  - SHA256: `be64d1512f231873468a80eb5cf7cad54b33c11e973879af340c2add5503ab96`
  - Rejected candidate: `/home/roma/IRC_SOOMAC3.0/models/response_adapter_v1_1_best_epoch4`
  - Candidate SHA256: `1bf60dd07c128223a773119b92a6a26bc02b8228ce9c8f65cc66f033e34a8188`

## Generation

- Decision: greedy, max_new_tokens=1024, XGrammar any_order=true
- Recommendation: base model, greedy, max_new_tokens=1024
- Response: greedy, max_new_tokens=512
