"""
2026.10.11
2026.10.11
5.13.1
1.8.0
__UNSLOTH_VERSIONING__
"""

# Unsloth auto generated code
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import os
import sys
import torch
import importlib.util
import math
if importlib.util.find_spec("unsloth_studio") is None:
    UNSLOTH_STUDIO_ENABLED = False
else:
    UNSLOTH_STUDIO_ENABLED = os.environ.get("UNSLOTH_STUDIO_DISABLED", "0") == "0"
pass
from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
import math

UNSLOTH_ENABLE_LOGGING = os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1"
UNSLOTH_ENABLE_CCE = os.environ.get("UNSLOTH_ENABLE_CCE", "1") == "1"
UNSLOTH_COMPILE_DISABLE = os.environ.get("UNSLOTH_COMPILE_DISABLE", "0") in ("1", "partial",)
UNSLOTH_COMPILE_LOCATION = os.environ.get("UNSLOTH_COMPILE_LOCATION", "unsloth_compiled_cache")
if UNSLOTH_COMPILE_LOCATION not in sys.path:
    sys.path.insert(0, UNSLOTH_COMPILE_LOCATION)

import logging
logger_compiler = logging.getLogger(__name__)
if UNSLOTH_ENABLE_LOGGING:
    logger_compiler.setLevel(logging.DEBUG)

global INFERENCE_RUNS
INFERENCE_RUNS = 0

try:
    import torch._dynamo.eval_frame as torch_dynamo_eval_frame
    torch_dynamo_eval_frame._stance.stance
    torch_compiler_set_stance = torch.compiler.set_stance
except:
    torch_dynamo_eval_frame = None
    torch_compiler_set_stance = None
pass

# The stance is process wide, so whether Unsloth owns it lives in unsloth_zoo, shared by
# every generated module: inference in one model's module, training in another's.
from unsloth_zoo.temporary_patches.utils import UNSLOTH_EAGER_STANCE_OWNED, unsloth_claim_eager_stance, unsloth_owns_stance, unsloth_release_stance

def unsloth_training_stance():
    # eager_on_recompile is global: left on by generate() or eval forwards, every recompile
    # in a later training step (new sequence length, model.train() without for_training)
    # would silently run the eager body. Training forwards restore the default stance,
    # unless the user chose eager_on_recompile themselves. Every set_stance call installs a
    # fresh stance object, so ours is still active only if the current one `is` it; a user
    # set_stance since then (even to eager_on_recompile again) is theirs and stays.
    global INFERENCE_RUNS
    INFERENCE_RUNS = 0
    if len(UNSLOTH_EAGER_STANCE_OWNED) != 0 and torch_dynamo_eval_frame is not None:
        current = torch_dynamo_eval_frame._stance
        # Ownership is only dropped once ours is reset: a temporary stance scope's exit restores
        # one of our exact objects, which a later training forward must still recognise.
        if current.stance == "eager_on_recompile" and unsloth_owns_stance(current):
            # Only the reset object: an outer scope may still restore another one of ours.
            # Put back what the caller had before Unsloth switched (force_eager, a backend, ...).
            previous = unsloth_release_stance(current)
            if previous is not None and hasattr(torch_dynamo_eval_frame, "_set_stance"):
                torch_dynamo_eval_frame._set_stance(previous)
            else:
                torch_compiler_set_stance(stance = "default", skip_guard_eval_unsafe = False)
pass

from unsloth_zoo import DEVICE_TYPE_TORCH, DEVICE_COUNT
from unsloth_zoo.temporary_patches.utils import UNSLOTH_DECODE_COMPILE


from unsloth_zoo.loss_utils import (
    HAS_CUT_CROSS_ENTROPY,
    fused_linear_cross_entropy,
    unsloth_fused_ce_loss,
)
from unsloth_zoo.fused_losses.forward_adapter import unsloth_fused_lm_head_loss
from unsloth_zoo.fused_losses.forward_install import _can_fuse_loss
from unsloth_zoo.compiler import lm_head_on_old_cuda_arch
from unsloth_zoo.fused_losses.cross_entropy_loss import unsloth_count_aware_cross_entropy, unsloth_loss_count_kwargs, unsloth_fsdp_unshard
from unsloth_zoo.fused_losses.aux_loss import unsloth_ga_scale_aux_loss

scaled_dot_product_attention = torch.nn.functional.scaled_dot_product_attention
@torch.compiler.disable(recursive = False)
def disable_compile_scaled_dot_product_attention(*args, **kwargs):
    return scaled_dot_product_attention(*args, **kwargs)
pass


from transformers.modeling_flash_attention_utils import is_flash_attn_available

if is_flash_attn_available():
    try:
        from transformers.modeling_flash_attention_utils import flash_attn_supports_top_left_mask
    except:
        flash_attn_supports_top_left_mask = None
    try:
        from transformers.modeling_flash_attention_utils import _flash_attention_forward
    except:
        _flash_attention_forward = None
    try:
        from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
    except:
        FlashAttentionKwargs = None
    try:
        from transformers.modeling_flash_attention_utils import flash_attn_varlen_func
    except:
        flash_attn_varlen_func = None
else:
    flash_attn_supports_top_left_mask = None
    _flash_attention_forward = None
    FlashAttentionKwargs = None
    flash_attn_varlen_func = None
pass


torch_compile_options = {'epilogue_fusion': True, 'max_autotune': False, 'shape_padding': True, 'trace.enabled': False, 'triton.cudagraphs': False, 'debug': False, 'dce': True, 'memory_planning': True, 'coordinate_descent_tuning': False, 'trace.graph_diagram': False, 'compile_threads': 32, 'group_fusion': True, 'disable_progress': True, 'verbose_progress': False, 'triton.multi_kernel': 0, 'triton.use_block_ptr': False, 'triton.enable_persistent_tma_matmul': False, 'triton.autotune_at_compile_time': False, 'triton.cooperative_reductions': False, 'cuda.compile_opt_level': '-O2', 'cuda.enable_cuda_lto': True, 'combo_kernels': True, 'benchmark_combo_kernel': True, 'combo_kernel_foreach_dynamic_shapes': True, 'triton.mix_order_reduction': False}

from torch.nn import CrossEntropyLoss
from unsloth_zoo.temporary_patches.utils import torch_compile_with_fallback

@torch_compile_with_fallback(fullgraph = True, dynamic = True, options = torch_compile_options)
def normal_cross_entropy_loss(self, hidden_states, labels):
    logits = self.lm_head(hidden_states)
    logits = logits.float()
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, self.config.vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss, logits
pass

# We need an empty logits flag to warn people logits will not be returned anymore unless asked ie
# os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
LOGITS_ERROR_STRING = \
    "Unsloth: Logits are empty from 2024.11 onwards. To get raw logits again, please "\
    'set the environment variable `UNSLOTH_RETURN_LOGITS` to `"1" BEFORE starting to train ie before `trainer.train()`. For example:\n'\
    "```\nimport os\n"\
    "os.environ['UNSLOTH_RETURN_LOGITS'] = '1'\n"\
    "trainer.train()\n```\n"\
    "No need to restart your console - just add `os.environ['UNSLOTH_RETURN_LOGITS'] = '1'` before trainer.train() and re-run the cell!"

def raise_logits_error(*args, **kwargs): raise NotImplementedError(LOGITS_ERROR_STRING)
def return_none(*args, **kwargs): return None
class EmptyLogits:
    def __init__(self): return
    def raise_getattr_error(self, attr):
        if attr == "to": return return_none
        # A catch-all __getattr__ makes hasattr() true for every name, dunders included, so the
        # sentinel answers yes to protocol probes it cannot honour. torch.distributed's output cast
        # tests `hasattr(x, "__dataclass_fields__")` and then calls `dataclasses.replace(x)` on
        # whatever said yes, so with FSDP2 mixed precision every step died in
        # `TypeError: replace() should be called on dataclass instances` (unsloth#409, reached
        # through `_fsdp_state._cast_output_dtype`). Protocol probes get an honest AttributeError;
        # ordinary attribute access still gets the callable that explains UNSLOTH_RETURN_LOGITS.
        if len(attr) > 4 and attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {attr!r}")
        return raise_logits_error
    __getitem__ = raise_logits_error
    __getattr__ = raise_getattr_error
    def __repr__(self): return LOGITS_ERROR_STRING
    def __str__ (self): return LOGITS_ERROR_STRING
    # Stateless pickling so accelerate gather_object works on the sentinel
    def __reduce__(self): return (type(self), ())
    # Gathered copies must compare equal in accelerate debug mode
    def __eq__(self, other): return type(other).__name__ == "EmptyLogits"
    __hash__ = object.__hash__
pass
EMPTY_LOGITS = EmptyLogits()
functions = dir(torch.Tensor)
for j, function in enumerate(functions):
    if function.startswith("__") and function.endswith("__"):
        exec(f"def raise_{j}(*args, **kwargs): print('{function}')", globals(), locals())
        try: exec(f"EMPTY_LOGITS.{function} = raise_{j}", globals(), locals())
        except: continue
pass
# The loop above stomps pickle hooks with stubs returning None; restore them.
for function in ("__reduce__", "__reduce_ex__", "__getstate__", "__setstate__"):
    try: delattr(EMPTY_LOGITS, function)
    except Exception: pass
pass


def mask_attention_mask_out(labels = None, attention_mask = None):
    # Only a mask over the labels' own positions: MusicGen's attention_mask is the text encoder's,
    # while its labels are decoder codebook frames.
    if (
        labels is not None and attention_mask is not None
        and tuple(attention_mask.shape) == tuple(labels.shape)
    ):
        attention_mask = attention_mask.to(device = labels.device)
        labels[attention_mask == 0] = -100
    return labels
pass


from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import functional as F
from unsloth_zoo.temporary_patches.utils import torch_compile_with_fallback
from unsloth_zoo.temporary_patches.utils import torch_compiler_disable_unless_decode
from unsloth_zoo.temporary_patches.common import torch_compile
import transformers
from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
from transformers.models.gemma4_unified.modeling_gemma4_unified import (Callable, Optional, torch, nn, init, ACT2FN, Cache, PreTrainedConfig, GenerationMixin, _preprocess_mask_arguments, blockwise_overlay, create_causal_mask, maybe_pad_block_sequence_ids, sliding_window_overlay, FlashAttentionKwargs, ModelOutput, ROPE_INIT_FUNCTIONS, dynamic_rope_update, ALL_ATTENTION_FUNCTIONS, PreTrainedModel, Unpack, TransformersKwargs, can_return_tuple, maybe_autocast, Gemma4UnifiedAudioConfig, Gemma4UnifiedConfig, Gemma4UnifiedTextConfig, Gemma4UnifiedVisionConfig, Gemma4UnifiedTextModelOutputWithPast, Gemma4UnifiedCausalLMOutputWithPast, Gemma4UnifiedPreTrainedModel, Gemma4UnifiedTextModel, Gemma4UnifiedForCausalLM, Gemma4UnifiedModel, Gemma4UnifiedForConditionalGeneration, create_masks_for_generate, create_causal_mask, create_masks_for_generate)

@torch_compile_with_fallback(fullgraph = True, dynamic = True, options = torch_compile_options)
def Gemma4UnifiedRMSNorm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    normed_output = self._norm(hidden_states.float())
    if self.with_scale:
        normed_output = normed_output * self.weight.float()
    return normed_output.type_as(hidden_states)

class Gemma4UnifiedRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale

        if self.with_scale:
            self.weight = nn.Parameter(torch.ones(dim), requires_grad=True)

    def _norm(self, hidden_states: torch.Tensor):
        mean_squared = hidden_states.pow(2).mean(-1, keepdim=True) + self.eps
        # Use torch.pow() (over torch.sqrt() or torch.rsqrt()) to address compiler differences between Torch and JAX
        return hidden_states * torch.pow(mean_squared, -0.5)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return Gemma4UnifiedRMSNorm_forward(self, hidden_states=hidden_states)


@torch_compile_with_fallback(fullgraph = False, dynamic = True, options = torch_compile_options)
@torch.no_grad()
@dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
def Gemma4UnifiedTextRotaryEmbedding_forward(self, x, position_ids, layer_type=None):
    inv_freq = getattr(self, f"{layer_type}_inv_freq")
    attention_scaling = getattr(self, f"{layer_type}_attention_scaling")

    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
    position_ids_expanded = position_ids[:, None, :].float()

    device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
    with maybe_autocast(device_type=device_type, enabled=False):  # Force float32
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * attention_scaling
        sin = emb.sin() * attention_scaling

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class Gemma4UnifiedTextRotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: Gemma4UnifiedTextConfig, device=None, layer_type=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.layer_types = set(config.layer_types)
        self.rope_init_fns: dict[str, Callable[..., tuple[torch.Tensor, float]]] = {}
        self.rope_type: dict[str, str] = {}

        for layer_type in self.layer_types:
            rope_params = self.config.rope_parameters[layer_type]
            if rope_params is None:
                continue

            if (rope_type := rope_params["rope_type"]) != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
            else:
                rope_init_fn = self.compute_default_rope_parameters

            self.rope_init_fns[layer_type] = rope_init_fn
            self.rope_type[layer_type] = rope_type

            rope_init_fn_kwargs = {"device": device, "layer_type": layer_type}
            if layer_type == "full_attention" and rope_type == "proportional":
                rope_init_fn_kwargs["head_dim_key"] = "global_head_dim"

            curr_inv_freq, curr_attention_scaling = rope_init_fn(self.config, **rope_init_fn_kwargs)
            self.register_buffer(f"{layer_type}_inv_freq", curr_inv_freq, persistent=False)
            self.register_buffer(f"{layer_type}_original_inv_freq", curr_inv_freq.clone(), persistent=False)
            setattr(self, f"{layer_type}_attention_scaling", curr_attention_scaling)

    @staticmethod
    def compute_default_rope_parameters(
        config: Gemma4UnifiedTextConfig | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
        layer_type: str | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
            layer_type (`str`, *optional*):
                The current layer type if the model has different RoPE parameters per type.
                Should not be used unless `config.layer_types is not None`

        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        # For backward compatibility standardize the `rope_parameters_dict` if it uses old format
        base = config.rope_parameters[layer_type]["rope_theta"]
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor


    def forward(self, x, position_ids, layer_type=None):
        return Gemma4UnifiedTextRotaryEmbedding_forward(self, x=x, position_ids=position_ids, layer_type=layer_type)


@torch_compile_with_fallback(fullgraph = True, dynamic = True, options = torch_compile_options)
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@torch_compile_with_fallback(fullgraph = True, dynamic = True, options = torch_compile_options)
def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        x (`torch.Tensor`): The tensor to embed.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


@torch_compile_with_fallback(fullgraph = True, dynamic = True, options = torch_compile_options)
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@torch_compile_with_fallback(fullgraph = True, dynamic = True, options = torch_compile_options)
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    dropout: float | int = 0.0,
    scaling: float | None = None,
    softcap: float | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = module.head_dim**-0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap
    if attention_mask is not None:

        if isinstance(attention_mask, dict):

            attention_mask = attention_mask.get(getattr(module, 'layer_type', None), None)

        if attention_mask is not None:

            attn_weights = attn_weights + attention_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype = torch.float32).to(attn_weights.dtype).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


@torch_compiler_disable_unless_decode
def Gemma4UnifiedTextAttention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: torch.Tensor,
    attention_mask: torch.Tensor | None,
    shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]],
    past_key_values: Cache | None = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    cos, sin = position_embeddings

    query_states = self.q_proj(hidden_states).view(hidden_shape)
    query_states = self.q_norm(query_states)
    query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
    query_states = query_states.transpose(1, 2)

    # For layers with shared KV (from kv sharing point onwards), we reuse the same keys/values states as the last non-sharing layer.
    # We cannot simply reuse the cached state if we have a Cache, as sliding layers will not remember the full states in their Cache
    # once we are past the sliding window - so we always use `shared_kv_states` instead, even when past_key_values is not None
    if self.is_kv_shared_layer:
        key_states, value_states = shared_kv_states[self.layer_type]
        # Device of past layer may be different from current one
        key_states = key_states.to(query_states.device)
        value_states = value_states.to(query_states.device)
    else:
        key_states = self.k_proj(hidden_states).view(hidden_shape)
        value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states

        key_states = self.k_norm(key_states)
        key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
        key_states = key_states.transpose(1, 2)

        value_states = self.v_norm(value_states)
        value_states = value_states.transpose(1, 2)

    if past_key_values is not None and not self.is_kv_shared_layer:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
    if self.store_full_length_kv:
        shared_kv_states[self.layer_type] = key_states, value_states

    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=self.attention_dropout if self.training else 0.0,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

class Gemma4UnifiedTextAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Gemma4UnifiedTextConfig, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
        self.config = config
        self.layer_idx = layer_idx
        self.is_sliding = self.layer_type == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None

        self.head_dim = config.global_head_dim if not self.is_sliding and config.global_head_dim else config.head_dim
        self.use_alternative_attention = config.attention_k_eq_v and not self.is_sliding
        num_key_value_heads = (
            config.num_global_key_value_heads if self.use_alternative_attention else config.num_key_value_heads
        )
        self.num_key_value_groups = config.num_attention_heads // num_key_value_heads
        self.scaling = 1.0
        self.attention_dropout = self.config.attention_dropout
        self.is_causal = config.use_bidirectional_attention != "all"

        # Shared kv cache
        first_kv_shared_layer_idx = self.config.num_hidden_layers - getattr(self.config, "num_kv_shared_layers", 0)
        self.is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx >= 0
        prev_layers = config.layer_types[:first_kv_shared_layer_idx]
        self.store_full_length_kv = not self.is_kv_shared_layer and layer_idx == len(prev_layers) - 1 - prev_layers[
            ::-1
        ].index(config.layer_types[layer_idx])

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.q_norm = Gemma4UnifiedRMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)

        # Layers sharing kv states don't need any weight matrices
        if not self.is_kv_shared_layer:
            self.k_norm = Gemma4UnifiedRMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
            self.v_norm = Gemma4UnifiedRMSNorm(self.head_dim, eps=config.rms_norm_eps, with_scale=False)

            self.k_proj = nn.Linear(
                config.hidden_size, num_key_value_heads * self.head_dim, bias=config.attention_bias
            )
            self.v_proj = (
                nn.Linear(config.hidden_size, num_key_value_heads * self.head_dim, bias=config.attention_bias)
                if not self.use_alternative_attention
                else None
            )

        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None,
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]],
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return Gemma4UnifiedTextAttention_forward(self, hidden_states=hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask, shared_kv_states=shared_kv_states, past_key_values=past_key_values, **kwargs)


@torch_compile_with_fallback(fullgraph = False, dynamic = True, options = torch_compile_options)
def Gemma4UnifiedTextMLP_forward(self, x):
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj

class Gemma4UnifiedTextMLP(nn.Module):
    def __init__(self, config: Gemma4UnifiedTextConfig, layer_idx: int):
        super().__init__()
        first_kv_shared_layer_idx = config.num_hidden_layers - config.num_kv_shared_layers
        is_kv_shared_layer = layer_idx >= first_kv_shared_layer_idx > 0
        use_double_wide_mlp = config.use_double_wide_mlp and is_kv_shared_layer
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size * (2 if use_double_wide_mlp else 1)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x):
        return Gemma4UnifiedTextMLP_forward(self, x=x)


@torch_compiler_disable_unless_decode
@can_return_tuple
def Gemma4UnifiedForCausalLM_forward(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    **kwargs: Unpack[TransformersKwargs],
) -> Gemma4UnifiedCausalLMOutputWithPast:
    r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, Gemma4UnifiedForCausalLM

    >>> model = Gemma4UnifiedForCausalLM.from_pretrained("google/gemma-4-12B-it")
    >>> tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-12B-it")

    >>> prompt = "What is your favorite condiment?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "What is your favorite condiment?"
    ```"""
    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    if not torch.compiler.is_compiling() and torch.is_grad_enabled() and getattr(self, "training", False):
        unsloth_training_stance()
    outputs: Gemma4UnifiedTextModelOutputWithPast = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :]) if os.environ.get('UNSLOTH_RETURN_LOGITS', '0') == '1' else EMPTY_LOGITS
    loss = None
    NOT_RETURN_LOGITS = os.environ.get('UNSLOTH_RETURN_LOGITS', '0') == '0'
    RETURN_HIDDEN_STATES = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1"
    
    n_items = None
    if (kwargs) != () and type(kwargs) is dict:
        n_items = (kwargs).get("num_items_in_batch", None)
        if n_items is None: n_items = (kwargs).get("n_items", None)
    if n_items is None:
        all_locals = locals()
        if 'loss_kwargs' in all_locals:
            __kwargs = all_locals['loss_kwargs']
            if type(__kwargs) is dict:
                n_items = __kwargs.get("num_items_in_batch", None)
                if n_items is None: n_items = __kwargs.get("n_items", None)
        if n_items is None and 'kwargs' in all_locals:
            __kwargs = all_locals['kwargs']
            if type(__kwargs) is dict:
                n_items = __kwargs.get("num_items_in_batch", None)
                if n_items is None: n_items = __kwargs.get("n_items", None)
        if n_items is None:
            all_locals = all_locals.values()
            for __kwargs in all_locals:
                if type(__kwargs) is dict:
                    n_items = __kwargs.get("num_items_in_batch", None)
                    if n_items is None: n_items = __kwargs.get("n_items", None)
                    break
    pass
    # Stock ForCausalLMLoss uses caller pre-shifted targets (context parallel, padding-free) as is.
    explicit_shift_labels = None
    if (kwargs) != () and type(kwargs) is dict:
        explicit_shift_labels = (kwargs).get("shift_labels", None)
        if not torch.is_tensor(explicit_shift_labels): explicit_shift_labels = None
    
    if labels is not None: unsloth_fsdp_unshard(self.lm_head)
    requires_grad_ = self.lm_head.weight.requires_grad
    requires_grad_ = requires_grad_ or self.lm_head.weight.dtype == torch.float32
    
    if RETURN_HIDDEN_STATES:
        logits = hidden_states[:, slice_indices, :]
    elif labels is None:
        
    
        # Set compiler stance to fail on recompiles for inference
        global INFERENCE_RUNS
        # Skipped while tracing (set_stance raises there, and the counter would guard every step)
        # and around a compiled decode step, which eager_on_recompile would otherwise freeze.
        if not torch.compiler.is_compiling() and not UNSLOTH_DECODE_COMPILE[0]:
            if torch_dynamo_eval_frame is not None:
                old_stance = torch_dynamo_eval_frame._stance.stance
            else:
                old_stance = None
            if old_stance is not None and INFERENCE_RUNS == 1:
                # Skip guards and return to eager -> we still need guards!
                # Already eager (another model's inference, or the user's choice): leave that stance
                # object, and so its ownership, as it is. Otherwise switch and claim the new object.
                if old_stance != "eager_on_recompile":
                    previous = getattr(torch_dynamo_eval_frame, "_stance", None)
                    torch_compiler_set_stance(stance = "eager_on_recompile", skip_guard_eval_unsafe = False)
                    # The stance object itself, or True where torch keeps no such object.
                    unsloth_claim_eager_stance(getattr(torch_dynamo_eval_frame, "_stance", True), previous)
                if UNSLOTH_ENABLE_LOGGING:
                    logger_compiler.info(
                        f"Unsloth: Removing compiler guards after 1 inference run. "\
                        f"DYNAMO_STANCE.stance = {torch_dynamo_eval_frame._stance.stance} "\
                        f"DYNAMO_STANCE.skip_guard_eval_unsafe = {torch_dynamo_eval_frame._stance.skip_guard_eval_unsafe}"
                    )
            elif old_stance == "eager_on_recompile":
                pass
            elif old_stance == "default" and INFERENCE_RUNS > 1:
                # Reset compiler stance
                torch_compiler_set_stance(stance = "default", skip_guard_eval_unsafe = False)
                if UNSLOTH_ENABLE_LOGGING:
                    logger_compiler.info(
                        f"Unsloth: Reseting guards. "\
                        f"DYNAMO_STANCE.stance = {torch_dynamo_eval_frame._stance.stance} "\
                        f"DYNAMO_STANCE.skip_guard_eval_unsafe = {torch_dynamo_eval_frame._stance.skip_guard_eval_unsafe}"
                    )
                INFERENCE_RUNS = 0
            INFERENCE_RUNS += 1
    
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        # Inference returns these logits to the caller, so they must carry the same
        # scale and softcap transforms the training branches below apply. Leaving
        # them off does not change greedy decoding, since tanh is monotonic, but it
        # does change the distribution: sampling, returned logprobs and any scoring
        # that reads the logits all see uncapped values.
        if () != ():
            logits = logits * ()
        if () != ():
            logits = logits / ()
        if (self.config.final_logit_softcapping) not in (None, (),):
            logits = logits / (self.config.final_logit_softcapping)
            logits = torch.tanh(logits)
            logits = logits * (self.config.final_logit_softcapping)
    elif (() == () and () == ()) and (UNSLOTH_ENABLE_CCE and HAS_CUT_CROSS_ENTROPY) and not lm_head_on_old_cuda_arch(self.lm_head.weight) and NOT_RETURN_LOGITS and _can_fuse_loss(self.loss_function) and labels is not None and not requires_grad_ and explicit_shift_labels is None and getattr(self.lm_head, "bias", None) is None:
        loss = fused_linear_cross_entropy(
            hidden_states      = hidden_states[:, slice_indices, :],
            lm_weight          = self.lm_head.weight,
            labels             = labels.to(self.lm_head.weight.device),
            num_items_in_batch = n_items,
            logit_softcapping  = None if (self.config.final_logit_softcapping) == () else (self.config.final_logit_softcapping),
        )
    elif _can_fuse_loss(self.loss_function) and labels is not None and NOT_RETURN_LOGITS:
        lm_head_weight = self.lm_head.weight
        lm_head_bias   = getattr(self.lm_head, "bias", None)
    
        # ========= NEW fused =========
        _hidden_states = hidden_states[:, slice_indices, :]
        _labels = labels if explicit_shift_labels is None else explicit_shift_labels
        torch._dynamo.mark_dynamic(_hidden_states, 1)
        torch._dynamo.mark_dynamic(_labels, _labels.dim() - 1)
        loss = unsloth_fused_ce_loss(
            trainer              = None,
            hidden_states        = _hidden_states,
            lm_head_weight       = lm_head_weight,
            lm_head_bias         = lm_head_bias,
            labels               = _labels,
            mask                 = None,
            n_items              = n_items,
            scaling              = getattr(self, "accelerator_scaler", None),
            target_gb            = None,
            torch_compile        = not UNSLOTH_COMPILE_DISABLE and not lm_head_on_old_cuda_arch(lm_head_weight),
            logit_scale_multiply = () if () != () else 0,
            logit_scale_divide   = () if () != () else 0,
            logit_softcapping    = (self.config.final_logit_softcapping) if (self.config.final_logit_softcapping) != () else 0,
            shift_labels         = explicit_shift_labels is None,
        )
    elif _can_fuse_loss(self.loss_function) and labels is not None and not NOT_RETURN_LOGITS:
        # UNSLOTH_RETURN_LOGITS=1 path. Prepended `logits = self.lm_head(...)`
        # already materialised the full lm_head matmul; apply the captured logit
        # scale/softcap transforms and route loss through self.loss_function on
        # those logits instead of letting unsloth_fused_ce_loss redo the matmul.
        if () != ():
            logits = logits * ()
        if () != ():
            logits = logits / ()
        if (self.config.final_logit_softcapping) not in (None, (),):
            logits = logits / (self.config.final_logit_softcapping)
            logits = torch.tanh(logits)
            logits = logits * (self.config.final_logit_softcapping)
        loss = self.loss_function(logits, labels.to(self.lm_head.weight.device), vocab_size=self.vocab_size, **kwargs)
    else:
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if () != ():
            logits = logits * ()
        if () != ():
            logits = logits / ()
        if (self.config.final_logit_softcapping) not in (None, (),):
            logits = logits / (self.config.final_logit_softcapping)
            logits = torch.tanh(logits)
            logits = logits * (self.config.final_logit_softcapping)
        loss = self.loss_function(logits, labels.to(self.lm_head.weight.device), vocab_size=self.vocab_size, **kwargs)


    return Gemma4UnifiedCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        shared_kv_states=outputs.shared_kv_states,
    )

class Gemma4UnifiedForCausalLM(Gemma4UnifiedPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    config: Gemma4UnifiedTextConfig
    base_model_prefix = "model"

    def __init__(self, config: Gemma4UnifiedTextConfig):
        super().__init__(config)
        self.model = Gemma4UnifiedTextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Gemma4UnifiedCausalLMOutputWithPast:
        return Gemma4UnifiedForCausalLM_forward(self, input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache, logits_to_keep=logits_to_keep, **kwargs)

Gemma4UnifiedForCausalLM.__UNSLOTH_SUPPORTS_RETURN_HIDDEN_STATES__ = True


@torch_compile_with_fallback(fullgraph = False, dynamic = True, options = torch_compile_options)
def Gemma4UnifiedMultimodalEmbedder_forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
    """Embeds token ids or soft tokens for multimodal content into language model space.
    Args:
        inputs_embeds: A torch.Tensor containing the soft tokens to embed.
    Returns:
        A torch.Tensor of embeddings with shape `[batch_size, seq_len, self.config.text_config.hidden_size]`.
    """
    # Additional dtype casting
    if (target_dtype := self.embedding_projection.weight.dtype).is_floating_point:
        inputs_embeds = inputs_embeds.to(target_dtype)
    embs_normed = self.embedding_pre_projection_norm(inputs_embeds)
    return self.embedding_projection(embs_normed)

class Gemma4UnifiedMultimodalEmbedder(nn.Module):
    """Embeds token ids or soft tokens for multimodal content into language model space."""

    def __init__(
        self,
        multimodal_config: Gemma4UnifiedAudioConfig | Gemma4UnifiedVisionConfig,
        text_config: Gemma4UnifiedTextConfig,
    ):
        super().__init__()

        # Both AudioConfig and VisionConfig define output_proj_dims directly
        self.multimodal_hidden_size = multimodal_config.output_proj_dims
        self.eps = multimodal_config.rms_norm_eps
        self.text_hidden_size = text_config.hidden_size
        self.embedding_projection = nn.Linear(self.multimodal_hidden_size, self.text_hidden_size, bias=False)
        self.embedding_pre_projection_norm = Gemma4UnifiedRMSNorm(
            self.multimodal_hidden_size, eps=self.eps, with_scale=False
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        return Gemma4UnifiedMultimodalEmbedder_forward(self, inputs_embeds=inputs_embeds)


@torch_compile_with_fallback(fullgraph = True, dynamic = True, options = torch_compile_options)
def get_block_sequence_ids_for_mask(mm_token_type_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
    mm_token_type_ids = mm_token_type_ids.to(device)

    is_vision = (mm_token_type_ids == 1) | (mm_token_type_ids == 2)
    is_prev_vision = torch.roll(is_vision, shifts=1, dims=-1)
    is_prev_vision[..., 0] = False
    new_vision_starts = is_vision & ~is_prev_vision
    vision_group_ids = torch.cumsum(new_vision_starts.int(), dim=1) - 1
    block_sequence_ids = torch.where(is_vision, vision_group_ids, -1)
    return block_sequence_ids


def create_masks_for_vision_model(
    config: PreTrainedConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None,
    position_ids: torch.Tensor | None,
    block_sequence_ids: torch.Tensor,
) -> dict:
    """Create full_attention and sliding_attention masks with correct composition.

    For global (full attention) layers:  causal only (no bidirectional)
    For local (sliding window) layers:  AND(sliding_window, OR(causal, blockwise))

    Unlike Gemma 3 (which applies bidirectional attention on all layers), Gemma 4
    explicitly disables bidirectional attention on global attention layers.
    """
    mask_kwargs = {
        "config": config,
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
    }

    # Full attention: causal only — no bidirectional blockwise overlay.
    full_mask = create_causal_mask(**mask_kwargs)

    # We need to manually pad the sequence IDs for the sliding mask
    # as it's passed as an `or_mask_function` which bypasses internal padding.
    early_exit, _, _, _, kv_length, _, kv_offset = _preprocess_mask_arguments(
        **mask_kwargs,
        layer_idx=0,
    )
    if early_exit:
        padded_block_sequence_ids = block_sequence_ids
    else:
        padded_block_sequence_ids = maybe_pad_block_sequence_ids(
            block_sequence_ids, attention_mask, kv_length, kv_offset
        )

    # Sliding attention: AND(sliding_window, OR(causal, blockwise))
    # Pass blockwise as or_mask_function (applied as step 2 in create_causal_mask)
    # Pass sliding_window as and_mask_function (applied as step 3, after OR)
    sliding_mask = create_causal_mask(
        **mask_kwargs,
        or_mask_function=blockwise_overlay(padded_block_sequence_ids),
        and_mask_function=sliding_window_overlay(config.sliding_window),
    )

    return {
        "full_attention": full_mask,
        "sliding_attention": sliding_mask,
    }


@torch_compiler_disable_unless_decode
@can_return_tuple
def Gemma4UnifiedForConditionalGeneration_forward(
    self,
    input_ids: torch.LongTensor | None = None,
    pixel_values: torch.FloatTensor | None = None,
    pixel_values_videos: torch.FloatTensor | None = None,
    input_features: torch.FloatTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    input_features_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    image_position_ids: torch.LongTensor | None = None,
    video_position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    mm_token_type_ids: torch.LongTensor | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    **kwargs: Unpack[TransformersKwargs],
) -> Gemma4UnifiedCausalLMOutputWithPast:
    r"""
    input_features_mask (`torch.FloatTensor]` of shape `(num_images, seq_length)`):
        The attention mask for the input audio.
    image_position_ids (`torch.LongTensor` of shape `(batch_size, max_patches, 2)`, *optional*):
        2D patch position coordinates from the image processor, with `(-1, -1)` indicating padding.
        Passed through to the vision encoder for positional embedding computation.
    video_position_ids (`torch.LongTensor` of shape `(num_videos, num_frames, max_patches, 2)`, *optional*):
        2D patch position coordinates from the video processor, with `(-1, -1)` indicating padding.
        Passed through to the vision encoder for positional embedding computation.
    """
    if not torch.compiler.is_compiling() and torch.is_grad_enabled() and getattr(self, "training", False):
        unsloth_training_stance()
    outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        input_features=input_features,
        attention_mask=attention_mask,
        input_features_mask=input_features_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        mm_token_type_ids=mm_token_type_ids,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        image_position_ids=image_position_ids,
        video_position_ids=video_position_ids,
        return_dict=True,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :]) if os.environ.get('UNSLOTH_RETURN_LOGITS', '0') == '1' else EMPTY_LOGITS
    loss = None
    NOT_RETURN_LOGITS = os.environ.get('UNSLOTH_RETURN_LOGITS', '0') == '0'
    RETURN_HIDDEN_STATES = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1"
    
    n_items = None
    if (kwargs) != () and type(kwargs) is dict:
        n_items = (kwargs).get("num_items_in_batch", None)
        if n_items is None: n_items = (kwargs).get("n_items", None)
    if n_items is None:
        all_locals = locals()
        if 'loss_kwargs' in all_locals:
            __kwargs = all_locals['loss_kwargs']
            if type(__kwargs) is dict:
                n_items = __kwargs.get("num_items_in_batch", None)
                if n_items is None: n_items = __kwargs.get("n_items", None)
        if n_items is None and 'kwargs' in all_locals:
            __kwargs = all_locals['kwargs']
            if type(__kwargs) is dict:
                n_items = __kwargs.get("num_items_in_batch", None)
                if n_items is None: n_items = __kwargs.get("n_items", None)
        if n_items is None:
            all_locals = all_locals.values()
            for __kwargs in all_locals:
                if type(__kwargs) is dict:
                    n_items = __kwargs.get("num_items_in_batch", None)
                    if n_items is None: n_items = __kwargs.get("n_items", None)
                    break
    pass
    # Stock ForCausalLMLoss uses caller pre-shifted targets (context parallel, padding-free) as is.
    explicit_shift_labels = None
    if (kwargs) != () and type(kwargs) is dict:
        explicit_shift_labels = (kwargs).get("shift_labels", None)
        if not torch.is_tensor(explicit_shift_labels): explicit_shift_labels = None
    
    if labels is not None: unsloth_fsdp_unshard(self.lm_head)
    requires_grad_ = self.lm_head.weight.requires_grad
    requires_grad_ = requires_grad_ or self.lm_head.weight.dtype == torch.float32
    
    if RETURN_HIDDEN_STATES:
        logits = hidden_states[:, slice_indices, :]
    elif labels is None:
        
    
        # Set compiler stance to fail on recompiles for inference
        global INFERENCE_RUNS
        # Skipped while tracing (set_stance raises there, and the counter would guard every step)
        # and around a compiled decode step, which eager_on_recompile would otherwise freeze.
        if not torch.compiler.is_compiling() and not UNSLOTH_DECODE_COMPILE[0]:
            if torch_dynamo_eval_frame is not None:
                old_stance = torch_dynamo_eval_frame._stance.stance
            else:
                old_stance = None
            if old_stance is not None and INFERENCE_RUNS == 1:
                # Skip guards and return to eager -> we still need guards!
                # Already eager (another model's inference, or the user's choice): leave that stance
                # object, and so its ownership, as it is. Otherwise switch and claim the new object.
                if old_stance != "eager_on_recompile":
                    previous = getattr(torch_dynamo_eval_frame, "_stance", None)
                    torch_compiler_set_stance(stance = "eager_on_recompile", skip_guard_eval_unsafe = False)
                    # The stance object itself, or True where torch keeps no such object.
                    unsloth_claim_eager_stance(getattr(torch_dynamo_eval_frame, "_stance", True), previous)
                if UNSLOTH_ENABLE_LOGGING:
                    logger_compiler.info(
                        f"Unsloth: Removing compiler guards after 1 inference run. "\
                        f"DYNAMO_STANCE.stance = {torch_dynamo_eval_frame._stance.stance} "\
                        f"DYNAMO_STANCE.skip_guard_eval_unsafe = {torch_dynamo_eval_frame._stance.skip_guard_eval_unsafe}"
                    )
            elif old_stance == "eager_on_recompile":
                pass
            elif old_stance == "default" and INFERENCE_RUNS > 1:
                # Reset compiler stance
                torch_compiler_set_stance(stance = "default", skip_guard_eval_unsafe = False)
                if UNSLOTH_ENABLE_LOGGING:
                    logger_compiler.info(
                        f"Unsloth: Reseting guards. "\
                        f"DYNAMO_STANCE.stance = {torch_dynamo_eval_frame._stance.stance} "\
                        f"DYNAMO_STANCE.skip_guard_eval_unsafe = {torch_dynamo_eval_frame._stance.skip_guard_eval_unsafe}"
                    )
                INFERENCE_RUNS = 0
            INFERENCE_RUNS += 1
    
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        # Inference returns these logits to the caller, so they must carry the same
        # scale and softcap transforms the training branches below apply. Leaving
        # them off does not change greedy decoding, since tanh is monotonic, but it
        # does change the distribution: sampling, returned logprobs and any scoring
        # that reads the logits all see uncapped values.
        if () != ():
            logits = logits * ()
        if () != ():
            logits = logits / ()
        if (self.config.get_text_config().final_logit_softcapping) not in (None, (),):
            logits = logits / (self.config.get_text_config().final_logit_softcapping)
            logits = torch.tanh(logits)
            logits = logits * (self.config.get_text_config().final_logit_softcapping)
    elif (() == () and () == ()) and (UNSLOTH_ENABLE_CCE and HAS_CUT_CROSS_ENTROPY) and not lm_head_on_old_cuda_arch(self.lm_head.weight) and NOT_RETURN_LOGITS and _can_fuse_loss(self.loss_function) and labels is not None and not requires_grad_ and explicit_shift_labels is None and getattr(self.lm_head, "bias", None) is None:
        loss = fused_linear_cross_entropy(
            hidden_states      = hidden_states[:, slice_indices, :],
            lm_weight          = self.lm_head.weight,
            labels             = labels.to(self.lm_head.weight.device),
            num_items_in_batch = n_items,
            logit_softcapping  = None if (self.config.get_text_config().final_logit_softcapping) == () else (self.config.get_text_config().final_logit_softcapping),
        )
    elif _can_fuse_loss(self.loss_function) and labels is not None and NOT_RETURN_LOGITS:
        lm_head_weight = self.lm_head.weight
        lm_head_bias   = getattr(self.lm_head, "bias", None)
    
        # ========= NEW fused =========
        _hidden_states = hidden_states[:, slice_indices, :]
        _labels = labels if explicit_shift_labels is None else explicit_shift_labels
        torch._dynamo.mark_dynamic(_hidden_states, 1)
        torch._dynamo.mark_dynamic(_labels, _labels.dim() - 1)
        loss = unsloth_fused_ce_loss(
            trainer              = None,
            hidden_states        = _hidden_states,
            lm_head_weight       = lm_head_weight,
            lm_head_bias         = lm_head_bias,
            labels               = _labels,
            mask                 = None,
            n_items              = n_items,
            scaling              = getattr(self, "accelerator_scaler", None),
            target_gb            = None,
            torch_compile        = not UNSLOTH_COMPILE_DISABLE and not lm_head_on_old_cuda_arch(lm_head_weight),
            logit_scale_multiply = () if () != () else 0,
            logit_scale_divide   = () if () != () else 0,
            logit_softcapping    = (self.config.get_text_config().final_logit_softcapping) if (self.config.get_text_config().final_logit_softcapping) != () else 0,
            shift_labels         = explicit_shift_labels is None,
        )
    elif _can_fuse_loss(self.loss_function) and labels is not None and not NOT_RETURN_LOGITS:
        # UNSLOTH_RETURN_LOGITS=1 path. Prepended `logits = self.lm_head(...)`
        # already materialised the full lm_head matmul; apply the captured logit
        # scale/softcap transforms and route loss through self.loss_function on
        # those logits instead of letting unsloth_fused_ce_loss redo the matmul.
        if () != ():
            logits = logits * ()
        if () != ():
            logits = logits / ()
        if (self.config.get_text_config().final_logit_softcapping) not in (None, (),):
            logits = logits / (self.config.get_text_config().final_logit_softcapping)
            logits = torch.tanh(logits)
            logits = logits * (self.config.get_text_config().final_logit_softcapping)
        loss = self.loss_function(logits, labels.to(self.lm_head.weight.device), vocab_size=self.config.get_text_config().vocab_size, **kwargs)
    else:
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if () != ():
            logits = logits * ()
        if () != ():
            logits = logits / ()
        if (self.config.get_text_config().final_logit_softcapping) not in (None, (),):
            logits = logits / (self.config.get_text_config().final_logit_softcapping)
            logits = torch.tanh(logits)
            logits = logits * (self.config.get_text_config().final_logit_softcapping)
        loss = self.loss_function(logits, labels.to(self.lm_head.weight.device), vocab_size=self.config.get_text_config().vocab_size, **kwargs)


    return Gemma4UnifiedCausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        image_hidden_states=outputs.image_hidden_states,
        audio_hidden_states=outputs.audio_hidden_states,
        shared_kv_states=outputs.shared_kv_states,
    )

class Gemma4UnifiedForConditionalGeneration(Gemma4UnifiedPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    accepts_loss_kwargs = False
    base_model_prefix = "model"

    def __init__(self, config: Gemma4UnifiedConfig):
        super().__init__(config)
        self.model = Gemma4UnifiedModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        image_position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        r"""
        image_position_ids (`torch.LongTensor` of shape `(batch_size, max_patches, 2)`, *optional*):
            2D patch position coordinates from the image processor, with `(-1, -1)` indicating padding.
            Passed through to the vision encoder for positional embedding computation.
        """
        return self.model.get_image_features(pixel_values, image_position_ids, **kwargs)


    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        input_features: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        input_features_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        image_position_ids: torch.LongTensor | None = None,
        video_position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        mm_token_type_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Gemma4UnifiedCausalLMOutputWithPast:
        return Gemma4UnifiedForConditionalGeneration_forward(self, input_ids=input_ids, pixel_values=pixel_values, pixel_values_videos=pixel_values_videos, input_features=input_features, attention_mask=attention_mask, input_features_mask=input_features_mask, position_ids=position_ids, image_position_ids=image_position_ids, video_position_ids=video_position_ids, past_key_values=past_key_values, mm_token_type_ids=mm_token_type_ids, inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache, logits_to_keep=logits_to_keep, **kwargs)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        position_ids=None,
        pixel_values=None,
        pixel_values_videos=None,
        input_features=None,
        attention_mask=None,
        input_features_mask=None,
        token_type_ids=None,
        use_cache=True,
        logits_to_keep=None,
        labels=None,
        is_first_iteration=False,
        **kwargs,
    ):
        # Overwritten -- custom `position_ids` and `pixel_values` handling
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            token_type_ids=token_type_ids,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        # If we're in cached decoding stage, multimodal inputs are already cached and can be dropped
        if is_first_iteration or not use_cache:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["pixel_values_videos"] = pixel_values_videos
            model_inputs["input_features"] = input_features
            model_inputs["input_features_mask"] = input_features_mask
        else:
            # Don't pass to not apply bidirectional mask on top
            model_inputs["mm_token_type_ids"] = None

        return model_inputs

    @staticmethod
    def create_masks_for_generate(
        config: PreTrainedConfig,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None,
        position_ids: torch.Tensor | None,
        mm_token_type_ids: torch.Tensor | None = None,
        is_first_iteration: bool | None = False,
        **kwargs,
    ) -> dict:
        mask_kwargs = {
            "config": config.get_text_config(),
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }

        text_config = config.get_text_config()
        use_bidir = getattr(text_config, "use_bidirectional_attention", None) == "vision"

        if use_bidir and mm_token_type_ids is not None:
            block_sequence_ids = get_block_sequence_ids_for_mask(mm_token_type_ids, device=inputs_embeds.device)
            return create_masks_for_vision_model(
                block_sequence_ids=block_sequence_ids,
                **mask_kwargs,
            )

        return create_masks_for_generate(**mask_kwargs)

Gemma4UnifiedForConditionalGeneration.__UNSLOTH_SUPPORTS_RETURN_HIDDEN_STATES__ = True
