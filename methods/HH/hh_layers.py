"""
Module wrappers for Qwen2 supporting Householder Quantization.
Matches the structure of FlatQuant's qwen_utils.py.
"""
import torch
import torch.nn as nn
from typing import Optional
from .hh_linear import HHLinear
from .hh_modules import HouseholderRotation
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP, Qwen2Attention, apply_rotary_pos_emb, repeat_kv
import math

class HHQwen2MLP(nn.Module):
    def __init__(self, args, module: Qwen2MLP, n_reflections_up_gate: int, n_reflections_down: int):
        super().__init__()
        self.args = args
        self.act_fn = module.act_fn
        
        # Shared input rotation for up and gate
        self.up_gate_rotation = HouseholderRotation(module.hidden_size, n_reflections_up_gate)
        self.up_gate_scale = nn.Parameter(torch.ones(module.hidden_size))
        
        # Wrapped layers
        self.up_proj = HHLinear(module.up_proj, n_reflections_up_gate, 
                                args=args,
                                external_rotation=self.up_gate_rotation,
                                external_scale=self.up_gate_scale)
        self.gate_proj = HHLinear(module.gate_proj, n_reflections_up_gate,
                                 args=args,
                                 external_rotation=self.up_gate_rotation,
                                 external_scale=self.up_gate_scale)
        
        # Down proj has its own rotation (usually on intermediate_size)
        self.down_proj = HHLinear(module.down_proj, n_reflections_down,
                                 args=args)
        
        self._ori_mode = False

    def forward(self, x):
        if self._ori_mode:
            return self._ori_forward(x)
        
        # Transformed forward
        # Compute shared rotation once
        x_rot = self.up_gate_rotation(x)
        
        gate_out = self.gate_proj(x, external_rotated=x_rot)
        up_out = self.up_proj(x, external_rotated=x_rot)
        
        # act_fn is typically SiLU
        h = self.act_fn(gate_out) * up_out
        return self.down_proj(h)

    def _ori_forward(self, x):
        # Collect stats if needed (for SmoothQuant)
        gate_out = torch.nn.functional.linear(x, self.gate_proj.weight, self.gate_proj.bias)
        up_out = torch.nn.functional.linear(x, self.up_proj.weight, self.up_proj.bias)
        h = self.act_fn(gate_out) * up_out
        return torch.nn.functional.linear(h, self.down_proj.weight, self.down_proj.bias)

    def reparameterize(self):
        # Reparameterize the shared rotation once
        self.up_gate_rotation.reparameterize(self.up_gate_scale)
        
        self.up_proj.reparameterize()
        self.gate_proj.reparameterize()
        self.down_proj.reparameterize()

    def fuse_up_gate_scale(self, target_norm: Optional[nn.LayerNorm] = None):
        """
        Absorb the shared scale into a LayerNorm (OSTQuant-style) so that the
        Householder rotation stays orthogonal during inference.
        """
        with torch.no_grad():
            scale = self.up_gate_scale.detach().clone()
            self.up_proj.reparameterize()
            self.gate_proj.reparameterize()
            self.up_proj.strip_scale(scale)
            self.gate_proj.strip_scale(scale)
            if target_norm is not None:
                target_norm.weight.mul_(scale.to(target_norm.weight))
            self.up_gate_scale.data.fill_(1.0)

class HHQwen2Attention(nn.Module):
    def __init__(self, args, module: Qwen2Attention, n_reflections_qkv: int, n_reflections_o: int):
        super().__init__()
        self.args = args
        self.config = module.config
        self.layer_idx = module.layer_idx
        self.head_dim = module.head_dim
        self.num_heads = module.config.num_attention_heads
        self.num_key_value_heads = module.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = module.attention_dropout
        
        # Shared input rotation for q, k, v
        self.qkv_rotation = HouseholderRotation(self.config.hidden_size, n_reflections_qkv)
        self.qkv_scale = nn.Parameter(torch.ones(self.config.hidden_size))
        self.scaling = self.head_dim**-0.5
        
        # Helper for K/V specialized args
        import copy
        k_args = copy.copy(args)
        k_args.a_bits = args.k_bits
        k_args.a_asym = args.k_asym
        
        v_args = copy.copy(args)
        v_args.a_bits = args.v_bits
        v_args.a_asym = args.v_asym
        v_args.w_bits = args.w_bits # Reuse w_bits

        self.q_proj = HHLinear(module.q_proj, n_reflections_qkv,
                               args=args,
                               external_rotation=self.qkv_rotation,
                               external_scale=self.qkv_scale)
        self.k_proj = HHLinear(module.k_proj, n_reflections_qkv,
                               args=k_args,
                               external_rotation=self.qkv_rotation,
                               external_scale=self.qkv_scale)
        self.v_proj = HHLinear(module.v_proj, n_reflections_qkv,
                               args=v_args,
                               external_rotation=self.qkv_rotation,
                               external_scale=self.qkv_scale)
        
        # o_proj has its own rotation (on num_heads * head_dim)
        self.o_proj = HHLinear(module.o_proj, n_reflections_o,
                               args=args)
        
        self._ori_mode = False

    def forward(self, hidden_states, attention_mask=None, position_ids=None, 
                past_key_value=None, output_attentions=False, use_cache=False, 
                cache_position=None, position_embeddings=None, **kwargs):
        
        if self._ori_mode:
            return self._ori_forward(hidden_states, attention_mask, position_ids, 
                                   past_key_value, output_attentions, use_cache, 
                                   cache_position, position_embeddings, **kwargs)

        bsz, q_len, _ = hidden_states.size()

        # Compute shared rotation once
        x_rot = self.qkv_rotation(hidden_states)

        query_states = self.q_proj(hidden_states, external_rotated=x_rot)
        key_states = self.k_proj(hidden_states, external_rotated=x_rot)
        value_states = self.v_proj(hidden_states, external_rotated=x_rot)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            # Fallback for older transformers or manual calls
            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            # Not implementing full rotary_emb here, assuming position_embeddings are passed by model wrapper
            cos, sin = kwargs.get('cos'), kwargs.get('sin')
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # print(f"DEBUG key_states shape: {key_states.shape}, groups: {self.num_key_value_groups}")
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states.float(), key_states.float().transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            # DEBUG: Check mask stats
            # print(f"DEBUG: Mask Max={attention_mask.max()}, Min={attention_mask.min()}")
            attn_weights = attn_weights + attention_mask

        # DEBUG: Check pre-softmax stats
        # print(f"DEBUG: Pre-Softmax Max={attn_weights.max()}, Min={attn_weights.min()}, IsInf={attn_weights.isinf().any()}")

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights

    def _ori_forward(self, hidden_states, attention_mask=None, position_ids=None, 
                     past_key_value=None, output_attentions=False, use_cache=False, 
                     cache_position=None, position_embeddings=None, **kwargs):
        # Implementation of original forward to collect stats if needed
        # Just use F.linear for simplicity
        bsz, q_len, _ = hidden_states.size()
        q = F.linear(hidden_states, self.q_proj.weight, self.q_proj.bias)
        k = F.linear(hidden_states, self.k_proj.weight, self.k_proj.bias)
        v = F.linear(hidden_states, self.v_proj.weight, self.v_proj.bias)
        
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_value.update(k, v, self.layer_idx, cache_kwargs)
            
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)
        
        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        
        attn_output = F.linear(attn_output, self.o_proj.weight, self.o_proj.bias)
        return attn_output, attn_weights

    def reparameterize(self):
        # QKV share the same rotation and scale
        # We must ensure the shared rotation is reparameterized ONLY ONCE
        # HHLinear.reparameterize calls self.rotation.reparameterize(self.scale)
        # Since HouseholderRotation.reparameterize is idempotent (has is_reparameterized flag),
        # multiple calls SHOULD be safe, BUT the scale might be different if they were different.
        # Here they share the same self.qkv_scale, so it's technically safe.
        # However, to be absolutely clean and avoid any precision drift:
        self.q_proj.reparameterize()
        self.k_proj.reparameterize()
        self.v_proj.reparameterize()
        self.o_proj.reparameterize()

    def fuse_qkv_scale(self, target_norm: Optional[nn.LayerNorm] = None):
        """
        Absorb the shared activation scale into a preceding LayerNorm, leaving
        the rotation orthogonal for inference.
        """
        with torch.no_grad():
            scale = self.qkv_scale.detach().clone()
            self.q_proj.reparameterize()
            self.k_proj.reparameterize()
            self.v_proj.reparameterize()
            self.q_proj.strip_scale(scale)
            self.k_proj.strip_scale(scale)
            self.v_proj.strip_scale(scale)
            if target_norm is not None:
                target_norm.weight.mul_(scale.to(target_norm.weight))
            self.qkv_scale.data.fill_(1.0)
