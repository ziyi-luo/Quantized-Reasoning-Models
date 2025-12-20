
from types import SimpleNamespace

import torch
import torch.nn as nn
import pytest

try:
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2ForCausalLM  # type: ignore
    TRANSFORMERS_AVAILABLE = True
except ImportError:  # pragma: no cover
    Qwen2Config = None  # type: ignore
    Qwen2ForCausalLM = None  # type: ignore
    TRANSFORMERS_AVAILABLE = False

if TRANSFORMERS_AVAILABLE:
    from HH.calibration import determine_rank_from_svd
    from HH.hh_layers import HHQwen2Attention, HHQwen2MLP
    from HH.hh_linear import HHLinear

    @pytest.fixture
    def dummy_args():
        return SimpleNamespace(
            w_bits=8,
            a_bits=8,
            a_asym=False,
            k_bits=8,
            k_asym=False,
            v_bits=8,
            v_asym=False,
        )


    def test_pca_rank_selection_prefers_low_h():
        base = torch.randn(256, 4)
        components = torch.randn(4, 32)
        data = base @ components  # rank <= 4
        k, _ = determine_rank_from_svd(data, threshold=0.95, max_rank=16)
        assert k <= 4


    def test_shared_rotations_and_scale_fusion(dummy_args):
        config = Qwen2Config(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
        model = Qwen2ForCausalLM(config)
        layer = model.model.layers[0]

        hh_attn = HHQwen2Attention(dummy_args, layer.self_attn, n_reflections_qkv=6, n_reflections_o=4)
        hh_mlp = HHQwen2MLP(dummy_args, layer.mlp, n_reflections_up_gate=5, n_reflections_down=5)

        # Shared Householder rotations
        assert hh_attn.q_proj.rotation is hh_attn.k_proj.rotation is hh_attn.v_proj.rotation
        assert hh_mlp.up_proj.rotation is hh_mlp.gate_proj.rotation

        # Fuse qkv scale into LayerNorm and ensure rotation scales are stripped
        ln_weight_before = layer.input_layernorm.weight.clone()
        hh_attn.reparameterize()
        hh_attn.fuse_qkv_scale(layer.input_layernorm)
        assert torch.allclose(hh_attn.qkv_scale, torch.ones_like(hh_attn.qkv_scale))
        assert torch.allclose(
            hh_attn.q_proj.rotation.scale_buffer, torch.ones_like(hh_attn.q_proj.rotation.scale_buffer)
        )
        ratio = layer.input_layernorm.weight / ln_weight_before
        assert torch.allclose(ratio, ratio.mean().expand_as(ratio), atol=1e-4, rtol=1e-2)

        # Fuse up/gate scale into post-attention LayerNorm
        mlp_ln_before = layer.post_attention_layernorm.weight.clone()
        hh_mlp.reparameterize()
        hh_mlp.fuse_up_gate_scale(layer.post_attention_layernorm)
        assert torch.allclose(hh_mlp.up_gate_scale, torch.ones_like(hh_mlp.up_gate_scale))
        assert torch.allclose(
            hh_mlp.up_proj.rotation.scale_buffer, torch.ones_like(hh_mlp.up_proj.rotation.scale_buffer)
        )
        ratio_mlp = layer.post_attention_layernorm.weight / mlp_ln_before
        assert torch.allclose(ratio_mlp, ratio_mlp.mean().expand_as(ratio_mlp), atol=1e-4, rtol=1e-2)


    def test_hhlinear_eval_no_nan(dummy_args):
        linear = nn.Linear(16, 8)
        hh = HHLinear(linear, n_reflections=4, args=dummy_args)
        sample = torch.randn(2, 4, 16)
        hh.calibrate(sample)
        hh.reparameterize()
        hh.strip_scale(torch.ones(16))
        out = hh(sample)
        assert torch.isfinite(out).all()
else:  # pragma: no cover
    def test_transformers_placeholder():
        pytest.skip("transformers is not installed")
