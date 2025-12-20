from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from HH import data_utils
from HH.hh_layers import HHQwen2Attention, HHQwen2MLP
from HH.calibration import determine_rank_from_svd


class StopForward(RuntimeError):
    """Private exception used to halt the model after the first layer."""


@dataclass
class VerificationArgs:
    model_path: str
    cali_dataset: str = "wikitext2"
    nsamples: int = 8
    cali_bsz: int = 2
    max_seq_len: int = 2048
    num_layers: int = 2
    w_bits: int = 8
    a_bits: int = 8
    a_asym: bool = True
    k_bits: int = 8
    k_asym: bool = True
    v_bits: int = 8
    v_asym: bool = True
    n_reflections_qkv: int = 16
    n_reflections_o: int = 16
    n_reflections_up_gate: int = 16
    n_reflections_down: int = 16
    svd_threshold: float = 0.95
    lr: float = 5e-4
    steps: int = 60
    seed: int = 0
    device: Optional[str] = None


def parse_args() -> VerificationArgs:
    parser = argparse.ArgumentParser(description="Verify HH on the first layers of DeepSeek (Qwen backbone).")
    parser.add_argument(
        "--model-path",
        required=False,
        default=os.environ.get("DEEPSEEK_MODEL_PATH", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
        help="Hugging Face repo id or local path to DeepSeek-R1-Distill-Qwen checkpoint.",
    )
    parser.add_argument("--cali-dataset", default="wikitext2", help="Calibration dataset (default: WikiText-2).")
    parser.add_argument("--nsamples", type=int, default=8, help="Number of calibration samples to capture.")
    parser.add_argument("--cali-bsz", type=int, default=2, help="Batch size used for calibration/optimization.")
    parser.add_argument("--max-seq-len", type=int, default=2048, help="Sequence length for calibration samples.")
    parser.add_argument("--num-layers", type=int, default=2, help="How many early layers to verify.")
    parser.add_argument("--steps", type=int, default=60, help="Optimization steps per layer.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for Householder optimization.")
    parser.add_argument("--svd-threshold", type=float, default=0.95, help="Energy threshold for PCA init.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--reflections", type=int, default=16, help="Number of reflections per rotation.")
    parser.add_argument("--device", default=None, help="Force device (e.g., cuda:0). Defaults to auto.")

    args_ns = parser.parse_args()
    reflections = args_ns.reflections
    return VerificationArgs(
        model_path=args_ns.model_path,
        cali_dataset=args_ns.cali_dataset,
        nsamples=args_ns.nsamples,
        cali_bsz=args_ns.cali_bsz,
        max_seq_len=args_ns.max_seq_len,
        num_layers=args_ns.num_layers,
        svd_threshold=args_ns.svd_threshold,
        lr=args_ns.lr,
        steps=args_ns.steps,
        seed=args_ns.seed,
        n_reflections_qkv=reflections,
        n_reflections_o=reflections,
        n_reflections_up_gate=reflections,
        n_reflections_down=reflections,
        device=args_ns.device,
    )


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_device(explicit_device: Optional[str]) -> torch.device:
    if explicit_device:
        return torch.device(explicit_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def capture_first_layer_inputs(
    model: AutoModelForCausalLM,
    trainloader: Iterable[torch.Tensor],
    args: VerificationArgs,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Use a Catcher module to store the inputs to the first transformer layer.
    """

    class Catcher(nn.Module):
        def __init__(self, module, storage, cache_dict):
            super().__init__()
            self.module = module
            self.storage = storage
            self.cache = cache_dict

        def forward(self, hidden_states, **kwargs):
            if self.cache["i"] >= self.storage.shape[0]:
                raise StopForward

            self.storage[self.cache["i"]] = hidden_states
            if self.cache["i"] == 0:
                self.cache["kwargs"] = {
                    "attention_mask": kwargs.get("attention_mask"),
                    "position_ids": kwargs.get("position_ids"),
                    "position_embeddings": kwargs.get("position_embeddings"),
                }
            self.cache["i"] += 1
            raise StopForward

        def __getattr__(self, name):
            if name in {"module", "storage", "cache", "forward"}:
                return super().__getattr__(name)
            return getattr(self.module, name)

    layers = model.model.layers
    hidden_size = model.config.hidden_size
    storage = torch.zeros(
        (args.nsamples, args.max_seq_len, hidden_size), device=device, dtype=torch.float16
    )
    cache: Dict[str, object] = {"i": 0, "kwargs": {}}

    layers[0] = Catcher(layers[0], storage, cache)

    with torch.no_grad():
        for batch in trainloader:
            if cache["i"] >= args.nsamples:
                break
            try:
                model(batch.to(device))
            except StopForward:
                pass

    layers[0] = layers[0].module
    if cache["i"] < args.nsamples:
        raise RuntimeError(f"Captured only {cache['i']} samples, expected {args.nsamples}.")

    kwargs_cache: Dict[str, torch.Tensor] = {}
    for key, value in cache["kwargs"].items():
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            kwargs_cache[key] = value.to(device)
        elif isinstance(value, tuple):
            kwargs_cache[key] = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in value)  # type: ignore

    return storage, kwargs_cache


def choose_reflections(tensor: torch.Tensor, max_reflections: int, threshold: float) -> int:
    reflections, _ = determine_rank_from_svd(
        tensor, threshold=threshold, max_rank=max_reflections
    )
    return max(1, min(max_reflections, reflections))


def expand_kwargs_for_batch(
    kwargs_template: Dict[str, torch.Tensor], batch_size: int
) -> Dict[str, torch.Tensor]:
    expanded: Dict[str, torch.Tensor] = {}
    for key, value in kwargs_template.items():
        if isinstance(value, torch.Tensor):
            if value.shape[0] == batch_size:
                expanded[key] = value
            else:
                expanded[key] = value.expand(batch_size, *value.shape[1:])
        elif isinstance(value, tuple):
            expanded[key] = tuple(
                tensor if tensor.shape[0] == batch_size else tensor.expand(batch_size, *tensor.shape[1:])
                for tensor in value
            )
    return expanded


def compute_fp_targets(
    layer: nn.Module,
    inputs: torch.Tensor,
    kwargs_cache: Dict[str, torch.Tensor],
    args: VerificationArgs,
    device: torch.device,
) -> torch.Tensor:
    outputs = torch.zeros_like(inputs, device=device)
    with torch.no_grad():
        for start in range(0, args.nsamples, args.cali_bsz):
            end = min(start + args.cali_bsz, args.nsamples)
            batch_inp = inputs[start:end].to(device).float()
            batch_kwargs = expand_kwargs_for_batch(kwargs_cache, batch_inp.size(0))
            out = layer(batch_inp, **batch_kwargs)[0]
            outputs[start:end] = out.to(outputs.dtype)
    return outputs


def initialize_rotations(
    layer: nn.Module,
    hh_attn: HHQwen2Attention,
    hh_mlp: HHQwen2MLP,
    inputs: torch.Tensor,
    kwargs_cache: Dict[str, torch.Tensor],
    args: VerificationArgs,
) -> None:
    init_batch = inputs[: min(args.cali_bsz, inputs.shape[0])].float()
    with torch.no_grad():
        normalized = layer.input_layernorm(init_batch)
        flat_attn = normalized.reshape(-1, normalized.shape[-1])
        hh_attn.qkv_rotation.initialize_from_activations(flat_attn, threshold=args.svd_threshold)
        hh_attn.o_proj.rotation.initialize_from_activations(flat_attn, threshold=args.svd_threshold)

        attn_kwargs = expand_kwargs_for_batch(kwargs_cache, normalized.size(0))
        attn_out = hh_attn(normalized, **attn_kwargs)[0]
        post_attn = layer.post_attention_layernorm(init_batch + attn_out)
        flat_post_attn = post_attn.reshape(-1, post_attn.shape[-1])
        hh_mlp.up_gate_rotation.initialize_from_activations(flat_post_attn, threshold=args.svd_threshold)

        rotated = hh_mlp.up_gate_rotation(post_attn)
        gate = hh_mlp.gate_proj(post_attn, external_rotated=rotated)
        up = hh_mlp.up_proj(post_attn, external_rotated=rotated)
        hidden = hh_mlp.act_fn(gate) * up
        flat_hidden = hidden.reshape(-1, hidden.shape[-1])
        hh_mlp.down_proj.rotation.initialize_from_activations(flat_hidden, threshold=args.svd_threshold)


def calibrate_hh_linears(hh_attn: HHQwen2Attention, hh_mlp: HHQwen2MLP, sample: torch.Tensor) -> None:
    with torch.no_grad():
        for proj in (hh_attn.q_proj, hh_attn.k_proj, hh_attn.v_proj, hh_attn.o_proj):
            proj.calibrate(sample)
        for proj in (hh_mlp.up_proj, hh_mlp.gate_proj):
            proj.calibrate(sample)
        gate = hh_mlp.gate_proj(sample)
        up = hh_mlp.up_proj(sample)
        hidden = hh_mlp.act_fn(gate) * up
        hh_mlp.down_proj.calibrate(hidden)


def gather_trainable_params(layer: nn.Module) -> Iterable[torch.nn.Parameter]:
    params = []
    if isinstance(layer.self_attn, HHQwen2Attention):
        params.extend([layer.self_attn.qkv_rotation.vectors, layer.self_attn.qkv_scale])
        params.extend([layer.self_attn.o_proj.rotation.vectors, layer.self_attn.o_proj.scale])
    if isinstance(layer.mlp, HHQwen2MLP):
        params.extend([layer.mlp.up_gate_rotation.vectors, layer.mlp.up_gate_scale])
        params.extend([layer.mlp.down_proj.rotation.vectors, layer.mlp.down_proj.scale])
    for param in params:
        param.requires_grad = True
    return params


def evaluate_mse(
    layer: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    kwargs_cache: Dict[str, torch.Tensor],
    args: VerificationArgs,
    device: torch.device,
) -> Tuple[float, int]:
    mse_total = 0.0
    nan_count = 0
    total_elems = 0
    loss_fn = nn.MSELoss(reduction="sum")
    with torch.no_grad():
        for start in range(0, args.nsamples, args.cali_bsz):
            end = min(start + args.cali_bsz, args.nsamples)
            batch_inp = inputs[start:end].to(device)
            batch_tgt = targets[start:end].to(device)
            batch_kwargs = expand_kwargs_for_batch(kwargs_cache, batch_inp.size(0))
            pred = layer(batch_inp, **batch_kwargs)[0]
            nan_count += torch.isnan(pred).sum().item()
            mse_total += loss_fn(pred, batch_tgt).item()
            total_elems += batch_tgt.numel()
    return mse_total / max(total_elems, 1), nan_count


def optimize_layer(
    layer: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    kwargs_cache: Dict[str, torch.Tensor],
    args: VerificationArgs,
    device: torch.device,
) -> Tuple[float, float, int]:
    params = list(gather_trainable_params(layer))
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    loss_fn = nn.MSELoss()

    baseline_mse, baseline_nans = evaluate_mse(layer, inputs, targets, kwargs_cache, args, device)
    logging.info("  Baseline MSE: %.6f | NaNs: %d", baseline_mse, baseline_nans)

    for step in range(args.steps):
        idx = torch.randint(0, args.nsamples, (args.cali_bsz,))
        batch_inp = inputs[idx].to(device)
        batch_tgt = targets[idx].to(device)
        batch_kwargs = expand_kwargs_for_batch(kwargs_cache, batch_inp.size(0))

        pred = layer(batch_inp, **batch_kwargs)[0]
        loss = loss_fn(pred, batch_tgt)
        if torch.isnan(loss):
            logging.error("  Encountered NaN loss at step %d.", step)
            break

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()

        if (step + 1) % max(1, args.steps // 5) == 0:
            logging.info("  Step %d/%d | Loss: %.6f", step + 1, args.steps, loss.item())

    final_mse, nan_count = evaluate_mse(layer, inputs, targets, kwargs_cache, args, device)
    logging.info("  Final MSE: %.6f | NaNs: %d", final_mse, nan_count)
    return baseline_mse, final_mse, nan_count


def update_inputs_for_next_layer(
    layer: nn.Module,
    inputs: torch.Tensor,
    kwargs_cache: Dict[str, torch.Tensor],
    args: VerificationArgs,
    device: torch.device,
) -> torch.Tensor:
    updated = torch.zeros_like(inputs)
    with torch.no_grad():
        if hasattr(layer, "self_attn") and isinstance(layer.self_attn, HHQwen2Attention):
            layer.self_attn.q_proj.reparameterize()
            layer.self_attn.k_proj.reparameterize()
            layer.self_attn.v_proj.reparameterize()
            layer.self_attn.o_proj.reparameterize()
        if hasattr(layer, "mlp") and isinstance(layer.mlp, HHQwen2MLP):
            layer.mlp.up_proj.reparameterize()
            layer.mlp.gate_proj.reparameterize()
            layer.mlp.down_proj.reparameterize()

        for start in range(0, args.nsamples, args.cali_bsz):
            end = min(start + args.cali_bsz, args.nsamples)
            batch_inp = inputs[start:end].to(device)
            batch_kwargs = expand_kwargs_for_batch(kwargs_cache, batch_inp.size(0))
            out = layer(batch_inp, **batch_kwargs)[0]
            updated[start:end] = out.to(updated.dtype)
    return updated


def verify_on_real_model(args: VerificationArgs) -> None:
    logging.info("Loading model from %s", args.model_path)
    device = prepare_device(args.device)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False

    logging.info("Loading calibration data (%s)...", args.cali_dataset)
    trainloader = data_utils.get_loaders(
        args.cali_dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=args.max_seq_len,
        model_path=args.model_path,
    )

    inputs, kwargs_cache = capture_first_layer_inputs(model, trainloader, args, device)
    logging.info("Captured %d samples for layer inputs. Max %.4f | Min %.4f", args.nsamples, inputs.max(), inputs.min())

    current_inputs = inputs
    for layer_idx in range(min(args.num_layers, len(model.model.layers))):
        logging.info("===== Optimizing Layer %d =====", layer_idx)
        layer = model.model.layers[layer_idx].to(device).float()
        ori_attn = layer.self_attn
        ori_mlp = layer.mlp

        fp_targets = compute_fp_targets(layer, current_inputs, kwargs_cache, args, device)

        # Determine reflection counts from PCA (OSTQuant-style)
        init_batch = current_inputs[: args.cali_bsz].float()
        normalized = layer.input_layernorm(init_batch)
        attn_kwargs = expand_kwargs_for_batch(kwargs_cache, normalized.size(0))
        attn_out_ori = ori_attn(normalized, **attn_kwargs)[0]

        h_qkv = choose_reflections(
            normalized.reshape(-1, normalized.shape[-1]),
            max_reflections=args.n_reflections_qkv,
            threshold=args.svd_threshold,
        )

        hh_attn = HHQwen2Attention(
            args, ori_attn, h_qkv, min(args.n_reflections_o, h_qkv)
        ).to(device=device, dtype=torch.float32)

        post_attn = layer.post_attention_layernorm(init_batch + attn_out_ori)
        h_up_gate = choose_reflections(
            post_attn.reshape(-1, post_attn.shape[-1]),
            max_reflections=args.n_reflections_up_gate,
            threshold=args.svd_threshold,
        )
        h_down = choose_reflections(
            post_attn.reshape(-1, post_attn.shape[-1]),
            max_reflections=args.n_reflections_down,
            threshold=args.svd_threshold,
        )

        hh_mlp = HHQwen2MLP(
            args, ori_mlp, h_up_gate, h_down
        ).to(device=device, dtype=torch.float32)
        layer.self_attn = hh_attn
        layer.mlp = hh_mlp

        initialize_rotations(layer, hh_attn, hh_mlp, current_inputs, kwargs_cache, args)
        calibrate_hh_linears(hh_attn, hh_mlp, current_inputs[: args.cali_bsz].float())

        _, final_mse, nan_count = optimize_layer(layer, current_inputs, fp_targets, kwargs_cache, args, device)
        if nan_count > 0:
            logging.warning("Layer %d produced %d NaN values during evaluation.", layer_idx, nan_count)

        # Fuse scales into LayerNorm/weights (OSTQuant-style)
        if hasattr(layer, "input_layernorm"):
            layer.self_attn.fuse_qkv_scale(layer.input_layernorm)
        else:
            layer.self_attn.fuse_qkv_scale(None)
        if hasattr(layer, "post_attention_layernorm"):
            layer.mlp.fuse_up_gate_scale(layer.post_attention_layernorm)
        else:
            layer.mlp.fuse_up_gate_scale(None)
        layer.self_attn.o_proj.reparameterize()
        layer.mlp.down_proj.reparameterize()

        current_inputs = update_inputs_for_next_layer(layer, current_inputs, kwargs_cache, args, device)
        model.model.layers[layer_idx] = layer.to(dtype)

    logging.info("Verification complete.")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
    )
    args = parse_args()
    set_seed(args.seed)
    verify_on_real_model(args)


if __name__ == "__main__":
    main()
