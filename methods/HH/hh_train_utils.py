"""
Refined training utilities for HH quantization.
Implements:
1. Alpha-scaling (SmoothQuant style)
2. SVD-based HH initialization
3. Training of Householder vectors
4. Reparameterization
"""
import os
import time
import gc
import torch
import torch.nn as nn
from .function_utils import get_init_scale

def get_module_hh_linear_layers(module):
    """Recursively find all HHLinear layers in a module."""
    layers = []
    for m in module.modules():
        if m.__class__.__name__ == 'HHLinear':
            layers.append(m)
    return layers

def cali_hh_quant(args, model, dataloader, dev, logger, start_layer_idx=0):
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False
    
    for param in model.parameters():
        param.requires_grad = False
    
    dtype = torch.float32
    if not args.deactive_amp:
        dtype = torch.bfloat16
        
    layers = model.model.layers
    layers[0] = layers[0].to(dev)
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    if hasattr(model.model, "rotary_emb"):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    
    captured_data = [] # (inp_cpu, kwargs_cpu)
    
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            # Capture for rotation optimization
            inp_cpu = inp.cpu()
            kwargs_cpu = {}
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor): kwargs_cpu[k] = v.cpu()
                elif isinstance(v, tuple): kwargs_cpu[k] = tuple(t.cpu() if isinstance(t, torch.Tensor) else t for t in v)
                else: kwargs_cpu[k] = v
            captured_data.append((inp_cpu, kwargs_cpu))
            raise ValueError
        def __getattr__(self, name):
            if name in ["module", "forward"]: return super().__getattr__(name)
            return getattr(self.module, name)
    
    layers[0] = Catcher(layers[0])
    with torch.no_grad():
        for batch in dataloader:
            if len(captured_data) >= args.nsamples: break
            try: model(batch[0].to(dev))
            except ValueError: pass
    
    layers[0] = layers[0].module.cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if hasattr(model.model, "rotary_emb"): model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()
    
    cur_hiddens = [d[0] for d in captured_data]
    loss_func = nn.MSELoss()
    
    for i in range(len(layers)):
        logger.info(f"========= Layer {i}/{len(layers)} =========")
        layer = layers[i].to(dev)
        
        # 1. Stats Collection (Original mode)
        layer._ori_mode = True
        input_max = None
        
        with torch.no_grad():
            for j in range(len(captured_data)):
                inp_j = cur_hiddens[j].to(dev)
                kwargs_j = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in captured_data[j][1].items()}
                for k, v in kwargs_j.items():
                    if isinstance(v, tuple): kwargs_j[k] = tuple(t.to(dev) if isinstance(t, torch.Tensor) else t for t in v)
                
                # Capture activation max for SmoothQuant scaling
                # We can hook or just use the input to the layer as a proxy for the first HH modules
                act_abs = inp_j.abs().reshape(-1, inp_j.shape[-1]).max(dim=0)[0]
                if input_max is None: input_max = act_abs
                else: input_max = torch.maximum(input_max, act_abs)
                
                # Get reference FP32 outputs
                _ = layer(inp_j, **kwargs_j) # This is _ori_forward now

        # 2. SmoothQuant-style Scaling Initialization
        # For simplicity, we apply scaling based on input_max to the shared scale parameters
        if i >= start_layer_idx:
            alpha = 0.5 # Default SmoothQuant alpha
            hh_linears = get_module_hh_linear_layers(layer)
            if len(hh_linears) > 0:
                logger.info("Initializing SmoothQuant-style scaling...")
                with torch.no_grad():
                    # For Attention (shared qkv_scale)
                    if hasattr(layer, 'qkv_scale'):
                        w_max = torch.cat([layer.q_proj.weight, layer.k_proj.weight, layer.v_proj.weight], dim=0).abs().max(dim=0)[0]
                        layer.qkv_scale.data.copy_(get_init_scale(w_max, input_max, alpha))
                    
                    # For MLP (shared up_gate_scale)
                    if hasattr(layer, 'up_gate_scale'):
                        w_max = torch.cat([layer.up_proj.weight, layer.gate_proj.weight], dim=0).abs().max(dim=0)[0]
                        layer.up_gate_scale.data.copy_(get_init_scale(w_max, input_max, alpha))
                        
                    # For O-proj and Down-proj (independent)
                    # These would need their own stats collection if we wanted to be rigorous.
                    # For now, we focus on the main input projections.

            # 3. Reference FP32 results for optimization
            layer._ori_mode = True
            ref_hiddens = []
            with torch.no_grad():
                for j in range(len(captured_data)):
                    inp_j = cur_hiddens[j].to(dev)
                    kwargs_j = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in captured_data[j][1].items()}
                    for k, v in kwargs_j.items():
                        if isinstance(v, tuple): kwargs_j[k] = tuple(t.to(dev) if isinstance(t, torch.Tensor) else t for t in v)
                    ref_hiddens.append(layer(inp_j, **kwargs_j)[0].cpu())

            # 4. Householder Optimization
            layer._ori_mode = False
            params = []
            for m in hh_linears:
                m.rotation.vectors.requires_grad = True
                m.scale.requires_grad = True
                params.extend([m.rotation.vectors, m.scale])
            
            if len(params) > 0:
                logger.info(f"Optimizing {len(hh_linears)} HH modules...")
                optimizer = torch.optim.AdamW(params, lr=args.hh_lr)
                for epoch in range(args.epochs):
                    total_mse = 0
                    for j in range(len(captured_data)):
                        inp_j = cur_hiddens[j].to(dev)
                        kwargs_j = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in captured_data[j][1].items()}
                        for k, v in kwargs_j.items():
                             if isinstance(v, tuple): kwargs_j[k] = tuple(t.to(dev) if isinstance(t, torch.Tensor) else t for t in v)
                        
                        out_quant = layer(inp_j, **kwargs_j)[0]
                        loss = loss_func(out_quant, ref_hiddens[j].to(dev))
                        total_mse += loss.item()
                        optimizer.zero_grad(); loss.backward(); optimizer.step()
                    if epoch % 10 == 0 or epoch == args.epochs - 1:
                        logger.info(f"Layer {i} Epoch {epoch} Loss: {total_mse/len(captured_data):.6f}")

                # 5. Reparameterization
                logger.info("Reparameterizing layer...")
                layer.reparameterize()
                for p in params: p.requires_grad = False

        # update hidden states for next layer using the REPARAMETERIZED (eval-ready) version
        layer.eval()
        layer._ori_mode = False
        new_hiddens = []
        with torch.no_grad():
            for j in range(len(captured_data)):
                inp_j = cur_hiddens[j].to(dev)
                kwargs_j = {k: (v.to(dev) if hasattr(v, 'to') else v) for k, v in captured_data[j][1].items()}
                for k, v in kwargs_j.items():
                    if isinstance(v, tuple): kwargs_j[k] = tuple(t.to(dev) if isinstance(t, torch.Tensor) else t for t in v)
                new_hiddens.append(layer(inp_j, **kwargs_j)[0].cpu())
        
        cur_hiddens = new_hiddens
        layers[i] = layer.cpu()
        torch.cuda.empty_cache(); gc.collect()

    model.config.use_cache = use_cache
    return model
