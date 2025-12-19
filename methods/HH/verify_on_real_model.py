import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys
import logging
from tqdm import tqdm
import functools
import gc

# Add parent dir to path to import utils and HH modules
# Current file: .../methods/HH/verify_on_real_model.py
# Parent: .../methods/HH
# Grandparent: .../methods -> contains 'utils'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from HH.hh_model_utils import apply_hh_to_qwen, reparameterize_model
from HH.hh_layers import HHQwen2Attention, HHQwen2MLP
from utils import data_utils # Import data_utils from methods/utils

logging.basicConfig(level=logging.INFO)

class Catcher(nn.Module):
    def __init__(self, module, inps, cache):
        super().__init__()
        self.module = module
        self.inps = inps
        self.cache = cache
        
    def forward(self, inp, **kwargs):
        print(f"DEBUG: Catcher called. Input shape: {inp.shape}. Kwargs keys: {list(kwargs.keys())}")
        self.inps[self.cache['i']] = inp
        self.cache['i'] += 1
        self.cache['attention_mask'] = kwargs.get('attention_mask')
        self.cache['position_ids'] = kwargs.get('position_ids')
        self.cache['position_embeddings'] = kwargs.get('position_embeddings')
        if 'position_embeddings' in kwargs:
             pe = kwargs['position_embeddings']
             if pe is not None:
                 print(f"DEBUG: Captured PE shape: {pe[0].shape}")
             else:
                 print("DEBUG: PE is None")
        raise ValueError("Stop Forward")

def verify_real_model():
    model_path = r"d:\dataset and model weight forllm\DeepSeek-R1-Distill-Qwen-1.5B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {model_path}...")
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    model.eval()
    
    # Arg simulation
    args = type('Args', (), {
        'w_bits': 8, 'a_bits': 8, 'a_asym': True,
        'k_bits': 8, 'k_asym': True,
        'v_bits': 8, 'v_asym': True,
        'nsamples': 128,       # Restore to 128
        'seed': 0,
        'cali_dataset': 'wikitext2',
        'cali_bsz': 4,         # Batch size 4
        'svd_threshold': 0.95
    })()

    # 1. Load Calibration Data (WikiText-2)
    print(f"Loading calibration data ({args.cali_dataset})...")
    # seqlen=2048 is standard for calibration
    trainloader = data_utils.get_loaders(
        args.cali_dataset, nsamples=args.nsamples,
        seed=args.seed, model=model_path,
        seqlen=2048 
    )
    
    # 2. Layer-wise Processing Loop
    print("Starting Layer-wise HH Optimization...")
    layers = model.model.layers
    
    # Prepare inputs for the first layer (Layer 0)
    # We capture outputs of Embedding + Rotary on the WikiText samples
    
    # Storage for inputs (on device to avoid repeated transfers, or CPU if OOM)
    # Using device for speed as 128*2048*1536*2B is ~800MB
    inps = torch.zeros(
        (args.nsamples, 2048, model.config.hidden_size), 
        dtype=torch.float16, device=device
    )
    cache = {'i': 0}
    
    # Replace Layer 0 with Catcher
    layers[0] = Catcher(layers[0], inps, cache)
    
    print("Capturing layer 0 inputs...")
    with torch.no_grad():
        for batch in trainloader:
            if cache['i'] >= args.nsamples:
                break
            try:
                # batch is (input_ids, target_ids)
                model(batch[0].to(device))
            except ValueError:
                pass
                
    # Restore Layer 0
    layers[0] = layers[0].module
    
    # Extract captured kwargs (assuming static mask/pos_ids for fixed seqlen)
    attention_mask = cache.get('attention_mask')
    position_ids = cache.get('position_ids')
    position_embeddings = cache.get('position_embeddings')
    
    # Prepare batch mask if needed (broadcast to batch size)
    if attention_mask is not None:
        # attention_mask is typically [1, 1, seq, seq]
        # optimization loop uses batch size 'cali_bsz'
        # We need to ensure it matches
        attention_mask_batch = attention_mask.repeat(args.cali_bsz, 1, 1, 1).float()
    else:
        attention_mask_batch = None
    
    if position_embeddings is not None and isinstance(position_embeddings, tuple):
         position_embeddings = tuple(t.float() for t in position_embeddings)

    print(f"Captured Inputs Shape: {inps.shape}")
    print(f"Captured Input Stats: Max={inps.max()}, Min={inps.min()}, NaNs={inps.isnan().sum()}")

    # Initialize fp_inps and fp_outs
    fp_inps = inps
    fp_outs = torch.zeros_like(inps)
    
    # We only process the first 2 layers as requested
    num_layers_to_verify = 2
    
    for i in range(num_layers_to_verify):
        print(f"\nProcessing Layer {i}...")
        layer = layers[i]
        
        # Move layer to device and FLOAT32 for stable optimization
        layer = layer.to(device).float()
        
        # 2.1 Get Target Output (Float16/Original)
        # Run original layer forward on all samples
        # Batching to avoid OOM
        print(f"  Generating targets for layer {i}...")
        with torch.no_grad():
            for j in range(0, args.nsamples, args.cali_bsz):
                batch_inp = fp_inps[j : j + args.cali_bsz].to(device).float()
                
                # Use captured kwargs
                out = layer(
                    batch_inp, 
                    attention_mask=attention_mask_batch if attention_mask_batch is not None else None,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings
                )[0]
                fp_outs[j : j + args.cali_bsz] = out.to(fp_outs.dtype) # Store as FP16
                
        # 2.2 Initialize HH params using Activations
        # Access submodules
        ori_attn = layer.self_attn
        ori_mlp = layer.mlp
        
        # Wrap Sub-modules
        hh_attn = HHQwen2Attention(args, ori_attn, n_reflections_qkv=32, n_reflections_o=32).to(device=device, dtype=torch.float32)
        hh_mlp = HHQwen2MLP(args, ori_mlp, n_reflections_up_gate=32, n_reflections_down=32).to(device=device, dtype=torch.float32)
        
        # Initialize Rotations using a subset of data (e.g. first batch)
        # Use first batch for PCA initialization to be fast and consistent with batching
        init_batch_size = args.cali_bsz
        init_data_batch = fp_inps[0:init_batch_size].to(device).float()
        init_data_flat = init_data_batch.reshape(-1, fp_inps.shape[-1])
        
        # Need intermediate activations for MLP/Attn-O init
        # We simulate forward for init_data
        with torch.no_grad():
            # Layer Norms
            hidden_states = init_data_batch # [BS, Seq, Hidden]
            hidden_states = layer.input_layernorm(hidden_states)
            attn_input = hidden_states.clone() # Input to QKV
            
            # Attn Call simulation for O-proj input?
            pass

        print("  Initializing Rotations...")
        hh_attn.qkv_rotation.initialize_from_activations(init_data_flat, threshold=args.svd_threshold)
        
        # MLP Input
        with torch.no_grad():
             # We need output of attention
             # Use the wrapped attn (identity init for O-proj/others)
             # Use proper batch shape [BS, Seq, Hidden]
             attn_out = hh_attn(attn_input, 
                                attention_mask=attention_mask_batch[0:init_batch_size] if attention_mask_batch is not None else None, 
                                position_ids=position_ids,
                                position_embeddings=position_embeddings)[0]
             
             print(f"DEBUG: Attn Out Stats: Max={attn_out.max()}, Min={attn_out.min()}, NaNs={attn_out.isnan().sum()}")
             
             # Residual
             hidden_states = init_data_batch + attn_out
             hidden_states = layer.post_attention_layernorm(hidden_states)
             mlp_input = hidden_states.clone()
             
             # Flatten MLP input for PCA
             mlp_input_flat = mlp_input.reshape(-1, model.config.hidden_size)

        hh_mlp.up_gate_rotation.initialize_from_activations(mlp_input_flat, threshold=args.svd_threshold)
        
        # Replace modules
        layer.self_attn = hh_attn
        layer.mlp = hh_mlp
        
        # 2.3 Optimize Layer
        
        # Calibrate Quantizers (using dummy input)
        print("  Calibrating Quantizers...")
        dtype = torch.float32
        dummy = torch.zeros(1, 1, model.config.hidden_size, device=device, dtype=dtype)
        # Attn
        for proj in [hh_attn.q_proj, hh_attn.k_proj, hh_attn.v_proj, hh_attn.o_proj]:
            # Adjust dummy size to input dim
            d = torch.zeros(1, 1, proj.in_features, device=device, dtype=dtype)
            proj.calibrate(d) 
        # MLP
        for proj in [hh_mlp.up_proj, hh_mlp.gate_proj, hh_mlp.down_proj]:
            d = torch.zeros(1, 1, proj.in_features, device=device, dtype=dtype)
            proj.calibrate(d)
        
        # Collect params
        learnable_params = []
        learnable_params.extend([hh_attn.qkv_rotation.vectors, hh_attn.qkv_scale])
        learnable_params.extend([hh_attn.o_proj.rotation.vectors, hh_attn.o_proj.scale])
        learnable_params.extend([hh_mlp.up_gate_rotation.vectors, hh_mlp.up_gate_scale])
        learnable_params.extend([hh_mlp.down_proj.rotation.vectors, hh_mlp.down_proj.scale])
        
        for p in learnable_params:
            p.data = p.data.float()
            p.requires_grad = True
            
        optimizer = torch.optim.AdamW(learnable_params, lr=1e-3)
        num_steps = 100 # Total steps
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
        
        print(f"  Optimizing Layer {i}...")
        pbar = tqdm(range(num_steps), leave=False)
        
        # Optimization Loop
        # We sample random batches from our calibration set
        
        for step in pbar:
            optimizer.zero_grad()
            
            # Sample batch index
            idx = torch.randint(0, args.nsamples, (args.cali_bsz,))
            
            # Construct batch
            # Note: fp_inps is (N, Seq, Hidden)
            batch_inp = fp_inps[idx].to(device).float()
            batch_target = fp_outs[idx].to(device).float()
            
            # Forward
            pred_out = layer(
                batch_inp, 
                attention_mask=attention_mask_batch,
                position_ids=position_ids,
                position_embeddings=position_embeddings
            )[0]
            
            loss = torch.nn.functional.mse_loss(pred_out, batch_target)
            
            # Normalize loss (FlatQuant style)
            loss_val = loss.item()
            loss = loss / (loss.detach() + 1e-12)
            
            if torch.isnan(loss):
                print("CRITICAL: NaN Loss!")
                break
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(learnable_params, 1.0)
            optimizer.step()
            scheduler.step()
            
            pbar.set_description(f"MSE: {loss_val:.4e}")
            
        print(f"  Layer {i} Final MSE: {loss_val:.4e}")
        
        # 2.4 Update Input for Next Layer
        print("  Updating inputs for next layer...")
        with torch.no_grad():
            # Reparameterize first
            hh_attn.reparameterize()
            hh_mlp.reparameterize()
            
            # Run forward on ALL samples to get input for layer i+1
            for j in range(0, args.nsamples, args.cali_bsz):
                 # Handle last partial batch if any (though 128 % 4 == 0)
                 end_j = min(j + args.cali_bsz, args.nsamples)
                 batch_inp = fp_inps[j : end_j].to(device).float()
                 
                 out = layer(
                    batch_inp,
                    attention_mask=attention_mask_batch[: (end_j - j)],
                    position_ids=position_ids,
                    position_embeddings=position_embeddings
                 )[0]
                 
                 # Store back to fp_inps (can overwrite safely since we process linearly)
                 fp_inps[j : end_j] = out.to(fp_inps.dtype) # Back to FP16 storage
        
        # Cleanup
        layer = layer.to(dtype=torch.float16)  # Cast back
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  Layer {i} complete.\n")

    print("Optimization Complete.")

if __name__ == "__main__":
    verify_real_model()
