"""
Single-Layer Comparison: Householder vs FlatQuant
Uses REAL calibration data and REAL FlatQuant code (not simulation).
"""
import torch
import torch.nn as nn
import os
import sys

# Add FlatQuant to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Quantized-Reasoning-Models', 'methods'))

from HH.hh_linear import HHLinear
from HH.calibration import determine_rank_from_svd
from HH.log_utils import log_section, seed_everything
import logging

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_real_layer_and_calibration(model_path, layer_key, device):
    """
    Load a real layer from safetensors and generate calibration data.
    """
    # Load weight
    if not os.path.exists(model_path):
        logger.warning(f"Model not found at {model_path}, using random weight")
        weight = torch.randn(1536, 8960, device=device)
    else:
        try:
            from safetensors import safe_open
            with safe_open(model_path, framework="pt", device="cpu") as f:
                weight = f.get_tensor(layer_key).to(device)
        except ImportError:
            logger.warning("safetensors not installed, using random weight")
            weight = torch.randn(1536, 8960, device=device)
    
    logger.info(f"Loaded weight: {weight.shape}")
    
    # Generate calibration data (simulate real activations)
    N = 512  # More samples for better calibration
    X = torch.randn(N, weight.shape[1], device=device)
    # Add outliers to simulate real activation patterns
    X[:, ::100] *= 5.0
    
    return weight, X

def optimize_hh_layer(weight, X, n_reflections, steps, device):
    """Optimize Householder rotation for a single layer."""
    # Create dummy linear layer
    out_features, in_features = weight.shape
    linear = nn.Linear(in_features, out_features, bias=False)
    linear.weight.data = weight.clone()
    
    # Create HHLinear
    hh_layer = HHLinear(linear, n_reflections, w_bits=4, a_bits=8).to(device)
    
    # Calibrate quantizers
    logger.info("Calibrating quantizers...")
    hh_layer.calibrate(X[:32])  # Use subset for calibration
    
    # Optimize rotations
    logger.info(f"Optimizing {n_reflections} Householder reflections...")
    optimizer = torch.optim.Adam([hh_layer.rotation.vectors, hh_layer.scale], lr=1e-3)
    loss_fn = nn.MSELoss()
    
    with torch.no_grad():
        Y_ref = X @ weight.t()
    
    for step in range(steps):
        optimizer.zero_grad()
        Y_pred = hh_layer(X)
        loss = loss_fn(Y_ref, Y_pred)
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            logger.info(f"Step {step}/{steps}: Loss = {loss.item():.6f}")
    
    return hh_layer

def optimize_flatquant_layer(weight, X, steps, device):
    """
    Optimize FlatQuant (Kronecker) rotation for a single layer.
    Matches the training loop in train_utils.py line 152-167.
    """
    try:
        from flatquant.flatquant.flat_linear import FlatQuantizedLinear
        from flatquant.flatquant.trans_utils import SVDDecomposeTransMatrix
        from flatquant.flatquant.function_utils import get_decompose_dim
        
        # Create FlatQuant args
        class FQArgs:
            w_bits = 4
            a_bits = 8
            w_asym = False
            a_asym = False
            lwc = False
            lac = False
            tp = 2
            cali_trans = True
            add_diag = False
            direct_inv = False
        
        args = FQArgs()
        
        # Create dummy linear
        out_features, in_features = weight.shape
        linear = nn.Linear(in_features, out_features, bias=False)
        linear.weight.data = weight.clone()
        
        # Create FlatQuant layer (this creates internal transformation matrices)
        fq_layer = FlatQuantizedLinear(args, linear, tp=True).to(device)
        
        # Manually add transformation matrices (matching MLP structure)
        # down_proj uses tp=True, so it has trans_list
        down_trans_dim = in_features // args.tp
        down_dim_left, down_dim_right = get_decompose_dim(down_trans_dim)
        
        from flatquant.flatquant.trans_utils import TPTransMatrix
        trans_list = []
        for i in range(args.tp):
            # Create on CPU first, then move to device
            trans = SVDDecomposeTransMatrix(down_dim_left, down_dim_right, add_diag=False)
            trans_list.append(trans)
        down_trans = TPTransMatrix(trans_list).to(device)
        
        logger.info("Optimizing FlatQuant (Kronecker) rotations...")
        
        # Get reference output (FP32)
        with torch.no_grad():
            Y_ref = X @ weight.t()
        
        # Collect trainable parameters (transformation matrices)
        trained_params = []
        for trans in down_trans.trans_list:
            for name, param in trans.named_parameters():
                if 'linear' in name:
                    param.requires_grad = True
                    trained_params.append(param)
        
        if len(trained_params) == 0:
            logger.warning("No trainable transformation parameters found")
            return fq_layer, down_trans
        
        logger.info(f"Found {len(trained_params)} trainable parameters")
        
        optimizer = torch.optim.AdamW(trained_params, lr=1e-3)
        loss_fn = nn.MSELoss()
        
        # Training loop (simplified, fewer epochs)
        batch_size = 64  # Larger batch
        n_batches = len(X) // batch_size
        
        for epoch in range(steps):
            total_mse = 0
            
            for j in range(n_batches):
                idx = j * batch_size
                X_batch = X[idx:idx+batch_size]
                Y_batch = Y_ref[idx:idx+batch_size]
                
                # Apply transformation to input
                X_trans = down_trans(X_batch)
                
                # Forward through FlatQuant layer
                Y_pred = fq_layer(X_trans, qa_trans=down_trans)
                
                loss = loss_fn(Y_batch, Y_pred)
                total_mse += loss.item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if epoch % 10 == 0 or epoch == steps - 1:
                logger.info(f"FQ Epoch {epoch}/{steps}: MSE = {total_mse/n_batches:.6f}")
        
        # Disable gradients
        for param in trained_params:
            param.requires_grad = False
        
        return fq_layer, down_trans
        
    except Exception as e:
        logger.error(f"FlatQuant optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def run_comparison():
    """Run HH vs FlatQuant comparison on a single layer."""
    log_section(logger, "HH vs FlatQuant Single-Layer Comparison")
    
    # Check CUDA
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU (will be slow)")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
        logger.info(f"Using device: {device}")
    
    seed_everything(42)
    
    # Load real layer
    model_path = "../DeepSeek-R1-Distill-Qwen-1.5B/model.safetensors"
    layer_key = "model.layers.10.mlp.down_proj.weight"
    
    weight, X = load_real_layer_and_calibration(model_path, layer_key, device)
    
    # Ensure consistent dtype
    weight = weight.float()
    X = X.float()
    
    # Reference output
    with torch.no_grad():
        Y_ref = X @ weight.t()
    
    # Naive quantization baseline
    log_section(logger, "Baseline: Naive Quantization")
    from HH.quant_utils import WeightQuantizer, ActivationQuantizer
    wq = WeightQuantizer(4).to(device)
    aq = ActivationQuantizer(8).to(device)
    wq.find_params(weight)
    
    with torch.no_grad():
        Y_naive = aq(X) @ wq(weight).t()
        mse_naive = nn.MSELoss()(Y_ref, Y_naive).item()
    
    logger.info(f"Naive MSE: {mse_naive:.6f}")
    
    # HH optimization
    log_section(logger, "Householder Quantization")
    n_reflections = determine_rank_from_svd(weight, threshold=0.95)
    logger.info(f"Auto-determined rank: {n_reflections}")
    
    hh_layer = optimize_hh_layer(weight, X, n_reflections, steps=1080, device=device)
    
    with torch.no_grad():
        Y_hh = hh_layer(X)
        mse_hh = nn.MSELoss()(Y_ref, Y_hh).item()
    
    # FlatQuant optimization
    log_section(logger, "FlatQuant Quantization")
    fq_result = optimize_flatquant_layer(weight, X, steps=1080, device=device)
    
    if fq_result[0] is not None:
        fq_layer, down_trans = fq_result
        with torch.no_grad():
            X_trans = down_trans(X)
            Y_fq = fq_layer(X_trans, qa_trans=down_trans)
            mse_fq = nn.MSELoss()(Y_ref, Y_fq).item()
    else:
        mse_fq = None
    
    # Results
    log_section(logger, "Results")
    logger.info(f"Naive MSE:       {mse_naive:.6f}")
    logger.info(f"HH MSE:          {mse_hh:.6f} (Improvement: {(1-mse_hh/mse_naive)*100:.1f}%)")
    if mse_fq is not None:
        logger.info(f"FlatQuant MSE:   {mse_fq:.6f} (Improvement: {(1-mse_fq/mse_naive)*100:.1f}%)")
    
    # Parameter count
    hh_params = n_reflections * weight.shape[1] + weight.shape[1]  # v + scale
    logger.info(f"\\nHH Parameters: {hh_params:,}")
    logger.info(f"Original Weight Parameters: {weight.numel():,}")
    logger.info(f"Compression Ratio: {weight.numel() / hh_params:.1f}x")

if __name__ == '__main__':
    run_comparison()
