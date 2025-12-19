import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from .hh_linear import HHLinear

def optimize_layer_rotation(
    activations: torch.Tensor,
    layer_weight: torch.Tensor,
    n_reflections: int,
    layer_bias: torch.Tensor = None,
    steps: int = 500,
    lr: float = 1e-3,
    batch_size: int = 1024,
    w_bits: int = 4,
    a_bits: int = 8,
    device: torch.device = torch.device('cuda')
):
    """
    Optimizes a HHLinear layer to match the original float output.
    
    Args:
        activations (torch.Tensor): Input activations X (N, In).
        layer_weight (torch.Tensor): Original weight W (Out, In).
        n_reflections (int): Algorithm determined rank.
        layer_bias (torch.Tensor, optional): Bias.
    
    Returns:
        hh_layer (HHLinear): Optimized, quantized-ready layer.
    """
    # 1. Setup Data
    activations = activations.to(device)
    layer_weight = layer_weight.to(device)
    if layer_bias is not None:
        layer_bias = layer_bias.to(device) # Note: Bias is not rotated in this formulation typically, or depends.
        # If X' = XQD, W' = WQD^-1, then X'W'.T = X W.T. Bias adds to output.
        # So bias doesn't need scaling usually unless we scale output. 
        # FlatQuant treats bias separately or ignores it in optimization loop if output MSE includes bias.
        
    dataset = TensorDataset(activations)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Setup HHLinear
    # Create a temporary linear layer to init HHLinear
    _tmp_linear = nn.Linear(layer_weight.shape[1], layer_weight.shape[0], bias=(layer_bias is not None))
    with torch.no_grad():
        _tmp_linear.weight.copy_(layer_weight)
        if layer_bias is not None:
            _tmp_linear.bias.copy_(layer_bias)
            
    hh_layer = HHLinear(_tmp_linear, n_reflections, w_bits=w_bits, a_bits=a_bits).to(device)
    
    # 3. Optimizer
    # Optimize local params (rotation.vectors, scale)
    optimizer = optim.Adam(hh_layer.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    # 4. Training Loop
    # We want to minimize || Y_float - Y_quant ||^2
    # Y_float = X @ W.T + B
    # Y_quant = HHLinear(X) -> Quantized output
    
    step = 0
    epoch = 0
    
    # Precompute float targets if possible? 
    # Can be large (OutDim * N). If N=2048, Out=8960 -> 72MB (Float32). Fits in GPU easily.
    # If N is huge, compute on fly.
    with torch.no_grad():
        Y_ref = torch.nn.functional.linear(activations, layer_weight, layer_bias)
        
    # Re-wrap Y_ref into dataset for easier batching
    dataset = TensorDataset(activations, Y_ref)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    while step < steps:
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            
            # Forward pass (Quantized Simulation)
            y_pred = hh_layer(x_batch)
            
            loss = loss_fn(y_batch, y_pred)
            loss.backward()
            optimizer.step()
            
            step += 1
            if step >= steps:
                break
                
        # print(f"Step {step} Loss: {loss.item()}")
            
    return hh_layer
