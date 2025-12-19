import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

def determine_rank_from_svd(data: torch.Tensor, threshold: float = 0.95, max_rank: int = 32) -> Tuple[int, torch.Tensor]:
    """
    Perform SVD on activations/weights and determine the number of components
    needed to explain `threshold` (e.g., 95%) of the variance.
    Returns (k, top_k_components).
    """
    if data.dim() == 3:
        # (B, L, D) -> (B*L, D)
        data = data.reshape(-1, data.shape[-1])
    
    N, D = data.shape
    
    # Subsample if too large to avoid OOM even on GPU
    if N > 10000:
        indices = torch.randperm(N)[:10000]
        X_sub = data[indices]
    else:
        X_sub = data

    # Move to GPU if available for faster SVD
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = X_sub.to(device).float()
        
    try:
        # We need V (right singular vectors) as principal components
        # X = U S V^T, so rows of V^T (columns of V) are components
        U, S, Vh = torch.linalg.svd(X, full_matrices=False)
        V = Vh.transpose(-2, -1) # (D, k_max)
    except Exception as e:
        # Fallback for very large matrices or errors
        X_cpu = X.cpu()
        U, S, Vh = torch.pca_lowrank(X_cpu, q=min(D, 256), center=False)
        V = Vh.to(device)

    total_energy = (S**2).sum()
    cumulative_energy = (S**2).cumsum(dim=0)
    ratio = cumulative_energy / total_energy
    
    # Find first index where ratio >= threshold
    mask = ratio >= threshold
    if not mask.any():
        k = len(S)
    else:
        k = mask.nonzero()[0].item() + 1
        
    k = max(1, min(k, max_rank, V.shape[1]))
    
    print(f"  [SVD Analysis] {data.shape} -> k={k} explains {ratio[k-1]:.2%} energy (Target: {threshold:.0%})")
    
    # Principal components are columns of V, we return them as rows (k, D)
    components = V[:, :k].t().contiguous()
    
    return int(k), components

def profile_model_activations(model, dataloader, n_samples=32) -> Dict[str, torch.Tensor]:
    """
    Profile model activations by running a forward pass with hooks.
    Returns a dictionary mapping layer names to their input activations.
    """
    model.eval()
    device = next(model.parameters()).device
    activations = {}
    hooks = []

    def get_hook(name):
        def hook(module, input, output):
            # input is a tuple, we want the first element
            inp = input[0].detach().cpu()
            if name not in activations:
                activations[name] = []
            activations[name].append(inp)
        return hook

    # Register hooks on all Linear layers
    # We target: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Filter to only keep the relevant ones if necessary, but usually all are fine
            hooks.append(module.register_forward_hook(get_hook(name)))

    print(f"Profiling activations on {n_samples} samples...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            if i >= n_samples:
                break
            if isinstance(batch, torch.Tensor):
                input_ids = batch.to(device)
            else:
                input_ids = batch['input_ids'].to(device)
            
            model(input_ids)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Concatenate results
    final_activations = {}
    for name, inps in activations.items():
        # inps is a list of (B, L, D) or (B, D)
        # We want to stack them and flatten to (-1, D)
        cat_inp = torch.cat(inps, dim=0)
        final_activations[name] = cat_inp

    return final_activations
