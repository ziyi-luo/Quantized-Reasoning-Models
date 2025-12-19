import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class HouseholderRotation(nn.Module):
    """
    Learnable Householder rotation specialized for quantization optimization.
    Applies Q = H_k ... H_1.
    """
    def __init__(self, n_features: int, n_reflections: int = 1, eps: float = 1e-6):
        super().__init__()
        self.n_features = n_features
        self.n_reflections = n_reflections
        self.eps = eps
        # Initialize with small random vectors
        self.vectors = nn.Parameter(0.01 * torch.randn(n_reflections, n_features))
        
        # State for reparameterization
        self.is_reparameterized = False
        self.register_buffer('W', torch.zeros(n_features, n_reflections))
        self.register_buffer('Y_scaled', torch.zeros(n_features, n_reflections))
        self.register_buffer('scale_buffer', torch.ones(n_features))
        self.register_buffer('is_initialized', torch.tensor(False))

    def from_vectors(self, vectors: torch.Tensor):
        """Initialize from a set of Householder vectors."""
        with torch.no_grad():
            self.vectors.copy_(vectors)
            self.is_initialized.fill_(True)

    def initialize_from_activations(self, data: torch.Tensor, k: int = None, threshold: float = 0.99):
        """
        Compute Principal Components (PCA) of activation data and initialize Householder vectors.
        
        Mathematical Principle:
        Given activation matrix $X \in \mathbb{R}^{N \times D}$ (where $N$ is batch size, $D$ is feature dim),
        we aim to find an orthogonal matrix $Q$ (parameterized by Householder reflectors) that 
        rotates the input space such that the variance is concentrated in the first $k$ dimensions.

        1. **Principal Component Analysis (PCA)**:
           We compute the Singular Value Decomposition (SVD) of the centered or uncentered activations:
           $$ X = U \Sigma V^T $$
           The rows of $V^T$ (columns of $V$) are the principal components $v_1, v_2, \dots, v_D$.
           We select the top $k$ components $V_k = [v_1, \dots, v_k] \in \mathbb{R}^{D \times k}$.

        2. **Householder QR Decomposition**:
           We construct $k$ Householder reflectors $H_1, \dots, H_k$ such that their product $Q^T = H_k \dots H_1$ 
           performs a QR decomposition on $V_k$:
           $$ Q^T V_k = R $$
           where $R$ is upper triangular. This implies that the first $k$ columns of $Q$ (the new basis) 
           span the same subspace as the top $k$ principal components.
           
           Each Householder reflector $H_i = I - 2 \frac{u_i u_i^T}{||u_i||^2}$ is defined by a vector $u_i$ computed to 
           zero out elements below the diagonal in the $i$-th column of the current matrix.

        Args:
            data: Activation tensor (N, D) or (B, L, D).
            k: Number of components to keep. If None, determined by threshold.
            threshold: Variance coverage threshold (0.0-1.0) to determine k if k is None.
        """
        if k is None and threshold is None:
            raise ValueError("Either k or threshold must be provided.")
            
        # Flatten data if needed
        if data.dim() == 3:
            data = data.reshape(-1, data.shape[-1])
            
        N, D = data.shape
        device = data.device
        
        if data.isnan().any() or data.isinf().any():
            print(f"WARNING: Activations contain NaNs or Infs! Max: {data.max()}, Min: {data.min()}")

        # Determine k and components via SVD/PCA
        # Use torch.pca_lowrank for efficiency on large matrices or full SVD for precision
        if N > 10000:
            indices = torch.randperm(N)[:10000]
            X_sub = data[indices]
        else:
            X_sub = data
            
        X_sub = X_sub.float()
        
        try:
            U, S, Vh = torch.linalg.svd(X_sub, full_matrices=False)
            V = Vh.mH 
            components = Vh 
        except Exception as e:
            print(f"SVD Failed: {e}. Fallback to PCA.")
            # Fallback
            X_cpu = X_sub.cpu()
            U, S, V = torch.pca_lowrank(X_cpu, q=min(D, 256), center=False)
            components = V.t().to(device) # (q, D)
            S = S.to(device)

        if components.isnan().any():
             print("WARNING: PCA Components contain NaNs!")

        # Determine k if needed
        if k is None:
            total_energy = (S**2).sum()
            cumulative_energy = (S**2).cumsum(dim=0)
            ratio = cumulative_energy / total_energy
            mask = ratio >= threshold
            if not mask.any():
                k = len(S)
            else:
                k = mask.nonzero()[0].item() + 1
            k = max(1, min(k, self.n_reflections))
            print(f"Initialized HH with k={k} (Threshold {threshold:.1%})")
        
        # Select top k components
        top_k_pc = components[:k, :] # (k, D)
        
        # Initialize
        self.initialize_from_pc(top_k_pc)

    def initialize_from_pc(self, pc: torch.Tensor):
        """
        Initialize Householder vectors from Principal Components using QR-like decomposition.
        pc: (n_pc, n_features) - Top principal components.
        """
        if pc.isnan().any():
            print("WARNING: Input PC to initialize_from_pc contain NaNs!")
            
        k = pc.shape[0]
        device = pc.device
        dtype = pc.dtype
        n = self.n_features
        
        X = pc.t().clone()
        vectors = torch.zeros(self.n_reflections, n, device=device, dtype=dtype)
        
        for i in range(min(k, self.n_reflections)):
            x = X[i:, i]
            norm_x = torch.norm(x)
            if norm_x < 1e-9:
                v = torch.zeros(n - i, device=device, dtype=dtype)
                v[0] = 1.0
            else:
                v = x.clone()
                v[0] = v[0] + torch.sign(v[0] + 1e-12) * norm_x
                v = v / (torch.norm(v) + 1e-12)
            
            vectors[i, i:] = v
            
            if i < k - 1:
                proj = v.unsqueeze(0) @ X[i:, i+1:]
                X[i:, i+1:] -= 2.0 * v.unsqueeze(1) @ proj
        
        if vectors.isnan().any():
            print("WARNING: Computed Householder vectors contain NaNs!")
            
        self.from_vectors(vectors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., n_features)
        if self.is_reparameterized:
            # Optimized WY application: x_trans = (x * scale) - (x @ W) @ Y_scaled.T
            # Force float32 to avoid overflow in intermediate matmuls
            orig_dtype = x.dtype
            x_f = x.float()
            W_f = self.W.float()
            Y_scaled_f = self.Y_scaled.float()
            scale_buffer_f = self.scale_buffer.float()
            
            x_scaled = x_f * scale_buffer_f
            xw = torch.matmul(x_f, W_f)
            out = x_scaled - torch.matmul(xw, Y_scaled_f.t())
            return out.to(orig_dtype)
        
        return self.apply_reflections(x)

    def apply_reflections(self, x: torch.Tensor) -> torch.Tensor:
        """Original Householder reflections loop, always uses parameters."""
        # Force float32 for stability
        orig_dtype = x.dtype
        out = x.float()
        vectors = self.vectors.float()
        
        for i in range(self.n_reflections):
            v = vectors[i]
            v_norm_sq = torch.dot(v, v) + self.eps
            proj = (out * v).sum(dim=-1, keepdim=True)
            scale = 2.0 * proj / v_norm_sq
            out = out - scale * v
            
        return out.to(orig_dtype)

    def reparameterize(self, scale: torch.Tensor):
        if self.is_reparameterized:
            return
        """
        Fold current reflections and external scale into Compact WY buffers.
        Q = I - W Y^T
        Effective transform at inference: X_trans = X @ Q @ D = X @ D - X @ W @ (D Y)^T
        """
        with torch.no_grad():
            W_mat, Y_mat = self.export_wy()
            
            if W_mat.isnan().any() or W_mat.isinf().any():
                print("WARNING: W_mat has NaNs/Infs in reparameterize!")
            else:
                print(f"W_mat stats: Max {W_mat.max().item():.4f}, Min {W_mat.min().item():.4f}")
                
            if Y_mat.isnan().any() or Y_mat.isinf().any():
                print("WARNING: Y_mat has NaNs/Infs in reparameterize!")
            else:
                print(f"Y_mat stats: Max {Y_mat.max().item():.4f}, Min {Y_mat.min().item():.4f}")
                
            # Y_scaled = D @ Y, where D = diag(scale)
            # scale is (n_features,)
            self.Y_scaled.copy_(Y_mat * scale.unsqueeze(1))
            self.W.copy_(W_mat)
            self.scale_buffer.copy_(scale)
            self.is_reparameterized = True

    def export_wy(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Export Compact WY representation: Q = I - W Y^T
        """
        # Force float32 for geometric stability
        vectors = self.vectors.float()
        k, n = vectors.shape
        device = vectors.device
        
        Y = vectors.t().contiguous() # (n, k)
        W = torch.zeros_like(Y)
        
        taus = 2.0 / (vectors.pow(2).sum(dim=1) + self.eps) # (k,)
        
        for i in range(k):
            v = Y[:, i]
            tau = taus[i]
            
            # w = tau * v - tau * W_prev @ (Y_prev^T @ v)
            # w_i = tau (v_i - Sum_{j<i} w_j (y_j^T v_i))
            
            term1 = tau * v
            if i > 0:
                # previous columns
                Y_prev = Y[:, :i] # (n, i)
                W_prev = W[:, :i] # (n, i)
                proj = (Y_prev.t() @ v) # (i,)
                correction = W_prev @ proj
                term1 = term1 - tau * correction
            
            W[:, i] = term1
            
        return W, Y
