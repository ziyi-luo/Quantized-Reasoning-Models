import torch
import torch_npu
import math
import time

class SmoothQuantLinearNPU(torch.nn.Module):
    def __init__(self, in_features, out_features, device="npu"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        
        # Initialize weights (FP16 for simulation, but in real scenario these would be quantized)
        self.weight = torch.randn(out_features, in_features, dtype=torch.float16, device=device) * 0.02
        self.bias = None # Optional
        
    def forward(self, x, smooth_scales=None):
        """
        x: Input tensor [Batch, Seq, In] (FP16)
        smooth_scales: Smoothing scales [In] (FP16). 
                       If provided, we assume x needs to be divided by these scales (migration factor),
                       and weights have been pre-multiplied by these scales.
                       
                       Standard SmoothQuant: X_smooth = X / s, W_smooth = W * s
                       
                       torch_npu.npu_dynamic_quant with smooth_scales parameter multiplies the input by smooth_scales.
                       So we pass 1.0 / smooth_scales to achieve division.
        """
        
        # 1. Weight Quantization (Simulated "Offline" Step)
        # In a real inference engine, this happens once or weights are stored in int8.
        
        if smooth_scales is not None:
            # Apply smoothing to weights: W_smooth = W * s
            # smooth_scales shape: (In_features)
            w_smooth = self.weight * smooth_scales.unsqueeze(0)
        else:
            w_smooth = self.weight

        # Quantize weights to Int8 (Per-Channel)
        # Scale shape: (Out_features, 1)
        w_max = w_smooth.abs().max(dim=1, keepdim=True)[0]
        w_scale = w_max / 127.0
        w_scale[w_scale == 0] = 1.0 # Avoid div by zero
        w_int8 = torch.round(w_smooth / w_scale).to(torch.int8)
        
        # w_scale needs to be 1D (float32) for npu_quant_matmul
        w_scale_1d = w_scale.squeeze(1).to(torch.float32)

        # 2. Activation Quantization (Online Step)
        # We use npu_dynamic_quant.
        # We pass 1/s to smooth_scales to perform X_smooth = X / s.
        
        if smooth_scales is not None:
            inv_smooth_scales = 1.0 / smooth_scales
            x_int8, x_scale = torch_npu.npu_dynamic_quant(x, smooth_scales=inv_smooth_scales)
        else:
            x_int8, x_scale = torch_npu.npu_dynamic_quant(x)

        # 3. Matrix Multiplication
        # npu_quant_matmul expects flattened input for per-token scaling.
        
        # Transpose weight for matmul (N, K) -> (K, N)
        w_int8_t = w_int8.t() 
        
        # Flatten x_int8 and x_scale for npu_quant_matmul
        # x_int8: (B, S, In) -> (B*S, In)
        # x_scale: (B, S) -> (B*S)
        input_shape = x_int8.shape
        x_int8_flat = x_int8.reshape(-1, input_shape[-1])
        x_scale_flat = x_scale.reshape(-1)

        # Run NPU Quant Matmul
        # out = (x_int8 * x_scale) @ (w_int8 * w_scale).T
        out_flat = torch_npu.npu_quant_matmul(
            x_int8_flat, 
            w_int8_t, 
            scale=w_scale_1d, 
            pertoken_scale=x_scale_flat,
            output_dtype=torch.float16
        )
        
        # Reshape output back to (B, S, Out)
        out = out_flat.reshape(input_shape[:-1] + (self.out_features,))
        
        return out

def test_smoothquant_accuracy():
    torch.npu.set_device("npu:0")
    device = "npu:0"
    
    # Configuration
    B, S, K, N = 2, 128, 1024, 1024
    dtype = torch.float16
    
    print(f"Testing SmoothQuant W8A8 on {device}")
    print(f"Shape: Batch={B}, Seq={S}, In={K}, Out={N}")
    
    # Create Model
    model = SmoothQuantLinearNPU(K, N, device=device)
    
    # Create Inputs
    x = torch.randn(B, S, K, dtype=dtype, device=device)
    
    # Create Random Smooth Scales
    # Scales are typically around 1.0, but can vary.
    smooth_scales = torch.rand(K, dtype=dtype, device=device) + 0.5
    
    # 1. Reference Output (FP16)
    # SmoothQuant Logic: Y = (X / s) @ (W * s).T
    # This is mathematically equivalent to X @ W.T (ignoring quantization errors)
    # So the reference is just the standard FP16 matmul.
    ref_out = torch.matmul(x, model.weight.t())
    
    # 2. NPU SmoothQuant Output
    try:
        npu_out = model(x, smooth_scales=smooth_scales)
    except Exception as e:
        print(f"NPU Inference Failed: {e}")
        return

    # 3. Compare
    # We expect some loss due to W8A8 quantization.
    # But it shouldn't be garbage.
    
    # Calculate Cosine Similarity
    cos_sim = torch.nn.functional.cosine_similarity(ref_out.flatten(), npu_out.flatten(), dim=0)
    
    # Calculate MSE
    mse = torch.nn.functional.mse_loss(ref_out, npu_out)
    
    # Calculate Relative Error
    rel_error = (ref_out - npu_out).abs().mean() / ref_out.abs().mean()
    
    print("-" * 40)
    print(f"Reference Mean: {ref_out.mean().item():.4f}")
    print(f"NPU Output Mean: {npu_out.mean().item():.4f}")
    print(f"Cosine Similarity: {cos_sim.item():.4f}")
    print(f"MSE: {mse.item():.6f}")
    print(f"Relative Error: {rel_error.item():.4f}")
    print("-" * 40)
    
    if cos_sim > 0.99:
        print("SUCCESS: High similarity verified.")
    else:
        print("WARNING: Similarity is lower than expected for W8A8.")

def test_performance():
    torch.npu.set_device("npu:0")
    device = "npu:0"
    dtype = torch.float16
    
    # Define test shapes
    # Decoding phase usually has Sequence Length = 1
    shapes = [
        (1, 1, 4096, 4096),   # Single batch decoding
        (16, 1, 4096, 4096),  # Batch 16 decoding
        (32, 1, 4096, 4096),  # Batch 32 decoding
        (1, 128, 4096, 4096)  # Small Prefill
    ]
    
    print("\n" + "="*80)
    print(f"{'Scenario':<30} | {'FP16 Latency (ms)':<20} | {'W8A8 Latency (ms)':<20} | {'Speedup':<10}")
    print("-" * 80)

    for B, S, K, N in shapes:
        model = SmoothQuantLinearNPU(K, N, device=device)
        x = torch.randn(B, S, K, dtype=dtype, device=device)
        smooth_scales = torch.rand(K, dtype=dtype, device=device) + 0.5
        
        # Warmup
        for _ in range(20):
            _ = torch.matmul(x, model.weight.t())
            _ = model(x, smooth_scales=smooth_scales)
        torch.npu.synchronize()
        
        # Test FP16
        start = time.time()
        iterations = 200
        for _ in range(iterations):
            _ = torch.matmul(x, model.weight.t())
        torch.npu.synchronize()
        fp16_avg = (time.time() - start) / iterations * 1000
        
        # Test W8A8
        start = time.time()
        for _ in range(iterations):
            _ = model(x, smooth_scales=smooth_scales)
        torch.npu.synchronize()
        w8a8_avg = (time.time() - start) / iterations * 1000
        
        scenario_str = f"B={B}, S={S}, K={K}, N={N}"
        print(f"{scenario_str:<30} | {fp16_avg:<20.3f} | {w8a8_avg:<20.3f} | {fp16_avg/w8a8_avg:<10.2f}x")
    print("="*80 + "\n")

if __name__ == "__main__":
    test_smoothquant_accuracy()
    test_performance()
