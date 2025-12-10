import torch
import torch_npu
import time
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# 1. 基础工具与 Profiler
# ==========================================

class Profiler:
    def __init__(self, device, enabled=True):
        self.device = device
        self.enabled = enabled
        self.records = {}
        self.current_name = None
        self.start_time = None

    def start(self, name):
        if not self.enabled: return
        self.sync()
        self.current_name = name
        self.start_time = time.time()

    def end(self):
        if not self.enabled or self.current_name is None: return
        self.sync()
        duration = time.time() - self.start_time
        if self.current_name not in self.records:
            self.records[self.current_name] = []
        self.records[self.current_name].append(duration)
        self.current_name = None

    def sync(self):
        if self.device.type == 'npu':
            torch.npu.synchronize()
        elif self.device.type == 'cuda':
            torch.cuda.synchronize()

    def print_stats(self):
        if not self.enabled: return
        print(f"\n{'='*15} Breakdown Analysis {'='*15}")
        print(f"{'Component':<25} | {'Avg Time (ms)':<15} | {'% of Total':<10}")
        print("-" * 56)
        
        total_time = 0
        avg_times = {}
        for name, times in self.records.items():
            # Skip warmup (first 10%)
            valid_times = times[int(len(times)*0.1):]
            if not valid_times: valid_times = times
            avg = sum(valid_times) / len(valid_times) * 1000
            avg_times[name] = avg
            total_time += avg
            
        for name, avg in avg_times.items():
            ratio = (avg / total_time) * 100
            print(f"{name:<25} | {avg:13.4f}   | {ratio:9.1f}%")
        print("-" * 56)
        print(f"{'Total Inference':<25} | {total_time:13.4f}   | 100.0%")
        print(f"{'='*56}\n")

# ==========================================
# 2. FlatQuant 核心组件模拟
# ==========================================

def kronecker_matmul(x, hadL, hadR):
    """
    FlatQuant 中的核心变换操作
    x: [..., M*N]
    hadL: [M, M]
    hadR: [N, N]
    """
    init_shape = x.shape
    # 1. Reshape
    x = x.reshape(-1, hadL.shape[0], hadR.shape[0])
    # 2. Matmul 1
    x = torch.matmul(x, hadR)
    # 3. Matmul 2
    x = torch.matmul(hadL.T, x)
    # 4. Reshape back
    return x.reshape(init_shape)

class MockTransMatrix(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # 简单的分解逻辑，模拟 get_decompose_dim
        a = int(math.sqrt(hidden_size))
        while hidden_size % a != 0:
            a -= 1
        b = hidden_size // a
        
        self.matrix_left = nn.Parameter(torch.randn(a, a))
        self.matrix_right = nn.Parameter(torch.randn(b, b))
        
    def forward(self, x):
        return kronecker_matmul(x, self.matrix_left, self.matrix_right)

class MockSingleTransMatrix(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.matrix = nn.Parameter(torch.randn(size, size))
        
    def forward(self, x):
        init_shape = x.shape
        x = x.reshape(-1, self.matrix.shape[0])
        return x.matmul(self.matrix).reshape(init_shape)

# 使用优化后的量化器
class ActivationQuantizerOptimized(torch.nn.Module):
    def __init__(self, bits=8, sym=False):
        super().__init__()
        self.bits = bits
        self.sym = sym
        q_max = 2 ** (bits - 1) - 1 if sym else 2 ** bits - 1
        self.register_buffer('q_max', torch.tensor(float(q_max)))
        self.enable = True

    def forward(self, x):
        if not self.enable: return x
        scale, zero = self.get_scale_zero(x)
        if self.sym:
            return self.sym_quant_dequant(x, scale, self.q_max)
        else:
            return self.asym_quant_dequant(x, scale, zero, self.q_max)

    def get_scale_zero(self, x):
        xmax = x.amax(dim=-1, keepdim=True).clamp_min(0)
        xmin = x.amin(dim=-1, keepdim=True).clamp_max(0)
        
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            scale = torch.where(tmp, 1.0, xmax / self.q_max)
            zero = torch.zeros_like(scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            if tmp.any():
                xmin = xmin.masked_fill(tmp, -1.0)
                xmax = xmax.masked_fill(tmp, 1.0)
            scale = (xmax - xmin) / self.q_max
            zero = torch.round(-xmin / scale)
        return scale, zero

    def sym_quant_dequant(self, x, scale, maxq):
        q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
        return scale * q

    def asym_quant_dequant(self, x, scale, zero, maxq):
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        return scale * (q - zero)

# ==========================================
# 2.1 NPU Native Quantizer
# ==========================================

def get_qmin_qmax(bits, sym):
    if sym:
        q_max = 2 ** (bits - 1) - 1
        q_min = -q_max - 1
    else:
        q_max = 2 ** bits - 1
        q_min = 0
    return q_max, q_min

def sym_dequant(q, scale):
    return q.to(scale.dtype) * scale

def asym_dequant(q, scale, zero):
    return (q.to(scale.dtype) - zero) * scale

class NPUActivationQuantizer(torch.nn.Module):
    """
    A NPU-friendly activation quantizer that mirrors the features of
    :class:`ActivationQuantizer` while preferentially dispatching to
    ``torch_npu``'s quantization kernels for better throughput.
    """

    def __init__(self, bits, sym=False, lac=False, groupsize=-1, clip_ratio=None):
        super().__init__()
        self.bits = bits
        self.q_max, self.q_min = get_qmin_qmax(bits, sym)
        self.sym = sym
        self.groupsize = groupsize
        self.lac = lac
        self._clip_ratio = clip_ratio
        self.enable = True

        self._fallback_quantizer = ActivationQuantizerOptimized(bits, sym)

        try:
            import torch_npu
            self._fast_path_ready = hasattr(torch_npu, "npu_dynamic_quant")
        except ImportError:
            self._fast_path_ready = False

    def forward(self, x, scale=None, zero=None):
        if self.bits == 16 or (not self.enable):
            return x

        if not self._fast_path_ready:
            return self._fallback_quantizer(x)

        # Flatten to 2D [N, H] for per-token quantization to ensure correct broadcasting
        original_shape = x.shape
        work_tensor = x.reshape(-1, x.shape[-1])
        
        # NPU ops require FP16 or BF16
        if work_tensor.dtype == torch.float32:
            work_tensor = work_tensor.half()

        try:
            # npu_dynamic_quant returns (quant, scale)
            # Signature: (input, *, smooth_scales=None)
            quant_outputs = torch_npu.npu_dynamic_quant(work_tensor)
        except Exception as e:
            print(f"NPU Quant Error: {e}")
            return self._fallback_quantizer(x)

        if self.sym:
            quantized_tensor = quant_outputs[0]
            used_scale = quant_outputs[1]
            used_zero = None
        else:
            quantized_tensor = quant_outputs[0]
            used_scale = quant_outputs[1]
            # npu_dynamic_quant performs symmetric quantization, so zero is 0
            used_zero = torch.zeros_like(used_scale)

        # Ensure scale and zero are broadcastable (N, 1)
        if used_scale is not None and isinstance(used_scale, torch.Tensor):
            if used_scale.ndim == 1:
                used_scale = used_scale.view(-1, 1)
            # Debug: Check for NaNs in scale
            if torch.isnan(used_scale).any():
                # print("Warning: NaN in NPU scale")
                used_scale = torch.nan_to_num(used_scale, nan=1.0)
                
        if used_zero is not None and isinstance(used_zero, torch.Tensor):
            if used_zero.ndim == 1:
                used_zero = used_zero.view(-1, 1)
            # Debug: Check for NaNs in zero
            if torch.isnan(used_zero).any():
                # print("Warning: NaN in NPU zero")
                used_zero = torch.nan_to_num(used_zero, nan=0.0)

        if self.sym:
            dequantized = sym_dequant(quantized_tensor, used_scale)
        else:
            dequantized = asym_dequant(quantized_tensor, used_scale, used_zero)
            
        # Reshape back and cast to original dtype
        return dequantized.reshape(original_shape).to(x.dtype)


def test_quantizer_correctness(device):
    print("\n>>> Testing Quantizer Correctness...")
    dtype = torch.float16
    # Use a fixed seed for reproducibility
    torch.manual_seed(42)
    x = torch.randn(16, 4096, device=device, dtype=dtype)
    
    # 1. Python Implementation
    print("Running Python Quantizer...")
    quant_py = ActivationQuantizerOptimized(bits=8, sym=False).to(device)
    out_py = quant_py(x)
    
    # 2. NPU Implementation
    print("Running NPU Quantizer...")
    quant_npu = NPUActivationQuantizer(bits=8, sym=False).to(device)
    out_npu = quant_npu(x)
    
    # Compare
    mse = F.mse_loss(out_py, out_npu)
    print(f"Input shape: {x.shape}")
    print(f"Python Output range: [{out_py.min():.4f}, {out_py.max():.4f}]")
    print(f"NPU Output range:    [{out_npu.min():.4f}, {out_npu.max():.4f}]")
    print(f"MSE (Python vs NPU): {mse.item():.6e}")
    
    if torch.isnan(mse):
        print("!!! MSE is NaN. Debugging NPU output...")
        print(f"NPU Output NaNs: {torch.isnan(out_npu).sum()}")
        print(f"NPU Output Infs: {torch.isinf(out_npu).sum()}")

# ==========================================
# 3. 模型定义
# ==========================================

class StandardMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x, profiler=None):
        if profiler: profiler.start("Linear (GateUp)")
        x = self.gate_up_proj(x)
        if profiler: profiler.end()

        if profiler: profiler.start("Activation (SwiGLU)")
        x, gate = x.chunk(2, dim=-1)
        x = F.silu(gate) * x
        if profiler: profiler.end()

        if profiler: profiler.start("Linear (Down)")
        x = self.down_proj(x)
        if profiler: profiler.end()
        return x

class FlatQuantMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, bits=8, sym=True, quantizer_cls=ActivationQuantizerOptimized):
        super().__init__()
        # 1. Transform Layers
        self.up_gate_trans = MockTransMatrix(hidden_size)
        self.down_trans = MockTransMatrix(intermediate_size) 
        
        # 2. Quantizers
        self.up_gate_quant = quantizer_cls(bits=bits, sym=sym)
        self.down_quant = quantizer_cls(bits=bits, sym=sym)
        
        # 3. Linear Layers
        self.gate_up_proj = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x, profiler=None):
        # --- Block 1: Up/Gate ---
        if profiler: profiler.start("Transform (Up)")
        x = self.up_gate_trans(x)
        if profiler: profiler.end()

        if profiler: profiler.start("Quant (Up)")
        x = self.up_gate_quant(x)
        if profiler: profiler.end()

        if profiler: profiler.start("Linear (GateUp)")
        x = self.gate_up_proj(x)
        if profiler: profiler.end()

        if profiler: profiler.start("Activation (SwiGLU)")
        x, gate = x.chunk(2, dim=-1)
        x = F.silu(gate) * x
        if profiler: profiler.end()

        # --- Block 2: Down ---
        if profiler: profiler.start("Transform (Down)")
        x = self.down_trans(x)
        if profiler: profiler.end()

        if profiler: profiler.start("Quant (Down)")
        x = self.down_quant(x)
        if profiler: profiler.end()

        if profiler: profiler.start("Linear (Down)")
        x = self.down_proj(x)
        if profiler: profiler.end()
        
        return x

class StandardAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, profiler=None):
        B, S, H = x.shape
        
        if profiler: profiler.start("Linear (QKV)")
        qkv = self.qkv_proj(x)
        if profiler: profiler.end()

        q, k, v = qkv.split([H, H, H], dim=-1)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        if profiler: profiler.start("Rotary Emb")
        q = q * 0.5 
        k = k * 0.5
        if profiler: profiler.end()

        if profiler: profiler.start("Attention")
        attn_output = F.scaled_dot_product_attention(q, k, v)
        if profiler: profiler.end()
        
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, S, H)

        if profiler: profiler.start("Linear (Output)")
        output = self.o_proj(attn_output)
        if profiler: profiler.end()
        
        return output

class FlatQuantAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim, bits=8, sym=True, quantizer_cls=ActivationQuantizerOptimized):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_heads = num_heads
        
        # Trans
        self.ln_trans = MockTransMatrix(hidden_size)
        self.kcache_trans = MockSingleTransMatrix(head_dim)
        self.vcache_trans = MockSingleTransMatrix(head_dim)
        self.o_trans = MockSingleTransMatrix(num_heads)
        
        # Quant
        self.qkv_quant = quantizer_cls(bits=bits, sym=sym)
        self.k_cache_quant = quantizer_cls(bits=bits, sym=sym)
        self.v_cache_quant = quantizer_cls(bits=bits, sym=sym)
        self.o_quant = quantizer_cls(bits=bits, sym=sym)
        
        # Linear
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, profiler=None):
        B, S, H = x.shape
        
        if profiler: profiler.start("Transform (LN)")
        x = self.ln_trans(x)
        if profiler: profiler.end()
        
        if profiler: profiler.start("Quant (QKV)")
        x = self.qkv_quant(x)
        if profiler: profiler.end()
        
        if profiler: profiler.start("Linear (QKV)")
        qkv = self.qkv_proj(x)
        if profiler: profiler.end()

        q, k, v = qkv.split([H, H, H], dim=-1)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        if profiler: profiler.start("Rotary Emb")
        q = q * 0.5 
        k = k * 0.5
        if profiler: profiler.end()
        
        if profiler: profiler.start("Transform (KV Cache)")
        q = self.kcache_trans(q)
        k = self.kcache_trans(k)
        if profiler: profiler.end()
        
        if profiler: profiler.start("Quant (KV Cache)")
        k = self.k_cache_quant(k)
        v = self.v_cache_quant(v)
        if profiler: profiler.end()

        if profiler: profiler.start("Attention")
        attn_output = F.scaled_dot_product_attention(q, k, v)
        if profiler: profiler.end()
        
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, S, H)
        
        if profiler: profiler.start("Transform (Output)")
        init_shape = attn_output.shape
        attn_output = attn_output.reshape(-1, self.num_heads, self.head_dim)
        # Optimization: Use left matmul to avoid transpose overhead, matching qwen2_flatquant.py
        # We want to transform the num_heads dimension (dim -2).
        # o_trans.matrix is [num_heads, num_heads].
        # We compute M.T @ attn_output.
        # Note: MockSingleTransMatrix stores 'matrix'. We use it directly.
        # attn_output: [Batch*Seq, NumHeads, HeadDim]
        # matrix: [NumHeads, NumHeads]
        # matmul(matrix.T, attn_output) -> [Batch*Seq, NumHeads, HeadDim]
        attn_output = torch.matmul(self.o_trans.matrix.T, attn_output)
        attn_output = attn_output.reshape(init_shape)
        if profiler: profiler.end()
        
        if profiler: profiler.start("Quant (Output)")
        attn_output = self.o_quant(attn_output)
        if profiler: profiler.end()

        if profiler: profiler.start("Linear (Output)")
        output = self.o_proj(attn_output)
        if profiler: profiler.end()
        
        return output

class SmoothQuantMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, bits=8, sym=True, quantizer_cls=ActivationQuantizerOptimized):
        super().__init__()
        
        self.up_gate_quant = quantizer_cls(bits=bits, sym=sym)
        self.down_quant = quantizer_cls(bits=bits, sym=sym)
        
        self.gate_up_proj = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x, profiler=None):
        if profiler: profiler.start("Quant (Up)")
        x = self.up_gate_quant(x)
        if profiler: profiler.end()

        if profiler: profiler.start("Linear (GateUp)")
        x = self.gate_up_proj(x)
        if profiler: profiler.end()

        if profiler: profiler.start("Activation (SwiGLU)")
        x, gate = x.chunk(2, dim=-1)
        x = F.silu(gate) * x
        if profiler: profiler.end()

        if profiler: profiler.start("Quant (Down)")
        x = self.down_quant(x)
        if profiler: profiler.end()

        if profiler: profiler.start("Linear (Down)")
        x = self.down_proj(x)
        if profiler: profiler.end()
        return x

class SmoothQuantAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim, bits=8, sym=True, quantizer_cls=ActivationQuantizerOptimized):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_heads = num_heads
        
        self.qkv_quant = quantizer_cls(bits=bits, sym=sym)
        self.k_cache_quant = quantizer_cls(bits=bits, sym=sym)
        self.v_cache_quant = quantizer_cls(bits=bits, sym=sym)
        self.o_quant = quantizer_cls(bits=bits, sym=sym)
        
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, profiler=None):
        B, S, H = x.shape
        
        if profiler: profiler.start("Quant (QKV)")
        x = self.qkv_quant(x)
        if profiler: profiler.end()
        
        if profiler: profiler.start("Linear (QKV)")
        qkv = self.qkv_proj(x)
        if profiler: profiler.end()

        q, k, v = qkv.split([H, H, H], dim=-1)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        if profiler: profiler.start("Rotary Emb")
        q = q * 0.5 
        k = k * 0.5
        if profiler: profiler.end()
        
        if profiler: profiler.start("Quant (KV Cache)")
        k = self.k_cache_quant(k)
        v = self.v_cache_quant(v)
        if profiler: profiler.end()

        if profiler: profiler.start("Attention")
        attn_output = F.scaled_dot_product_attention(q, k, v)
        if profiler: profiler.end()
        
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(B, S, H)
        
        if profiler: profiler.start("Quant (Output)")
        attn_output = self.o_quant(attn_output)
        if profiler: profiler.end()

        if profiler: profiler.start("Linear (Output)")
        output = self.o_proj(attn_output)
        if profiler: profiler.end()
        
        return output

# ==========================================
# 4. Benchmark 逻辑
# ==========================================

def run_single_benchmark(model, x, label, device, iterations=50):
    print(f"\n>>> Benchmarking {label}...")
    model = model.to(device).to(x.dtype)
    profiler = Profiler(device)
    
    # Warmup
    for _ in range(5): model(x)
    
    # Run
    start = time.time()
    for _ in range(iterations):
        model(x, profiler)
    profiler.sync()
    total_time = (time.time() - start) / iterations * 1000
    
    profiler.print_stats()
    print(f"{label} Total Avg Time: {total_time:.4f} ms")
    return total_time

def check_precision(model_ref, model_test, x, label):
    model_ref.eval()
    model_test.eval()
    with torch.no_grad():
        out_ref = model_ref(x)
        out_test = model_test(x)
        mse = F.mse_loss(out_ref, out_test)
        print(f">>> Precision Check [{label}]: MSE = {mse.item():.6e}")
    return mse.item()

def run_benchmark():
    # Device Setup
    try:
        import torch_npu
        device = torch.device("npu:0")
        print("Running on NPU...")
    except ImportError:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print("Running on CUDA...")
        else:
            device = torch.device("cpu")
            print("Running on CPU...")

    # Config
    B, S, H = 4, 2048, 4096
    Intermediate = 11008 # Llama 2 7B size
    dtype = torch.float16
    iterations = 50
    num_heads = 32
    head_dim = H // num_heads

    print(f"Config: Batch={B}, Seq={S}, Hidden={H}, Intermediate={Intermediate}")
    
    # Run Correctness Test First
    test_quantizer_correctness(device)

    x = torch.randn(B, S, H, device=device, dtype=dtype)
    
    # --- MLP Benchmarks ---
    print(f"\n{'='*20} MLP Benchmarks {'='*20}")
    t_mlp_std = run_single_benchmark(
        StandardMLP(H, Intermediate), x, "Standard MLP", device, iterations
    )
    t_mlp_sym = run_single_benchmark(
        FlatQuantMLP(H, Intermediate, sym=True), x, "FlatQuant MLP (Sym)", device, iterations
    )
    t_mlp_asym = run_single_benchmark(
        FlatQuantMLP(H, Intermediate, sym=False), x, "FlatQuant MLP (Asym)", device, iterations
    )
    t_mlp_npu = run_single_benchmark(
        FlatQuantMLP(H, Intermediate, sym=False, quantizer_cls=NPUActivationQuantizer), x, "FlatQuant MLP (NPU Asym)", device, iterations
    )
    
    # SmoothQuant MLP
    t_sq_mlp = run_single_benchmark(
        SmoothQuantMLP(H, Intermediate, sym=True), x, "SmoothQuant MLP (Sym)", device, iterations
    )
    t_sq_mlp_npu = run_single_benchmark(
        SmoothQuantMLP(H, Intermediate, sym=True, quantizer_cls=NPUActivationQuantizer), x, "SmoothQuant MLP (NPU Sym)", device, iterations
    )
    
    # Precision Check MLP
    check_precision(
        FlatQuantMLP(H, Intermediate, sym=False, quantizer_cls=ActivationQuantizerOptimized).to(device).to(dtype),
        FlatQuantMLP(H, Intermediate, sym=False, quantizer_cls=NPUActivationQuantizer).to(device).to(dtype),
        x, "MLP Asym vs NPU Asym"
    )

    # --- Attention Benchmarks ---
    print(f"\n{'='*20} Attention Benchmarks {'='*20}")
    t_attn_std = run_single_benchmark(
        StandardAttention(H, num_heads, head_dim), x, "Standard Attention", device, iterations
    )
    t_attn_sym = run_single_benchmark(
        FlatQuantAttention(H, num_heads, head_dim, sym=True), x, "FlatQuant Attention (Sym)", device, iterations
    )
    t_attn_asym = run_single_benchmark(
        FlatQuantAttention(H, num_heads, head_dim, sym=False), x, "FlatQuant Attention (Asym)", device, iterations
    )
    t_attn_npu = run_single_benchmark(
        FlatQuantAttention(H, num_heads, head_dim, sym=False, quantizer_cls=NPUActivationQuantizer), x, "FlatQuant Attention (NPU Asym)", device, iterations
    )
    
    # SmoothQuant Attention
    t_sq_attn = run_single_benchmark(
        SmoothQuantAttention(H, num_heads, head_dim, sym=True), x, "SmoothQuant Attention (Sym)", device, iterations
    )
    t_sq_attn_npu = run_single_benchmark(
        SmoothQuantAttention(H, num_heads, head_dim, sym=True, quantizer_cls=NPUActivationQuantizer), x, "SmoothQuant Attention (NPU Sym)", device, iterations
    )

    # --- Final Summary ---
    print(f"\n{'='*30} Final Summary (Prefill Phase: S={S}) {'='*30}")
    print(f"{'Module':<15} | {'Type':<10} | {'Time (ms)':<10} | {'Slowdown':<10}")
    print("-" * 55)
    
    # MLP
    print(f"{'MLP':<15} | {'Standard':<10} | {t_mlp_std:10.4f} | {'1.00x':<10}")
    print(f"{'MLP':<15} | {'FQ Sym':<10} | {t_mlp_sym:10.4f} | {t_mlp_sym/t_mlp_std:10.2f}x")
    print(f"{'MLP':<15} | {'FQ Asym':<10} | {t_mlp_asym:10.4f} | {t_mlp_asym/t_mlp_std:10.2f}x")
    print(f"{'MLP':<15} | {'FQ NPU':<10} | {t_mlp_npu:10.4f} | {t_mlp_npu/t_mlp_std:10.2f}x")
    print(f"{'MLP':<15} | {'SQ Sym':<10} | {t_sq_mlp:10.4f} | {t_sq_mlp/t_mlp_std:10.2f}x")
    print(f"{'MLP':<15} | {'SQ NPU':<10} | {t_sq_mlp_npu:10.4f} | {t_sq_mlp_npu/t_mlp_std:10.2f}x")
    print("-" * 55)
    
    # Attention
    print(f"{'Attention':<15} | {'Standard':<10} | {t_attn_std:10.4f} | {'1.00x':<10}")
    print(f"{'Attention':<15} | {'FQ Sym':<10} | {t_attn_sym:10.4f} | {t_attn_sym/t_attn_std:10.2f}x")
    print(f"{'Attention':<15} | {'FQ Asym':<10} | {t_attn_asym:10.4f} | {t_attn_asym/t_attn_std:10.2f}x")
    print(f"{'Attention':<15} | {'FQ NPU':<10} | {t_attn_npu:10.4f} | {t_attn_npu/t_attn_std:10.2f}x")
    print(f"{'Attention':<15} | {'SQ Sym':<10} | {t_sq_attn:10.4f} | {t_sq_attn/t_attn_std:10.2f}x")
    print(f"{'Attention':<15} | {'SQ NPU':<10} | {t_sq_attn_npu:10.4f} | {t_sq_attn_npu/t_attn_std:10.2f}x")
    print(f"{'='*55}\n")

    # ==========================================
    # 5. Decoding Benchmark (S=1)
    # ==========================================
    print(f"\n{'#'*60}")
    print(f"{'Decoding Phase Benchmark (Batch=1, Seq=1)':^60}")
    print(f"{'#'*60}\n")
    
    B_dec, S_dec = 1, 1
    x_dec = torch.randn(B_dec, S_dec, H, device=device, dtype=dtype)
    
    # --- MLP Decoding ---
    print(f"\n{'='*20} MLP Decoding (S=1) {'='*20}")
    t_mlp_std_dec = run_single_benchmark(
        StandardMLP(H, Intermediate), x_dec, "Standard MLP (Dec)", device, iterations
    )
    t_mlp_asym_dec = run_single_benchmark(
        FlatQuantMLP(H, Intermediate, sym=False), x_dec, "FlatQuant MLP Asym (Dec)", device, iterations
    )
    t_mlp_npu_dec = run_single_benchmark(
        FlatQuantMLP(H, Intermediate, sym=False, quantizer_cls=NPUActivationQuantizer), x_dec, "FlatQuant MLP NPU (Dec)", device, iterations
    )
    t_sq_mlp_dec = run_single_benchmark(
        SmoothQuantMLP(H, Intermediate, sym=True), x_dec, "SmoothQuant MLP Sym (Dec)", device, iterations
    )
    t_sq_mlp_npu_dec = run_single_benchmark(
        SmoothQuantMLP(H, Intermediate, sym=True, quantizer_cls=NPUActivationQuantizer), x_dec, "SmoothQuant MLP NPU (Dec)", device, iterations
    )

    # --- Attention Decoding ---
    print(f"\n{'='*20} Attention Decoding (S=1) {'='*20}")
    t_attn_std_dec = run_single_benchmark(
        StandardAttention(H, num_heads, head_dim), x_dec, "Standard Attention (Dec)", device, iterations
    )
    t_attn_asym_dec = run_single_benchmark(
        FlatQuantAttention(H, num_heads, head_dim, sym=False), x_dec, "FlatQuant Attention Asym (Dec)", device, iterations
    )
    t_attn_npu_dec = run_single_benchmark(
        FlatQuantAttention(H, num_heads, head_dim, sym=False, quantizer_cls=NPUActivationQuantizer), x_dec, "FlatQuant Attention NPU (Dec)", device, iterations
    )
    t_sq_attn_dec = run_single_benchmark(
        SmoothQuantAttention(H, num_heads, head_dim, sym=True), x_dec, "SmoothQuant Attention Sym (Dec)", device, iterations
    )
    t_sq_attn_npu_dec = run_single_benchmark(
        SmoothQuantAttention(H, num_heads, head_dim, sym=True, quantizer_cls=NPUActivationQuantizer), x_dec, "SmoothQuant Attention NPU (Dec)", device, iterations
    )

    # --- Decoding Summary ---
    print(f"\n{'='*30} Final Summary (Decoding Phase: S=1) {'='*30}")
    print(f"{'Module':<15} | {'Type':<10} | {'Time (ms)':<10} | {'Slowdown':<10}")
    print("-" * 55)
    
    # MLP
    print(f"{'MLP':<15} | {'Standard':<10} | {t_mlp_std_dec:10.4f} | {'1.00x':<10}")
    print(f"{'MLP':<15} | {'FQ Asym':<10} | {t_mlp_asym_dec:10.4f} | {t_mlp_asym_dec/t_mlp_std_dec:10.2f}x")
    print(f"{'MLP':<15} | {'FQ NPU':<10} | {t_mlp_npu_dec:10.4f} | {t_mlp_npu_dec/t_mlp_std_dec:10.2f}x")
    print(f"{'MLP':<15} | {'SQ Sym':<10} | {t_sq_mlp_dec:10.4f} | {t_sq_mlp_dec/t_mlp_std_dec:10.2f}x")
    print(f"{'MLP':<15} | {'SQ NPU':<10} | {t_sq_mlp_npu_dec:10.4f} | {t_sq_mlp_npu_dec/t_mlp_std_dec:10.2f}x")
    print("-" * 55)
    
    # Attention
    print(f"{'Attention':<15} | {'Standard':<10} | {t_attn_std_dec:10.4f} | {'1.00x':<10}")
    print(f"{'Attention':<15} | {'FQ Asym':<10} | {t_attn_asym_dec:10.4f} | {t_attn_asym_dec/t_attn_std_dec:10.2f}x")
    print(f"{'Attention':<15} | {'FQ NPU':<10} | {t_attn_npu_dec:10.4f} | {t_attn_npu_dec/t_attn_std_dec:10.2f}x")
    print(f"{'Attention':<15} | {'SQ Sym':<10} | {t_sq_attn_dec:10.4f} | {t_sq_attn_dec/t_attn_std_dec:10.2f}x")
    print(f"{'Attention':<15} | {'SQ NPU':<10} | {t_sq_attn_npu_dec:10.4f} | {t_sq_attn_npu_dec/t_attn_std_dec:10.2f}x")
    print(f"{'='*55}\n")

if __name__ == "__main__":
    torch.manual_seed(42)
    with torch.no_grad():
        run_benchmark()