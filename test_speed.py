import torch
import time
import copy

# 模拟依赖函数
def round_ste(x: torch.Tensor):
    return (x.round() - x).detach() + x

def get_qmin_qmax(bits, sym):
    if sym:
        q_max = 2 ** (bits - 1) - 1
        q_min = -q_max -1
    else:
        q_max, q_min = 2 ** bits - 1, 0
    return q_max, q_min

def sym_quant(x, scale, maxq):
    scale = scale.to(x.device)
    q = torch.clamp(round_ste(x / scale), -(maxq + 1), maxq)
    return q, scale

def sym_dequant(q, scale):
    return scale * q

def sym_quant_dequant(x, scale, maxq):
    return sym_dequant(*sym_quant(x, scale, maxq))

def asym_quant(x, scale, zero, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(round_ste(x / scale) + zero, 0, maxq)
    return q, scale, zero

def asym_dequant(q, scale, zero):
    return scale * (q - zero)

def asym_quant_dequant(x, scale, zero, maxq):
    return asym_dequant(*asym_quant(x, scale, zero, maxq))

# --- 修改前的原始类 (模拟) ---
class ActivationQuantizerOriginal(torch.nn.Module):
    '''
        A class for quantizing the activations. We only support (both sym. and asym.) per-token quantization
        for the activations.
    '''
    def __init__(self, bits, sym=False, lac=False, groupsize=-1, clip_ratio=None, ):
        super(ActivationQuantizerOriginal, self).__init__()
        self.bits = bits
        self.q_max, self.q_min = get_qmin_qmax(bits, sym)
        self.sym = sym
        self.groupsize = groupsize
        self.lac = lac
        self._clip_ratio = clip_ratio
        if self.lac:
            init_value = 4.
            self.sigmoid = torch.nn.Sigmoid()
            self.clip_factor_a_max = torch.nn.Parameter(torch.ones((1, ))*init_value, requires_grad=True)
            self.clip_factor_a_min = torch.nn.Parameter(torch.ones((1, ))*init_value, requires_grad=True)
        
        self.enable = True

    def forward(self, x, scale=None, zero=None):
        if self.bits == 16 or (not self.enable):
            return x
        if self.groupsize != -1:
            init_shape = x.shape
            x = x.reshape(-1, self.groupsize)
        fq_x = self.fake_quant(x, scale, zero)
        if self.groupsize != -1:
            fq_x = fq_x.reshape(*init_shape)
        return fq_x

    def fake_quant(self, x, scale=None, zero=None):
        x_dtype = x.dtype
        if scale is None or zero is None:
            scale, zero = self.get_scale_zero(x)
        if self.sym:
            # return sym_quant_dequant(x, scale, self.q_max.to(x)).to(x_dtype)
            return sym_quant_dequant(x, scale, self.q_max).to(x_dtype)
        else:
            # 这里和相关位置需要固定数据，避免触发动态计算报错
            # return asym_quant_dequant(x, scale, zero, self.q_max.to(x)).to(x_dtype)  # TODO
            return asym_quant_dequant(x, scale, zero, self.q_max).to(x_dtype)

    def get_scale_zero(self, x):
        # q_max = self.q_max.to(x)
        q_max = self.q_max
        init_shape = x.shape
        reshaped_x = x.reshape((-1, x.shape[-1]))
        xmax, xmin = reshaped_x.amax(1, keepdim=True), reshaped_x.amin(1, keepdim=True)
        tmp = torch.zeros_like(xmax)
        xmax, xmin = torch.maximum(xmax, tmp), torch.minimum(xmin, tmp)
        # # if self.groupsize > 0:
        # #     assert x.shape[-1] % self.groupsize == 0
        # #     x = x.reshape((-1, self.groupsize))
        # #     # TODO: add padding
        if self.lac:
            xmax = xmax * self.sigmoid(self.clip_factor_a_max)
            xmin = xmin * self.sigmoid(self.clip_factor_a_min)
        elif self._clip_ratio is not None:
            xmax = xmax * self._clip_ratio
            xmin = xmin * self._clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            scale = (xmax / q_max)
            scale[tmp] = 1
            scale = scale.repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
            zero = torch.zeros_like(scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            # 这里会触发nonzero对应的底层算子aclnnNonzeroV2报错
            # 在新版本的npu代码推理不会出现此问题
            xmin[tmp] = -1
            xmax[tmp] = +1
            # xmin = torch.where(tmp, torch.full_like(xmin, -1), xmin)
            # xmax = torch.where(tmp, torch.full_like(xmax, +1), xmax)
            scale = (xmax - xmin) / q_max
            zero = torch.round(-xmin / scale)

            # expand 回原始形状
            scale = scale.repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
            zero = zero.repeat(1, reshaped_x.shape[-1]).reshape(init_shape)

        return scale, zero

# --- 修改后的优化类 ---
class ActivationQuantizerOptimized(torch.nn.Module):
    def __init__(self, bits=4, sym=False, lac=False, groupsize=-1, clip_ratio=None):
        super().__init__()
        self.bits = bits
        self.sym = sym
        self.groupsize = groupsize
        self.lac = lac
        self._clip_ratio = clip_ratio
        self.q_max_val, _ = get_qmin_qmax(bits, sym)
        # 优化: 注册为 buffer，避免每次 forward 都要 to(device)
        self.register_buffer('q_max', torch.tensor(float(self.q_max_val)))
        
        if self.lac:
            init_value = 4.
            self.sigmoid = torch.nn.Sigmoid()
            self.clip_factor_a_max = torch.nn.Parameter(torch.ones((1, ))*init_value, requires_grad=True)
            self.clip_factor_a_min = torch.nn.Parameter(torch.ones((1, ))*init_value, requires_grad=True)
        
        self.enable = True

    def forward(self, x, scale=None, zero=None):
        if self.bits == 16 or (not self.enable):
            return x
        
        # 记录原始形状
        init_shape = x.shape
        
        if self.groupsize != -1:
            # 如果有 groupsize，必须 reshape
            x = x.reshape(-1, self.groupsize)
            
        fq_x = self.fake_quant(x, scale, zero)
            
        if self.groupsize != -1:
            fq_x = fq_x.reshape(*init_shape)
            
        return fq_x

    def fake_quant(self, x, scale=None, zero=None):
        x_dtype = x.dtype
        if scale is None or zero is None:
            scale, zero = self.get_scale_zero(x)
        
        # 优化: self.q_max 已经在 device 上，不需要 .to(x.device)
        if self.sym:
            return sym_quant_dequant(x, scale, self.q_max).to(x_dtype)
        else:
            return asym_quant_dequant(x, scale, zero, self.q_max).to(x_dtype)

    def get_scale_zero(self, x):
        # q_max 已经在 device 上
        q_max = self.q_max
        
        # 优化1: 直接在最后一维统计
        xmax = x.amax(dim=-1, keepdim=True)
        xmin = x.amin(dim=-1, keepdim=True)
        
        # 优化2: clamp
        xmax = torch.clamp_min(xmax, 0)
        xmin = torch.clamp_max(xmin, 0)
        
        if self.lac:
            xmax = xmax * self.sigmoid(self.clip_factor_a_max)
            xmin = xmin * self.sigmoid(self.clip_factor_a_min)
        elif self._clip_ratio is not None:
            xmax = xmax * self._clip_ratio
            xmin = xmin * self._clip_ratio
        
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            scale = (xmax / q_max)
            
            # 优化3: torch.where 替代索引赋值，直接使用标量避免创建 Tensor
            scale = torch.where(tmp, 1.0, scale)
            
            # 优化4: 移除 repeat
            zero = torch.zeros_like(scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            
            # 优化3: torch.where 替代索引赋值，直接使用标量
            # 注意: 如果 NPU 不支持标量 where，可以使用 masked_fill_ (in-place)
            # xmin = torch.where(tmp, -1.0, xmin)
            # xmax = torch.where(tmp, 1.0, xmax)
            
            # 使用 masked_fill_ 通常更快且避免创建新 tensor (如果 tmp 是 bool)
            if tmp.any():
                xmin = xmin.masked_fill(tmp, -1.0)
                xmax = xmax.masked_fill(tmp, 1.0)
            
            scale = (xmax - xmin) / q_max
            zero = torch.round(-xmin / scale)

            # 优化4: 移除 repeat，利用广播机制
        
        return scale, zero

def run_benchmark(device, sym_mode, lac=False, groupsize=-1, clip_ratio=None, shape=(4, 2048, 4096), description="Standard"):
    mode_str = "Symmetric" if sym_mode else "Asymmetric"
    extra_info = []
    if lac: extra_info.append("LAC")
    if groupsize != -1: extra_info.append(f"GroupSize={groupsize}")
    if clip_ratio: extra_info.append(f"ClipRatio={clip_ratio}")
    info_str = ", ".join(extra_info) if extra_info else description
    
    print(f"\n{'='*20} Testing {mode_str} Quantization [{info_str}] {'='*20}")
    
    # 构造测试数据 [Batch, Seq, Hidden]
    B, S, H = shape
    x = torch.randn(B, S, H, dtype=torch.float16, device=device)
    
    # 构造一些 0 值区域以触发 tmp 条件
    x[0, 0, :] = 0 

    model_orig = ActivationQuantizerOriginal(bits=4, sym=sym_mode, lac=lac, groupsize=groupsize, clip_ratio=clip_ratio).to(device)
    model_opt = ActivationQuantizerOptimized(bits=4, sym=sym_mode, lac=lac, groupsize=groupsize, clip_ratio=clip_ratio).to(device)

    print(f"Input shape: {x.shape}, dtype: {x.dtype}")

    # --- 准确度测试 ---
    print(f"--- Accuracy Check ({mode_str}) ---")
    out_orig = model_orig(x)
    out_opt = model_opt(x)
    
    # NPU 上 allclose 可能不支持 fp16，转为 fp32 比较
    if torch.allclose(out_orig.float(), out_opt.float(), atol=1e-3):
        print("✅ Outputs match!")
    else:
        print("❌ Outputs do not match!")
        diff = (out_orig.float() - out_opt.float()).abs().max()
        print(f"Max difference: {diff}")

    # --- 速度测试 ---
    print(f"--- Speed Benchmark ({mode_str}, 100 iterations) ---")
    iterations = 100
    
    # Warmup
    for _ in range(10):
        model_orig(x)
        model_opt(x)
    
    if device.type == 'npu':
        torch.npu.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Test Original
    start_time = time.time()
    for _ in range(iterations):
        model_orig(x)
    if device.type == 'npu':
        torch.npu.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()
    orig_time = time.time() - start_time
    print(f"Original Time:  {orig_time:.4f} s")

    # Test Optimized
    start_time = time.time()
    for _ in range(iterations):
        model_opt(x)
    if device.type == 'npu':
        torch.npu.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()
    opt_time = time.time() - start_time
    print(f"Optimized Time: {opt_time:.4f} s")

    print(f"Speedup: {orig_time / opt_time:.2f}x")

def test_speed_accuracy():
    # 尝试使用 NPU，否则使用 CPU
    try:
        import torch_npu
        device = torch.device("npu:0")
        print("Running on NPU...")
    except ImportError:
        try:
            if torch.cuda.is_available():
                device = torch.device("cuda:0")
                print("Running on CUDA...")
            else:
                raise ImportError
        except ImportError:
            device = torch.device("cpu")
            print("Running on CPU (注意: CPU上速度差异可能不明显，主要验证NPU算子陷阱)...")

    # 1. 基础测试
    run_benchmark(device, sym_mode=False, description="Standard (B=4, S=2048)")
    run_benchmark(device, sym_mode=True, description="Standard (B=4, S=2048)")
    
    # 2. 测试 GroupSize
    run_benchmark(device, sym_mode=False, groupsize=128, description="GroupSize=128")
    
    # 3. 测试 LAC
    run_benchmark(device, sym_mode=True, lac=True, description="LAC")
    
    # 4. 测试 Clip Ratio
    run_benchmark(device, sym_mode=False, clip_ratio=0.9, description="ClipRatio=0.9")

    # 5. vllm 推理场景覆盖
    print("\n" + "="*40)
    print("Running vllm Inference Scenarios")
    print("="*40)

    # Prefill 阶段 (Batch=4, Seq=2048, Hidden=4096) - 7B/8B
    run_benchmark(device, sym_mode=False, shape=(4, 2048, 4096), description="Prefill 7B (B=4, S=2048, H=4096)")
    
    # Decode 阶段 (Batch=64, Seq=1, Hidden=4096) - 7B/8B
    run_benchmark(device, sym_mode=False, shape=(64, 1, 4096), description="Decode 7B (B=64, S=1, H=4096)")
    
    # Prefill 阶段 (Batch=2, Seq=4096, Hidden=5120) - 13B/14B
    run_benchmark(device, sym_mode=False, shape=(2, 4096, 5120), description="Prefill 14B (B=2, S=4096, H=5120)")
    
    # Decode 阶段 (Batch=128, Seq=1, Hidden=5120) - 13B/14B
    run_benchmark(device, sym_mode=False, shape=(128, 1, 5120), description="Decode 14B (B=128, S=1, H=5120)")

    # Decode 阶段 (Batch=256, Seq=1, Hidden=8192) - 70B
    run_benchmark(device, sym_mode=False, shape=(256, 1, 8192), description="Decode 70B (B=256, S=1, H=8192)")
    
    # Decode 阶段 + GroupSize (Batch=64, Seq=1, Hidden=4096, GroupSize=128)
    run_benchmark(device, sym_mode=False, shape=(64, 1, 4096), groupsize=128, description="Decode 7B + GroupSize=128")

if __name__ == "__main__":
    test_speed_accuracy()
