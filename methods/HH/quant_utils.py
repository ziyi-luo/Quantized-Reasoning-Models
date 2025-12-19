import torch
import torch.nn as nn

def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x

def get_qmin_qmax(bits, sym):
    if sym:
        q_max = torch.tensor(2 ** (bits - 1) - 1)
        q_min = -q_max - 1
    else:
        q_max, q_min = torch.tensor(2 ** bits - 1), 0
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

class ActivationQuantizer(nn.Module):
    """
    Activation Quantizer matching FlatQuant.
    Supports Per-Token (if groupsize=-1/default logic), LAC, Sym/Asym.
    """
    def __init__(self, n_bits, sym=False, lac=False, groupsize=-1, clip_ratio=None, num_groups=1):
        super(ActivationQuantizer, self).__init__()
        self.bits = n_bits
        self.q_max, self.q_min = get_qmin_qmax(n_bits, sym)
        self.sym = sym
        self.groupsize = groupsize
        self.num_groups = num_groups
        self.lac = lac
        self._clip_ratio = clip_ratio
        
        if self.lac:
            init_value = 4.
            self.sigmoid = nn.Sigmoid()
            self.clip_factor_a_max = nn.Parameter(torch.ones((num_groups, ))*init_value, requires_grad=True)
            self.clip_factor_a_min = nn.Parameter(torch.ones((num_groups, ))*init_value, requires_grad=True)
        
        self.enable = True

    def forward(self, x):
        if self.bits >= 16 or (not self.enable):
            return x
        init_shape = x.shape
        # Flatten logic similar to FlatQuant if groupsize is used
        # For Per-Token (FlatQuant "groupsize=-1"), we usually handle last dim.
        # But here FlatQuant reshapes: x.reshape(-1, self.num_groups, init_shape[-1]...)
        # If num_groups=1, it is just x.
        
        # FlatQuant Logic:
        # x = x.reshape(-1, self.num_groups, init_shape[-1] if self.groupsize == -1 else self.groupsize)
        # If per-token, num_groups=1, groupsize=-1.
        # x -> (N*Seq, 1, Hidden)
        
        x_reshaped = x.reshape(-1, self.num_groups, init_shape[-1] if self.groupsize == -1 else self.groupsize)
        fq_x = self.fake_quant(x_reshaped)
        return fq_x.reshape(*init_shape)

    def fake_quant(self, x):
        x_dtype = x.dtype
        scale, zero = self.get_scale_zero(x)
        if self.sym:
            return sym_quant_dequant(x, scale, self.q_max.to(x)).to(x_dtype)
        else:
            return asym_quant_dequant(x, scale, zero, self.q_max.to(x)).to(x_dtype)

    def get_scale_zero(self, x):
        q_max = self.q_max.to(x)
        init_shape = x.shape
        
        # Max/Min along last dim (feature dim)
        xmax, xmin = x.amax(-1, keepdim=True), x.amin(-1, keepdim=True)
        tmp = torch.zeros_like(xmax)
        xmax, xmin = torch.maximum(xmax, tmp), torch.minimum(xmin, tmp)
        
        if self.lac:
            xmax = xmax * self.sigmoid(self.clip_factor_a_max).reshape(1, -1, 1)
            xmin = xmin * self.sigmoid(self.clip_factor_a_min).reshape(1, -1, 1)
        elif self._clip_ratio is not None:
             xmax = xmax * self._clip_ratio
             xmin = xmin * self._clip_ratio
             
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            scale = (xmax / q_max)
            scale[tmp] = 1
            # Broadcast back to shape
            scale = scale.repeat(1, 1, x.shape[-1]).reshape(init_shape)
            zero = torch.zeros_like(scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            scale = (xmax - xmin) / q_max
            zero = torch.round(-xmin / scale)
            
            scale = scale.repeat(1, 1, x.shape[-1]).reshape(init_shape)
            zero = zero.repeat(1, 1, x.shape[-1]).reshape(init_shape)
            
        return scale, zero

class WeightQuantizer(nn.Module):
    """
    Weight Quantizer matching FlatQuant.
    Supports MSE-based optimization (grid search) for clipping.
    """
    def __init__(self, n_bits, groupsize=-1, sym=True, mse=True, norm=2.4, grid=100, maxshrink=0.8):
        super(WeightQuantizer, self).__init__()
        self.bits = n_bits
        self.groupsize = groupsize
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        
        # Just placeholder buffers
        self.register_buffer('scale', torch.zeros(1))
        self.register_buffer('zero', torch.zeros(1))
        self.register_buffer('maxq', torch.tensor(0))
        
        if sym:
            self.maxq = torch.tensor(2**(n_bits-1)-1)
        else:
            self.maxq = torch.tensor(2**n_bits - 1)
            
        self.enable = True
            
    def configure(self, bits, groupsize=-1, sym=True, mse=False, norm=2.4, grid=100, maxshrink=0.8):
        # Allow reconfiguration
        self.bits = bits
        self.groupsize = groupsize
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if sym:
            self.maxq = torch.tensor(2**(bits-1)-1)
        else:
            self.maxq = torch.tensor(2**bits - 1)

    def find_params(self, x):
        if self.bits >= 16 or (not self.enable):
            return
            
        if self.groupsize != -1:
            # Flatten to (N, Groupsize) if needed
            # For per-channel, x is (Out, In). groupsize=-1 (all In).
            # FlatQuant logic: x = x.reshape(-1, self.groupsize)
            pass 
            
        dev = x.device
        self.maxq = self.maxq.to(dev)
        shape = x.shape
        
        # Init min/max
        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)
        
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)
            
        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax
                
                if self.sym:
                    scale1 = xmax1 / self.maxq
                    zero1 = torch.zeros_like(scale1)
                    q = sym_quant_dequant(x, scale1.unsqueeze(1), self.maxq)
                else:
                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    q = asym_quant_dequant(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                    
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                
                is_better = err < best
                if torch.any(is_better):
                    best[is_better] = err[is_better]
                    self.scale[is_better] = scale1[is_better]
                    self.zero[is_better] = zero1[is_better]

        # Broadcast scale back to (Out, 1) for shape compatibility during quantize
        # FlatQuant: self.scale = self.scale.reshape(shape) where shape is [-1] + [1]*(...)
        # We need (Out, 1)
        self.scale = self.scale.reshape(-1, 1)
        self.zero = self.zero.reshape(-1, 1)
        return

    def quantize(self, x):
        x_dtype = x.dtype
        if self.enable and self.bits < 16:
            # Assumes x matches the shape expected by scale
            if self.sym:
                x = sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
            else:
                x = asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)
        return x
        
    def forward(self, x):
        return self.quantize(x)
