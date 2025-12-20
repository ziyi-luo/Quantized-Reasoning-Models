import torch
import torch.nn as nn
import torch.nn.functional as F
from .hh_modules import HouseholderRotation
from .quant_utils import WeightQuantizer, ActivationQuantizer

class HHLinear(nn.Module):
    """
    Linear layer wrapper for Householder Quantization.
    
    Training Mode:
        - Learns Householder Rotation (Q) and Diagonal Scale (D).
        - Forward pass applies:
            X_rot = X @ Q @ D
            W_rot = W @ Q @ D^-1
            Y = Quant(X_rot) @ Quant(W_rot).T
            
    Eval Mode:
        - Folds Q and D into weights if possible (or exports them).
        - Usually for WA (Weight-Activation) quantization, we keep Q at inference:
            X_inf = X @ Q @ D
            Y = Linear_Quantized(X_inf)
            Since W was optimized to match (X @ Q @ D), the linear layer weights are pre-rotated.
            
    Args:
        linear (nn.Linear): Original linear layer.
        n_reflections (int): Number of Householder reflections.
        w_bits (int): Weight bits (4).
        a_bits (int): Activation bits (8).
    """
    def __init__(self, linear: nn.Linear, n_reflections: int, 
                 args=None,
                 external_rotation=None, external_scale=None, 
                 weight_quantizer=None, act_quantizer=None):
        super().__init__()
        self.args = args
        w_bits = args.w_bits if args else 4
        a_bits = args.a_bits if args else 8
        a_sym = not (args.a_asym if args else False)
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        
        # Copy original weight/bias (keep them floating point reference)
        self.register_buffer('weight', linear.weight.data.clone())
        if linear.bias is not None:
            self.register_buffer('bias', linear.bias.data.clone())
        else:
            self.bias = None
            
        # Learnable Parameters (can be external)
        if external_rotation is not None:
            self.rotation = external_rotation
            self.owns_rotation = False
        else:
            self.rotation = HouseholderRotation(self.in_features, n_reflections)
            self.owns_rotation = True
            
        if external_scale is not None:
            self.scale = external_scale
            self.owns_scale = False
        else:
            self.scale = nn.Parameter(torch.ones(self.in_features))
            self.owns_scale = True
            
        # Quantizers
        self.weight_quantizer = weight_quantizer or WeightQuantizer(n_bits=w_bits)
        self.act_quantizer = act_quantizer or ActivationQuantizer(n_bits=a_bits, sym=a_sym)
        
        # State
        self.is_reparameterized = False
        
    def forward(self, x, external_rotated=None):
        """
        Forward pass. Use eval mode if reparameterized.
        """
        if not self.is_reparameterized:
            return self._train_forward(x)
        else:
            return self._eval_forward(x, external_rotated=external_rotated)

    def _eval_forward(self, x, external_rotated=None):
        """Efficient inference path."""
        if x.isnan().any() or x.isinf().any():
             print(f"WARNING: Input x to _eval_forward has NaNs/Infs!")
             
        # Use externally rotated activation if available
        if external_rotated is not None:
            x_trans = external_rotated
        else:
            # Translate activations: x -> (x @ Q @ D) using Optimized WY
            # rotation.forward already handles the scale when reparameterized
            x_trans = self.rotation(x)
            
        if x_trans.isnan().any() or x_trans.isinf().any():
             print(f"WARNING: x_trans in _eval_forward has NaNs/Infs!")
        
        # Quantize activations
        x_q = self.act_quantizer(x_trans)
        
        if x_q.isnan().any() or x_q.isinf().any():
             print(f"WARNING: x_q in _eval_forward has NaNs/Infs!")
        
        # Linear layer with pre-transformed and pre-quantized weights
        input_dtype = x_q.dtype
        w_q = self.weight.to(input_dtype)
        bias = self.bias.to(input_dtype) if self.bias is not None else None
        
        out = F.linear(x_q, w_q, bias)
        
        if out.isnan().any() or out.isinf().any():
             print(f"WARNING: Linear output in _eval_forward has NaNs/Infs!")
             
        return out

    def _train_forward(self, x):
        """Dynamic transformation for calibration/training."""
        # 1. Transform activations: x' = x @ Q @ D
        # rotation already includes D in train mode if we use scale explicitly
        # but in training, rotation is just reflections.
        x_rot = self.rotation(x)
        x_trans = x_rot * self.scale
        
        # 2. Quantize activations
        x_q = self.act_quantizer(x_trans)
        
        # 3. Transform and quantize weights: W' = W @ Q @ D^-1
        # In train mode, we do this on the fly
        w_rot = self.rotation(self.weight)
        inv_scale = 1.0 / (self.scale + 1e-6)
        w_trans = w_rot * inv_scale
        w_q = self.weight_quantizer(w_trans)
        
        input_dtype = x_q.dtype
        w_q = w_q.to(input_dtype)
        bias = self.bias.to(input_dtype) if self.bias is not None else None
        return F.linear(x_q, w_q, bias)
    
    def reparameterize(self):
        """Fold current rotation, scale, and weight quantization into weights for inference."""
        if self.is_reparameterized:
            return
            
        with torch.no_grad():
            # 1. Transform: W' = W @ Q @ D^-1
            # Use apply_reflections directly to avoid issues with shared rotations being in eval mode
            if self.weight.isnan().any() or self.weight.isinf().any():
                 print(f"WARNING: weight has NaNs/Infs BEFORE rotation! {self.weight.dtype}")
            
            w_rot = self.rotation.apply_reflections(self.weight)
            
            if w_rot.isnan().any() or w_rot.isinf().any():
                 print(f"WARNING: w_rot has NaNs/Infs! Max: {w_rot.max()}, Min: {w_rot.min()}")
            
            inv_scale = 1.0 / (self.scale + 1e-6)
            w_trans = w_rot * inv_scale
            
            # Ensure quantizer is initialized
            if self.weight_quantizer.scale.sum() == 0:
                # Just find params on the fly if not calibrated
                self.weight_quantizer.find_params(w_trans)
            
            # 2. Quantize: W'' = Quant_Dequant(W')
            w_q_deq = self.weight_quantizer(w_trans)
            
            if w_q_deq.isnan().any() or w_q_deq.isinf().any():
                 print(f"WARNING: w_q_deq has NaNs/Infs!")
            
            # 3. Update weight buffer
            self.weight.data.copy_(w_q_deq)
            
            # 4. Reparameterize Rotation to absorb scale
            self.rotation.reparameterize(self.scale)
            
            self.is_reparameterized = True
            
        # Note: self.rotation and self.scale are kept for activation transformation.
    
    def calibrate(self, x):
        """
        Calibrate quantizers with input data.
        Call this ONCE with calibration data before training/inference.
        """
        with torch.no_grad():
            # Transform
            x_rot = self.rotation(x)
            x_scaled = x_rot * self.scale
            
            w_rot = self.rotation(self.weight)
            inv_scale = 1.0 / (self.scale + 1e-6)
            w_scaled = w_rot * inv_scale
            
            # Find optimal quantization parameters (MSE search)
            self.weight_quantizer.find_params(w_scaled)

    def get_rotated_weight(self):
        """Return the transformed (but not quantized) weight W'."""
        with torch.no_grad():
            w_rot = self.rotation(self.weight)
            inv_scale = 1.0 / (self.scale + 1e-6)
            return w_rot * inv_scale

    def strip_scale(self, scale_vec: torch.Tensor):
        """
        Remove activation scaling from the rotation so that the scale
        can be fused into preceding modules (e.g., LayerNorm weights).
        This keeps the WY buffers consistent and avoids double-scaling.
        """
        with torch.no_grad():
            if torch.allclose(self.rotation.scale_buffer, torch.ones_like(self.rotation.scale_buffer)):
                return
            scale_flat = scale_vec.view(-1, 1).to(self.rotation.Y_scaled)
            self.rotation.Y_scaled.div_(scale_flat)
            self.rotation.scale_buffer.fill_(1.0)
            if self.owns_scale:
                self.scale.data.fill_(1.0)
            else:
                self.scale.fill_(1.0)

    def export_wy_and_scale(self):
        """Export WY params and the merged scale for inference."""
        W, Y = self.rotation.export_wy()
        return W, Y, self.scale
