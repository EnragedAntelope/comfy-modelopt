"""
Native quantization utilities for comfy-modelopt.

Provides real weight quantization using PyTorch operations.
Works with or without comfy_kitchen installed.
When comfy_kitchen IS available, integrates with QuantizedTensor for acceleration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

# ==============================================================================
# Constants
# ==============================================================================

# FP8 E4M3 representable range
FP8_E4M3_MAX = 448.0

# FP8 E5M2 representable range  
FP8_E5M2_MAX = 57344.0

# NVFP4 uses FP8 scale + FP4 values
NVFP4_E4M3_SCALE_MAX = 448.0
NVFP4_E2M1_MAX = 6.0  # Max representable value in E2M1


# ==============================================================================
# Quantization Functions
# ==============================================================================

def compute_scale_fp8(tensor: torch.Tensor, dtype: torch.dtype = torch.float8_e4m3fn) -> torch.Tensor:
    """Compute per-tensor FP8 scale: scale = amax / max_representable."""
    amax = tensor.abs().max()
    if dtype == torch.float8_e4m3fn:
        max_val = FP8_E4M3_MAX
    elif dtype == torch.float8_e5m2:
        max_val = FP8_E5M2_MAX
    else:
        raise ValueError(f"Unsupported FP8 dtype: {dtype}")
    scale = amax / max_val
    # Prevent zero scale
    scale = torch.clamp(scale, min=1e-12)
    return scale.float()


def quantize_fp8(tensor: torch.Tensor, scale: Optional[torch.Tensor] = None, 
                 dtype: torch.dtype = torch.float8_e4m3fn) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize tensor to FP8.
    
    Returns:
        (quantized_tensor, scale)
    """
    if scale is None:
        scale = compute_scale_fp8(tensor, dtype)
    
    # Quantize: q = round(tensor / scale)
    # Then clamp to FP8 range
    scaled = tensor / scale.to(tensor.dtype)
    
    if dtype == torch.float8_e4m3fn:
        max_val = FP8_E4M3_MAX
    else:
        max_val = FP8_E5M2_MAX
    
    scaled = torch.clamp(scaled, min=-max_val, max=max_val)
    qdata = scaled.to(dtype)
    
    return qdata, scale


def dequantize_fp8(qdata: torch.Tensor, scale: torch.Tensor, 
                   out_dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Dequantize FP8 tensor back to floating point."""
    return qdata.to(out_dtype) * scale.to(out_dtype)


def quantize_int8(tensor: torch.Tensor, axis: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize tensor to INT8.
    
    Args:
        tensor: Input tensor
        axis: If None, per-tensor quantization. If 0, per-output-channel.
    
    Returns:
        (quantized_tensor_int8, scale)
    """
    if axis is None:
        # Per-tensor quantization
        amax = tensor.abs().max()
        scale = amax / 127.0
        scale = torch.clamp(scale, min=1e-12)
        
        qdata = torch.round(tensor / scale.to(tensor.dtype)).to(torch.int8)
        return qdata, scale.float()
    
    elif axis == 0:
        # Per-output-channel quantization (for Linear weights: [out_features, in_features])
        amax = tensor.abs().amax(dim=1, keepdim=True)  # [out_features, 1]
        scale = amax / 127.0
        scale = torch.clamp(scale, min=1e-12)
        
        qdata = torch.round(tensor / scale.to(tensor.dtype)).to(torch.int8)
        return qdata, scale.float()
    
    else:
        raise ValueError(f"Unsupported axis: {axis}")


def dequantize_int8(qdata: torch.Tensor, scale: torch.Tensor,
                    out_dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Dequantize INT8 tensor."""
    return qdata.to(out_dtype) * scale.to(out_dtype)


def quantize_mxfp8(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize to MXFP8 (Microscaling FP8 with 32-element blocks).
    
    MXFP8 uses E8M0 block scales (power-of-2 only) and E4M3 values.
    Each block of 32 elements shares a scale.
    
    Returns:
        (quantized_tensor_float8, block_scale)
    """
    if tensor.dim() != 2:
        raise ValueError(f"MXFP8 requires 2D tensor, got {tensor.dim()}D")
    
    orig_shape = tensor.shape  # [M, K]
    orig_dtype = tensor.dtype
    
    # Pad to multiple of 32 if needed
    M, K = orig_shape
    K_padded = ((K + 31) // 32) * 32
    
    if K_padded != K:
        padded = torch.zeros((M, K_padded), dtype=orig_dtype, device=tensor.device)
        padded[:, :K] = tensor
        tensor = padded
    
    # Reshape to [M, num_blocks, 32]
    num_blocks = K_padded // 32
    tensor_blocks = tensor.reshape(M, num_blocks, 32)
    
    # Compute per-block amax and E8M0 scale
    block_amax = tensor_blocks.abs().amax(dim=2, keepdim=True)  # [M, num_blocks, 1]
    
    # E8M0 scale: power-of-2, range [2^-127, 2^127]
    # scale = 2^round(log2(amax / 448))
    target = block_amax / FP8_E4M3_MAX
    log2_target = torch.log2(torch.clamp(target, min=1e-45))
    exponent = torch.round(log2_target)
    exponent = torch.clamp(exponent, min=-127, max=127)
    block_scale = torch.pow(2.0, exponent).to(torch.float32)  # [M, num_blocks, 1]
    
    # Quantize each block
    scaled = tensor_blocks / block_scale.to(orig_dtype)
    scaled = torch.clamp(scaled, min=-FP8_E4M3_MAX, max=FP8_E4M3_MAX)
    qdata_blocks = scaled.to(torch.float8_e4m3fn)
    
    # Reshape back
    qdata = qdata_blocks.reshape(M, K_padded)
    block_scale = block_scale.reshape(M, num_blocks)
    
    return qdata, block_scale


def dequantize_mxfp8(qdata: torch.Tensor, block_scale: torch.Tensor,
                     out_dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Dequantize MXFP8 tensor."""
    M, K = qdata.shape
    num_blocks = K // 32
    
    qdata_blocks = qdata.reshape(M, num_blocks, 32)
    scale_blocks = block_scale.reshape(M, num_blocks, 1)
    
    dequant = qdata_blocks.to(out_dtype) * scale_blocks.to(out_dtype)
    return dequant.reshape(M, K)


def quantize_nvfp4(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize to NVFP4 (NVIDIA 4-bit floating point).
    
    NVFP4 uses:
    - Per-tensor E4M3 scale (coarse)
    - Per-16-element-block E2M1 scale (fine)
    - Values stored as E2M1 (4-bit)
    
    Returns:
        (quantized_tensor_uint8_packed, tensor_scale, block_scale)
    """
    if tensor.dim() != 2:
        raise ValueError(f"NVFP4 requires 2D tensor, got {tensor.dim()}D")
    
    orig_shape = tensor.shape  # [M, K]
    orig_dtype = tensor.dtype
    M, K = orig_shape
    
    # Compute per-tensor E4M3 scale
    tensor_amax = tensor.abs().max()
    tensor_scale = tensor_amax / (NVFP4_E4M3_SCALE_MAX * NVFP4_E2M1_MAX)
    tensor_scale = torch.clamp(tensor_scale, min=1e-12).float()
    
    # Pad to multiple of 16
    K_padded = ((K + 15) // 16) * 16
    if K_padded != K:
        padded = torch.zeros((M, K_padded), dtype=orig_dtype, device=tensor.device)
        padded[:, :K] = tensor
        tensor = padded
    
    num_blocks = K_padded // 16
    tensor_blocks = tensor.reshape(M, num_blocks, 16)
    
    # Compute per-block E2M1 scale  
    block_amax = tensor_blocks.abs().amax(dim=2, keepdim=True)
    block_scale = block_amax / NVFP4_E2M1_MAX
    block_scale = torch.clamp(block_scale, min=1e-12).float()
    
    # Quantize: first apply tensor_scale, then block_scale
    scaled = tensor_blocks / tensor_scale.to(orig_dtype)
    scaled = scaled / block_scale.to(orig_dtype)
    scaled = torch.clamp(scaled, min=-NVFP4_E2M1_MAX, max=NVFP4_E2M1_MAX)
    
    # Round to nearest E2M1 representable value
    # E2M1 values: 0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6
    qdata_f = torch.round(scaled * 2) / 2  # Round to nearest 0.5
    qdata_f = torch.clamp(qdata_f, min=-6.0, max=6.0)
    
    # Pack two E2M1 values into one uint8
    # For now, store as float16 and pack later during save
    # TODO: Implement proper bit-packing
    qdata = qdata_f.to(torch.float16).reshape(M, K_padded)
    block_scale = block_scale.reshape(M, num_blocks)
    
    return qdata, tensor_scale, block_scale


def dequantize_nvfp4(qdata: torch.Tensor, tensor_scale: torch.Tensor,
                     block_scale: torch.Tensor, out_dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Dequantize NVFP4 tensor."""
    M, K = qdata.shape
    num_blocks = K // 16
    
    qdata_blocks = qdata.reshape(M, num_blocks, 16)
    scale_blocks = block_scale.reshape(M, num_blocks, 1)
    
    dequant = qdata_blocks.to(out_dtype) * tensor_scale.to(out_dtype)
    dequant = dequant * scale_blocks.to(out_dtype)
    
    return dequant.reshape(M, K)


def quantize_int4(tensor: torch.Tensor, axis: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize to INT4 (4-bit integers, packed into int8).
    
    Args:
        tensor: 2D weight tensor [out_features, in_features]
        axis: 0 for per-output-channel, None for per-tensor
    
    Returns:
        (quantized_tensor_int8_packed, scale)
    """
    if tensor.dim() != 2:
        raise ValueError(f"INT4 quantization requires 2D tensor, got {tensor.dim()}D")
    
    if axis == 0:
        amax = tensor.abs().amax(dim=1, keepdim=True)
    elif axis is None:
        amax = tensor.abs().max()
    else:
        raise ValueError(f"Unsupported axis: {axis}")
    
    scale = amax / 7.0  # INT4 range: [-7, 7] (keeping 1 value for padding)
    scale = torch.clamp(scale, min=1e-12)
    
    qdata_f = torch.round(tensor / scale.to(tensor.dtype))
    qdata_f = torch.clamp(qdata_f, min=-7, max=7)
    
    # For now store as int8. Proper packing (2 values per byte) can be done during save
    qdata = qdata_f.to(torch.int8)
    
    return qdata, scale.float()


def dequantize_int4(qdata: torch.Tensor, scale: torch.Tensor,
                    out_dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Dequantize INT4 tensor (stored as int8)."""
    return qdata.to(out_dtype) * scale.to(out_dtype)


# ==============================================================================
# SVD Outlier Absorption
# ==============================================================================

def apply_svd_outlier_absorption(weight: torch.Tensor, outlier_ratio: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply SVD-based outlier absorption to a weight matrix.
    
    Decomposes W = U @ diag(S) @ V^T, keeps top-k singular components
    in high precision (outliers), returns residual for quantization.
    
    Args:
        weight: 2D weight tensor [M, K]
        outlier_ratio: Fraction of dimensions to keep as outliers (0.0-1.0)
    
    Returns:
        (residual, svd_u, svd_s, svd_v)
        residual: W - outlier_part, to be quantized
        svd_u: [M, k] outlier left singular vectors
        svd_s: [k] outlier singular values
        svd_v: [K, k] outlier right singular vectors
    """
    if weight.dim() != 2:
        raise ValueError(f"SVD requires 2D tensor, got {weight.dim()}D")
    
    M, K = weight.shape
    k = max(1, int(min(M, K) * outlier_ratio))
    
    # Compute SVD
    with torch.no_grad():
        U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)
    
    # Extract top-k components
    u_out = U[:, :k]           # [M, k]
    s_out = S[:k]              # [k]
    v_out = Vh.conj().t()[:, :k]  # [K, k]
    
    # Reconstruct outlier part
    outlier_part = (u_out * s_out.unsqueeze(0)) @ v_out.t()  # [M, K]
    
    # Compute residual
    residual = weight - outlier_part.to(weight.dtype)
    
    # Cast outlier buffers to FP16 for storage
    return residual, u_out.half(), s_out.half(), v_out.half()


def reconstruct_svd_outlier(input: torch.Tensor, svd_u: torch.Tensor,
                              svd_s: torch.Tensor, svd_v: torch.Tensor) -> torch.Tensor:
    """
    Compute outlier contribution for a given input.
    
    Equation: ((input @ V) * S) @ U^T
    
    Args:
        input: [batch, in_features]
        svd_u: [out_features, k]
        svd_s: [k]
        svd_v: [in_features, k]
    
    Returns:
        [batch, out_features] outlier contribution
    """
    # input @ V -> [batch, k]
    out = torch.matmul(input, svd_v.to(input.dtype))
    # Multiply by singular values
    out = out * svd_s.to(input.dtype).unsqueeze(0)
    # @ U^T -> [batch, out_features]
    out = torch.matmul(out, svd_u.to(input.dtype).t())
    return out

# ==============================================================================
# Format Dispatch
# ==============================================================================

QUANTIZE_FUNCTIONS = {
    'fp8': quantize_fp8,
    'mxfp8': quantize_mxfp8,
    'nvfp4': quantize_nvfp4,
    'int8': quantize_int8,
    'int4': quantize_int4,
}

DEQUANTIZE_FUNCTIONS = {
    'fp8': dequantize_fp8,
    'mxfp8': dequantize_mxfp8,
    'nvfp4': dequantize_nvfp4,
    'int8': dequantize_int8,
    'int4': dequantize_int4,
}


def get_quantize_fn(precision: str):
    """Get quantization function for precision."""
    precision = precision.lower()
    if precision not in QUANTIZE_FUNCTIONS:
        raise ValueError(f"Unknown precision: {precision}. Supported: {list(QUANTIZE_FUNCTIONS.keys())}")
    return QUANTIZE_FUNCTIONS[precision]


def get_dequantize_fn(precision: str):
    """Get dequantization function for precision."""
    precision = precision.lower()
    if precision not in DEQUANTIZE_FUNCTIONS:
        raise ValueError(f"Unknown precision: {precision}")
    return DEQUANTIZE_FUNCTIONS[precision]


# ==============================================================================
# ModelOpt Scale Extraction
# ==============================================================================

def extract_modelopt_scales(model: nn.Module) -> Dict[str, Dict[str, Any]]:
    """
    Extract scales from ModelOpt TensorQuantizers after calibration.
    
    Returns dict mapping layer name -> {
        'weight_amax': float or None,
        'input_amax': float or None,
        'weight_scale': float or None,
        'input_scale': float or None,
    }
    """
    try:
        from modelopt.torch.quantization.nn import TensorQuantizer
    except ImportError:
        raise ImportError("nvidia-modelopt not installed")
    
    scales = {}
    
    for name, module in model.named_modules():
        if isinstance(module, TensorQuantizer):
            # Parse the name to find the parent layer
            # Names look like: "input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight_quantizer"
            parts = name.split('.')
            
            # The quantizer is attached to a layer, find which type
            if 'weight_quantizer' in parts[-1]:
                layer_name = '.'.join(parts[:-1])
                quant_type = 'weight'
            elif 'input_quantizer' in parts[-1]:
                layer_name = '.'.join(parts[:-1])
                quant_type = 'input'
            elif 'output_quantizer' in parts[-1]:
                continue  # We typically don't quantize outputs
            else:
                continue
            
            if layer_name not in scales:
                scales[layer_name] = {}
            
            # Extract amax
            amax = None
            if hasattr(module, 'amax') and module.amax is not None:
                if isinstance(module.amax, torch.Tensor):
                    amax = module.amax.detach().cpu().item()
                else:
                    amax = float(module.amax)
            
            scales[layer_name][f'{quant_type}_amax'] = amax
            
            # Compute scale from amax
            if amax is not None:
                # For FP8 (4,3) format: max_representable = 448.0
                max_repr = 448.0
                scale = amax / max_repr
                scales[layer_name][f'{quant_type}_scale'] = scale
    
    return scales


def strip_modelopt_quantizers(model: nn.Module) -> int:
    """
    Remove all ModelOpt TensorQuantizer submodules from model.
    
    Returns number of quantizers removed.
    """
    try:
        from modelopt.torch.quantization.nn import TensorQuantizer
    except ImportError:
        return 0
    
    removed = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, TensorQuantizer):
                delattr(module, child_name)
                removed += 1
    
    return removed


def strip_modelopt_state(model: nn.Module) -> None:
    """Remove ModelOpt state attributes from model and all submodules."""
    for module in [model] + list(model.modules()):
        for attr in ['_modelopt_state', '_modelopt_state_version']:
            if hasattr(module, attr):
                delattr(module, attr)


# ==============================================================================
# Layer Replacement
# ==============================================================================

class QuantizedLinear(nn.Module):
    """
    Linear layer with quantized weights.
    
    Stores weights in quantized format and dequantizes on forward pass.
    Compatible with ComfyUI's execution pipeline.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 precision: str = 'fp8', device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.precision = precision
        
        # Register scales as buffers so they persist in state_dict
        self.register_buffer('weight_scale', None)
        self.register_buffer('input_scale', None)
        self.register_buffer('block_scale', None)  # For MXFP8/NVFP4
        self.register_buffer('tensor_scale', None)  # For NVFP4
        
        # SVD outlier absorption buffers (optional)
        self.register_buffer('use_svd', torch.tensor(False))
        self.register_buffer('svd_u', None)
        self.register_buffer('svd_s', None)
        self.register_buffer('svd_v', None)
        
        # Placeholder buffers - will be replaced
        self.register_buffer('weight_q', None)
        if bias:
            self.register_buffer('bias', None)
        else:
            self.bias = None
    
    def set_quantized_weight(self, weight_q: torch.Tensor, 
                             weight_scale: Optional[torch.Tensor] = None,
                             input_scale: Optional[torch.Tensor] = None,
                             block_scale: Optional[torch.Tensor] = None,
                             tensor_scale: Optional[torch.Tensor] = None,
                             svd_u: Optional[torch.Tensor] = None,
                             svd_s: Optional[torch.Tensor] = None,
                             svd_v: Optional[torch.Tensor] = None):
        """Set the quantized weight and scales."""
        self.weight_q = weight_q
        if weight_scale is not None:
            self.weight_scale = weight_scale
        if input_scale is not None:
            self.input_scale = input_scale
        if block_scale is not None:
            self.block_scale = block_scale
        if tensor_scale is not None:
            self.tensor_scale = tensor_scale
        if svd_u is not None:
            self.svd_u = svd_u
            self.use_svd = torch.tensor(True)
        if svd_s is not None:
            self.svd_s = svd_s
        if svd_v is not None:
            self.svd_v = svd_v
    
    def get_dequantized_weight(self, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """Dequantize weight for computation."""
        if self.weight_q is None:
            raise RuntimeError("Quantized weight not set")
        
        dequant_fn = get_dequantize_fn(self.precision)
        
        if self.precision == 'nvfp4':
            return dequant_fn(self.weight_q, self.tensor_scale, self.block_scale, dtype)
        elif self.precision == 'mxfp8':
            return dequant_fn(self.weight_q, self.block_scale, dtype)
        else:
            return dequant_fn(self.weight_q, self.weight_scale, dtype)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.use_svd.item():
            return self._forward_svd(input)
        weight = self.get_dequantized_weight(input.dtype)
        return F.linear(input, weight, self.bias)
    
    def _forward_svd(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with SVD outlier absorption."""
        # Outlier contribution: ((x @ V) * S) @ U^T
        outlier = reconstruct_svd_outlier(input, self.svd_u, self.svd_s, self.svd_v)
        # Residual contribution
        residual_weight = self.get_dequantized_weight(input.dtype)
        residual = F.linear(input, residual_weight, self.bias)
        return outlier + residual
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'bias={self.bias is not None}, precision={self.precision}'


class QuantizedConv2d(nn.Module):
    """Conv2d layer with quantized weights."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros',
                 precision: str = 'fp8', device=None, dtype=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.precision = precision
        
        # Register scales as buffers
        self.register_buffer('weight_scale', None)
        self.register_buffer('input_scale', None)
        self.register_buffer('block_scale', None)
        self.register_buffer('tensor_scale', None)
        
        # SVD outlier absorption buffers (optional, for consistency with QuantizedLinear)
        self.register_buffer('use_svd', torch.tensor(False))
        self.register_buffer('svd_u', None)
        self.register_buffer('svd_s', None)
        self.register_buffer('svd_v', None)
        self.register_buffer('weight_q', None)
        if bias:
            self.register_buffer('bias', None)
        else:
            self.bias = None
    
    def set_quantized_weight(self, weight_q: torch.Tensor,
                             weight_scale: Optional[torch.Tensor] = None,
                             input_scale: Optional[torch.Tensor] = None,
                             block_scale: Optional[torch.Tensor] = None,
                             tensor_scale: Optional[torch.Tensor] = None,
                             svd_u: Optional[torch.Tensor] = None,
                             svd_s: Optional[torch.Tensor] = None,
                             svd_v: Optional[torch.Tensor] = None):
        """Set quantized weight and scales."""
        self.weight_q = weight_q
        if weight_scale is not None:
            self.weight_scale = weight_scale
        if input_scale is not None:
            self.input_scale = input_scale
        if block_scale is not None:
            self.block_scale = block_scale
        if tensor_scale is not None:
            self.tensor_scale = tensor_scale
        if svd_u is not None:
            self.svd_u = svd_u
            self.use_svd = torch.tensor(True)
        if svd_s is not None:
            self.svd_s = svd_s
        if svd_v is not None:
            self.svd_v = svd_v
    def get_dequantized_weight(self, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        if self.weight_q is None:
            raise RuntimeError("Quantized weight not set")

        # Blockwise formats (MXFP8/NVFP4) need flattening for Conv2d weights
        if self.precision in ['nvfp4', 'mxfp8'] and self.block_scale is not None:
            dequant_fn = get_dequantize_fn(self.precision)
            orig_shape = self.weight_q.shape
            flat = self.weight_q.reshape(-1, orig_shape[-1])
            if self.precision == 'nvfp4':
                dequant = dequant_fn(flat, self.tensor_scale, self.block_scale, dtype)
            else:
                dequant = dequant_fn(flat, self.block_scale, dtype)
            return dequant.reshape(orig_shape)

        # Fallback: blockwise format with no block_scale -> use FP8
        if self.precision in ['nvfp4', 'mxfp8'] and self.block_scale is None:
            return dequantize_fp8(self.weight_q, self.weight_scale, dtype)

        # Standard per-tensor dequantization (FP8, INT8, INT4)
        dequant_fn = get_dequantize_fn(self.precision)
        return dequant_fn(self.weight_q, self.weight_scale, dtype)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.get_dequantized_weight(input.dtype)
        return F.conv2d(input, weight, self.bias, self.stride, 
                       self.padding, self.dilation, self.groups)
    
    def extra_repr(self) -> str:
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ', precision={precision}'
        return s.format(**self.__dict__)


def replace_layer_with_quantized(parent_module: nn.Module, child_name: str,
                                  quant_type: str = 'fp8') -> nn.Module:
    """
    Replace a Linear or Conv2d layer with its quantized equivalent.
    
    Args:
        parent_module: Parent module containing the child
        child_name: Name of child to replace
        quant_type: Quantization precision
    
    Returns:
        The new quantized module
    """
    child = getattr(parent_module, child_name)
    
    if isinstance(child, nn.Linear):
        new_module = QuantizedLinear(
            in_features=child.in_features,
            out_features=child.out_features,
            bias=child.bias is not None,
            precision=quant_type,
            device=child.weight.device,
            dtype=child.weight.dtype
        )
        # Copy bias
        if child.bias is not None:
            new_module.bias = child.bias.detach().clone()
    
    elif isinstance(child, nn.Conv2d):
        new_module = QuantizedConv2d(
            in_channels=child.in_channels,
            out_channels=child.out_channels,
            kernel_size=child.kernel_size,
            stride=child.stride,
            padding=child.padding,
            dilation=child.dilation,
            groups=child.groups,
            bias=child.bias is not None,
            padding_mode=child.padding_mode,
            precision=quant_type,
            device=child.weight.device,
            dtype=child.weight.dtype
        )
        if child.bias is not None:
            new_module.bias = child.bias.detach().clone()
    
    else:
        raise TypeError(f"Cannot quantize layer of type {type(child)}")
    
    setattr(parent_module, child_name, new_module)
    return new_module


# ==============================================================================
# Full Model Quantization
# ==============================================================================

def quantize_model_weights(model: nn.Module, precision: str = 'fp8',
                           scales: Optional[Dict[str, Dict[str, Any]]] = None,
                           use_svd: bool = False, svd_ratio: float = 0.01) -> Tuple[nn.Module, Dict]:
    """
    Quantize all Linear and Conv2d weights in a model.
    
    Args:
        model: The model to quantize
        precision: Quantization precision
        scales: Optional dict of pre-computed scales from ModelOpt calibration
    
    Returns:
        (quantized_model, metadata)
    """
    precision = precision.lower()
    
    metadata = {
        'precision': precision,
        'quantized_layers': [],
        'skipped_layers': [],
    }
    
    quantized_count = 0
    skipped_count = 0
    
    for name, module in list(model.named_modules()):
        # Only quantize Linear and Conv2d layers
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue
        
        # Skip if no weights
        if not hasattr(module, 'weight') or module.weight is None:
            continue
        
        weight = module.weight.data
        
        # Apply SVD outlier absorption if enabled
        svd_u = svd_s = svd_v = None
        if use_svd and weight.dim() == 2 and weight.numel() >= 4096:
            try:
                residual, svd_u, svd_s, svd_v = apply_svd_outlier_absorption(weight, svd_ratio)
                weight = residual
            except Exception as svd_err:
                print(f"  Warning: SVD failed for {name}: {svd_err}")
                svd_u = svd_s = svd_v = None
        
        # Skip small layers (not worth quantizing)
        if weight.numel() < 1024:
            metadata['skipped_layers'].append(name)
            skipped_count += 1
            continue
        
        try:
            # Get scale from ModelOpt if available
            layer_scales = scales.get(name, {}) if scales else {}
            weight_amax = layer_scales.get('weight_amax')
            
            # Quantize weight
            if precision == 'fp8':
                if weight_amax is not None:
                    scale = torch.tensor(weight_amax / FP8_E4M3_MAX, dtype=torch.float32, device=weight.device)
                else:
                    scale = None
                qdata, computed_scale = quantize_fp8(weight, scale)
                
            elif precision == 'mxfp8':
                if weight.dim() == 2:
                    qdata, block_scale = quantize_mxfp8(weight)
                else:
                    # Conv2d weights: fall back to FP8 since MXFP8 requires 2D
                    print(f"  Note: Falling back to FP8 for Conv2d layer {name} (MXFP8 requires 2D)")
                    qdata, computed_scale = quantize_fp8(weight)
                    block_scale = None  # FP8 uses weight_scale instead
                    
            elif precision == 'nvfp4':
                if weight.dim() == 2:
                    qdata, tensor_scale, block_scale = quantize_nvfp4(weight)
                else:
                    # Conv2d weights: fall back to FP8 since NVFP4 requires 2D
                    print(f"  Note: Falling back to FP8 for Conv2d layer {name} (NVFP4 requires 2D)")
                    qdata, computed_scale = quantize_fp8(weight)
                    tensor_scale = None
                    block_scale = None  # FP8 uses weight_scale instead
                    
            elif precision == 'int8':
                if weight.dim() == 2:
                    qdata, scale = quantize_int8(weight, axis=0)  # Per-channel
                else:
                    qdata, scale = quantize_int8(weight, axis=None)  # Per-tensor
                    
            elif precision == 'int4':
                if weight.dim() == 2:
                    qdata, scale = quantize_int4(weight, axis=0)
                else:
                    qdata, scale = quantize_int4(weight, axis=None)
            
            else:
                raise ValueError(f"Unsupported precision: {precision}")
            
            # Replace layer with quantized version
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model
            
            new_module = replace_layer_with_quantized(parent, child_name, precision)
            
            # Set quantized weight
            if precision == 'fp8':
                new_module.set_quantized_weight(qdata, weight_scale=computed_scale, svd_u=svd_u, svd_s=svd_s, svd_v=svd_v)
            elif precision == 'mxfp8':
                if block_scale is not None:
                    new_module.set_quantized_weight(qdata, block_scale=block_scale, svd_u=svd_u, svd_s=svd_s, svd_v=svd_v)
                else:
                    # Fallback to FP8
                    new_module.set_quantized_weight(qdata, weight_scale=computed_scale, svd_u=svd_u, svd_s=svd_s, svd_v=svd_v)
            elif precision == 'nvfp4':
                if block_scale is not None:
                    new_module.set_quantized_weight(qdata, tensor_scale=tensor_scale, block_scale=block_scale, svd_u=svd_u, svd_s=svd_s, svd_v=svd_v)
                else:
                    # Fallback to FP8
                    new_module.set_quantized_weight(qdata, weight_scale=computed_scale, svd_u=svd_u, svd_s=svd_s, svd_v=svd_v)
            elif precision in ['int8', 'int4']:
                new_module.set_quantized_weight(qdata, weight_scale=scale, svd_u=svd_u, svd_s=svd_s, svd_v=svd_v)
            
            metadata['quantized_layers'].append({
                'name': name,
                'orig_shape': list(weight.shape),
                'orig_dtype': str(weight.dtype),
                'quantized_dtype': str(qdata.dtype),
                'compression_ratio': weight.element_size() * weight.numel() / (qdata.element_size() * qdata.numel()),
            })
            quantized_count += 1
            
        except Exception as e:
            print(f"  Warning: Failed to quantize layer {name}: {e}")
            metadata['skipped_layers'].append(name)
            skipped_count += 1
    
    metadata['quantized_count'] = quantized_count
    metadata['skipped_count'] = skipped_count
    
    return model, metadata


# ==============================================================================
# Integration with comfy_kitchen (when available)
# ==============================================================================

def try_comfy_kitchen_quantize(tensor: torch.Tensor, precision: str, 
                                scale: Optional[torch.Tensor] = None) -> Optional[Any]:
    """
    Try to quantize using comfy_kitchen if available.
    
    Returns QuantizedTensor if comfy_kitchen is available, None otherwise.
    """
    try:
        from comfy.quant_ops import (
            QuantizedTensor,
            TensorCoreFP8E4M3Layout,
            TensorCoreMXFP8Layout,
            TensorCoreNVFP4Layout,
        )
        
        if precision == 'fp8':
            if scale is None:
                scale = 'recalculate'
            qdata, params = TensorCoreFP8E4M3Layout.quantize(tensor, scale=scale)
            return QuantizedTensor(qdata, 'TensorCoreFP8E4M3Layout', params)
        
        elif precision == 'mxfp8':
            if tensor.dim() != 2:
                return None
            qdata, params = TensorCoreMXFP8Layout.quantize(tensor)
            return QuantizedTensor(qdata, 'TensorCoreMXFP8Layout', params)
        
        elif precision == 'nvfp4':
            if tensor.dim() != 2:
                return None
            if scale is None:
                scale = 'recalculate'
            qdata, params = TensorCoreNVFP4Layout.quantize(tensor, scale=scale)
            return QuantizedTensor(qdata, 'TensorCoreNVFP4Layout', params)
        
    except Exception:
        pass
    
    return None
