"""
ComfyUI ModelOpt Integration Nodes

This package provides ComfyUI nodes for loading and working with models
quantized using NVIDIA TensorRT Model Optimizer (ModelOpt).

Supported Features:
- Loading quantized models (INT8, FP8, INT4)
- GPU capability detection
- Model caching for performance
- Comprehensive error handling

Supported Models:
- SDXL (Stable Diffusion XL) - INT8 quantization
- SD1.5 (Stable Diffusion 1.5) - INT8 quantization
- SD3 (Stable Diffusion 3) - INT8 quantization

Note: FLUX, Qwen Image, and WAN 2.2 are NOT officially supported by ModelOpt.
"""

from .loader import ModelOptLoader
from .utils import get_gpu_compute_capability, validate_model_file

__version__ = "0.1.0"
__all__ = ["ModelOptLoader", "get_gpu_compute_capability", "validate_model_file"]
