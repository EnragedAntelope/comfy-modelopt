"""
ComfyUI ModelOpt Integration Nodes

This package provides ComfyUI nodes for loading and quantizing diffusion models
with NVIDIA TensorRT Model Optimizer (ModelOpt).

Nodes:
- ModelOptUNetLoader: Load pre-quantized UNet models
- ModelOptQuantizeUNet: Quantize UNet models to INT8/FP8/INT4
- ModelOptSaveQuantized: Save quantized models
- ModelOptCalibrationHelper: Collect calibration data

Quantize your diffusion models for ~2x faster inference with minimal quality loss.
Supports INT8, FP8, and INT4 quantization formats.
"""

from .loader import NODE_CLASS_MAPPINGS as LOADER_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as LOADER_DISPLAY
from .quantizer import NODE_CLASS_MAPPINGS as QUANTIZER_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as QUANTIZER_DISPLAY

# Combine all node mappings
NODE_CLASS_MAPPINGS = {
    **LOADER_MAPPINGS,
    **QUANTIZER_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **LOADER_DISPLAY,
    **QUANTIZER_DISPLAY,
}

# Export for ComfyUI
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

__version__ = "0.1.1"
