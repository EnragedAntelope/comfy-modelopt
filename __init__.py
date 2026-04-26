"""
ComfyUI NVIDIA ModelOpt Integration

Custom nodes for quantizing and loading models optimized with NVIDIA ModelOpt.

Installation:
1. Place this folder in ComfyUI/custom_nodes/
2. Install dependencies: pip install -r requirements.txt
3. Restart ComfyUI

For more information, see: https://github.com/EnragedAntelope/comfy-modelopt
"""

import warnings
import os
import sys

# Suppress misleading ModelOpt FP8 CUDA extension build warning on Windows.
# ModelOpt tries to JIT-compile a CUDA extension for FP8 using system nvcc.
# This fails on Windows with modern MSVC (unsupported compiler version),
# but it gracefully falls back to simulated/fake FP8 which works correctly.
# The warning is multi-line and noisy, so we suppress it by module.
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="modelopt.torch.utils.cpp_extension",
)

# Also suppress the distributed/elastic multiprocessing redirect warning
# that appears on Windows (harmless, but clutters output)
warnings.filterwarnings(
    "ignore",
    message="Redirects are currently not supported in Windows or MacOs",
)

# Suppress setuptools distutils deprecation warning on Windows
warnings.filterwarnings(
    "ignore",
    message="_get_vc_env is private",
)

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Display startup message
print("NVIDIA ModelOpt for ComfyUI loaded successfully")
