"""
ComfyUI NVIDIA ModelOpt Integration

Custom nodes for quantizing and loading models optimized with NVIDIA ModelOpt.

Installation:
1. Place this folder in ComfyUI/custom_nodes/
2. Install dependencies: pip install -r requirements.txt
3. Restart ComfyUI

For more information, see: https://github.com/EnragedAntelope/comfy-modelopt
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Display startup message
print("NVIDIA ModelOpt for ComfyUI loaded successfully")
