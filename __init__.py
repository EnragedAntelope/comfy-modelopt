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
__version__ = "0.1.1"

# Display startup message
print("\n" + "="*60)
print("NVIDIA ModelOpt for ComfyUI")
print("="*60)
print("Loaded nodes:")
for node_name in NODE_CLASS_MAPPINGS.keys():
    print(f"  • {node_name}")
print("\nQuantize diffusion models with INT8, FP8, or INT4")
print("Achieve ~2x faster inference with minimal quality loss")
print("="*60 + "\n")
