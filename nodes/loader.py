"""
ModelOpt UNet Loader for ComfyUI

Loads UNet models that have been quantized with NVIDIA ModelOpt.
Works with SDXL, SD1.5, and SD3 models.

User provides quantized UNet separately from VAE and text encoders.
"""

import os
import torch
import folder_paths
import comfy.sd
import comfy.utils
from comfy.cli_args import args

from .utils import (
    get_gpu_compute_capability,
    get_gpu_info,
    validate_model_file,
    check_precision_compatibility,
    format_bytes,
)


class ModelOptUNetLoader:
    """
    Load a UNet model quantized with NVIDIA ModelOpt.

    Supports:
    - SDXL UNet (INT8/FP8)
    - SD1.5 UNet (INT8/FP8)
    - SD3 UNet (INT8)

    Note: VAE and text encoders must be loaded separately using standard ComfyUI nodes.
    ModelOpt only quantizes the UNet component.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Register modelopt folder path if not already registered
        if "modelopt_unet" not in folder_paths.folder_names_and_paths:
            modelopt_path = os.path.join(folder_paths.models_dir, "modelopt_unet")
            os.makedirs(modelopt_path, exist_ok=True)
            folder_paths.folder_names_and_paths["modelopt_unet"] = (
                [modelopt_path],
                folder_paths.supported_pt_extensions
            )

        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("modelopt_unet"), {
                    "tooltip": "Select a ModelOpt quantized UNet model from models/modelopt_unet/"
                }),
                "precision": (["auto", "fp8", "fp16", "int8", "int4"], {
                    "default": "auto",
                    "tooltip": (
                        "Quantization precision:\n\n"
                        "• auto: Detect from model metadata\n"
                        "• fp8: Best quality, requires Ada Lovelace+ (RTX 4000 series)\n"
                        "• int8: Production ready, Turing+ (RTX 2000 series)\n"
                        "• int4: Maximum compression, Turing+\n"
                        "• fp16: No quantization (baseline)"
                    )
                }),
            },
            "optional": {
                "enable_caching": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Cache loaded models in memory to speed up subsequent loads"
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_unet"
    CATEGORY = "loaders/modelopt"
    DESCRIPTION = "Load UNet quantized with NVIDIA ModelOpt (SDXL, SD1.5, SD3). Use with standard VAE and text encoder loaders."

    # Model cache
    _model_cache = {}

    def load_unet(self, unet_name, precision="auto", enable_caching=True):
        """Load ModelOpt quantized UNet"""

        # Validate GPU
        gpu_info = get_gpu_info()
        if not gpu_info["available"]:
            raise RuntimeError(
                "❌ No CUDA device available!\n\n"
                "ModelOpt requires an NVIDIA GPU with CUDA support.\n"
                "Detected: No CUDA device\n\n"
                "Please ensure:\n"
                "• NVIDIA GPU is properly installed\n"
                "• CUDA drivers are installed\n"
                "• PyTorch CUDA version matches your CUDA installation"
            )

        # Check precision compatibility
        if precision != "auto":
            is_compatible, message = check_precision_compatibility(precision)
            if not is_compatible:
                raise RuntimeError(
                    f"❌ Precision {precision.upper()} not supported on this GPU\n\n"
                    f"{message}\n\n"
                    f"Your GPU: {gpu_info['name']}\n"
                    f"Compute Capability: {gpu_info['compute_capability']}\n\n"
                    f"Please select a compatible precision or use 'auto'."
                )

        # Get model path
        unet_path = folder_paths.get_full_path("modelopt_unet", unet_name)

        if unet_path is None:
            raise FileNotFoundError(
                f"❌ Model not found: {unet_name}\n\n"
                f"Please place your ModelOpt quantized UNet in:\n"
                f"  ComfyUI/models/modelopt_unet/\n\n"
                f"Supported formats: .safetensors, .pt, .pth, .ckpt"
            )

        # Validate model file
        is_valid, error = validate_model_file(unet_path)
        if not is_valid:
            raise RuntimeError(f"❌ Invalid model file:\n{error}")

        # Check cache
        cache_key = f"{unet_path}_{precision}"
        if enable_caching and cache_key in self._model_cache:
            print(f"✓ Loading {unet_name} from cache")
            return (self._model_cache[cache_key],)

        # Load the model
        print(f"Loading ModelOpt UNet: {unet_name}")
        print(f"  Precision: {precision}")
        print(f"  GPU: {gpu_info['name']} (SM {gpu_info['compute_capability']})")

        try:
            # Load state dict
            sd = comfy.utils.load_torch_file(unet_path, safe_load=True)

            # Detect model type and precision from metadata
            model_type, detected_precision = self._detect_model_info(sd, unet_path)

            if precision == "auto":
                precision = detected_precision
                print(f"  Auto-detected precision: {precision}")

            # Create model
            model = self._create_model_from_state_dict(sd, model_type, precision)

            # Cache if enabled
            if enable_caching:
                self._model_cache[cache_key] = model

            file_size = os.path.getsize(unet_path)
            print(f"✓ Successfully loaded {unet_name}")
            print(f"  Model type: {model_type}")
            print(f"  File size: {format_bytes(file_size)}")
            print(f"  Precision: {precision}")

            return (model,)

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"ModelOpt Loader Error:\n{error_trace}")

            raise RuntimeError(
                f"❌ Failed to load ModelOpt UNet: {unet_name}\n\n"
                f"Error: {str(e)}\n\n"
                f"Common issues:\n"
                f"• Model was quantized for different GPU architecture\n"
                f"• Corrupted model file\n"
                f"• Insufficient VRAM ({gpu_info['vram_gb']:.1f}GB available)\n"
                f"• Missing ModelOpt dependencies\n\n"
                f"Check console for detailed error trace."
            )

    def _detect_model_info(self, state_dict, model_path):
        """Detect model type and precision from state dict"""

        # Check for model type hints in state dict keys
        keys = list(state_dict.keys())

        # Detect model architecture
        model_type = "unknown"
        if any("joint_blocks" in k for k in keys):
            model_type = "sd3"
        elif any("label_emb" in k for k in keys):
            model_type = "sdxl"
        elif any("time_embed" in k for k in keys):
            model_type = "sd15"

        # Detect precision from tensor dtypes or metadata
        precision = "fp16"  # default

        # Check for quantization metadata
        if "modelopt_metadata" in state_dict:
            metadata = state_dict["modelopt_metadata"]
            if isinstance(metadata, dict):
                precision = metadata.get("precision", "fp16")

        # Check tensor dtypes as fallback
        for key in keys[:10]:  # Check first 10 tensors
            if isinstance(state_dict[key], torch.Tensor):
                dtype = state_dict[key].dtype
                if dtype == torch.float8_e4m3fn or dtype == torch.float8_e5m2:
                    precision = "fp8"
                    break
                elif dtype == torch.int8:
                    precision = "int8"
                    break
                elif dtype == torch.float16:
                    precision = "fp16"

        return model_type, precision

    def _create_model_from_state_dict(self, state_dict, model_type, precision):
        """Create ComfyUI model from state dict"""

        # Use ComfyUI's model loading infrastructure
        # This creates a model wrapper compatible with ComfyUI's execution

        # For now, we use ComfyUI's standard model detection
        # In production, you might need custom model classes for quantized models

        model_config = comfy.sd.load_model_weights(state_dict, "")

        if model_config is None:
            # Fallback: try to detect config from state dict
            model_config = self._detect_model_config(state_dict, model_type)

        return model_config

    def _detect_model_config(self, state_dict, model_type):
        """Detect model configuration from state dict"""
        # Placeholder for custom config detection
        # In production, implement proper config detection based on model architecture
        raise NotImplementedError(
            f"Could not automatically detect model configuration for {model_type}. "
            f"Please ensure the model file contains proper metadata."
        )


# Register the node
NODE_CLASS_MAPPINGS = {
    "ModelOptUNetLoader": ModelOptUNetLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelOptUNetLoader": "ModelOpt UNet Loader",
}
