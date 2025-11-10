"""
ModelOpt UNet Loader for ComfyUI

Loads UNet/diffusion models that have been quantized with NVIDIA ModelOpt.

User provides quantized UNet separately from VAE and text encoders.
Compatible with various diffusion model architectures.
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
    Load a UNet/diffusion model quantized with NVIDIA ModelOpt.

    Supports INT8, FP8, and INT4 quantized models.

    Note: VAE and text encoders must be loaded separately using standard ComfyUI nodes.
    ModelOpt only quantizes the UNet/diffusion model component.
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
                "base_model": ("MODEL", {
                    "tooltip": "Original unquantized model (same architecture as the quantized model)"
                }),
                "unet_name": (folder_paths.get_filename_list("modelopt_unet"), {
                    "tooltip": "Select a ModelOpt quantized UNet model from models/modelopt_unet/"
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
    RETURN_NAMES = ("quantized_model",)
    FUNCTION = "load_unet"
    CATEGORY = "loaders/modelopt"
    DESCRIPTION = "Load quantized UNet by restoring ModelOpt state into base model. Requires original unquantized model as input."

    # Model cache
    _model_cache = {}

    def load_unet(self, base_model, unet_name, enable_caching=True):
        """Load ModelOpt quantized UNet by restoring into base model"""

        # Import ModelOpt
        try:
            import modelopt.torch.opt as mto
        except ImportError:
            raise RuntimeError(
                "❌ ModelOpt not installed!\n\n"
                "Please install: pip install nvidia-modelopt"
            )

        # Validate GPU
        gpu_info = get_gpu_info()
        if not gpu_info["available"]:
            raise RuntimeError(
                "❌ No CUDA device available!\n\n"
                "ModelOpt requires an NVIDIA GPU with CUDA support."
            )

        # Get model path
        unet_path = folder_paths.get_full_path("modelopt_unet", unet_name)

        if unet_path is None:
            raise FileNotFoundError(
                f"❌ Model not found: {unet_name}\n\n"
                f"Please place your ModelOpt quantized UNet in:\n"
                f"  ComfyUI/models/modelopt_unet/\n\n"
                f"Supported formats: .pt, .pth"
            )

        # Validate model file
        is_valid, error = validate_model_file(unet_path)
        if not is_valid:
            raise RuntimeError(f"❌ Invalid model file:\n{error}")

        # Check cache
        cache_key = f"{unet_path}"
        if enable_caching and cache_key in self._model_cache:
            print(f"✓ Loading {unet_name} from cache")
            return (self._model_cache[cache_key],)

        print(f"\n{'='*60}")
        print(f"Loading Quantized Model with ModelOpt")
        print(f"{'='*60}")
        print(f"  Quantized model: {unet_name}")
        print(f"  GPU: {gpu_info['name']} (SM {gpu_info['compute_capability']})")

        try:
            # Clone the base model to avoid modifying the original
            print(f"\n  Cloning base model...")
            quantized_model = base_model.clone()

            # Extract diffusion model from ComfyUI's model wrapper
            # ComfyUI structure: model (ModelPatcher) -> model.model (BaseModel) -> model.model.diffusion_model (UNet)
            diffusion_model = quantized_model.model.diffusion_model

            print(f"  Base model architecture: {type(diffusion_model).__name__}")
            param_count = sum(p.numel() for p in diffusion_model.parameters())
            print(f"  Parameters: {param_count:,} ({param_count/1e9:.2f}B)")

            # Restore quantized state using ModelOpt
            print(f"\n  Restoring quantized state with mto.restore()...")
            mto.restore(diffusion_model, unet_path)

            # Verify quantizers were restored
            from modelopt.torch.quantization.nn import TensorQuantizer
            quantizer_count = sum(1 for m in diffusion_model.modules() if isinstance(m, TensorQuantizer))

            if quantizer_count == 0:
                raise RuntimeError(
                    f"❌ No quantizers found after restoration!\n\n"
                    f"This may indicate:\n"
                    f"• The saved model was not properly quantized\n"
                    f"• Architecture mismatch between base model and saved model\n"
                    f"• Corrupted saved model file\n\n"
                    f"Please ensure:\n"
                    f"1. The base_model has the same architecture as when quantized\n"
                    f"2. The saved model was created with ModelOptSaveQuantized node\n"
                    f"3. The saved file is not corrupted"
                )

            print(f"  ✓ Quantizers restored: {quantizer_count}")

            # The diffusion_model is already modified in place by mto.restore()
            # It's already part of quantized_model.model.diffusion_model

            # Cache if enabled
            if enable_caching:
                self._model_cache[cache_key] = quantized_model

            file_size = os.path.getsize(unet_path)
            print(f"\n✓ Successfully loaded quantized model!")
            print(f"  File size: {format_bytes(file_size)}")
            print(f"  Quantizers: {quantizer_count}")
            print(f"  Ready for inference")
            print(f"{'='*60}\n")

            return (quantized_model,)

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"ModelOpt Loader Error:\n{error_trace}")

            raise RuntimeError(
                f"❌ Failed to load ModelOpt UNet: {unet_name}\n\n"
                f"Error: {str(e)}\n\n"
                f"Common issues:\n"
                f"• Base model architecture doesn't match saved model\n"
                f"• Saved model not created with ModelOptSaveQuantized\n"
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
