"""
ModelOpt Quantization Nodes for ComfyUI

Convert/quantize UNet models to ModelOpt optimized formats.
Supports INT8, FP8, and INT4 quantization.
"""

import os
import torch
import folder_paths
import comfy.model_management
import comfy.utils

from .utils import (
    get_gpu_info,
    check_precision_compatibility,
    format_bytes,
)


class ModelOptQuantizeUNet:
    """
    Quantize a UNet/diffusion model using NVIDIA ModelOpt.

    This node performs Post-Training Quantization (PTQ) on a loaded UNet/diffusion model.
    The quantized model can then be saved and reused for faster inference (~2x speedup).

    Supports INT8, FP8, and INT4 quantization formats.

    Note: Quantization requires calibration data. Use ModelOptCalibrationHelper
    to generate calibration samples, or the node will use default random calibration.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Input model to quantize (connect from UNet Loader or Checkpoint Loader)"
                }),
                "precision": (["int8", "fp8", "int4"], {
                    "default": "int8",
                    "tooltip": (
                        "Quantization format:\n\n"
                        "• int8: Best compatibility, Turing+ GPUs, ~2x speedup\n"
                        "• fp8: Best quality/speed balance, Ada Lovelace+, ~2x speedup\n"
                        "• int4: Maximum compression, experimental, ~4x memory savings"
                    )
                }),
                "calibration_steps": ("INT", {
                    "default": 32,
                    "min": 8,
                    "max": 512,
                    "step": 8,
                    "tooltip": (
                        "Number of calibration steps for quantization.\n"
                        "Higher = better accuracy but slower conversion.\n\n"
                        "Recommended:\n"
                        "• Quick test: 16-32 steps\n"
                        "• Production: 64-128 steps\n"
                        "• Best quality: 256+ steps"
                    )
                }),
            },
            "optional": {
                "calibration_data": ("LATENT", {
                    "tooltip": "Optional: Calibration latent samples. If not provided, random data will be used."
                }),
                "skip_layers": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": (
                        "Comma-separated layer names to skip quantization.\n"
                        "Example: 'out.0,time_embed'\n"
                        "Leave empty to quantize all layers."
                    )
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("quantized_model",)
    FUNCTION = "quantize"
    CATEGORY = "modelopt"
    DESCRIPTION = "Quantize UNet model with NVIDIA ModelOpt (INT8/FP8/INT4)"

    def quantize(self, model, precision, calibration_steps, calibration_data=None, skip_layers=""):
        """Quantize the model using ModelOpt"""

        # Validate GPU and precision
        gpu_info = get_gpu_info()
        if not gpu_info["available"]:
            raise RuntimeError(
                "❌ No CUDA device available!\n\n"
                "ModelOpt quantization requires an NVIDIA GPU."
            )

        is_compatible, message = check_precision_compatibility(precision)
        if not is_compatible:
            raise RuntimeError(
                f"❌ {precision.upper()} quantization not supported on this GPU\n\n"
                f"{message}\n\n"
                f"Your GPU: {gpu_info['name']} (SM {gpu_info['compute_capability']})\n\n"
                f"Please select a compatible precision."
            )

        # Check ModelOpt availability
        try:
            import modelopt.torch.quantization as mtq
        except ImportError:
            raise ImportError(
                "❌ NVIDIA ModelOpt not installed!\n\n"
                "Please install ModelOpt:\n"
                "  pip install nvidia-modelopt[all]\n\n"
                "Note: Requires PyTorch with CUDA support."
            )

        print(f"\n{'='*60}")
        print(f"ModelOpt Quantization")
        print(f"{'='*60}")
        print(f"Precision: {precision.upper()}")
        print(f"Calibration steps: {calibration_steps}")
        print(f"GPU: {gpu_info['name']} (SM {gpu_info['compute_capability']})")
        print(f"VRAM: {gpu_info['vram_gb']:.1f}GB")

        try:
            # Get the actual UNet/diffusion model from ComfyUI's model wrapper
            # ComfyUI structure: model (ModelPatcher) -> model.model (BaseModel) -> model.model.diffusion_model (UNet)
            base_model = model.model
            diffusion_model = base_model.diffusion_model

            # Parse skip layers
            skip_layer_list = []
            if skip_layers:
                skip_layer_list = [s.strip() for s in skip_layers.split(",") if s.strip()]
                print(f"Skipping layers: {skip_layer_list}")

            # Select quantization config
            print(f"\nPreparing quantization config...")
            if precision == "int8":
                quant_cfg = mtq.INT8_DEFAULT_CFG
            elif precision == "fp8":
                quant_cfg = mtq.FP8_DEFAULT_CFG
            elif precision == "int4":
                quant_cfg = mtq.INT4_AWQ_CFG
            else:
                raise ValueError(f"Unsupported precision: {precision}")

            # Customize config to skip certain layers if specified
            if skip_layer_list:
                import copy
                quant_cfg = copy.deepcopy(quant_cfg)
                for layer_name in skip_layer_list:
                    quant_cfg["quant_cfg"][f"*{layer_name}*"] = {"enable": False}

            # Prepare calibration data
            print(f"Preparing calibration data...")
            if calibration_data is not None:
                # Use provided calibration latents
                calib_latents = calibration_data["samples"]
            else:
                # Generate random calibration data
                # Determine latent shape based on model type
                latent_shape = self._get_latent_shape(base_model)
                calib_latents = torch.randn(
                    calibration_steps,
                    *latent_shape,
                    device=comfy.model_management.get_torch_device()
                )
                print(f"  Using random calibration data: {calib_latents.shape}")

            # Define calibration forward loop
            def forward_loop(diffusion_model_to_calibrate):
                """Calibration forward pass for the diffusion model"""
                print(f"Running calibration...")
                device = comfy.model_management.get_torch_device()

                with torch.no_grad():
                    for i in range(min(calibration_steps, len(calib_latents))):
                        if i % 10 == 0:
                            print(f"  Step {i}/{calibration_steps}")

                        # Get latent sample
                        if calibration_data is not None:
                            latent = calib_latents[i:i+1].to(device)
                        else:
                            latent = calib_latents[i:i+1]

                        # Create dummy timestep and context
                        # Timestep as tensor for diffusion models
                        timestep = torch.tensor([999.0], device=device)

                        # Context dimensions vary by model
                        # Try to detect appropriate context size
                        context_dim = getattr(diffusion_model_to_calibrate, 'context_dim', 768)
                        context = torch.randn(1, 77, context_dim, device=device)

                        # Forward pass - diffusion models typically accept (x, timesteps, context, **kwargs)
                        try:
                            _ = diffusion_model_to_calibrate(latent, timestep, context)
                        except Exception as e:
                            # Try alternative signature if first fails
                            try:
                                _ = diffusion_model_to_calibrate(x=latent, timesteps=timestep, context=context)
                            except Exception as e2:
                                print(f"  Warning: Calibration step {i} failed: {e2}")
                                # Continue with other steps

                print(f"  Calibration complete!")

            # Quantize the diffusion model
            print(f"\nQuantizing diffusion model to {precision.upper()}...")
            print(f"This may take several minutes...")

            quantized_diffusion_model = mtq.quantize(
                diffusion_model,
                quant_cfg,
                forward_loop
            )

            # Replace the diffusion model in the ComfyUI model structure
            # Clone the model patcher and replace the diffusion model
            quantized_comfy_model = model.clone()
            quantized_comfy_model.model.diffusion_model = quantized_diffusion_model

            print(f"\n✓ Quantization complete!")
            print(f"{'='*60}\n")

            return (quantized_comfy_model,)

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"\nModelOpt Quantization Error:\n{error_trace}")

            raise RuntimeError(
                f"❌ Quantization failed!\n\n"
                f"Error: {str(e)}\n\n"
                f"Common issues:\n"
                f"• Insufficient VRAM (try reducing calibration_steps)\n"
                f"• Incompatible model architecture\n"
                f"• GPU doesn't support {precision.upper()}\n"
                f"• Missing ModelOpt dependencies\n\n"
                f"Check console for detailed error trace."
            )

    def _get_latent_shape(self, model):
        """Determine latent shape based on model architecture"""
        # Default SDXL latent shape: (4, 128, 128)
        # Default SD1.5 latent shape: (4, 64, 64)
        # Default SD3 latent shape: (16, 128, 128)

        # Try to detect from model config
        if hasattr(model, 'latent_format'):
            if hasattr(model.latent_format, 'latent_channels'):
                channels = model.latent_format.latent_channels
                # Guess reasonable spatial dimensions
                if channels == 16:  # SD3
                    return (channels, 128, 128)
                elif channels == 4:  # SDXL/SD1.5
                    return (channels, 128, 128)

        # Default to SDXL shape
        return (4, 128, 128)


class ModelOptSaveQuantized:
    """
    Save a quantized model to disk.

    Saves the model in a format that can be loaded with ModelOptUNetLoader.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Quantized model to save (from ModelOptQuantizeUNet)"
                }),
                "filename": ("STRING", {
                    "default": "quantized_unet.safetensors",
                    "tooltip": "Filename for the saved model (will be saved in models/modelopt_unet/)"
                }),
                "save_format": (["safetensors", "pytorch"], {
                    "default": "safetensors",
                    "tooltip": (
                        "File format:\n"
                        "• safetensors: Recommended, safer and faster\n"
                        "• pytorch: Standard .pt format"
                    )
                }),
            },
            "optional": {
                "metadata": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional metadata to save with the model (JSON format)"
                }),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_model"
    OUTPUT_NODE = True
    CATEGORY = "modelopt"
    DESCRIPTION = "Save ModelOpt quantized model to disk"

    def save_model(self, model, filename, save_format="safetensors", metadata=""):
        """Save the quantized model"""

        # Ensure modelopt_unet directory exists
        save_dir = os.path.join(folder_paths.models_dir, "modelopt_unet")
        os.makedirs(save_dir, exist_ok=True)

        # Add extension if not present
        if save_format == "safetensors" and not filename.endswith(".safetensors"):
            filename = filename + ".safetensors"
        elif save_format == "pytorch" and not filename.endswith((".pt", ".pth")):
            filename = filename + ".pt"

        save_path = os.path.join(save_dir, filename)

        print(f"\nSaving quantized model...")
        print(f"  Path: {save_path}")
        print(f"  Format: {save_format}")

        try:
            # Get state dict from model
            state_dict = model.model.state_dict()

            # Add metadata if provided
            if metadata:
                try:
                    import json
                    metadata_dict = json.loads(metadata)
                    state_dict["modelopt_metadata"] = metadata_dict
                except json.JSONDecodeError:
                    print(f"  Warning: Invalid metadata JSON, skipping")

            # Save based on format
            if save_format == "safetensors":
                from safetensors.torch import save_file
                save_file(state_dict, save_path)
            else:
                torch.save(state_dict, save_path)

            file_size = os.path.getsize(save_path)
            print(f"✓ Model saved successfully!")
            print(f"  Size: {format_bytes(file_size)}")
            print(f"  Location: models/modelopt_unet/{filename}\n")

            return {}

        except Exception as e:
            import traceback
            print(f"Save Error:\n{traceback.format_exc()}")
            raise RuntimeError(
                f"❌ Failed to save model!\n\n"
                f"Error: {str(e)}\n\n"
                f"Please check:\n"
                f"• Write permissions for {save_dir}\n"
                f"• Sufficient disk space\n"
                f"• Valid filename"
            )


class ModelOptCalibrationHelper:
    """
    Helper node to prepare calibration data for quantization.

    Collects latent samples from your workflow to use as calibration data.
    This improves quantization quality compared to random calibration.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "Input latent samples to collect for calibration"
                }),
                "max_samples": ("INT", {
                    "default": 64,
                    "min": 8,
                    "max": 512,
                    "step": 8,
                    "tooltip": "Maximum number of samples to collect"
                }),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("calibration_data",)
    FUNCTION = "collect_samples"
    CATEGORY = "modelopt"
    DESCRIPTION = "Collect latent samples for ModelOpt calibration"

    def __init__(self):
        self.samples = []

    def collect_samples(self, latent, max_samples):
        """Collect calibration samples"""

        samples = latent["samples"]

        # Add to collection
        if len(self.samples) < max_samples:
            self.samples.append(samples)
            print(f"Collected calibration sample {len(self.samples)}/{max_samples}")

        # Concatenate all collected samples
        if len(self.samples) > 0:
            all_samples = torch.cat(self.samples[:max_samples], dim=0)
            return ({"samples": all_samples},)

        return (latent,)


# Register nodes
NODE_CLASS_MAPPINGS = {
    "ModelOptQuantizeUNet": ModelOptQuantizeUNet,
    "ModelOptSaveQuantized": ModelOptSaveQuantized,
    "ModelOptCalibrationHelper": ModelOptCalibrationHelper,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelOptQuantizeUNet": "ModelOpt Quantize UNet",
    "ModelOptSaveQuantized": "ModelOpt Save Quantized Model",
    "ModelOptCalibrationHelper": "ModelOpt Calibration Helper",
}
