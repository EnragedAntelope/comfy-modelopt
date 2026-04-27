"""
ModelOpt Native Quantized UNet Loader for ComfyUI

Loads quantized UNet models saved with native quantization.

Standalone loading only (v0.6.0+).

Reconstructs UNet from architecture metadata stored in the .safetensors file.
No base model checkpoint required - the quantized file is self-contained.

To use: quantize with ModelOptSaveQuantized (v0.6.0+) to embed architecture config.
"""

import os
import torch
import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_management
import comfy.model_base
import comfy.model_patcher
import comfy.ops

from .utils import (
    get_gpu_info,
    validate_model_file,
    format_bytes,
)
from .native_quant import (
    QuantizedLinear,
    QuantizedConv2d,
)
from .quant_saveload import load_quantized_model, reconstruct_base_model


class ModelOptUNetLoader:
    """Standalone loading only (v0.6.0+). No base_model required."""

    _model_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        # Register modelopt folder path if not already registered
        if "modelopt_unet" not in folder_paths.folder_names_and_paths:
            modelopt_path = os.path.join(folder_paths.models_dir, "modelopt_unet")
            os.makedirs(modelopt_path, exist_ok=True)
            folder_paths.folder_names_and_paths["modelopt_unet"] = (
                [modelopt_path],
                {'.safetensors', '.sft'}
            )

        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("modelopt_unet"), {
                    "tooltip": "Quantized UNet checkpoint (.safetensors) from ModelOptSaveQuantized (v0.6.0+). Contains embedded architecture config for standalone loading. Place files in ComfyUI/models/modelopt_unet/"
                }),
                "enable_caching": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Keep quantized model cached in VRAM to skip re-loading on subsequent generations. Disable to free memory between uses."
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("quantized_model",)
    FUNCTION = "load_unet"
    CATEGORY = "loaders/modelopt"
    DESCRIPTION = (
        "Load a quantized UNet standalone (v0.6.0+). No base model required - "
        "the checkpoint contains embedded architecture config. "
        "Connect output to KSampler for inference. "
        "Place .safetensors files in ComfyUI/models/modelopt_unet/"
    )

    def load_unet(self, unet_name, enable_caching=True):
        """Load native quantized UNet - standalone mode only."""

        gpu_info = get_gpu_info()
        if not gpu_info["available"]:
            raise RuntimeError(
                "No CUDA device available!\n\n"
                "Quantized models require an NVIDIA GPU with CUDA support."
            )

        # Get model path
        unet_path = folder_paths.get_full_path("modelopt_unet", unet_name)
        if unet_path is None:
            raise FileNotFoundError(
                f"Model not found: {unet_name}\n\n"
                f"Please place your quantized UNet in:\n"
                f"  ComfyUI/models/modelopt_unet/\n\n"
                f"Supported formats: .safetensors"
            )

        # Validate model file
        is_valid, error = validate_model_file(unet_path, valid_extensions=['.safetensors', '.sft'])
        if not is_valid:
            raise RuntimeError(f"Invalid model file:\n{error}")

        # Check cache
        cache_key = f"{unet_path}"
        if enable_caching and cache_key in self._model_cache:
            print(f"Loading {unet_name} from cache")
            return (self._model_cache[cache_key],)

        print(f"\n{'='*60}")
        print("Loading Native Quantized Model")
        print(f"{'='*60}")
        print(f"  Quantized model: {unet_name}")
        print(f"  GPU: {gpu_info['name']} (SM {gpu_info['compute_capability']})")

        try:
            # Load quantized weights and metadata
            print(f"\n  Loading quantized weights from {unet_name}...")
            device = comfy.model_management.get_torch_device()
            q_state_dict, metadata = load_quantized_model(unet_path, device=device)

            precision = metadata.get('precision', 'fp8')
            print(f"  Precision: {precision.upper()}")

            # Check if we have model config for standalone reconstruction
            model_config = metadata.get('_model_config', None)
            has_standalone_config = (
                model_config is not None
                and isinstance(model_config, dict)
                and 'unet_config' in model_config
            )

            if has_standalone_config:
                # --- STANDALONE reconstruction ---
                print("  Mode: STANDALONE (reconstructing from stored config)")
                model_patcher = self._load_standalone(
                    q_state_dict, metadata, model_config, unet_name, precision, device
                )
            else:
                raise RuntimeError(
                    "This checkpoint was saved without architecture metadata (pre-v0.6.0).\n\n"
                    "Please re-quantize and save with ModelOptSaveQuantized (v0.6.0+) "
                    "to embed architecture config."
                )

            # Cache if enabled
            if enable_caching:
                self._model_cache[cache_key] = model_patcher

            file_size = os.path.getsize(unet_path)
            quantized_count = sum(
                1 for m in model_patcher.model.diffusion_model.modules()
                if isinstance(m, (QuantizedLinear, QuantizedConv2d))
            )

            print("\nSuccessfully loaded quantized model!")
            print(f"  File size: {format_bytes(file_size)}")
            print(f"  Quantized layers: {quantized_count}")
            print("  Ready for inference")
            print(f"{'='*60}\n")

            return (model_patcher,)

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Loader Error:\n{error_trace}")

            raise RuntimeError(
                f"Failed to load quantized UNet: {unet_name}\n\n"
                f"Error: {str(e)}\n\n"
                f"Common issues:\n"
                f"\u2022 Corrupted model file\n"
                f"\u2022 Insufficient VRAM ({gpu_info['vram_gb']:.1f}GB available)\n"
                f"\u2022 Incompatible ComfyUI version\n\n"
                f"Check console for detailed error trace."
            )

    # ------------------------------------------------------------------
    # PATH 1: Standalone reconstruction
    # ------------------------------------------------------------------

    def _load_standalone(self, q_state_dict, metadata, model_config,
                         unet_name, precision, device):
        """
        Reconstruct the model from stored architecture config, then apply
        quantized weights. No base model needed.
        """
        import comfy.model_patcher

        print("  Reconstructing model architecture from stored config...")

        # Reconstruct BaseModel from stored config using FP16 as working dtype
        base_model = reconstruct_base_model(
            model_config,
            device=device,
            dtype=torch.float16
        )

        diffusion_model = base_model.diffusion_model
        diffusion_model = diffusion_model.to(device)
        diffusion_model.eval()

        # Load all non-quantized state dict entries directly into the model
        # (norms, embeddings, time_embed, output layers, etc.)
        # Load with strict=False - quantized layers will be handled separately
        non_quantized_sd = {}
        quantized_keys = set()
        for key in q_state_dict.keys():
            if any(suffix in key for suffix in [
                '.weight_q', '.weight_scale', '.input_scale',
                '.block_scale', '.tensor_scale', '.svd_u', '.svd_s', '.svd_v',
                '.use_svd'
            ]):
                quantized_keys.add(key.rsplit('.', 1)[0])  # base name
            else:
                non_quantized_sd[key] = q_state_dict[key]

        # Load non-quantized params (with relaxed matching)
        missing, unexpected = diffusion_model.load_state_dict(
            non_quantized_sd, strict=False
        )
        if missing:
            # Missing keys for quantized layers are expected
            print(f"    Non-quantized params loaded "
                  f"({len(non_quantized_sd)} tensors)")

        # Apply quantized weights
        print("  Applying quantized weights...")
        self._apply_quantized_weights(diffusion_model, q_state_dict, precision)

        # Verify
        quantized_count = sum(
            1 for m in diffusion_model.modules()
            if isinstance(m, (QuantizedLinear, QuantizedConv2d))
        )
        if quantized_count == 0:
            raise RuntimeError(
                "No quantized layers found after loading!\n\n"
                "The quantized checkpoint may be corrupted or incompatible."
            )
        print(f"  Quantized layers: {quantized_count}")

        # Wrap in ModelPatcher
        load_device = comfy.model_management.get_torch_device()
        offload_device = comfy.model_management.unet_offload_device()
        model_patcher = comfy.model_patcher.ModelPatcher(
            base_model,
            load_device=load_device,
            offload_device=offload_device,
        )

        return model_patcher

    # ------------------------------------------------------------------
    # REMOVED: _load_overlay() method - standalone loading only (v0.6.0+)
    # This method is deprecated and no longer used.
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Quantized weight application (shared by both paths)
    # ------------------------------------------------------------------

    def _apply_quantized_weights(self, model: torch.nn.Module,
                                  q_state_dict: dict, precision: str) -> None:
        """
        Overlay quantized weights from state_dict onto model layers.

        For each layer in the model:
        1. Check if quantized version exists in state_dict
        2. Replace layer with QuantizedLinear/QuantizedConv2d
        3. Load quantized weight and scales
        """
        precision = precision.lower()

        for name, module in list(model.named_modules()):
            if not isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                continue

            # Check if we have quantized weights for this layer
            weight_key = f"{name}.weight_q" if name else "weight_q"

            # Try different key formats
            if weight_key in q_state_dict:
                pass
            elif f"{name}.weight" in q_state_dict:
                weight_key = f"{name}.weight"
            else:
                continue

            # Get the quantized weight
            weight_q = q_state_dict[weight_key]

            # Get scales
            weight_scale = q_state_dict.get(f"{name}.weight_scale")
            input_scale = q_state_dict.get(f"{name}.input_scale")
            block_scale = q_state_dict.get(f"{name}.block_scale")
            tensor_scale = q_state_dict.get(f"{name}.tensor_scale")

            # Get bias
            bias_key = f"{name}.bias"
            bias = q_state_dict.get(bias_key)
            if bias is None and hasattr(module, 'bias') and module.bias is not None:
                bias = module.bias

            # Replace layer with quantized version
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]

            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model

            if isinstance(module, torch.nn.Linear):
                new_layer = QuantizedLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=bias is not None,
                    precision=precision,
                    device=module.weight.device,
                    dtype=module.weight.dtype
                )
            elif isinstance(module, torch.nn.Conv2d):
                new_layer = QuantizedConv2d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                    bias=bias is not None,
                    padding_mode=module.padding_mode,
                    precision=precision,
                    device=module.weight.device,
                    dtype=module.weight.dtype
                )

            # Set quantized weight and scales
            new_layer.weight_q = weight_q
            if bias is not None:
                new_layer.bias = bias

            # Set quantized weight and scales (only if present in state_dict)
            if weight_scale is not None:
                new_layer.weight_scale = weight_scale
            if input_scale is not None:
                new_layer.input_scale = input_scale
            if block_scale is not None:
                new_layer.block_scale = block_scale
            if tensor_scale is not None:
                new_layer.tensor_scale = tensor_scale

            # Load SVD buffers if present
            svd_u = q_state_dict.get(f"{name}.svd_u")
            svd_s = q_state_dict.get(f"{name}.svd_s")
            svd_v = q_state_dict.get(f"{name}.svd_v")
            use_svd = q_state_dict.get(f"{name}.use_svd")

            if use_svd is not None:
                new_layer.use_svd = use_svd
            if svd_u is not None:
                new_layer.svd_u = svd_u
            if svd_s is not None:
                new_layer.svd_s = svd_s
            if svd_v is not None:
                new_layer.svd_v = svd_v

            # Replace in parent
            setattr(parent, child_name, new_layer)

    def _detect_model_info(self, state_dict, model_path):
        """Detect model type and precision from state dict."""
        keys = list(state_dict.keys())

        model_type = "unknown"
        if any("joint_blocks" in k for k in keys):
            model_type = "sd3"
        elif any("label_emb" in k for k in keys):
            model_type = "sdxl"
        elif any("time_embed" in k for k in keys):
            model_type = "sd15"

        precision = "fp16"
        # Check metadata for precision info
        for key in keys[:10]:
            if isinstance(state_dict[key], torch.Tensor):
                dtype = state_dict[key].dtype
                if dtype == torch.float8_e4m3fn or dtype == torch.float8_e5m2:
                    precision = "fp8"
                    break
                elif dtype == torch.int8:
                    precision = "int8"
                    break
                elif dtype == torch.uint8:
                    precision = "nvfp4"
                    break

        return {
            "model_type": model_type,
            "precision": precision,
            "file_size": os.path.getsize(model_path),
        }


# Register nodes
NODE_CLASS_MAPPINGS = {
    "ModelOptUNetLoader": ModelOptUNetLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelOptUNetLoader": "ModelOpt UNet Loader",
}
