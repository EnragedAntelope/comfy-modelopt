# -*- coding: utf-8 -*-
"""
Native Quantization Node for comfy-modelopt

Uses NVIDIA ModelOpt for calibration (scale extraction), then applies
real weight quantization using PyTorch operations for self-contained
checkpoints that don't require a base model.

Architecture:
  1. ModelOpt calibration: Run forward passes, extract optimal scales
  2. Native quantization: Replace weights with quantized storage
  3. Save: safetensors with quantized weights + metadata
  4. Load: Overlay quantized weights onto standard ComfyUI model
"""

import os
import copy
import torch
import folder_paths
import comfy.model_management
import comfy.utils

from .utils import (
    get_gpu_info,
    check_precision_compatibility,
    format_bytes,
    introspect_diffusion_model,
)

from .native_quant import (
    extract_modelopt_scales,
    strip_modelopt_quantizers,
    strip_modelopt_state,
    quantize_model_weights,
)

from .quant_saveload import save_quantized_model, get_modelopt_metadata, capture_model_config


class ModelOptQuantizeUNet:
    """
    Quantize a UNet/diffusion model using real native quantization.
    
    Supports INT8, FP8, MXFP8, INT4, and NVFP4 formats.
    Produces self-contained quantized checkpoints.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "Input model to quantize"
                }),
                "precision": (["int8", "fp8", "int4", "nvfp4", "mxfp8"], {
                    "default": "mxfp8",
                    "tooltip": (
                        "int8: Best compatibility, Turing+ GPUs\n"
                        "fp8: Good quality/speed, Ada Lovelace+ (RTX 40-series+)\n"
                        "mxfp8: Blockwise FP8, Blackwell (RTX 50-series) — RECOMMENDED\n"
                        "int4: Maximum compression, experimental\n"
                        "nvfp4: Native 4-bit float, Blackwell (RTX 50-series)"
                    )
                }),
                "calibration_steps": ("INT", {
                    "default": 32,
                    "min": 8,
                    "max": 512,
                    "step": 8,
                }),
                "algorithm": (["max", "mse", "awq_lite"], {
                    "default": "max",
                    "tooltip": (
                        "max: Default, uses max absolute value for scale (fastest)\n"
                        "mse: Minimizes mean squared error (better quality, slower)\n"
                        "awq_lite: Activation-aware weight quantization (best quality, slowest)"
                    )
                }),
                "use_svd": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable SVD outlier absorption for better 4-bit quality"
                }),
                "svd_ratio": ("FLOAT", {
                    "default": 0.01,
                    "min": 0.001,
                    "max": 0.1,
                    "step": 0.001,
                    "tooltip": "Fraction of dimensions to keep as outliers (0.001-0.1)"
                }),
            },
            "optional": {
                "calibration_data": ("LATENT", {
                    "tooltip": "Optional calibration latent samples"
                }),
                "skip_layers": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("quantized_model",)
    FUNCTION = "quantize"
    CATEGORY = "modelopt"
    DESCRIPTION = "Quantize UNet model with native real quantization"

    def quantize(self, model, precision, calibration_steps, algorithm="max", use_svd=False, svd_ratio=0.01, calibration_data=None, skip_layers=""):
        gpu_info = get_gpu_info()
        if not gpu_info["available"]:
            raise RuntimeError("No CUDA device available!")

        is_compatible, message = check_precision_compatibility(precision)
        if not is_compatible:
            raise RuntimeError(f"{precision.upper()} not supported: {message}")

        try:
            import modelopt.torch.quantization as mtq
            import modelopt.torch.opt as mto
            import modelopt
            modelopt_version = getattr(modelopt, '__version__', 'unknown')
        except ImportError:
            raise ImportError("nvidia-modelopt not installed. Run: pip install nvidia-modelopt[all]")

        print(f"\n{'='*60}")
        print(f"ModelOpt Native Quantization v{modelopt_version}")
        print(f"{'='*60}")
        print(f"Precision: {precision.upper()}")
        print(f"Algorithm: {algorithm}")
        print(f"Calibration steps: {calibration_steps}")
        print(f"GPU: {gpu_info['name']} (SM {gpu_info['compute_capability']})")
        print(f"VRAM: {gpu_info['vram_gb']:.1f}GB")

        try:
            base_model = model.model
            diffusion_model = base_model.diffusion_model

            print("\nIntrospecting model architecture...")
            model_info = introspect_diffusion_model(diffusion_model)
            print(f"  Architecture: {model_info['architecture']}")
            print(f"  Parameters: {model_info['param_count_billions']:.2f}B")
            print(f"  Y dimension: {model_info['y_dim']}")
            print(f"  Context dimension: {model_info['context_dim']}")
            print(f"  Has y param: {model_info['has_y_param']}")
            
            # Capture original model size BEFORE quantization (for accurate compression ratio)
            original_params = sum(p.numel() for p in diffusion_model.parameters())
            original_bytes = original_params * 2  # Assume FP16 original
            print(f"  Original parameters: {original_params:,} ({format_bytes(original_bytes)} @ FP16)")
            # Parse skip layers
            skip_layer_list = []
            if skip_layers:
                skip_layer_list = [s.strip() for s in skip_layers.split(",") if s.strip()]
                print(f"Skipping layers: {skip_layer_list}")

            # Select proper config based on user precision choice
            print(f"\nPreparing {precision.upper()} quantization config...")
            
            if precision == "int8":
                if hasattr(mtq, 'INT8_DEFAULT_CFG'):
                    quant_cfg = mtq.INT8_DEFAULT_CFG
                    print("  Using ModelOpt built-in INT8_DEFAULT_CFG")
                else:
                    quant_cfg = {
                        "quant_cfg": {
                            "*weight_quantizer": {"num_bits": 8, "axis": 0},
                            "*input_quantizer": {"num_bits": 8, "axis": 0},
                            "*output_quantizer": {"enable": False},
                            "default": {"enable": False},
                        },
                        "algorithm": "max",
                    }
            elif precision == "fp8":
                if hasattr(mtq, 'FP8_DEFAULT_CFG'):
                    quant_cfg = mtq.FP8_DEFAULT_CFG
                    print("  Using ModelOpt built-in FP8_DEFAULT_CFG")
                else:
                    quant_cfg = {
                        "quant_cfg": {
                            "*weight_quantizer": {"num_bits": (4, 3), "axis": None},
                            "*input_quantizer": {"num_bits": (4, 3), "axis": None},
                            "*output_quantizer": {"enable": False},
                            "default": {"enable": False},
                        },
                        "algorithm": "max",
                    }
            elif precision == "int4":
                if hasattr(mtq, 'INT4_AWQ_CFG'):
                    quant_cfg = mtq.INT4_AWQ_CFG
                    print("  Using ModelOpt built-in INT4_AWQ_CFG")
                else:
                    quant_cfg = {
                        "quant_cfg": {
                            "*weight_quantizer": {"num_bits": 4, "axis": 0},
                            "*input_quantizer": {"enable": False},
                            "*output_quantizer": {"enable": False},
                            "default": {"enable": False},
                        },
                        "algorithm": {"method": "awq_lite", "alpha_step": 0.1},
                    }
            elif precision == "nvfp4":
                if hasattr(mtq, 'NVFP4_DEFAULT_CFG'):
                    quant_cfg = mtq.NVFP4_DEFAULT_CFG
                    print("  Using ModelOpt built-in NVFP4_DEFAULT_CFG")
                else:
                    raise RuntimeError(
                        "NVFP4 not supported by this ModelOpt version. "
                        "Please upgrade to nvidia-modelopt>=0.43.0"
                    )
            elif precision == "mxfp8":
                if hasattr(mtq, 'MXFP8_DEFAULT_CFG'):
                    quant_cfg = mtq.MXFP8_DEFAULT_CFG
                    print("  Using ModelOpt built-in MXFP8_DEFAULT_CFG")
                else:
                    raise RuntimeError(
                        "MXFP8 not supported by this ModelOpt version. "
                        "Please upgrade to nvidia-modelopt>=0.43.0"
                    )
            else:
                raise ValueError(f"Unsupported precision: {precision}")

            # Override algorithm if user selected non-default
            if algorithm != "max":
                quant_cfg = copy.deepcopy(quant_cfg)
                quant_cfg["algorithm"] = algorithm
                print(f"  Using calibration algorithm: {algorithm}")

            # Add skip layers to config
            if skip_layer_list:
                quant_cfg = copy.deepcopy(quant_cfg)
                if "quant_cfg" not in quant_cfg:
                    quant_cfg["quant_cfg"] = {}
                for layer_name in skip_layer_list:
                    quant_cfg["quant_cfg"][f"*{layer_name}*"] = {"enable": False}

            # Unwrap ComfyUI modules
            print("\nUnwrapping ComfyUI modules for ModelOpt compatibility...")
            unwrapped_count = self._unwrap_comfy_ops(diffusion_model)
            print(f"  Replaced {unwrapped_count} wrapped modules")

            # Move to GPU and eval mode
            device = comfy.model_management.get_torch_device()
            diffusion_model = diffusion_model.to(device)
            diffusion_model.eval()

            # Convert to FP32 for quantization
            original_dtype = next(diffusion_model.parameters()).dtype
            print(f"  Original dtype: {original_dtype}")
            if original_dtype != torch.float32:
                print("  Converting to FP32 for calibration...")
                diffusion_model = diffusion_model.to(torch.float32)

            # Prepare calibration data
            print("\nPreparing calibration data...")
            if calibration_data is not None:
                calib_latents = calibration_data["samples"]
                print(f"  Using provided calibration data: {calib_latents.shape}")
                if calib_latents.shape[0] > calibration_steps:
                    calib_latents = calib_latents[:calibration_steps]
            else:
                latent_shape = (
                    calibration_steps,
                    model_info['latent_channels'],
                    model_info['latent_spatial'],
                    model_info['latent_spatial']
                )
                calib_latents = torch.randn(
                    *latent_shape,
                    device=device,
                    dtype=torch.float32
                )
                print(f"  Using random calibration data: {calib_latents.shape}")

            actual_batch_size = calib_latents.shape[0]
            print(f"  Batch size for calibration: {actual_batch_size}")

            def forward_loop(diffusion_model_to_calibrate):
                """Calibration forward pass"""
                print(f"Running calibration ({calibration_steps} steps)...")
                model_dtype = next(diffusion_model_to_calibrate.parameters()).dtype

                with torch.no_grad():
                    for i in range(min(calibration_steps, len(calib_latents))):
                        if i % 10 == 0 or i == calibration_steps - 1:
                            print(f"  Step {i}/{calibration_steps}")

                        latent = calib_latents[i:i+1].to(device, dtype=model_dtype)
                        timestep = torch.tensor([999.0], device=device, dtype=model_dtype)
                        context_dim = model_info['context_dim']
                        context = torch.randn(1, 77, context_dim, device=device, dtype=model_dtype)

                        try:
                            if model_info['has_y_param'] and model_info['y_dim'] is not None:
                                y = torch.randn(1, model_info['y_dim'], device=device, dtype=model_dtype)
                                _ = diffusion_model_to_calibrate(latent, timestep, context, y=y)
                            else:
                                _ = diffusion_model_to_calibrate(latent, timestep, context)
                        except Exception as e:
                            if i == 0:
                                print(f"  Warning: Step 0 failed: {e}")
                                print(f"  Input shapes: latent={latent.shape}, timestep={timestep.shape}, context={context.shape}")
                                if model_info['has_y_param']:
                                    print(f"  y shape: {y.shape}")
                            # Continue with remaining steps
                            
                print("  Calibration complete!")

            # Run ModelOpt calibration
            print(f"\nRunning ModelOpt calibration with {precision.upper()} config...")
            print("  This may take several minutes for large models...")
            
            torch.cuda.empty_cache()
            
            calibrated_model = mtq.quantize(
                diffusion_model,
                quant_cfg,
                forward_loop
            )

            # Extract scales from ModelOpt quantizers
            print("\nExtracting calibration scales from ModelOpt...")
            scales = extract_modelopt_scales(calibrated_model)
            print(f"  Extracted scales for {len(scales)} layers")

            # Apply native quantization
            print(f"\nApplying native {precision.upper()} quantization...")
            quantized_model, quant_metadata = quantize_model_weights(
                calibrated_model,
                precision=precision,
                scales=scales,
                use_svd=use_svd,
                svd_ratio=svd_ratio
            )

            # Strip ModelOpt state
            print("  Stripping ModelOpt quantizers...")
            removed = strip_modelopt_quantizers(quantized_model)
            strip_modelopt_state(quantized_model)
            print(f"  Removed {removed} ModelOpt quantizer submodules")

            # Restore original dtype
            if original_dtype != torch.float32:
                print(f"  Restoring original dtype ({original_dtype})...")
                quantized_model = quantized_model.to(original_dtype)

            # Print summary
            print(f"\n{'='*60}")
            print("Quantization Summary")
            print(f"{'='*60}")
            print(f"  Layers quantized: {quant_metadata['quantized_count']}")
            print(f"  Layers skipped: {quant_metadata['skipped_count']}")
            
            total_params = sum(p.numel() for p in quantized_model.parameters())
            total_bytes = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
            print(f"  Total parameters: {total_params:,}")
            print(f"  Model size: {format_bytes(total_bytes)}")
            print(f"{'='*60}\n")

            # Replace model in ComfyUI structure
            quantized_comfy_model = model.clone()
            quantized_comfy_model.model.diffusion_model = quantized_model

            # Store metadata on the model for saving
            quantized_comfy_model.model.modelopt_metadata = {
                'precision': precision,
                'algorithm': algorithm,
                'calibration_steps': calibration_steps,
                'architecture': model_info,
                'quantization_metadata': quant_metadata,
                'original_params': original_params,
                'original_bytes': original_bytes,  # FP16 assumption
            }
            return (quantized_comfy_model,)

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"\nModelOpt Quantization Error:\n{error_trace}")
            raise RuntimeError(f"Quantization failed: {str(e)}")

    def _unwrap_comfy_ops(self, model):
        """Replace ComfyUI wrapped modules with standard PyTorch"""
        replaced_count = 0
        
        def replace_in_module(parent_module):
            nonlocal replaced_count
            for child_name in list(parent_module._modules.keys()):
                child = parent_module._modules[child_name]
                if child is None:
                    continue
                
                child_module_path = child.__class__.__module__
                child_class_name = child.__class__.__name__
                
                if child_module_path == 'comfy.ops' and child_class_name == 'Linear' and isinstance(child, torch.nn.Linear):
                    standard_linear = torch.nn.Linear(
                        in_features=child.in_features,
                        out_features=child.out_features,
                        bias=child.bias is not None,
                        device=child.weight.device,
                        dtype=child.weight.dtype
                    )
                    with torch.no_grad():
                        standard_linear.weight.copy_(child.weight)
                        if child.bias is not None:
                            standard_linear.bias.copy_(child.bias)
                    parent_module._modules[child_name] = standard_linear
                    replaced_count += 1
                elif child_module_path == 'comfy.ops' and child_class_name == 'Conv2d' and isinstance(child, torch.nn.Conv2d):
                    standard_conv = torch.nn.Conv2d(
                        in_channels=child.in_channels,
                        out_channels=child.out_channels,
                        kernel_size=child.kernel_size,
                        stride=child.stride,
                        padding=child.padding,
                        dilation=child.dilation,
                        groups=child.groups,
                        bias=child.bias is not None,
                        padding_mode=child.padding_mode,
                        device=child.weight.device,
                        dtype=child.weight.dtype
                    )
                    with torch.no_grad():
                        standard_conv.weight.copy_(child.weight)
                        if child.bias is not None:
                            standard_conv.bias.copy_(child.bias)
                    parent_module._modules[child_name] = standard_conv
                    replaced_count += 1
                else:
                    replace_in_module(child)
        
        replace_in_module(model)
        return replaced_count


class ModelOptSaveQuantized:
    """Save quantized model to disk as safetensors (self-contained)"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {}),
                "filename": ("STRING", {
                    "default": "quantized_unet",
                }),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_model"
    OUTPUT_NODE = True
    CATEGORY = "modelopt"
    def save_model(self, model, filename):
        save_dir = os.path.join(folder_paths.models_dir, "modelopt_unet")
        os.makedirs(save_dir, exist_ok=True)

        if not filename.endswith('.safetensors'):
            filename = filename + '.safetensors'
        
        save_path = os.path.join(save_dir, filename)

        print(f"\nSaving quantized model to {save_path}")
        
        try:
            diffusion_model = model.model.diffusion_model
            
            # Get existing metadata from quantization step
            metadata = getattr(model.model, 'modelopt_metadata', {})
            
            # Capture full model config for standalone reconstruction
            try:
                model_config = capture_model_config(model)
                metadata['_model_config'] = model_config
                print("  Captured model architecture config for standalone loading")
            except Exception as cfg_err:
                print(f"  Warning: Could not capture model config: {cfg_err}")
                print("  Loader will require a base model for this checkpoint")
            
            # Get architecture layer metadata
            arch_metadata = get_modelopt_metadata(diffusion_model)
            metadata['architecture'] = arch_metadata
            
            # Save
            save_quantized_model(diffusion_model, save_path, metadata)
            
            file_size = os.path.getsize(save_path)
            print(f"  Saved! Size: {format_bytes(file_size)}")
            
            # Print compression info using ORIGINAL size captured before quantization
            orig_size = metadata.get('original_bytes', file_size)  # Fallback to file_size if not stored
            print(f"  Original est. size: {format_bytes(orig_size)}")
            print(f"  Compression: {orig_size / file_size:.1f}x")
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"Save failed: {e}")
        
        # OUTPUT_NODE with no outputs - return empty tuple
        return ()

class ModelOptCalibrationHelper:
    """
    Collect calibration data from latents for better ModelOpt quantization.
    
    Accumulates individual latent samples across multiple workflow runs.
    Connect to a KSampler output and run the queue multiple times to
    build a diverse calibration dataset.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "Latent samples from KSampler output"
                }),
                "max_samples": ("INT", {
                    "default": 64,
                    "min": 8,
                    "max": 512,
                    "step": 8,
                    "tooltip": "Maximum number of individual samples to collect"
                }),
            }
        }

    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("calibration_data", "collected_count")
    FUNCTION = "collect_samples"
    CATEGORY = "modelopt"
    DESCRIPTION = "Accumulate latent samples for calibration. Run queue multiple times to build dataset."

    def __init__(self):
        self.samples = []

    def collect_samples(self, latent, max_samples):
        incoming = latent["samples"]  # [B, C, H, W]

        # Split batch into individual samples and accumulate
        batch_size = incoming.shape[0]
        for i in range(batch_size):
            if len(self.samples) >= max_samples:
                break
            self.samples.append(incoming[i:i+1].clone())

        print(f"Calibration: collected {len(self.samples)}/{max_samples} samples"
              f" (got {batch_size} from this run)")

        if len(self.samples) == 0:
            return ({"samples": incoming}, 0)

        # Concatenate all accumulated samples
        all_samples = torch.cat(self.samples[:max_samples], dim=0)
        return ({"samples": all_samples}, len(self.samples))


# Register nodes
NODE_CLASS_MAPPINGS = {
    "ModelOptQuantizeUNet": ModelOptQuantizeUNet,
    "ModelOptSaveQuantized": ModelOptSaveQuantized,
    "ModelOptCalibrationHelper": ModelOptCalibrationHelper,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ModelOptQuantizeUNet": "ModelOpt Quantize UNet",
    "ModelOptSaveQuantized": "ModelOpt Save Quantized",
    "ModelOptCalibrationHelper": "ModelOpt Calibration Helper",
}
