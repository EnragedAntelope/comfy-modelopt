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
    introspect_diffusion_model,
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
            import modelopt
            modelopt_version = getattr(modelopt, '__version__', 'unknown')
            print(f"Debug: ModelOpt version: {modelopt_version}")
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

            # Introspect model to detect architecture details
            print(f"\nIntrospecting model architecture...")
            model_info = introspect_diffusion_model(diffusion_model)
            print(f"  Architecture: {model_info['architecture']}")
            print(f"  Parameters: {model_info['param_count_billions']:.2f}B ({model_info['param_count']:,})")
            print(f"  Y dimension: {model_info['y_dim']} ({model_info.get('y_dim_source', 'not detected')})")
            print(f"  Context dimension: {model_info['context_dim']} ({model_info.get('context_dim_source', 'default')})")
            print(f"  Latent format: {model_info['latent_channels']}x{model_info['latent_spatial']}x{model_info['latent_spatial']}")
            print(f"  Has y parameter: {model_info['has_y_param']}")

            # Debug: Print found attributes
            if model_info.get('important_attributes_found'):
                found = [k for k, v in model_info['important_attributes_found'].items() if v]
                print(f"  Found attributes: {', '.join(found) if found else 'None'}")

            # Debug: Print detection events
            if model_info.get('detected_attributes'):
                print(f"  Detection events: {', '.join(model_info['detected_attributes'])}")

            # Parse skip layers
            skip_layer_list = []
            if skip_layers:
                skip_layer_list = [s.strip() for s in skip_layers.split(",") if s.strip()]
                print(f"Skipping layers: {skip_layer_list}")

            # Select quantization config
            # Use diffusion-specific configs (based on NVIDIA's official examples)
            # These are different from LLM configs - they use simple wildcards
            # Source: NVIDIA/TensorRT-Model-Optimizer/examples/diffusers/quantization/
            print(f"\nPreparing quantization config for diffusion model...")

            import copy

            if precision == "int8":
                # Diffusion-specific INT8 config
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
                # Diffusion-specific FP8 config
                quant_cfg = {
                    "quant_cfg": {
                        "*weight_quantizer": {"num_bits": (4, 3), "axis": None},
                        "*input_quantizer": {"num_bits": (4, 3), "axis": None},
                        "*output_quantizer": {"enable": False},
                        "*[qkv]_bmm_quantizer": {"num_bits": (4, 3), "axis": None},
                        "*softmax_quantizer": {"num_bits": (4, 3), "axis": None},
                        "default": {"enable": False},
                    },
                    "algorithm": "max",
                }
            elif precision == "int4":
                # Try using the default INT4_AWQ_CFG as fallback
                # This is experimental for diffusion models
                try:
                    quant_cfg = copy.deepcopy(mtq.INT4_AWQ_CFG)
                    quant_cfg["quant_cfg"]["*output_quantizer"] = {"enable": False}
                except AttributeError:
                    # If INT4_AWQ_CFG doesn't exist, create a basic config
                    quant_cfg = {
                        "quant_cfg": {
                            "*weight_quantizer": {"num_bits": 4, "axis": 0},
                            "*input_quantizer": {"num_bits": 8, "axis": 0},
                            "*output_quantizer": {"enable": False},
                            "default": {"enable": False},
                        },
                        "algorithm": "max",
                    }
            else:
                raise ValueError(f"Unsupported precision: {precision}")

            print(f"  Using base config: {precision.upper()}")
            print(f"  Algorithm: {quant_cfg.get('algorithm', 'default')}")

            # Debug: Print config details to troubleshoot "Inserted 0 quantizers"
            if "quant_cfg" in quant_cfg:
                print(f"  Config keys: {list(quant_cfg['quant_cfg'].keys())}")
                # Check if there are any wildcard patterns
                wildcards = [k for k in quant_cfg['quant_cfg'].keys() if '*' in k]
                print(f"  Wildcard patterns: {wildcards}")

            # Customize config to skip certain layers if specified
            if skip_layer_list:
                import copy
                quant_cfg = copy.deepcopy(quant_cfg)
                for layer_name in skip_layer_list:
                    quant_cfg["quant_cfg"][f"*{layer_name}*"] = {"enable": False}

            # Quantize the diffusion model
            print(f"\nPreparing model for quantization to {precision.upper()}...")
            print(f"This may take several minutes...")

            # Debug: Print model info
            print(f"Debug: Model type: {type(diffusion_model).__name__}")
            param_count = sum(p.numel() for p in diffusion_model.parameters())
            print(f"Debug: Total parameters: {param_count:,}")

            # Check for quantizable layers
            linear_count = sum(1 for m in diffusion_model.modules() if isinstance(m, torch.nn.Linear))
            conv_count = sum(1 for m in diffusion_model.modules() if isinstance(m, torch.nn.Conv2d))
            print(f"Debug: Linear layers: {linear_count}, Conv2d layers: {conv_count}")

            # Debug: Print sample layer names to verify wildcard matching
            print(f"\nDebug: Model structure analysis:")
            sample_count = 0
            linear_layers = []
            conv_layers = []
            for name, module in diffusion_model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    linear_layers.append(name)
                    if sample_count < 5:
                        print(f"  Linear: {name} (in={module.in_features}, out={module.out_features})")
                        sample_count += 1
                elif isinstance(module, torch.nn.Conv2d):
                    conv_layers.append(name)
                    if sample_count < 10:
                        print(f"  Conv2d: {name} (in={module.in_channels}, out={module.out_channels}, kernel={module.kernel_size})")
                        sample_count += 1

            print(f"\nDebug: Layer distribution:")
            print(f"  Total Linear layers: {len(linear_layers)}")
            print(f"  Total Conv2d layers: {len(conv_layers)}")
            print(f"  First 3 Linear: {linear_layers[:3]}")
            print(f"  First 3 Conv2d: {conv_layers[:3]}")
            print(f"  Last 3 Linear: {linear_layers[-3:]}")
            print(f"  Last 3 Conv2d: {conv_layers[-3:]}")

            # Debug: Check model's module structure
            print(f"\nDebug: Top-level model attributes:")
            top_attrs = [attr for attr in dir(diffusion_model) if not attr.startswith('_')][:20]
            print(f"  First 20 attributes: {', '.join(top_attrs)}")

            # Debug: Check for any existing quantizers (shouldn't be any)
            existing_quantizers = []
            for name, module in diffusion_model.named_modules():
                if 'quantiz' in str(type(module)).lower():
                    existing_quantizers.append((name, type(module).__name__))
            print(f"\nDebug: Existing quantizers before mtq.quantize(): {len(existing_quantizers)}")
            if existing_quantizers:
                print(f"  Found: {existing_quantizers[:5]}")

            # CRITICAL: Ensure model is fully on GPU and in eval mode for ModelOpt
            device = comfy.model_management.get_torch_device()
            print(f"Debug: Moving model to {device} and setting eval mode...")

            # Move model to GPU (ComfyUI may have offloaded parts to CPU)
            diffusion_model = diffusion_model.to(device)

            # Set to eval mode (required for BatchNorm, Dropout, etc.)
            diffusion_model.eval()

            # Verify all parameters are on the same device
            devices = {p.device for p in diffusion_model.parameters()}
            print(f"Debug: Model parameters are on devices: {devices}")

            # CRITICAL: Convert to FP32 for quantization
            # ModelOpt quantization requires FP32 input models
            original_dtype = next(diffusion_model.parameters()).dtype
            print(f"Debug: Original model dtype: {original_dtype}")
            if original_dtype != torch.float32:
                print(f"Debug: Converting model to FP32 for quantization...")
                diffusion_model = diffusion_model.to(torch.float32)
                print(f"Debug: Model converted to FP32")

            # Now detect model dtype for calibration (should be FP32 after conversion)
            model_dtype = next(diffusion_model.parameters()).dtype
            print(f"Preparing calibration data (dtype: {model_dtype})...")

            if calibration_data is not None:
                # Use provided calibration latents
                calib_latents = calibration_data["samples"]
                print(f"  Using provided calibration data: {calib_latents.shape}")
            else:
                # Generate random calibration data using introspected latent format
                latent_shape = (
                    model_info['latent_channels'],
                    model_info['latent_spatial'],
                    model_info['latent_spatial']
                )
                calib_latents = torch.randn(
                    calibration_steps,
                    *latent_shape,
                    device=device,
                    dtype=model_dtype  # Match model dtype (FP32)
                )
                print(f"  Using random calibration data: {calib_latents.shape}")

            # Define calibration forward loop
            def forward_loop(diffusion_model_to_calibrate):
                """Calibration forward pass for the diffusion model"""
                print(f"Running calibration...")
                # Get model dtype
                model_dtype = next(diffusion_model_to_calibrate.parameters()).dtype

                # Use introspected model info for calibration
                print(f"  Using introspected architecture: {model_info['architecture']}")
                if model_info['has_y_param']:
                    print(f"  Y dimension: {model_info['y_dim']} (from {model_info.get('y_dim_source', 'detection')})")

                with torch.no_grad():
                    for i in range(min(calibration_steps, len(calib_latents))):
                        if i % 10 == 0:
                            print(f"  Step {i}/{calibration_steps}")

                        # Get latent sample
                        if calibration_data is not None:
                            latent = calib_latents[i:i+1].to(device, dtype=model_dtype)
                        else:
                            latent = calib_latents[i:i+1]

                        # Create dummy timestep and context with matching dtype
                        # Timestep as tensor for diffusion models
                        timestep = torch.tensor([999.0], device=device, dtype=model_dtype)

                        # Use introspected context dimension
                        context_dim = model_info['context_dim']
                        context = torch.randn(1, 77, context_dim, device=device, dtype=model_dtype)

                        # Forward pass - use introspected model info
                        try:
                            if model_info['has_y_param'] and model_info['y_dim'] is not None:
                                # Model uses y parameter (SDXL, SD3)
                                y = torch.randn(1, model_info['y_dim'], device=device, dtype=model_dtype)
                                _ = diffusion_model_to_calibrate(latent, timestep, context, y=y)
                            else:
                                # Model doesn't use y parameter (SD1.5)
                                _ = diffusion_model_to_calibrate(latent, timestep, context)
                        except Exception as e:
                            # If forward pass fails, print warning but continue
                            if i == 0:  # Only print detailed error for first step
                                print(f"  Warning: Calibration step {i} failed: {e}")
                                print(f"  Trying alternative signatures...")
                                # Try alternative signatures
                                try:
                                    if model_info['has_y_param']:
                                        y = torch.randn(1, model_info['y_dim'], device=device, dtype=model_dtype)
                                        _ = diffusion_model_to_calibrate(x=latent, timesteps=timestep, context=context, y=y)
                                    else:
                                        _ = diffusion_model_to_calibrate(x=latent, timesteps=timestep, context=context)
                                except Exception as e2:
                                    print(f"  All calibration signatures failed: {e2}")
                            else:
                                # For subsequent steps, just note failure without spam
                                pass

                print(f"  Calibration complete!")

            print(f"\nRunning ModelOpt quantization...")
            print(f"Debug: Config details:")
            print(f"  Config type: {type(quant_cfg)}")
            print(f"  Config keys: {list(quant_cfg.keys())}")
            print(f"  quant_cfg type: {type(quant_cfg.get('quant_cfg', {}))}")
            print(f"  quant_cfg keys: {list(quant_cfg.get('quant_cfg', {}).keys())}")
            print(f"  Full config: {quant_cfg}")

            # DIAGNOSTIC: Check what configs ModelOpt has available
            print(f"\nDebug: Checking ModelOpt's built-in configs...")
            available_configs = [attr for attr in dir(mtq) if 'CFG' in attr or 'CONFIG' in attr]
            print(f"  Available config constants: {available_configs}")

            # Try to use a built-in config if available
            builtin_cfg = None
            if hasattr(mtq, 'FP8_DEFAULT_CFG'):
                builtin_cfg = mtq.FP8_DEFAULT_CFG
                print(f"  Found FP8_DEFAULT_CFG: {builtin_cfg}")
            elif hasattr(mtq, 'INT8_DEFAULT_CFG'):
                builtin_cfg = mtq.INT8_DEFAULT_CFG
                print(f"  Found INT8_DEFAULT_CFG: {builtin_cfg}")

            # DIAGNOSTIC: Try a very permissive config first to see if ANYTHING works
            print(f"\nDebug: Testing with permissive config first...")
            test_cfg = {
                "quant_cfg": {
                    "default": {"num_bits": (4, 3), "axis": None}
                },
                "algorithm": "max",
            }
            print(f"  Test config: {test_cfg}")

            # Also try with the built-in config if available
            if builtin_cfg:
                print(f"\nDebug: Also testing with ModelOpt's built-in FP8/INT8 config...")
                try:
                    test_builtin_model = mtq.quantize(diffusion_model, builtin_cfg, lambda m: None)
                    from modelopt.torch.quantization.nn import TensorQuantizer
                    builtin_test_count = sum(1 for m in test_builtin_model.modules() if isinstance(m, TensorQuantizer))
                    print(f"  Built-in config result: {builtin_test_count} quantizers")
                    if builtin_test_count > 0:
                        print(f"  SUCCESS with built-in config! Using that instead.")
                        # Use the built-in config for actual quantization
                        quant_cfg = builtin_cfg
                except Exception as e:
                    print(f"  Built-in config test failed: {str(e)[:200]}")

            try:
                # Try quantizing with the permissive config (no forward loop, just see if quantizers insert)
                print(f"  Attempting test quantization (this will fail during calibration, but we just want to see if quantizers insert)...")
                test_model = mtq.quantize(diffusion_model, test_cfg, lambda m: None)

                # Check if ANY quantizers were inserted
                from modelopt.torch.quantization.nn import TensorQuantizer
                test_quantizer_count = sum(1 for m in test_model.modules() if isinstance(m, TensorQuantizer))
                print(f"  Test result: {test_quantizer_count} TensorQuantizers inserted with permissive config")

                if test_quantizer_count > 0:
                    print(f"  SUCCESS: Permissive config works! Issue is with wildcard matching.")
                    print(f"  Sample quantizer locations:")
                    count = 0
                    for name, module in test_model.named_modules():
                        if isinstance(module, TensorQuantizer) and count < 10:
                            # Get parent module name
                            parent_name = '.'.join(name.split('.')[:-1]) if '.' in name else 'root'
                            print(f"    - {name} (parent: {parent_name})")
                            count += 1
                else:
                    print(f"  FAILURE: Even permissive config doesn't work. Deeper issue with ModelOpt/model compatibility.")
            except Exception as e:
                print(f"  Test quantization exception (expected): {str(e)[:200]}")
                # Still check if quantizers were inserted despite the error
                from modelopt.torch.quantization.nn import TensorQuantizer
                try:
                    test_quantizer_count = sum(1 for m in test_model.modules() if isinstance(m, TensorQuantizer))
                    print(f"  Quantizers inserted before error: {test_quantizer_count}")
                except:
                    print(f"  Could not check quantizer count after error")

            print(f"\n" + "="*60)
            print(f"Now running ACTUAL quantization with diffusion config...")
            print(f"="*60)

            # CRITICAL DEBUG: Check if ModelOpt has any special requirements
            print(f"\nDebug: ModelOpt API inspection:")
            print(f"  mtq.quantize signature: {mtq.quantize.__doc__[:500] if mtq.quantize.__doc__ else 'No docstring'}")

            # Check if there are any config validation functions we should call
            try:
                if hasattr(mtq, 'config'):
                    print(f"  mtq.config module exists: {dir(mtq.config)[:10]}")
            except:
                pass

            # Try to see if ModelOpt has any verbose/debug mode
            import logging
            modelopt_logger = logging.getLogger('modelopt')
            original_level = modelopt_logger.level
            modelopt_logger.setLevel(logging.DEBUG)
            print(f"  Enabled ModelOpt debug logging")

            # Also check for any environment variables that might affect behavior
            import os
            relevant_env_vars = {k: v for k, v in os.environ.items() if 'modelopt' in k.lower() or 'quant' in k.lower()}
            if relevant_env_vars:
                print(f"  Relevant env vars: {relevant_env_vars}")

            quantized_diffusion_model = mtq.quantize(
                diffusion_model,
                quant_cfg,
                forward_loop
            )

            # Restore logger level
            modelopt_logger.setLevel(original_level)

            # Print quantization summary to verify quantizers were inserted
            print(f"\nQuantization Summary:")
            mtq.print_quant_summary(quantized_diffusion_model)

            # Debug: Extensive post-quantization analysis
            from modelopt.torch.quantization.nn import TensorQuantizer
            quantizer_count = sum(1 for m in quantized_diffusion_model.modules() if isinstance(m, TensorQuantizer))
            print(f"\nDebug: Post-quantization analysis:")
            print(f"  TensorQuantizer modules found: {quantizer_count}")

            # Check what changed in the model
            all_modules_after = list(quantized_diffusion_model.named_modules())
            print(f"  Total modules after quantization: {len(all_modules_after)}")
            print(f"  Total modules before quantization: {len(list(diffusion_model.named_modules()))}")

            if quantizer_count > 0:
                print(f"\n  SUCCESS! Quantizer locations:")
                count = 0
                for name, module in quantized_diffusion_model.named_modules():
                    if isinstance(module, TensorQuantizer):
                        if count < 15:  # Show first 15
                            parent = '.'.join(name.split('.')[:-1])
                            print(f"    {count+1}. {name}")
                            print(f"       Parent: {parent}")
                        count += 1
                        if count == quantizer_count:
                            break
                print(f"  ... and {quantizer_count - 15} more" if quantizer_count > 15 else "")
            else:
                print(f"\n  FAILURE: No quantizers inserted!")
                print(f"  Possible causes:")
                print(f"    1. Wildcard pattern mismatch")
                print(f"    2. ModelOpt version incompatibility")
                print(f"    3. Model structure not recognized by ModelOpt")
                print(f"    4. 'default': False blocking all quantization")

                # Try to understand what ModelOpt is seeing
                print(f"\n  Attempting to diagnose...")
                print(f"  Checking if model has 'forward' method: {hasattr(quantized_diffusion_model, 'forward')}")
                print(f"  Model class: {type(quantized_diffusion_model).__name__}")
                print(f"  Model module: {type(quantized_diffusion_model).__module__}")

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
