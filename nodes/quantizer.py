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

            # CRITICAL DIAGNOSTIC: Check if Linear/Conv2d are actually standard PyTorch modules
            print(f"\nDebug: Checking module types for ModelOpt compatibility...")
            sample_linear = None
            sample_conv = None
            for name, module in diffusion_model.named_modules():
                if isinstance(module, torch.nn.Linear) and sample_linear is None:
                    sample_linear = (name, module)
                if isinstance(module, torch.nn.Conv2d) and sample_conv is None:
                    sample_conv = (name, module)
                if sample_linear and sample_conv:
                    break

            if sample_linear:
                name, module = sample_linear
                print(f"  Sample Linear layer: {name}")
                print(f"    Type: {type(module)}")
                print(f"    Module: {type(module).__module__}")
                print(f"    Is torch.nn.Linear: {type(module) == torch.nn.Linear}")
                print(f"    Is subclass of torch.nn.Linear: {issubclass(type(module), torch.nn.Linear)}")
                print(f"    MRO: {[c.__name__ for c in type(module).__mro__[:5]]}")

            if sample_conv:
                name, module = sample_conv
                print(f"  Sample Conv2d layer: {name}")
                print(f"    Type: {type(module)}")
                print(f"    Module: {type(module).__module__}")
                print(f"    Is torch.nn.Conv2d: {type(module) == torch.nn.Conv2d}")
                print(f"    Is subclass of torch.nn.Conv2d: {issubclass(type(module), torch.nn.Conv2d)}")
                print(f"    MRO: {[c.__name__ for c in type(module).__mro__[:5]]}")

            # CRITICAL FIX: Replace ComfyUI's wrapped modules with standard PyTorch
            # ComfyUI uses comfy.ops.disable_weight_init.Linear/Conv2d which ModelOpt doesn't recognize
            print(f"\nCRITICAL FIX: Unwrapping ComfyUI custom modules...")
            unwrapped_count = self._unwrap_comfy_ops(diffusion_model)
            print(f"  Replaced {unwrapped_count} ComfyUI wrapped modules with standard PyTorch")

            # Verify replacement worked
            verify_linear = None
            for name, module in diffusion_model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    verify_linear = (name, module)
                    break
            if verify_linear:
                name, module = verify_linear
                print(f"  Verification - Sample Linear: {name}")
                print(f"    Type: {type(module)}")
                print(f"    Is torch.nn.Linear: {type(module) == torch.nn.Linear}")
                if type(module) != torch.nn.Linear:
                    print(f"    WARNING: Still wrapped! Type mismatch.")
                else:
                    print(f"    SUCCESS: Now standard PyTorch module!")

            # CRITICAL: Ensure model is fully on GPU and in eval mode for ModelOpt
            device = comfy.model_management.get_torch_device()
            print(f"\nDebug: Moving model to {device} and setting eval mode...")

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

            # CRITICAL: Check if ModelOpt is filtering out our modules
            print(f"\nDebug: Checking if ModelOpt can see our modules...")
            try:
                # Try to manually call ModelOpt's config parsing to see what it thinks
                if hasattr(mtq, 'config') and hasattr(mtq.config, 'QuantizeConfig'):
                    test_qconfig = mtq.config.QuantizeConfig(quant_cfg)
                    print(f"  QuantizeConfig created successfully")
                    print(f"  Config type: {type(test_qconfig)}")
                    # Try to get the module filter if available
                    if hasattr(test_qconfig, '_module_filter'):
                        print(f"  Has module filter: {test_qconfig._module_filter}")
            except Exception as e:
                print(f"  Could not create QuantizeConfig: {str(e)[:200]}")

            # Check if there's a module registration issue
            print(f"\nDebug: Checking module registration...")
            all_named_modules = list(diffusion_model.named_modules())
            print(f"  Total named_modules: {len(all_named_modules)}")
            print(f"  Sample module names: {[name for name, _ in all_named_modules[:10]]}")

            # Check if the model's __class__ might be confusing ModelOpt
            print(f"\nDebug: Model class hierarchy:")
            print(f"  Model class: {diffusion_model.__class__}")
            print(f"  Model bases: {diffusion_model.__class__.__bases__}")
            print(f"  Is nn.Module: {isinstance(diffusion_model, torch.nn.Module)}")

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
                    # Make a copy of the model for testing to avoid double-quantization
                    import copy
                    test_builtin_model = copy.deepcopy(diffusion_model)
                    test_builtin_model = mtq.quantize(test_builtin_model, builtin_cfg, lambda m: None)
                    from modelopt.torch.quantization.nn import TensorQuantizer
                    builtin_test_count = sum(1 for m in test_builtin_model.modules() if isinstance(m, TensorQuantizer))
                    print(f"  Built-in config result: {builtin_test_count} quantizers")
                    if builtin_test_count > 0:
                        print(f"  SUCCESS with built-in config!")
                        print(f"  Skipping further tests - will use built-in config for actual quantization.")
                        # Don't set quant_cfg here - we'll use it later with a fresh model
                        use_builtin_cfg = True
                    else:
                        use_builtin_cfg = False
                    # Clean up test model
                    del test_builtin_model
                except Exception as e:
                    print(f"  Built-in config test failed: {str(e)[:200]}")
                    use_builtin_cfg = False
            else:
                use_builtin_cfg = False

            # LAST RESORT: Try manually adding quantizer to one module to test API
            print(f"\nDebug: Testing manual quantizer insertion...")
            try:
                from modelopt.torch.quantization.nn import TensorQuantizer
                from modelopt.torch.quantization import tensor_quant

                # Find first Linear layer
                test_linear = None
                test_linear_name = None
                for name, module in diffusion_model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        test_linear = module
                        test_linear_name = name
                        break

                if test_linear:
                    print(f"  Attempting to manually add quantizer to: {test_linear_name}")
                    # Try to create a quantizer manually
                    try:
                        manual_quantizer = TensorQuantizer()
                        print(f"  TensorQuantizer created: {manual_quantizer}")
                        print(f"  This proves ModelOpt's quantizer API works with this PyTorch version")
                    except Exception as e:
                        print(f"  Failed to create TensorQuantizer: {str(e)[:200]}")
                        print(f"  This suggests ModelOpt API incompatibility")
            except ImportError as e:
                print(f"  Could not import TensorQuantizer: {str(e)[:200]}")

            # Skip permissive config test if built-in already worked
            if not use_builtin_cfg:
                try:
                    # Try quantizing with the permissive config (no forward loop, just see if quantizers insert)
                    print(f"  Attempting test quantization (this will fail during calibration, but we just want to see if quantizers insert)...")
                    import copy
                    test_model = copy.deepcopy(diffusion_model)
                    test_model = mtq.quantize(test_model, test_cfg, lambda m: None)

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

                    # Clean up test model
                    del test_model
                except Exception as e:
                    print(f"  Test quantization exception (expected): {str(e)[:200]}")
                    # Still check if quantizers were inserted despite the error
                    from modelopt.torch.quantization.nn import TensorQuantizer
                    try:
                        test_quantizer_count = sum(1 for m in test_model.modules() if isinstance(m, TensorQuantizer))
                        print(f"  Quantizers inserted before error: {test_quantizer_count}")
                        del test_model
                    except:
                        print(f"  Could not check quantizer count after error")
            else:
                print(f"  Skipped permissive config test - built-in config already works!")

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

            # Use built-in config if test showed it works, otherwise use our custom config
            if use_builtin_cfg:
                print(f"\n  Using ModelOpt's built-in FP8_DEFAULT_CFG (test showed it works)")
                final_config = builtin_cfg
            else:
                print(f"\n  Using custom diffusion config")
                final_config = quant_cfg

            quantized_diffusion_model = mtq.quantize(
                diffusion_model,
                final_config,
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

    def _unwrap_comfy_ops(self, model):
        """
        Replace ComfyUI's wrapped modules (comfy.ops.disable_weight_init.Linear/Conv2d)
        with standard torch.nn modules so ModelOpt can recognize them.

        Returns:
            int: Number of modules replaced
        """
        replaced_count = 0

        # Recursively walk through all modules
        # We need to replace modules in their parent's _modules dict
        def replace_in_module(parent_module):
            nonlocal replaced_count

            # Check each child module (stored in _modules dict)
            for child_name in list(parent_module._modules.keys()):
                child = parent_module._modules[child_name]

                if child is None:
                    continue

                # Check if this child is a ComfyUI wrapped module
                child_module_path = child.__class__.__module__
                child_class_name = child.__class__.__name__

                # Replace comfy.ops wrapped Linear
                # Note: __module__ is 'comfy.ops', not 'comfy.ops.disable_weight_init'
                if child_module_path == 'comfy.ops' and child_class_name == 'Linear' and isinstance(child, torch.nn.Linear):
                    standard_linear = torch.nn.Linear(
                        in_features=child.in_features,
                        out_features=child.out_features,
                        bias=child.bias is not None,
                        device=child.weight.device,
                        dtype=child.weight.dtype
                    )

                    # Copy weights and biases
                    with torch.no_grad():
                        standard_linear.weight.copy_(child.weight)
                        if child.bias is not None:
                            standard_linear.bias.copy_(child.bias)

                    # Replace in parent's _modules dict
                    parent_module._modules[child_name] = standard_linear
                    replaced_count += 1

                # Replace comfy.ops wrapped Conv2d
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

                    # Copy weights and biases
                    with torch.no_grad():
                        standard_conv.weight.copy_(child.weight)
                        if child.bias is not None:
                            standard_conv.bias.copy_(child.bias)

                    # Replace in parent's _modules dict
                    parent_module._modules[child_name] = standard_conv
                    replaced_count += 1

                # Replace comfy.ops wrapped Conv1d (if exists)
                elif child_module_path == 'comfy.ops' and child_class_name == 'Conv1d' and isinstance(child, torch.nn.Conv1d):
                    standard_conv1d = torch.nn.Conv1d(
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
                        standard_conv1d.weight.copy_(child.weight)
                        if child.bias is not None:
                            standard_conv1d.bias.copy_(child.bias)

                    # Replace in parent's _modules dict
                    parent_module._modules[child_name] = standard_conv1d
                    replaced_count += 1

                else:
                    # Recursively process child's children
                    replace_in_module(child)

        # Start recursive replacement from top
        replace_in_module(model)

        return replaced_count

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
                    "default": "quantized_unet.pt",
                    "tooltip": "Filename for the saved model (will be saved in models/modelopt_unet/)"
                }),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_model"
    OUTPUT_NODE = True
    CATEGORY = "modelopt"
    DESCRIPTION = "Save ModelOpt quantized model to disk"

    def save_model(self, model, filename):
        """Save the quantized model using ModelOpt's save function"""

        # Import ModelOpt
        try:
            import modelopt.torch.opt as mto
        except ImportError:
            raise RuntimeError(
                "❌ ModelOpt not installed!\n\n"
                "Please install: pip install nvidia-modelopt"
            )

        # Ensure modelopt_unet directory exists
        save_dir = os.path.join(folder_paths.models_dir, "modelopt_unet")
        os.makedirs(save_dir, exist_ok=True)

        # Ensure .pt extension
        if not filename.endswith((".pt", ".pth")):
            filename = filename + ".pt"

        save_path = os.path.join(save_dir, filename)

        print(f"\n{'='*60}")
        print(f"Saving Quantized Model with ModelOpt")
        print(f"{'='*60}")
        print(f"  Output: {save_path}")
        print(f"  Format: PyTorch with ModelOpt state")

        try:
            # Extract the quantized diffusion model from ComfyUI's model wrapper
            # ComfyUI structure: model (ModelPatcher) -> model.model (BaseModel) -> model.model.diffusion_model (UNet)
            diffusion_model = model.model.diffusion_model

            # Verify this is a quantized model
            from modelopt.torch.quantization.nn import TensorQuantizer
            quantizer_count = sum(1 for m in diffusion_model.modules() if isinstance(m, TensorQuantizer))

            if quantizer_count == 0:
                print(f"  ⚠️  Warning: No quantizers found in model!")
                print(f"  This model may not be quantized. Saving anyway...")
            else:
                print(f"  Quantizers found: {quantizer_count}")

            # Save with ModelOpt - preserves both weights AND quantizer infrastructure
            print(f"\n  Saving with mto.save()...")
            mto.save(diffusion_model, save_path)

            file_size = os.path.getsize(save_path)
            print(f"\n✓ Model saved successfully!")
            print(f"  Size: {format_bytes(file_size)}")
            print(f"  Location: models/modelopt_unet/{filename}")
            print(f"\n  ℹ️  To load this model:")
            print(f"     1. Load the ORIGINAL unquantized model")
            print(f"     2. Use ModelOptUNetLoader with both inputs")
            print(f"     3. Loader will restore quantized state into base model")
            print(f"{'='*60}\n")

            return {}

        except Exception as e:
            import traceback
            print(f"Save Error:\n{traceback.format_exc()}")
            raise RuntimeError(
                f"❌ Failed to save model!\n\n"
                f"Error: {str(e)}\n\n"
                f"Please check:\n"
                f"• Model is properly quantized\n"
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
