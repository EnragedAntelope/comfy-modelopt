"""
Utility functions for ModelOpt ComfyUI integration.

Provides helper functions for GPU capability detection, model validation,
and file handling.
"""

import os
import hashlib
import torch


def get_gpu_compute_capability():
    """
    Get the compute capability of the current GPU.

    Returns:
        float: Compute capability (e.g., 8.9 for Ada Lovelace)
        None: If no CUDA device is available

    Example:
        >>> cap = get_gpu_compute_capability()
        >>> if cap >= 8.9:
        >>>     print("FP8 quantization supported!")
    """
    if not torch.cuda.is_available():
        return None

    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)

    # Convert (major, minor) tuple to float (e.g., (8, 9) -> 8.9)
    return float(f"{capability[0]}.{capability[1]}")


def get_gpu_info():
    """
    Get detailed information about the current GPU.

    Returns:
        dict: GPU information including name, compute capability, and VRAM

    Example:
        >>> info = get_gpu_info()
        >>> print(f"GPU: {info['name']}, VRAM: {info['vram_gb']:.1f}GB")
    """
    if not torch.cuda.is_available():
        return {
            "available": False,
            "name": "No CUDA device",
            "compute_capability": None,
            "vram_gb": 0
        }

    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    capability = get_gpu_compute_capability()

    return {
        "available": True,
        "name": props.name,
        "compute_capability": capability,
        "vram_gb": props.total_memory / (1024**3),
        "device_id": device
    }


def validate_model_file(model_path, valid_extensions=None):
    """
    Validate that a model file exists and has a valid extension.

    Args:
        model_path (str): Path to the model file
        valid_extensions (list): List of valid file extensions (default: common formats)

    Returns:
        tuple: (is_valid: bool, error_message: str or None)

    Example:
        >>> is_valid, error = validate_model_file("model.safetensors")
        >>> if not is_valid:
        >>>     print(f"Error: {error}")
    """
    if valid_extensions is None:
        valid_extensions = ['.plan', '.engine', '.onnx', '.safetensors', '.pt', '.pth']

    # Check if file exists
    if not os.path.exists(model_path):
        return False, f"Model file not found: {model_path}"

    # Check if it's a file (not a directory)
    if not os.path.isfile(model_path):
        return False, f"Path is not a file: {model_path}"

    # Check file extension
    file_ext = os.path.splitext(model_path)[1].lower()
    if file_ext not in valid_extensions:
        return False, f"Invalid file format '{file_ext}'. Expected: {', '.join(valid_extensions)}"

    # Check file size (warn if too large or empty)
    file_size = os.path.getsize(model_path)
    if file_size == 0:
        return False, "Model file is empty (0 bytes)"

    if file_size > 20 * 1024**3:  # 20GB
        print(f"Warning: Large model file ({file_size / (1024**3):.2f}GB)")

    return True, None


def hash_file(file_path, chunk_size=8192):
    """
    Calculate MD5 hash of a file for caching purposes.

    Args:
        file_path (str): Path to the file
        chunk_size (int): Chunk size for reading file (default: 8192 bytes)

    Returns:
        str: MD5 hash of the file

    Example:
        >>> hash1 = hash_file("model.safetensors")
        >>> hash2 = hash_file("model.safetensors")
        >>> assert hash1 == hash2  # Same file = same hash
    """
    hasher = hashlib.md5()

    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)

    return hasher.hexdigest()


def check_precision_compatibility(precision, compute_capability=None):
    """
    Check if a quantization precision is compatible with the current GPU.

    Args:
        precision (str): Quantization precision ('fp8', 'int8', 'int4', 'nvfp4')
        compute_capability (float): GPU compute capability (auto-detected if None)

    Returns:
        tuple: (is_compatible: bool, message: str)

    Example:
        >>> compat, msg = check_precision_compatibility('fp8')
        >>> if not compat:
        >>>     print(f"Incompatible: {msg}")
    """
    if compute_capability is None:
        compute_capability = get_gpu_compute_capability()

    if compute_capability is None:
        return False, "No CUDA device available"

    precision = precision.lower()

    requirements = {
        'fp8': (8.9, "Ada Lovelace (RTX 40-series) or Hopper (H100)"),
        'int8': (7.5, "Turing (RTX 20-series, T4) or newer"),
        'int4': (7.5, "Turing (RTX 20-series, T4) or newer"),
        'nvfp4': (12.0, "Blackwell (B200, GB200)"),
        'fp16': (0.0, "Any CUDA device"),
        'fp32': (0.0, "Any CUDA device"),
    }

    if precision not in requirements:
        return False, f"Unknown precision format: {precision}"

    required_cap, gpu_name = requirements[precision]

    if compute_capability < required_cap:
        return False, (
            f"{precision.upper()} requires Compute Capability {required_cap}+ ({gpu_name})\n"
            f"Your GPU: Compute Capability {compute_capability}"
        )

    return True, f"{precision.upper()} is supported on this GPU (Compute Capability {compute_capability})"


def format_bytes(bytes_value):
    """
    Format bytes into human-readable string.

    Args:
        bytes_value (int): Number of bytes

    Returns:
        str: Formatted string (e.g., "1.5 GB")

    Example:
        >>> print(format_bytes(1536 * 1024**2))
        1.50 GB
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def get_model_size_estimate(num_parameters, precision='fp16'):
    """
    Estimate model size based on parameter count and precision.

    Args:
        num_parameters (int): Number of model parameters (e.g., 7B = 7_000_000_000)
        precision (str): Model precision ('fp32', 'fp16', 'fp8', 'int8', 'int4')

    Returns:
        str: Estimated model size in human-readable format

    Example:
        >>> size = get_model_size_estimate(7_000_000_000, 'fp8')
        >>> print(f"Estimated size: {size}")
        Estimated size: 6.52 GB
    """
    bytes_per_param = {
        'fp32': 4,
        'fp16': 2,
        'bf16': 2,
        'fp8': 1,
        'int8': 1,
        'int4': 0.5,
        'nvfp4': 0.5,
    }

    precision = precision.lower()
    if precision not in bytes_per_param:
        precision = 'fp16'  # Default to fp16

    total_bytes = num_parameters * bytes_per_param[precision]
    return format_bytes(total_bytes)


def scan_model_directory(base_path, extensions):
    """
    Recursively scan directory for models with specific extensions.

    Args:
        base_path (str): Base directory to scan
        extensions (list): List of file extensions to match (e.g., ['.engine', '.plan'])

    Returns:
        dict: Dictionary mapping relative paths to absolute paths

    Example:
        >>> models = scan_model_directory('/models/tensorrt', ['.engine', '.plan'])
        >>> for name, path in models.items():
        >>>     print(f"{name}: {path}")
    """
    models = {}

    if not os.path.exists(base_path):
        return models

    for root, dirs, files in os.walk(base_path):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, base_path)
                models[rel_path] = abs_path

    return models


def introspect_diffusion_model(model):
    """
    Comprehensive introspection of a diffusion model to detect architecture details.

    This function analyzes the model structure to automatically determine:
    - Model architecture type (SDXL, SD1.5, SD3, FLUX, etc.)
    - Y dimension (pooled embeddings size) for class-conditional models
    - Context dimension for text conditioning
    - Latent channels and recommended spatial dimensions
    - Forward pass signature

    Args:
        model: The UNet/diffusion model to introspect

    Returns:
        dict: Model architecture information

    Example:
        >>> info = introspect_diffusion_model(unet)
        >>> print(f"Architecture: {info['architecture']}")
        >>> print(f"Y dimension: {info['y_dim']}")
    """
    info = {
        "architecture": "unknown",
        "y_dim": None,
        "context_dim": 768,
        "latent_channels": 4,
        "latent_spatial": 128,
        "has_y_param": False,
        "model_channels": None,
        "num_heads": None,
        "detected_attributes": [],  # Track what we actually found
    }

    # DEBUG: List all top-level attributes to help troubleshoot
    all_attrs = [attr for attr in dir(model) if not attr.startswith('_')]
    info["all_model_attributes"] = all_attrs

    # Check for specific important attributes
    important_attrs = ['adm_in_channels', 'y_dim', 'label_emb', 'time_embed', 'model_channels',
                       'context_dim', 'in_channels', 'conv_in', 'num_heads', 'num_res_blocks']
    found_attrs = {attr: hasattr(model, attr) for attr in important_attrs}
    info["important_attributes_found"] = found_attrs

    # Detect y dimension (pooled embeddings) from various attributes
    y_dim_candidates = []

    # Method 1: Check adm_in_channels (SDXL attribute)
    if hasattr(model, 'adm_in_channels'):
        y_dim_candidates.append(("adm_in_channels", model.adm_in_channels))

    # Method 2: Check y_dim attribute
    if hasattr(model, 'y_dim'):
        y_dim_candidates.append(("y_dim", model.y_dim))

    # Method 3: Check label_emb layer dimensions
    if hasattr(model, 'label_emb'):
        label_emb = model.label_emb
        # Try to get input dimension
        if hasattr(label_emb, 'in_features'):
            y_dim_candidates.append(("label_emb.in_features", label_emb.in_features))
        elif hasattr(label_emb, '0') and hasattr(label_emb[0], 'in_features'):
            # Sequential module - check first layer
            y_dim_candidates.append(("label_emb[0].in_features", label_emb[0].in_features))
        elif isinstance(label_emb, torch.nn.Sequential) and len(label_emb) > 0:
            first_layer = label_emb[0]
            if hasattr(first_layer, 'in_features'):
                y_dim_candidates.append(("label_emb[0].in_features", first_layer.in_features))

    # Method 4: Check time_embed layer (sometimes contains adm projection)
    if hasattr(model, 'time_embed'):
        time_embed = model.time_embed
        # Look for layers that might be the adm projection
        if isinstance(time_embed, torch.nn.Sequential):
            for i, layer in enumerate(time_embed):
                if isinstance(layer, torch.nn.Linear):
                    # Check if this might be processing concatenated time + y
                    # SDXL concatenates time_emb (1280) + y (2816) = 4096 input
                    if hasattr(layer, 'in_features'):
                        in_feat = layer.in_features
                        # Common patterns:
                        # SDXL: 4096 input (1280 time + 2816 y) -> y_dim = 2816
                        # SD1.5: 1280 input (no y)
                        if in_feat > 2000:  # Likely has y concatenated
                            # Try to infer y_dim
                            # Usually time_embed_dim + y_dim = in_features
                            if hasattr(model, 'model_channels'):
                                time_embed_dim = model.model_channels * 4
                                potential_y_dim = in_feat - time_embed_dim
                                if potential_y_dim > 0:
                                    y_dim_candidates.append((f"inferred_from_time_embed[{i}]", potential_y_dim))

    # Select most reliable y_dim
    if y_dim_candidates:
        # Prefer explicit attributes over inferred values
        priority_order = ["adm_in_channels", "y_dim", "label_emb"]
        for priority_name in priority_order:
            for source, value in y_dim_candidates:
                if priority_name in source:
                    info["y_dim"] = value
                    info["y_dim_source"] = source
                    info["has_y_param"] = True
                    break
            if info["y_dim"] is not None:
                break

        # If still None, use first candidate
        if info["y_dim"] is None and y_dim_candidates:
            info["y_dim"] = y_dim_candidates[0][1]
            info["y_dim_source"] = y_dim_candidates[0][0]
            info["has_y_param"] = True

    # Detect context dimension
    if hasattr(model, 'context_dim'):
        info["context_dim"] = model.context_dim
    elif hasattr(model, 'transformer_depth'):
        # Might be a transformer-based model
        info["context_dim"] = 768  # Common default

    # Detect model channels (base channel count)
    if hasattr(model, 'model_channels'):
        info["model_channels"] = model.model_channels

    # Detect number of attention heads
    if hasattr(model, 'num_heads'):
        info["num_heads"] = model.num_heads

    # Try to determine architecture based on detected features
    if info["y_dim"] == 2816:
        info["architecture"] = "SDXL-like"
    elif info["y_dim"] == 1280:
        info["architecture"] = "SDXL-compact"
    elif info["y_dim"] is None and info["context_dim"] == 768:
        info["architecture"] = "SD1.5-like"
    elif info["context_dim"] == 4096:
        info["architecture"] = "SD3-like"

    # Detect latent format
    # Check for in_channels attribute
    if hasattr(model, 'in_channels'):
        info["latent_channels"] = model.in_channels
    elif hasattr(model, 'conv_in') and hasattr(model.conv_in, 'in_channels'):
        info["latent_channels"] = model.conv_in.in_channels

    # Infer recommended spatial dimensions based on model size
    if info["latent_channels"] == 16:
        info["latent_spatial"] = 128  # SD3 uses 16 channels
    elif info["latent_channels"] == 4:
        info["latent_spatial"] = 128  # SDXL/SD1.5 use 4 channels

    # Count parameters to help identify model
    param_count = sum(p.numel() for p in model.parameters())
    info["param_count"] = param_count
    info["param_count_billions"] = param_count / 1e9

    # Use parameter count as additional signal
    # SDXL: ~2.5-2.6B parameters
    # SD1.5: ~900M parameters
    # SD3: ~2B parameters
    if param_count > 2e9 and info["y_dim"] is None:
        # Likely SDXL but failed to detect y_dim
        # Try to infer y_dim = 2816 for SDXL
        info["y_dim"] = 2816
        info["y_dim_source"] = "inferred_from_param_count_SDXL"
        info["has_y_param"] = True
        info["architecture"] = "SDXL-like (inferred from 2.5B params)"

    # FINAL CHECK: Test forward pass to definitively determine if y is needed
    # This is the most reliable method
    if info["y_dim"] is None or not info["has_y_param"]:
        print(f"  Attempting test forward pass to detect y parameter requirement...")
        try:
            with torch.no_grad():
                # Create dummy inputs
                device = next(model.parameters()).device
                dtype = next(model.parameters()).dtype

                test_latent = torch.randn(1, info["latent_channels"], 64, 64, device=device, dtype=dtype)
                test_timestep = torch.tensor([999.0], device=device, dtype=dtype)
                test_context = torch.randn(1, 77, info["context_dim"], device=device, dtype=dtype)

                # Try without y first
                try:
                    _ = model(test_latent, test_timestep, test_context)
                    # Success without y - doesn't need it
                    info["has_y_param"] = False
                    info["detected_attributes"].append("test_forward_pass_no_y_SUCCESS")
                except TypeError as e:
                    error_str = str(e)
                    if "must specify y" in error_str or "class-conditional" in error_str:
                        # Needs y parameter!
                        info["has_y_param"] = True
                        # Try to infer y_dim
                        if info["y_dim"] is None:
                            # Try common SDXL value
                            info["y_dim"] = 2816
                            info["y_dim_source"] = "inferred_from_forward_pass_error"
                        info["detected_attributes"].append("test_forward_pass_NEEDS_Y")
                    elif "unexpected keyword argument" in error_str:
                        # Model uses different signature
                        info["detected_attributes"].append(f"test_forward_pass_different_signature: {error_str}")
                except Exception as e:
                    info["detected_attributes"].append(f"test_forward_pass_failed: {str(e)[:100]}")
        except Exception as e:
            info["detected_attributes"].append(f"test_forward_pass_setup_failed: {str(e)[:100]}")

    # Final architecture determination
    if info["y_dim"] == 2816 or info["param_count_billions"] > 2.0:
        info["architecture"] = "SDXL-like"
    elif info["y_dim"] == 1280:
        info["architecture"] = "SDXL-compact"
    elif not info["has_y_param"] and info["context_dim"] == 768:
        info["architecture"] = "SD1.5-like"
    elif info["context_dim"] == 4096:
        info["architecture"] = "SD3-like"

    return info
