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
