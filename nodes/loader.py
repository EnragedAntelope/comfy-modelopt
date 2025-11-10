"""
ModelOpt Model Loader Node for ComfyUI V3

This module provides a ComfyUI V3 node for loading models quantized with
NVIDIA TensorRT Model Optimizer (ModelOpt).

Supported formats:
- TensorRT engine files (.plan, .engine)
- ONNX models (.onnx)
- Safetensors with quantization metadata (.safetensors)
- PyTorch checkpoints (.pt, .pth)

Supported quantization formats:
- INT8 (Turing+, SM 7.5+)
- FP8 (Ada Lovelace+, SM 8.9+)
- INT4 (Turing+, SM 7.5+)
- NVFP4 (Blackwell, SM 12.0)
"""

import os
from typing import Optional

try:
    from comfy_api.latest import ComfyExtension, io
    COMFY_V3_AVAILABLE = True
except ImportError:
    COMFY_V3_AVAILABLE = False
    print("Warning: ComfyUI V3 API not available. Using legacy mode.")

try:
    import folder_paths
    COMFY_FOLDER_PATHS_AVAILABLE = True
except ImportError:
    COMFY_FOLDER_PATHS_AVAILABLE = False

from .utils import (
    get_gpu_compute_capability,
    get_gpu_info,
    validate_model_file,
    check_precision_compatibility,
    scan_model_directory,
    format_bytes
)


# Legacy V1/V2 compatibility
class ModelOptLoaderLegacy:
    """
    Legacy ComfyUI V1/V2 loader for ModelOpt quantized models.

    Use this if ComfyUI V3 API is not available.
    """

    @classmethod
    def INPUT_TYPES(cls):
        """Define input types for the node"""
        model_list = cls.get_available_models()

        return {
            "required": {
                "model_name": (model_list, {
                    "tooltip": "Select a ModelOpt quantized model"
                }),
                "precision": (["auto", "fp8", "fp16", "int8", "int4"], {
                    "default": "auto",
                    "tooltip": (
                        "Quantization precision:\n"
                        "• auto: Detect from model metadata\n"
                        "• fp8: Best quality, requires Ada Lovelace+\n"
                        "• int8: Good balance, Turing+\n"
                        "• int4: Maximum compression, Turing+\n"
                        "• fp16: No quantization"
                    )
                }),
                "batch_size": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 8,
                    "step": 1,
                    "tooltip": "Batch size for inference"
                }),
            },
            "optional": {
                "enable_caching": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Cache loaded models to avoid repeated loading"
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "info")
    FUNCTION = "load_model"
    CATEGORY = "loaders/modelopt"
    DESCRIPTION = "Load models quantized with NVIDIA ModelOpt"

    # Class-level cache for loaded models
    _model_cache = {}
    _model_hashes = {}

    @classmethod
    def get_available_models(cls):
        """Scan for available ModelOpt models"""
        if not COMFY_FOLDER_PATHS_AVAILABLE:
            return ["ComfyUI not detected"]

        try:
            # Try to get tensorrt model directory
            model_dirs = folder_paths.get_folder_paths("tensorrt")
            if not model_dirs:
                model_dirs = folder_paths.get_folder_paths("checkpoints")

            if not model_dirs:
                return ["No model directory found"]

            extensions = ['.plan', '.engine', '.onnx', '.safetensors', '.pt', '.pth']
            models = []

            for model_dir in model_dirs:
                found_models = scan_model_directory(model_dir, extensions)
                models.extend(found_models.keys())

            return sorted(models) if models else ["No models found"]

        except Exception as e:
            print(f"Error scanning for models: {e}")
            return ["Error scanning models"]

    def load_model(self, model_name, precision, batch_size, enable_caching=True):
        """Load a ModelOpt quantized model"""

        # Validate GPU
        gpu_info = get_gpu_info()
        if not gpu_info["available"]:
            raise RuntimeError(
                "No CUDA device available!\n"
                "ModelOpt requires an NVIDIA GPU with CUDA support."
            )

        # Validate precision compatibility
        if precision != "auto":
            is_compatible, message = check_precision_compatibility(precision)
            if not is_compatible:
                raise RuntimeError(
                    f"Precision {precision.upper()} not supported on this GPU:\n{message}"
                )

        # Get full model path
        if not COMFY_FOLDER_PATHS_AVAILABLE:
            model_path = model_name
        else:
            try:
                model_dirs = folder_paths.get_folder_paths("tensorrt")
                if not model_dirs:
                    model_dirs = folder_paths.get_folder_paths("checkpoints")

                model_path = None
                for model_dir in model_dirs:
                    candidate = os.path.join(model_dir, model_name)
                    if os.path.exists(candidate):
                        model_path = candidate
                        break

                if model_path is None:
                    raise FileNotFoundError(f"Model not found: {model_name}")

            except Exception as e:
                raise RuntimeError(f"Error locating model: {e}")

        # Validate model file
        is_valid, error = validate_model_file(model_path)
        if not is_valid:
            raise RuntimeError(f"Invalid model file:\n{error}")

        # Load model (placeholder - actual implementation would load the model)
        try:
            model = self._load_model_impl(model_path, precision, batch_size, enable_caching)

            # Generate info string
            file_size = os.path.getsize(model_path)
            info = (
                f"Model: {model_name}\n"
                f"Precision: {precision}\n"
                f"Batch Size: {batch_size}\n"
                f"File Size: {format_bytes(file_size)}\n"
                f"GPU: {gpu_info['name']} (SM {gpu_info['compute_capability']})"
            )

            return (model, info)

        except Exception as e:
            import traceback
            print(f"ModelOpt Loader Error:\n{traceback.format_exc()}")
            raise RuntimeError(
                f"Failed to load model:\n{str(e)}\n\n"
                f"Common issues:\n"
                f"• GPU mismatch (model built for different GPU)\n"
                f"• Insufficient VRAM ({gpu_info['vram_gb']:.1f}GB available)\n"
                f"• Corrupted model file\n"
                f"• Missing dependencies"
            )

    def _load_model_impl(self, model_path, precision, batch_size, enable_caching):
        """
        Actual model loading implementation (placeholder).

        In a real implementation, this would:
        1. Check cache if caching enabled
        2. Load the model based on file extension
        3. Apply precision settings
        4. Configure batch size
        5. Return the loaded model
        """
        # Placeholder: Return a dummy model object
        # In production, this would load the actual TensorRT/ONNX/PyTorch model

        print(f"Loading model from: {model_path}")
        print(f"Precision: {precision}, Batch size: {batch_size}")

        # This is where you would implement actual loading logic:
        # - For .engine/.plan: Load TensorRT engine
        # - For .onnx: Load ONNX model
        # - For .safetensors/.pt: Load PyTorch checkpoint with ModelOpt

        class DummyModel:
            def __init__(self, path, precision, batch_size):
                self.path = path
                self.precision = precision
                self.batch_size = batch_size

        return DummyModel(model_path, precision, batch_size)


# V3 Node Implementation
if COMFY_V3_AVAILABLE:
    class ModelOptLoader(io.ComfyNode):
        """
        # ModelOpt Model Loader

        Load models quantized with NVIDIA TensorRT Model Optimizer (ModelOpt).

        ## Supported Models
        - SDXL (Stable Diffusion XL) - INT8 quantization
        - SD1.5 (Stable Diffusion 1.5) - INT8 quantization
        - SD3 (Stable Diffusion 3) - INT8 quantization

        ## Supported Formats
        - TensorRT engines (.plan, .engine)
        - ONNX models (.onnx)
        - Safetensors with quantization (.safetensors)
        - PyTorch checkpoints (.pt, .pth)

        ## Requirements
        - NVIDIA GPU with Compute Capability 7.5+ (Turing or newer)
        - For FP8: Ada Lovelace (RTX 40-series) or Hopper (H100)
        - CUDA 12.0+
        - TensorRT 8.6+

        ## Usage
        1. Place quantized models in `models/tensorrt/`
        2. Select model from dropdown
        3. Choose precision (or use 'auto' to detect)
        4. Set batch size for your workflow
        """

        @classmethod
        def define_schema(cls) -> io.Schema:
            """Define the V3 node schema"""

            # Get available models
            model_list = cls._get_available_models()

            return io.Schema(
                node_id="ModelOptLoader",
                display_name="ModelOpt Model Loader",
                category="loaders/modelopt",
                description="Load models quantized with NVIDIA TensorRT Model Optimizer",
                inputs=[
                    io.Combo.Input(
                        "model_name",
                        options=model_list,
                        tooltip="Select a ModelOpt quantized model from the models directory"
                    ),
                    io.Combo.Input(
                        "precision",
                        options=["auto", "fp8", "fp16", "int8", "int4", "nvfp4"],
                        default="auto",
                        tooltip=(
                            "Quantization precision:\n\n"
                            "• auto: Detect from model metadata\n"
                            "• fp8: Best quality, requires Ada Lovelace+ (SM 8.9)\n"
                            "• int8: Good balance, Turing+ (SM 7.5)\n"
                            "• int4: Maximum compression, Turing+ (SM 7.5)\n"
                            "• nvfp4: Latest format, Blackwell only (SM 12.0)\n"
                            "• fp16: No quantization (baseline)"
                        )
                    ),
                    io.Int.Input(
                        "batch_size",
                        default=1,
                        min=1,
                        max=16,
                        tooltip=(
                            "Batch size for inference.\n"
                            "Higher values = better throughput but more VRAM.\n"
                            "Recommended: 1-4 for most workflows"
                        )
                    ),
                    io.Bool.Input(
                        "enable_caching",
                        default=True,
                        tooltip=(
                            "Cache loaded models to avoid repeated loading.\n"
                            "Improves performance but uses more RAM."
                        )
                    ),
                ],
                outputs=[
                    io.Model.Output(
                        display_name="MODEL",
                        tooltip="Loaded quantized model ready for inference"
                    ),
                    io.String.Output(
                        display_name="info",
                        tooltip="Model information and loading statistics"
                    )
                ],
                is_output_node=False
            )

        @classmethod
        def execute(cls, model_name, precision, batch_size, enable_caching) -> io.NodeOutput:
            """Execute the model loading"""

            # Reuse the legacy implementation for actual loading
            legacy_loader = ModelOptLoaderLegacy()
            model, info = legacy_loader.load_model(
                model_name=model_name,
                precision=precision,
                batch_size=batch_size,
                enable_caching=enable_caching
            )

            return io.NodeOutput(model, info)

        @classmethod
        def _get_available_models(cls):
            """Get list of available models"""
            legacy_loader = ModelOptLoaderLegacy()
            return legacy_loader.get_available_models()


    # Extension entry point
    class ModelOptExtension(ComfyExtension):
        """ComfyUI V3 Extension for ModelOpt nodes"""

        async def get_node_list(self) -> list[type[io.ComfyNode]]:
            """Return list of nodes provided by this extension"""
            return [ModelOptLoader]


    async def comfy_entrypoint() -> ComfyExtension:
        """Entry point for ComfyUI V3"""
        return ModelOptExtension()


# Export appropriate class based on ComfyUI version
if COMFY_V3_AVAILABLE:
    # V3 is available, use V3 node
    __all__ = ["ModelOptLoader", "ModelOptExtension", "comfy_entrypoint"]
else:
    # Fallback to legacy mode
    ModelOptLoader = ModelOptLoaderLegacy
    __all__ = ["ModelOptLoader"]
