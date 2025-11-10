# NVIDIA TensorRT Model Optimizer (ModelOpt) for ComfyUI: Complete Technical Guide

## Executive Summary

NVIDIA TensorRT Model Optimizer (ModelOpt) v0.27.0-v0.33 is an open-source model compression library with **limited support for diffusion models**. While it provides excellent INT8 quantization for SDXL and SD1.5 via NeMo integration, **FLUX, Qwen Image, and WAN 2.2 are NOT officially supported**. The library primarily focuses on LLM optimization with FP8, INT8, INT4, and NVFP4 quantization formats. For ComfyUI integration, you'll need custom loader nodes following V3 schema patterns with robust validation and error handling.

## 1. ModelOpt Current Capabilities (November 2025)

### Supported Model Architectures

**Diffusion Models (Limited Support):**
- **SDXL (Stable Diffusion XL)**: ✅ Fully supported via NeMo framework with INT8 quantization achieving ~2x speedup
- **SD1.5 (Stable Diffusion 1.5)**: ✅ Supported via NeMo with INT8 quantization
- **Stable Diffusion 3**: ✅ Mentioned in v0.15 release with INT8 support
- **FLUX.1**: ❌ **NOT officially supported** (community FP8/NF4 alternatives exist)
- **Qwen Image**: ❌ **NOT officially supported** (20B parameter model, community FP8 available)
- **WAN 2.2**: ❌ **NOT officially supported** (video generation model)

**LLM Architectures (Primary Focus):**
ModelOpt has extensive LLM support including Llama 3.1/3.2/3.3/4, DeepSeek-R1, Qwen 2.5/3, Mistral, Mixtral, Phi-3/3.5, Gemma 2, Nemotron, Arctic 2, and DBRX.

### Quantization Formats Supported

**Linux Platform:**

| Format | Precision | Requirements | Best Use Case | Quality Loss |
|--------|-----------|--------------|---------------|--------------|
| **NVFP4** | 4-bit float | Blackwell GPUs (SM 12.0) | Maximum compression on latest hardware | \<1% from FP8 |
| **FP8** | 8-bit float | Hopper/Ada GPUs (SM 8.9+) | Best quality/performance balance | \<1% |
| **INT8** | 8-bit integer | SM 7.5+ (Turing+) | Broad hardware compatibility, production-ready | 1-3% |
| **INT4 AWQ** | 4-bit integer | SM 7.5+ | Maximum VRAM savings, small-batch inference | 2-5% |
| **W4A8** | Mixed 4/8-bit | SM 8.9+ | Balanced memory and performance | Low |

**Key Details:**
- **FP8 formats**: E4M3 (better precision) and E5M2 (larger range)
- **INT4 AWQ**: Activation-Aware Weight Quantization with block-wise quantization (e.g., block size 128)
- **NVFP4**: Added January 28, 2025 when ModelOpt became open source
- **Windows platform**: Limited to INT4, FP8, INT8 via ONNX export for DirectML

### Model Types Supported

**For LLMs:**
- ✅ Base checkpoints (full pre-trained models)
- ✅ Fine-tuned models (SFT, instruction-tuned)
- ✅ LoRAs via QLoRA workflow with NF4 (NeMo integration)
- ✅ Multi-GPU trained models (tensor/pipeline parallelism)

**For Diffusion Models:**
- ✅ Base checkpoints (SDXL, SD1.5 only)
- ❌ LoRAs: NOT directly supported (must be applied at framework level post-quantization)
- ❌ ControlNets: NOT supported for quantization
- ❌ Other adapters (IP-Adapter, T2I-Adapter): NOT supported

**Critical Limitation**: ModelOpt only quantizes the UNet for diffusion models. VAE and text encoders remain FP16.

### Output Formats

**1. TensorRT-LLM Checkpoint Format:**
```python
from modelopt.torch.export import export_tensorrt_llm_checkpoint

export_tensorrt_llm_checkpoint(
    model,                              # Quantized model
    decoder_type="llama",               # Model architecture type
    dtype=torch.float16,                # Dtype for unquantized layers
    export_dir="./checkpoint",
    inference_tensor_parallel=2,        # Target TP for inference
    inference_pipeline_parallel=1       # Target PP for inference
)
```
**Output structure**: `config.json` + `rank{N}.safetensors` per GPU

**2. Unified HuggingFace Checkpoint:**
```python
from modelopt.torch.export import export_hf_checkpoint

export_hf_checkpoint(model, export_dir="./quantized_model")
tokenizer.save_pretrained("./quantized_model")
```
**Compatible with**: TensorRT-LLM v0.17.0+, vLLM v0.6.5+, SGLang (main branch since Jan 6, 2025)

**3. ONNX Format:**
Optimized ONNX models for DirectML (Windows), compatible with ONNX Runtime and Microsoft Olive.

**4. PyTorch Checkpoint:**
Native PyTorch state dict with quantization metadata when `mto.enable_huggingface_checkpointing()` is enabled.

### Current Limitations

**Diffusion Model Limitations:**
- Only SDXL and SD1.5 officially supported (no FLUX, Qwen, WAN 2.2)
- Only INT8 quantization available (no FP8 or FP4 despite LLM support)
- NeMo framework dependency (complex conversion workflow)
- No LoRA, ControlNet, or adapter quantization support
- Only UNet quantization (VAE and text encoders remain FP16)

**Platform Limitations:**
- Windows support deprecated for TensorRT-LLM (v0.18.0+)
- SBSA (ARM) has limited ONNX PTQ support
- FP8 only on SM 8.9+ GPUs (Hopper, Ada architectures)
- NVFP4 only on Blackwell architecture

**Deployment Constraints:**
- vLLM: FP8 without FP8 KV cache only, requires `quantization="modelopt"` flag
- SGLang: Must build from main branch (January 2025+)
- TensorRT-LLM version requirements vary by format (NVFP4 needs v0.17.0+)

## 2. PyTorch Requirements

### Minimum PyTorch Version

**Core Requirements:**
- **PyTorch**: >= 2.6 (recommended: latest stable)
- **Python**: >=3.10, <3.13
- **Architecture**: x86_64, aarch64 (SBSA)
- **OS**: Linux (primary), Windows via ModelOpt-Windows

**Installation:**
```bash
# Default installation (installs latest PyTorch)
pip install -U "nvidia-modelopt[all]"

# For specific CUDA version, install PyTorch first
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -U "nvidia-modelopt[all]"
```

**Version Compatibility:**
- ModelOpt v0.31+ deprecates torch<2.4
- TensorRT-LLM 1.2.0 requires ModelOpt v0.33
- ONNX Runtime 1.22+ optional for ONNX deployment

### CUDA Version Dependencies

**Minimum Requirements:**
- **CUDA**: >= 12.0
- **Compute Capability**: >= 5.3 (TensorRT requirement)
- **Recommended**: >= 8.9 for accelerated Triton kernels (~40% faster quantization)

**Special Features by Architecture:**
- **Hopper (H100, H200)**: FP8 with Transformer Engine, SM 9.0
- **Ada Lovelace (RTX 4000)**: FP8 support, SM 8.9
- **Ampere (A100, RTX 3000)**: TF32/BF16 optimizations, SM 8.0-8.6
- **Turing (T4, RTX 2000)**: Minimum support, SM 7.5
- **Blackwell (B200, GB200)**: NVFP4 support, SM 12.0

**Accelerated Quantization with Triton:**
```bash
pip install triton  # For SM 8.9+ devices
```
Automatically used for NVFP4 format when available.

**First-Time Compilation:**
ModelOpt compiles CUDA kernels on first use (takes a few minutes). Pre-compile with:
```bash
python -c "import modelopt.torch.quantization.extensions as ext; ext.precompile()"
```

### Mixed Precision Support

**Mixed precision is OPTIONAL but RECOMMENDED:**
- Works with any base dtype: FP32, FP16, BF16
- BF16 preferred over FP16 (same dynamic range as FP32, prevents overflow)
- Quantized inference typically uses: quantized weights (INT4/INT8/FP8) + BF16/FP16 activations

**Usage:**
```python
import torch
from torch.cuda.amp import autocast

# BF16 recommended (Ampere+ GPUs)
model = model.to(torch.bfloat16)

# Inference with autocast
with torch.inference_mode():
    with autocast(dtype=torch.bfloat16):
        outputs = model(inputs)
```

**Hardware Requirements:**
- BF16: Ampere or newer
- FP16: Volta or newer
- TF32: Automatically enabled on Ampere+ for matmul

### torch.compile Requirements

**Compatibility:**
- Generally compatible with PyTorch 2.0+ `torch.compile`
- Status: Some edge cases may fail (feature still maturing)
- Test thoroughly before production use

**Usage:**
```python
# Quantize first, then compile
model = mtq.quantize(model, config, forward_loop)
compiled_model = torch.compile(model, mode='max-autotune')
```

**Known Issues:**
- Some quantizer operations may not optimize well
- BF16/FP16 conversion can cause compilation errors (PyTorch issue #123830)
- Workaround: Use export modes or skip compile for problematic layers

**Recommended Optimizations:**
```python
# Backend optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True  # Ampere+

# Weight folding for faster inference
mtq.fold_weight(quantized_model)  # Note: Cannot export to ONNX after folding

# Compression for real speedup (v0.31+)
compressed_model = mtq.compress(quantized_model)  # Enables real INT8/FP8 GEMM
```

## 3. Loading Mechanisms

### Native PyTorch Loading for Quantized Models

ModelOpt uses "fake quantization" - it simulates low-precision computation within PyTorch. The quantized model remains a PyTorch model with TensorQuantizer modules inserted.

**Basic Quantization Workflow:**
```python
import modelopt.torch.quantization as mtq
import modelopt.torch.opt as mto
from transformers import AutoModelForCausalLM

# 1. Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# 2. Select quantization config
config = mtq.FP8_DEFAULT_CFG  # or INT8_DEFAULT_CFG, INT4_AWQ_CFG, NVFP4_DEFAULT_CFG

# 3. Prepare calibration dataset
from modelopt.torch.utils import dataset_utils
calib_dataloader = dataset_utils.get_dataset_dataloader(
    "cnn_dailymail",
    tokenizer,
    batch_size=1,
    num_samples=512
)

# 4. Define forward loop
def forward_loop(model):
    for data in calib_dataloader:
        model(**data)

# 5. Quantize model (in-place)
model = mtq.quantize(model, config, forward_loop)

# 6. Save quantized model
mto.save(model, "quantized_model.pt")

# 7. Load quantized model
model = mto.restore(model, "quantized_model.pt")
```

### Safetensors vs Diffusers Format

**Safetensors Format (Primary for LLMs):**
- Used by TensorRT-LLM, vLLM, SGLang deployment
- Structure:
  ```
  checkpoint_dir/
  ├── config.json              # Model config with quantization info
  ├── rank0.safetensors        # Quantized weights + scaling factors
  ├── rank1.safetensors        # (for multi-GPU models)
  └── tokenizer files
  ```

**Diffusers Format (for Diffusion Models):**
```python
from diffusers import AutoModel, NVIDIAModelOptConfig
import torch

# Quantize during loading
quantization_config = NVIDIAModelOptConfig(
    quant_type="FP8",
    quant_method="modelopt",
    modules_to_not_convert=["pos_embed.proj.weight"],  # Skip specific layers
    disable_conv_quantization=True  # Optional: disable conv layers
)

transformer = AutoModel.from_pretrained(
    "model_id",
    subfolder="transformer",
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16
)

# Save quantized model
model.save_pretrained('path/to/saved_model', safe_serialization=False)
```

### ComfyUI Model Loading Infrastructure

**Key Challenge**: ComfyUI's standard model loading assumes PyTorch .safetensors or .ckpt formats. ModelOpt quantized models require special handling.

**Integration Strategy:**
1. **Custom loader nodes** that handle ModelOpt-specific formats
2. **Wrapper classes** that adapt quantized models to ComfyUI's expected model interface
3. **Lazy loading** with caching to avoid repeated quantization overhead

**Pattern from TensorRT loader:**
```python
import folder_paths

def get_model_list():
    """Scan directory for quantized model files"""
    model_dir = folder_paths.get_folder_paths("tensorrt")
    models = []
    for root, dirs, files in os.walk(model_dir[0]):
        for file in files:
            if file.endswith(('.plan', '.engine', '.safetensors')):
                models.append(file)
    return models

class ModelOptModelLoader:
    def __init__(self):
        self.loaded_model = None
        self.model_hash = None

    def load_if_changed(self, model_path):
        new_hash = hash_file(model_path)
        if new_hash != self.model_hash:
            self.loaded_model = self._load_engine(model_path)
            self.model_hash = new_hash
        return self.loaded_model
```

### Tensor Format Requirements

**TensorQuantizer Modules:**
ModelOpt inserts TensorQuantizer modules into nn.Linear and nn.Conv layers:
- `input_quantizer`: Quantizes input activations
- `weight_quantizer`: Quantizes weights
- `output_quantizer`: Typically disabled

**Quantizer States:**
- Stored in module's `extra_state` (v0.31+)
- Includes scaling factors and calibration parameters
- Parallel state tracking for distributed models

**Special Handling:**
```python
# For distributed models
# Parallel state: data_parallel_group, tensor_parallel_group

# Loading requires same GPU count for tensor-parallel models
# Example: Model trained on 8 GPUs needs 8 GPUs for PTQ calibration
```

### Memory Requirements

**Memory Footprint:**
- Quantized models require **both half and full precision copies** during mixed precision training
- Actual memory savings only in forward activations
- Real memory reduction requires deployment to TensorRT-LLM/TensorRT/vLLM

**Memory Optimization Techniques:**

1. **Low Memory Mode (PTQ):**
```bash
python llm_ptq.py --low_memory_mode  # Reduces peak memory during PTQ
```

2. **CPU Offloading:**
```python
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

with init_empty_weights():
    model = AutoModelForCausalLM.from_pretrained("model_path")

model = load_checkpoint_and_dispatch(
    model,
    checkpoint="path/to/checkpoint",
    device_map="auto"  # Automatic CPU/GPU memory management
)
```

3. **Gradient Checkpointing:**
```python
from modelopt.torch.opt import enable_huggingface_checkpointing
enable_huggingface_checkpointing()
```

**Quantization Memory Reduction:**
- **FP8**: ~2x memory reduction (8-bit vs 16-bit)
- **INT8**: ~2x reduction with better performance
- **INT4**: ~4x reduction (4-bit vs 16-bit)
- **NVFP4**: ~4x reduction with improved quality vs INT4

## 4. ComfyUI V3 Integration

### V3 Node Schema Implementation

**Key Breaking Changes from V1/V2:**
- Inputs/outputs defined by objects instead of dictionaries
- Execution method fixed to `execute()` as class method
- Node registration uses `comfy_entrypoint()` returning `ComfyExtension`
- All nodes must inherit from `ComfyNode`
- Stateless design: class methods (receive `cls`) instead of instance methods

**V3 Node Template:**
```python
from comfy_api.latest import ComfyExtension, io

class ModelOptLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ModelOptLoader",
            display_name="ModelOpt TensorRT Loader",
            category="loaders/tensorrt",
            description="Load quantized models optimized with NVIDIA ModelOpt",
            inputs=[
                io.String.Input(
                    "model_path",
                    default="",
                    tooltip="Path to the quantized model file"
                ),
                io.Combo.Input(
                    "precision",
                    options=["fp16", "fp8", "int8", "int4"],
                    default="fp16",
                    tooltip="Quantization precision of the model"
                ),
                io.Int.Input(
                    "batch_size",
                    default=1,
                    min=1,
                    max=8,
                    display_mode=io.NumberDisplay.number
                ),
            ],
            outputs=[
                io.Model.Output(display_name="MODEL"),
                io.String.Output(display_name="model_info")
            ],
            is_output_node=False
        )

    @classmethod
    def execute(cls, model_path, precision, batch_size) -> io.NodeOutput:
        model = load_model(model_path, precision, batch_size)
        info = f"Loaded {model_path} with {precision} precision"
        return io.NodeOutput(model, info)

class MyExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [ModelOptLoader]

async def comfy_entrypoint() -> ComfyExtension:
    return MyExtension()
```

### Custom Model Loaders for Non-Standard Formats

**Pattern 1: Directory-Based Model Discovery**
```python
import folder_paths

def scan_model_directory(base_path, extensions):
    """Recursively scan for models with specific extensions"""
    models = {}
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                rel_path = os.path.relpath(os.path.join(root, file), base_path)
                models[rel_path] = os.path.join(root, file)
    return models
```

**Pattern 2: Metadata Parsing**
```python
def parse_engine_metadata(engine_path):
    """Extract model type, batch size, resolution from filename"""
    # Example: "dyn-b-1-4-2-h-512-1024-768-w-512-1024-768.engine"
    parts = os.path.basename(engine_path).split('-')
    metadata = {
        'type': parts[0],  # 'dyn' or 'stat'
        'batch_range': parse_range(parts[2:5]),
        'height_range': parse_range(parts[7:10]),
        'width_range': parse_range(parts[12:15])
    }
    return metadata
```

**Pattern 3: Lazy Loading with Caching**
```python
class ModelOptModelLoader:
    def __init__(self):
        self.loaded_model = None
        self.model_hash = None

    def load_if_changed(self, model_path):
        new_hash = hash_file(model_path)
        if new_hash != self.model_hash:
            self.loaded_model = self._load_engine(model_path)
            self.model_hash = new_hash
        return self.loaded_model
```

### Input Validation Patterns

**V3 Validation Method:**
```python
@classmethod
def validate_inputs(cls, model_path, precision):
    """Validate inputs before execution"""
    # File path validation
    if not os.path.exists(model_path):
        return f"Model file not found: {model_path}"

    # File extension validation
    valid_extensions = ['.plan', '.engine', '.onnx', '.safetensors']
    if not any(model_path.endswith(ext) for ext in valid_extensions):
        return f"Invalid file format. Expected: {', '.join(valid_extensions)}"

    # File size check
    file_size = os.path.getsize(model_path) / (1024**3)  # GB
    if file_size > 10:
        print(f"Warning: Large model file ({file_size:.2f}GB)")

    # Precision compatibility check
    if precision == "fp8":
        gpu_arch = get_compute_capability()
        if gpu_arch < 8.9:
            return (
                f"FP8 requires Compute Capability 8.9+ (Ada Lovelace or newer)\n"
                f"Your GPU: Compute Capability {gpu_arch}"
            )

    return True
```

**Dynamic Combo Options:**
```python
@classmethod
def INPUT_TYPES(cls):
    """Generate dropdown options dynamically"""
    return {
        "required": {
            "model_name": (cls.get_available_models(), {
                "tooltip": "Select a quantized model"
            }),
            "precision": (["fp16", "fp8", "int8", "int4"], {
                "default": "fp16",
                "tooltip": "Model quantization precision"
            })
        }
    }

@classmethod
def get_available_models(cls):
    """Scan and return available models"""
    model_dir = folder_paths.get_folder_paths("tensorrt")[0]
    models = [f for f in os.listdir(model_dir) if f.endswith(('.plan', '.engine'))]
    return sorted(models) if models else ["No models found"]
```

### Error Handling Patterns

**Graceful Failure with User-Friendly Messages:**
```python
@classmethod
def execute(cls, model_path, precision, batch_size):
    """Execute with comprehensive error handling"""
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at: {model_path}\n"
                f"Please ensure the file exists in the models/tensorrt directory"
            )

        try:
            model = cls._load_engine(model_path)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to load TensorRT engine:\n{str(e)}\n\n"
                f"Common causes:\n"
                f"- GPU mismatch (engine built for different GPU)\n"
                f"- TensorRT version mismatch\n"
                f"- Corrupted model file"
            )

        try:
            result = cls._run_inference(model, batch_size)
        except torch.cuda.OutOfMemoryError:
            raise RuntimeError(
                f"Out of VRAM! Model requires more memory.\n"
                f"Try:\n"
                f"- Reducing batch_size (current: {batch_size})\n"
                f"- Using lower precision (current: {precision})\n"
                f"- Closing other applications"
            )

        return io.NodeOutput(result)

    except Exception as e:
        import traceback
        print(f"ModelOpt Error: {traceback.format_exc()}")
        raise RuntimeError(
            f"ModelOpt Loader Error: {str(e)}\n"
            f"Check the console for detailed error information."
        )
```

**Progress Reporting:**
```python
from comfy.utils import ProgressBar

@classmethod
def execute(cls, model_path):
    """Execute with progress updates"""
    pbar = ProgressBar(3)

    pbar.update_absolute(0, 3, "Loading model...")
    model = load_model(model_path)

    pbar.update_absolute(1, 3, "Initializing inference engine...")
    engine = initialize_engine(model)

    pbar.update_absolute(2, 3, "Warming up...")
    warmup(engine)

    pbar.update_absolute(3, 3, "Complete!")
    return io.NodeOutput(engine)
```

### UI Helper Text Implementation

**Comprehensive Tooltips:**
```python
@classmethod
def define_schema(cls) -> io.Schema:
    return io.Schema(
        node_id="ModelOptQuantizer",
        display_name="ModelOpt Quantizer",
        description="Quantize models using NVIDIA ModelOpt for optimized inference",
        inputs=[
            io.String.Input(
                "model_path",
                default="",
                tooltip="Path to the source model file (safetensors or checkpoint)"
            ),
            io.Combo.Input(
                "quant_format",
                options=["fp8", "int8", "int4", "w4a8"],
                default="fp8",
                tooltip=(
                    "Quantization format:\n"
                    "• fp8: Best quality, requires Ada Lovelace+\n"
                    "• int8: Good balance of speed and quality\n"
                    "• int4: Maximum speed, lower quality\n"
                    "• w4a8: Weights in INT4, activations in INT8"
                )
            ),
            io.Int.Input(
                "calibration_steps",
                default=512,
                min=1,
                max=2048,
                step=64,
                tooltip=(
                    "Number of calibration samples for quantization.\n"
                    "Higher values = better accuracy but slower conversion.\n"
                    "Recommended: 512 for testing, 1024+ for production"
                )
            ),
        ],
        outputs=[
            io.Model.Output(
                display_name="Quantized Model",
                tooltip="Quantized model ready for inference"
            ),
            io.String.Output(
                display_name="Statistics",
                tooltip="Quantization statistics and metrics"
            )
        ]
    )
```

**Node-Level Documentation:**
```python
class ModelOptLoader(io.ComfyNode):
    """
    # ModelOpt TensorRT Loader

    Load models quantized with NVIDIA ModelOpt for optimized inference on NVIDIA GPUs.

    ## Features
    - Support for FP8, INT8, INT4 quantization
    - Dynamic and static batch sizes
    - Automatic GPU architecture detection

    ## Requirements
    - NVIDIA GPU with Compute Capability 8.0+
    - TensorRT 8.6+
    - For FP8: Ada Lovelace architecture (RTX 40-series)

    ## Usage
    1. Place quantized models in `models/tensorrt/`
    2. Select model from dropdown
    3. Choose precision matching your model
    4. Set batch size for your workflow
    """
```

## 5. Conversion Process

### Programmatic Conversion

**Basic PTQ Workflow:**
```python
import torch
import modelopt.torch.quantization as mtq
from transformers import AutoModelForCausalLM
from modelopt.torch.utils import dataset_utils

# 1. Load model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# 2. Select quantization config
config = mtq.FP8_DEFAULT_CFG  # or INT8_DEFAULT_CFG, INT4_AWQ_CFG, NVFP4_DEFAULT_CFG

# 3. Prepare calibration dataset
calib_dataloader = dataset_utils.get_dataset_dataloader(
    "cnn_dailymail",
    tokenizer,
    batch_size=1,
    num_samples=512
)

# 4. Define forward loop
def forward_loop(model):
    for data in calib_dataloader:
        model(**data)

# 5. Quantize model (in-place)
model = mtq.quantize(model, config, forward_loop)

# 6. Export
from modelopt.torch.export import export_hf_checkpoint
export_hf_checkpoint(model, export_dir="./quantized_model")
```

### Conversion Options and Parameters

**Available Quantization Configs:**
- `FP8_DEFAULT_CFG`: Best quality, requires SM 8.9+
- `NVFP4_DEFAULT_CFG`: Maximum compression, Blackwell only
- `INT8_DEFAULT_CFG`: Broad compatibility
- `INT4_AWQ_CFG`: Weight-only, small-batch inference
- `W4A8_AWQ_BETA_CFG`: Mixed precision

**Custom Quantization Config:**
```python
import copy

# Create custom config
CUSTOM_INT4_AWQ_CFG = copy.deepcopy(mtq.INT4_AWQ_CFG)

# Skip quantizing specific layers
CUSTOM_INT4_AWQ_CFG["quant_cfg"]["*lm_head*"] = {"enable": False}

# Apply custom config
model = mtq.quantize(model, CUSTOM_INT4_AWQ_CFG, forward_loop)
```

**AutoQuantize (Mixed Precision):**
```python
from modelopt.torch.quantization import auto_quantize

# Define constraints
constraints = {"effective_bits": 4.8}  # Target effective precision

# Specify quantization formats to search
quantization_formats = [mtq.NVFP4_DEFAULT_CFG, mtq.FP8_DEFAULT_CFG]

# Auto-search optimal per-layer quantization
model, state_dict = auto_quantize(
    model,
    constraints=constraints,
    quantization_formats=quantization_formats,
    data_loader=calib_dataloader,
    forward_step=forward_step
)
```

### Batch Conversion Best Practices

1. **Use representative calibration datasets**:
   - CNN/DailyMail for summarization models
   - Pile for general language models
   - Domain-specific data for specialized models
   - Size: 128-512 samples typical (512+ for production)

2. **Multi-GPU considerations**:
   - If model trained with tensor parallelism, calibration needs same GPU count
   - Single GPU calibration possible for most models

3. **Memory optimization**:
   - Use `accelerate` package for large models
   - Adjust `device_map` settings
   - Can use CPU for quantization (very slow, not recommended)

4. **Error prevention**:
   - Clean output directory before each run
   - Ensure CUDA/cuDNN version compatibility
   - Verify compute capability meets format requirements

### Conversion Time Estimates

**PTQ Calibration Duration (128-512 samples):**
- **Small models (7B)**: 5-15 minutes on H100
- **Medium models (13B-30B)**: 15-30 minutes
- **Large models (70B)**: 30-60 minutes
- **Very large models (405B)**: 1-2 hours
- **Diffusion models (SD3.5)**: 10-25 minutes
- **SVD/SVD-XT**: Up to 1 hour

**Engine Building Time (TensorRT compilation):**
- **First time**: 3-10 minutes for LLMs, 10-25 minutes for diffusion
- **Subsequent builds**: Much faster (engine cached)

### Resource Requirements During Conversion

**Peak VRAM Usage:**
- Calibration: Model size + 20-30% overhead
- Example: 7B FP16 model needs ~18-20GB during calibration
- **Must fit entire model in GPU memory** for efficient calibration

**Temporary Storage:**
- Calibration cache: 1-5GB
- Intermediate files: 2-10GB
- Clean output directory to avoid corruption

### Metadata Preservation

ModelOpt exports preserve:
- Model configuration (JSON)
- Tokenizer configuration (YAML)
- Quantized weights (safetensors format)
- Scaling factors and calibration parameters

**Export formats maintain full metadata:**
```python
# Hugging Face checkpoint format
export_hf_checkpoint(model, export_dir)

# TensorRT-LLM checkpoint
export_tensorrt_llm_checkpoint(model, decoder_type, dtype, export_dir)

# ONNX (for Windows/DirectML)
torch.onnx.export(model, ...)
```

## 6. Hardware Requirements

### GPU Requirements

**Minimum Compute Capability: SM 7.5+ (Turing and newer)**

**Compute Capability by Quantization Format:**
- **INT8/INT4**: SM 7.5+ (T4, RTX 2000+)
- **FP8**: SM 8.9+ (Ada Lovelace, Hopper)
- **FP4**: SM 9.0+ (Hopper) or SM 12.0 (Blackwell)

**Specific GPU Models:**

| GPU | Architecture | SM | VRAM | Best For |
|-----|--------------|-----|------|----------|
| H100 | Hopper | 9.0 | 80GB | FP8, NVFP4, large models |
| H200 | Hopper | 9.0 | 480GB variant | Massive models, highest performance |
| RTX 4090 | Ada Lovelace | 8.9 | 24GB | FP8, consumer high-end |
| RTX 4080 | Ada Lovelace | 8.9 | 16GB | FP8, consumer mid-range |
| A100 | Ampere | 8.0 | 40/80GB | INT8, datacenter |
| RTX A6000 | Ampere | 8.6 | 48GB | INT8, workstation |
| T4 | Turing | 7.5 | 16GB | INT8, minimum support |

**VRAM Requirements by Model Size (without quantization):**
- **7B parameters**: ~14GB FP16, requires 24GB GPU minimum
- **13B parameters**: ~26GB FP16, requires 40-48GB GPU
- **70B parameters**: ~140GB FP16, requires multiple GPUs or quantization
- **SDXL**: FP16 ~6GB, FP8 ~3GB

**PTQ Calibration Memory:**
Quantization process needs 20-24GB VRAM for calibration on top of model size.

### CPU Requirements

- **Core count**: Equal to or greater than number of GPUs
- **RAM**: 64GB minimum, 512GB+ recommended for multi-GPU
- CPU primarily used for data loading during calibration

### Disk Space Requirements

**Storage by Model Type:**
- **FP32**: ~4 bytes/parameter (7B = 28GB, 70B = 280GB)
- **FP16**: ~2 bytes/parameter (7B = 14GB, 70B = 140GB)
- **FP8**: ~1 byte/parameter (7B = 7GB, 70B = 70GB)
- **INT8**: ~1 byte/parameter
- **INT4**: ~0.5 bytes/parameter (7B = 3.5GB, 70B = 35GB)

**Total Workflow Storage:**
Original model + quantized checkpoint + TensorRT engine + calibration cache
- Example 7B workflow: ~40-50GB total

### Performance Differences by Hardware

**Quantization Performance Improvements (vs FP16 baseline):**

| Model | GPU | Quantization | Batch | Speedup | VRAM Reduction |
|-------|-----|--------------|-------|---------|----------------|
| Llama 3 7B | H100 | FP8 | 1 | 1.5x | ~50% |
| Llama 3 7B | H100 | FP8 | 32 | 1.8x | ~50% |
| Llama 3 7B | H100 | INT4 AWQ | 1 | 2.5x | ~75% |
| Llama 3 70B | H100 | FP8 | 32 | 2.0x | ~50% |
| Llama 3 70B | H100 | INT4 AWQ | 1 | 3.0x | ~75% |
| SD3.5 Large | RTX | FP8 | 1 | 2.3x | 40% (19GB→11GB) |

**Batch Size Impact:**
- **Small batch (≤4)**: Memory-bound, INT4 AWQ best
- **Large batch (≥16)**: Compute-bound, FP8 best

## 7. Implementation References

### Official NVIDIA TensorRT Model Optimizer Repository

**Primary Resource:** https://github.com/NVIDIA/TensorRT-Model-Optimizer

**Key Examples in Repository:**
- `examples/llm_ptq/notebooks/`: Jupyter notebooks for PTQ workflows
  - `1_FP4-FP8_PTQ_Min-Max_Calibration.ipynb`
  - `2_PTQ_AWQ_Calibration.ipynb`
  - `3_PTQ_AutoQuantization.ipynb`
- `examples/diffusers/`: Stable Diffusion quantization
- `examples/vlm_ptq/`: Vision-language models
- `examples/onnx_ptq/`: ONNX model quantization
- `examples/llm_qat/`: QAT workflows

**Python Scripts:**
- `hf_ptq.py`: HuggingFace model PTQ
- `modelopt_to_tensorrt_llm.py`: Build TensorRT-LLM engines
- `nemo_example.sh`: NeMo PTQ workflow

## Key Recommendations for Your ComfyUI Node Pack

### Critical Findings

1. **Limited Diffusion Support**: ModelOpt only officially supports SDXL and SD1.5 with INT8 quantization. **FLUX, Qwen Image, and WAN 2.2 are NOT supported.**

2. **NeMo Dependency**: Diffusion model quantization requires NeMo framework conversion, adding complexity to your workflow.

3. **No Adapter Support**: LoRAs, ControlNets, and other adapters cannot be quantized directly. They must be applied at inference time after loading the quantized base model.

4. **Format Limitations**: For diffusion models, only INT8 is supported (no FP8 or FP4 despite these formats being available for LLMs).

### Implementation Strategy

**For SDXL/SD1.5:**
- Use official ModelOpt INT8 quantization via NeMo
- Achieve ~2x speedup with minimal quality loss
- Deploy via TensorRT for optimal performance

**For FLUX/Qwen/WAN:**
- ModelOpt is not an option
- Consider community FP8 checkpoints
- Implement GGUF loader nodes instead
- Document limitations clearly to users

**For ComfyUI Integration:**
- Follow V3 schema patterns strictly
- Implement robust validation with GPU capability checks
- Provide clear error messages with troubleshooting steps
- Use progress bars for long operations
- Cache loaded models to avoid repeated loading overhead

### Version Compatibility Matrix

| Component | Minimum Version | Recommended Version |
|-----------|----------------|---------------------|
| PyTorch | 2.6+ | Latest stable |
| CUDA | 12.0+ | 12.4+ |
| ModelOpt | 0.27.0+ | 0.33+ |
| TensorRT-LLM | 0.13.0+ | 1.2.0+ |
| Python | 3.10+ | 3.11 |

**GPU Requirements:**
- FP8: Ada Lovelace (SM 8.9+) or Hopper (SM 9.0+)
- INT8: Turing (SM 7.5+) or newer
- FP4: Blackwell (SM 12.0) only

---

This comprehensive guide provides the foundation for building a robust ModelOpt integration for ComfyUI. Focus on SDXL/SD1.5 support where ModelOpt excels, and clearly document the lack of support for modern models like FLUX and Qwen Image.
