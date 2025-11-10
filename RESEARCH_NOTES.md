# ModelOpt ComfyUI Integration - R&D Notes

**Purpose**: Track technical findings, known issues, and solutions for future development without relearning.

**Last Updated**: 2025-01-10

---

## ✅ DEFINITIVELY LEARNED

### Model Introspection (WORKS)

**Context Dimension Detection** - SOLVED ✓
- **Problem**: Defaulted to 768, but SDXL needs 2048
- **Solution**: Inspect cross-attention layers (`attn2`, `to_k`, `to_q`) and extract `in_features`
- **Implementation**: `nodes/utils.py:381-422`
- **Result**: Correctly detects 2048 from `input_blocks.4.1.transformer_blocks.0.attn2.to_k`

**Y Parameter Detection** - SOLVED ✓
- **Problem**: Class-conditional models (SDXL) require y parameter (pooled embeddings)
- **Solution**: Multi-method detection:
  1. Check `adm_in_channels` attribute (most reliable)
  2. Check parameter count (>2B = SDXL = y_dim 2816)
  3. Test forward pass and catch "must specify y" error
- **Implementation**: `nodes/utils.py:313-473`
- **Result**: Correctly detects y_dim=2816 for SDXL

**Architecture Identification** - WORKS ✓
- Parameter count: 2.57B → SDXL-like
- Y dimension: 2816 → SDXL
- Context dimension: 2048 → SDXL
- Latent format: 4x128x128 → SD/SDXL

### ModelOpt Configuration (TESTED)

**Config Format** - CONFIRMED ✓
- Diffusion models use simpler configs than LLMs
- No LLM-specific patterns (`*lm_head*`, `*mlp.gate.*`, `*router*`)
- Correct format for FP8:
```python
{
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
```

**Config Testing Results** - ALL FAILED ✓
- Built-in `FP8_DEFAULT_CFG`: **0 quantizers**
- Permissive config (`default` enabled): **0 quantizers**
- Diffusion-specific config: **0 quantizers**
- **Conclusion**: NOT a config/wildcard problem

### ModelOpt API Compatibility (TESTED)

**TensorQuantizer Creation** - WORKS ✓
```
TensorQuantizer created: TensorQuantizer(8 bit fake per-tensor amax=dynamic calibrator=MaxCalibrator quant)
```
- ModelOpt v0.37.0 API works with PyTorch version
- Can manually create quantizers
- Problem is with `mtq.quantize()` not recognizing modules

---

## 🔴 CRITICAL ISSUE IDENTIFIED

### **ComfyUI Custom Module Wrappers**

**Root Cause** - DEFINITIVELY IDENTIFIED ✓

ComfyUI uses custom wrapped modules instead of standard PyTorch:
```
Standard PyTorch:        torch.nn.Linear, torch.nn.Conv2d
ComfyUI Wrapped:         comfy.ops.disable_weight_init.Linear
                         comfy.ops.disable_weight_init.Conv2d
```

**Evidence**:
```
Sample Linear layer: time_embed.0
  Type: <class 'comfy.ops.disable_weight_init.Linear'>
  Is torch.nn.Linear: False  ❌
  Is subclass of torch.nn.Linear: True  ✓
  MRO: ['Linear', 'Linear', 'Module', 'CastWeightBiasOp', 'object']
```

**Why This Breaks ModelOpt**:
- ModelOpt's internal logic checks: `type(module) == torch.nn.Linear`
- ComfyUI's modules fail this check (wrong type)
- ModelOpt uses `isinstance()` checks, which WOULD work
- But ModelOpt has additional type filtering that rejects wrapped modules

**Test Results**:
- 743 Linear layers detected → ALL wrapped
- 51 Conv2d layers detected → ALL wrapped
- 0 quantizers inserted → ModelOpt can't see them

**Model Details**:
- Class: `comfy.ldm.modules.diffusionmodules.openaimodel.UNetModel`
- Module: `comfy.ops.disable_weight_init`
- Properly inherits from `torch.nn.Module`

---

## 🛠️ POTENTIAL SOLUTIONS

### Option 1: Replace Wrapped Modules with Standard PyTorch (RECOMMENDED)

**Approach**: Before quantization, recursively replace ComfyUI's wrapped modules with standard torch.nn versions.

**Pros**:
- Clean solution - makes model look like standard PyTorch to ModelOpt
- Doesn't modify ModelOpt internals
- Should work with all ModelOpt versions

**Cons**:
- Need to preserve weights during replacement
- May lose ComfyUI-specific optimizations (weight init disabling)
- Need to handle module replacement carefully

**Implementation Strategy**:
```python
def unwrap_comfy_ops(model):
    """Replace ComfyUI wrapped modules with standard PyTorch"""
    for name, module in model.named_modules():
        if type(module).__module__ == 'comfy.ops.disable_weight_init':
            if 'Linear' in type(module).__name__:
                # Create standard Linear with same weights
                standard = torch.nn.Linear(...)
                standard.weight.data = module.weight.data
                # Replace in parent
            elif 'Conv2d' in type(module).__name__:
                # Similar for Conv2d
```

### Option 2: Monkeypatch ModelOpt's Module Filtering

**Approach**: Override ModelOpt's internal type checking to accept subclasses.

**Pros**:
- Minimal changes to model
- Preserves ComfyUI's module wrapping

**Cons**:
- Fragile - depends on ModelOpt internals
- May break with ModelOpt updates
- Harder to maintain

### Option 3: Convert to HuggingFace Diffusers UNet

**Approach**: Convert ComfyUI's UNet to HuggingFace format before quantization.

**Pros**:
- Guaranteed compatibility with ModelOpt (designed for HF models)
- Can use NVIDIA's official examples directly

**Cons**:
- Complex conversion process
- May lose ComfyUI-specific features
- Requires maintaining conversion logic

---

## 📊 HARDWARE/SOFTWARE ENVIRONMENT

**Confirmed Working**:
- GPU: NVIDIA GeForce RTX 5090 (SM 12.0, Blackwell)
- VRAM: 31.8GB
- ModelOpt: v0.37.0
- Python: 3.10-3.12 (per requirements)
- PyTorch: CUDA-enabled (version TBD)
- ComfyUI: Custom UNet implementation

**GPU Capabilities**:
- FP8: Supported (SM 12.0 > 8.9 required)
- INT8: Supported
- NVFP4: Supported (SM 12.0 ≥ 12.0 required)

---

## 🔍 TECHNICAL DETAILS

### ModelOpt Quantization Process

1. **Module Discovery**: ModelOpt scans model with `named_modules()`
2. **Type Checking**: Filters for quantizable types (Linear, Conv2d)
   - **Issue**: Uses exact type match, not isinstance()
3. **Wildcard Matching**: Applies config patterns to module names
4. **Quantizer Insertion**: Wraps modules with TensorQuantizer
5. **Calibration**: Runs forward passes to collect statistics

**Where ComfyUI Breaks**: Step 2 (type checking rejects wrapped modules)

### ComfyUI's Module Wrapping

**Purpose**: `disable_weight_init` optimizes memory by preventing unnecessary weight initialization.

**Implementation**: Subclasses of torch.nn with custom `__init__` logic.

**Side Effect**: Breaks tools expecting exact `torch.nn.Linear` type.

---

## ⚠️ KNOWN LIMITATIONS

### Current Implementation

1. **Cannot Quantize ComfyUI Models** ❌
   - Root cause: Module type mismatch
   - Affects: ALL ComfyUI-loaded models
   - Status: CRITICAL BLOCKER

2. **No Verification of Quantization Quality**
   - Even if quantization works, no validation that output quality is acceptable
   - Need to add: PSNR, SSIM, or visual comparison tests

3. **No Support for Non-UNet Components**
   - VAE: Not quantizable (per ModelOpt design)
   - CLIP: Not quantizable
   - Only UNet backbone can be quantized

### ModelOpt Limitations (By Design)

1. **Requires FP32 Input**
   - Must convert FP16 models to FP32 before quantization
   - Increases VRAM temporarily

2. **Calibration Required**
   - Need forward passes with representative data
   - Currently using random data (suboptimal)

3. **Format-Specific GPU Requirements**
   - FP8: Ada Lovelace+ (SM 8.9+)
   - NVFP4: Blackwell+ (SM 12.0+)

---

## 🎯 NEXT STEPS

### Immediate (To Get Working)

1. **Implement Module Unwrapping** (Option 1) - RECOMMENDED
   - Write `unwrap_comfy_ops()` function
   - Replace wrapped Linear/Conv2d with standard versions
   - Preserve weights and biases
   - Test with SDXL model

2. **Verify Quantization Success**
   - Check for >0 quantizers inserted
   - Verify model still runs
   - Test image generation quality

3. **Add Quality Validation**
   - Generate test images before/after quantization
   - Compare quality metrics
   - Document any quality degradation

### Future Improvements

1. **Better Calibration Data**
   - Use real prompts instead of random data
   - Implement CalibrationHelper properly
   - Collect diverse dataset

2. **Support More Architectures**
   - SD1.5 testing
   - SD3 support
   - FLUX support (if compatible)

3. **Performance Benchmarking**
   - Measure actual speedup vs unquantized
   - VRAM usage comparison
   - Quality vs performance tradeoffs

4. **Error Recovery**
   - Graceful fallback if quantization fails
   - Better error messages for users
   - Automatic compatibility detection

---

## 📝 DEVELOPMENT LOG

### v0.3.0 (2025-11-10) - CRITICAL FIX ATTEMPT (FAILED)
- **ATTEMPTED FIX**: Module unwrapping for ModelOpt compatibility
- Added `_unwrap_comfy_ops()` to replace ComfyUI wrapped modules
- **BUG IDENTIFIED**: Wrong module path check (`'comfy.ops.disable_weight_init'` vs `'comfy.ops'`)
- Function replaced 0 modules due to incorrect string comparison
- **Status**: BUG FIXED, READY FOR RE-TEST

### v0.2.2 (2025-11-10)
- Fixed context dimension detection (2048 for SDXL)
- Fixed quantization config (diffusion-specific vs LLM)
- Added comprehensive diagnostics
- **IDENTIFIED ROOT CAUSE**: ComfyUI module wrapping

### v0.2.1
- Enhanced model introspection (y parameter, param count)
- Added test forward pass for architecture detection

### v0.2.0
- Initial diffusion model introspection
- Architecture identification (SDXL/SD1.5/SD3)

### v0.1.5
- Fixed y parameter handling for SDXL
- Fixed device management and dtype conversion

### v0.1.0-0.1.2
- Initial implementation
- Basic node structure
- ComfyUI Manager integration

---

## 🔗 REFERENCES

### NVIDIA Documentation
- [TensorRT Model Optimizer Examples](https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/diffusers/quantization)
- [Diffusion Quantization Config](https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/examples/diffusers/quantization/config.py)
- [ModelOpt API Docs](https://nvidia.github.io/TensorRT-Model-Optimizer/)

### ComfyUI
- [ComfyUI Repo](https://github.com/comfyanonymous/ComfyUI)
- Module: `comfy.ops.disable_weight_init`
- Model: `comfy.ldm.modules.diffusionmodules.openaimodel.UNetModel`

### Related Issues
- ModelOpt expects exact torch.nn types
- ComfyUI optimizes with custom wrappers
- Fundamental incompatibility requiring adapter layer

---

**Status**: ROOT CAUSE IDENTIFIED - Ready to implement fix (module unwrapping)
