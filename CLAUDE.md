# comfy-modelopt — Project Context for AI Assistants

## Project Overview

ComfyUI custom node pack integrating NVIDIA ModelOpt for quantizing diffusion models directly in ComfyUI workflows. Uses a **hybrid architecture**: ModelOpt for calibration (scale extraction) + native PyTorch quantization for real weight storage.

**Repository**: https://github.com/EnragedAntelope/comfy-modelopt
**License**: MIT
**Current Version**: v0.6.0 (Standalone loading)
**Status**: ACTIVE — Standalone loading overhaul complete

## Architecture (v0.6.0+)

```
┌─ Save Phase ─────────────────────────────────────────────────┐
│ ModelOpt calibration → Extract scales → Native quantize      │
│ → Strip ModelOpt → Save safetensors + model config metadata  │
└──────────────────────────────────────────────────────────────┘
┌─ Load Phase (STANDALONE) ────────────────────────────────────┐
│ Read safetensors → Extract model config metadata →            │
│ Reconstruct UNetModel from config → Load quantized weights →  │
│ Return ModelPatcher ready for inference                       │
│                                                               │
│ No base model checkpoint required!                            │
└──────────────────────────────────────────────────────────────┘
```

**Critical Integration Pattern**:
1. Unwrap ComfyUI modules (`comfy.ops.disable_weight_init.Linear/Conv2d` → `torch.nn.Linear/Conv2d`)
2. Run ModelOpt calibration to extract optimal scales (`amax` values)
3. Apply native quantization: replace weights with quantized storage
4. Strip all ModelOpt quantizer submodules
5. Save as safetensors with metadata + **full model config** (unet_config, model_type, latent_format)
6. Load by **reconstructing the model from stored config** — no base model needed

## Custom Nodes (4)

| Node | File | Purpose | Status |
|------|------|---------|--------|
| ModelOptQuantizeUNet | quantizer.py | Quantize UNet using ModelOpt calibration + native quantization | ✅ Working |
| ModelOptSaveQuantized | quantizer.py | Save as self-contained `.safetensors` (stores model config) | ✅ Working |
| ModelOptUNetLoader | loader.py | **Standalone** (v0.6.0+) or overlay onto base model | ✅ Working |
| ModelOptCalibrationHelper | quantizer.py | Collect calibration data from latents | ✅ Fixed |

### Key Change in v0.6.0: Standalone Loading

**Before (v0.5.x)**: `ModelOptUNetLoader` required a `base_model` input. You had to keep the original checkpoint and wire it into your workflow alongside the quantized file.

**After (v0.6.0)**: `ModelOptUNetLoader` works **standalone** — no `base_model` required. The quantized `.safetensors` file stores the full model architecture config. The loader reconstructs the UNetModel from stored config, then loads quantized weights. This means:
- Delete the original checkpoint after quantization
- No need to wire up a base model in your ComfyUI workflow
- Quantized models are truly self-contained

**Backward compatibility**: Checkpoints saved with v0.5.x still work via the overlay path (connect `base_model` as before).

## Core Modules

| Module | Purpose |
|--------|---------|
| `nodes/native_quant.py` | Native quantization/dequantization functions, QuantizedLinear/QuantizedConv2d layers |
| `nodes/quant_saveload.py` | Safetensors save/load + StoredModelConfig + model reconstruction |
| `nodes/quantizer.py` | ComfyUI node: ModelOpt calibration + native quantization |
| `nodes/loader.py` | ComfyUI node: Load quantized UNet (standalone or overlay) |
| `nodes/utils.py` | GPU detection, model introspection, validation |

## Key Technical Decisions

1. **Hybrid architecture** (v0.5.0): ModelOpt for calibration (best-in-class scale extraction) + native PyTorch for storage (real quantized weights, self-contained checkpoints)
2. **Standalone loading** (v0.6.0): Store unet_config + model_type in safetensors metadata. Reconstruct BaseModel from config on load. No base model checkpoint needed.
3. **Self-contained checkpoints**: `.safetensors` format with metadata
4. **Conv2d fallback**: MXFP8/NVFP4 require 2D tensors; Conv2d weights automatically fall back to FP8
5. **Scale extraction**: Use ModelOpt's `amax` values to compute scales: `scale = amax / max_representable`

## Model Reconstruction Architecture (v0.6.0)

```
safetensors metadata
  ├── unet_config: Dict (UNetModel.__init__ kwargs)
  ├── model_type: str (e.g. "EPS", "FLOW", "V_PREDICTION")
  ├── latent_format_cls: str (e.g. "comfy.latent_formats.SDXL")
  └── sampling_settings: Dict

reconstruct_base_model(metadata)
  → StoredModelConfig (wraps stored data)
  → comfy.model_base.BaseModel(config, model_type)
  → UNetModel(**unet_config, operations=ops)
  → Load quantized weights via _apply_quantized_weights()
  → Return ModelPatcher
```

## Development Conventions

- **Module unwrapping**: Always call `_unwrap_comfy_ops()` before ModelOpt calibration
- **Model chain**: `model (ModelPatcher) → model.model (BaseModel) → model.model.diffusion_model (UNet)`
- **Save format**: Use `.safetensors` with metadata header (NOT `.pt`)
- **Calibration**: FP32 conversion required before quantization
- **Architecture detection**: `introspect_diffusion_model()` in utils.py
- **Standalone save**: `capture_model_config()` must be called during save to capture unet_config + model_type
- **Standalone load**: `reconstruct_base_model()` + `_apply_quantized_weights()`

## Dependencies

- `nvidia-modelopt[all]>=0.43.0` — for calibration
- `torch>=2.0` with CUDA — PyTorch backend
- `safetensors>=0.3.0` — checkpoint format
- `accelerate>=0.20.0`
- `transformers>=4.30.0`
- `diffusers>=0.20.0`
- Linux: `triton` (faster quantization)
- Windows: `onnxruntime-gpu`, `torch-directml`

## Known Issues

1. **Conv2d with MXFP8/NVFP4**: Automatically falls back to FP8 since blockwise formats require 2D tensors
2. **SVD only for Linear layers**: SVD outlier absorption is only applied to 2D Linear weights (not Conv2d)
3. **Without comfy_kitchen**: Dequantization happens in PyTorch eager mode. No native CUDA acceleration until PyTorch cu130+ wheels are available.
