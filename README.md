# NVIDIA ModelOpt for ComfyUI

Quantize Stable Diffusion UNets directly in ComfyUI workflows using NVIDIA ModelOpt for calibration and native PyTorch quantization for real weight storage. Reduces model size by ~50-75% with minimal quality loss.

**Status**: Active development — Native quantization rewrite complete. Self-contained safetensors checkpoints.

---

## What It Does

This custom node pack adds quantization nodes to ComfyUI:

- **Quantize** a loaded UNet to INT8, FP8, MXFP8, INT4, or NVFP4 (Blackwell only)
- **Save** the quantized UNet as a self-contained `.safetensors` file
- **Load** a saved quantized UNet back for inference
- **No base model required** — checkpoints are fully self-contained

Only the UNet/diffusion model is quantized. VAE and text encoders are loaded separately via standard ComfyUI nodes.

---

## Hardware Support

| Format | GPU Required | Notes |
|--------|-------------|-------|
| INT8 | Turing+ (RTX 20-series+) | Best compatibility |
| FP8 | Ada Lovelace+ (RTX 40-series+) | Good quality/speed balance |
| MXFP8 | Blackwell (RTX 50-series+) | **Blockwise FP8 — recommended for RTX 5090** |
| INT4 | Ampere+ (RTX 30-series+) | Maximum compression, experimental |
| NVFP4 | Blackwell (RTX 50-series+) | Native 4-bit float, highest compression |

---

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.6+ with CUDA 12.8+ (12.8 required for RTX 50-series)
- An NVIDIA GPU (see table above)

### Step 1: Install PyTorch

Choose the command for your CUDA version:

```bash
# CUDA 12.8 (recommended for RTX 50-series)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.6 (RTX 40-series and below)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

### Step 2: Install ModelOpt

```bash
pip install nvidia-modelopt[all]>=0.43.0
```

### Step 3: Install This Custom Node Pack

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/EnragedAntelope/comfy-modelopt.git
```

Or download and extract the ZIP into `ComfyUI/custom_nodes/comfy-modelopt/`.

### Windows Users

Also install Windows-specific helpers:

```bash
pip install triton-windows>=3.6.0 ninja>=1.11.0
```

### ComfyUI Manager

If you use ComfyUI Manager, click **"Install Custom Nodes"** and search for `comfy-modelopt`.

---

## Basic Workflow

### Quantize and Save

1. Add a **Checkpoint Loader** node (standard ComfyUI)
2. Connect it to **ModelOpt Quantize UNet**
   - Select precision: `mxfp8` (recommended for RTX 50-series), `fp8`, `int8`, `int4`, or `nvfp4`
   - Set calibration steps: 32 (default), higher = slightly better quality
   - Select algorithm: `max` (fastest), `mse` (better quality), or `awq_lite` (best quality, slowest)
   - Enable SVD: Turn on `use_svd` for better 4-bit quality (INT4/NVFP4 recommended)
3. Connect **ModelOpt Quantize UNet** output to **ModelOpt Save Quantized**
   - Filename: `my_model_fp8`
4. Queue prompt. The quantized model is saved to `ComfyUI/models/modelopt_unet/`.

### Load and Generate

1. Add a **Checkpoint Loader** with the **original unquantized model**
2. Connect it to **ModelOpt UNet Loader**
   - Select the saved `.safetensors` file from the dropdown
3. Connect **ModelOpt UNet Loader** output to a **KSampler**
4. Add VAE and CLIP loaders as usual. Queue prompt.

> **Note**: The loader overlays quantized weights onto the base model. The base model provides the architecture; the `.safetensors` file provides the quantized weights. Both are needed for loading.

---

## Nodes Reference

| Node | What It Does |
|------|-------------|
| **ModelOpt Quantize UNet** | Quantizes the UNet portion of a loaded model using real native quantization |
| **ModelOpt Save Quantized** | Saves the quantized UNet as a self-contained `.safetensors` file |
| **ModelOpt UNet Loader** | Loads a saved quantized UNet by overlaying quantized weights onto a base model |
| **ModelOpt Calibration Helper** | (Optional) Collect latent samples for better calibration data |

### SVD Outlier Absorption

For **INT4** and **NVFP4** quantization, enable `use_svd` in the **ModelOpt Quantize UNet** node. This uses SVD decomposition to keep outlier dimensions in high precision (FP16) while quantizing the residual to 4-bit, significantly improving quality at maximum compression.

- **SVD Ratio**: Controls what fraction of singular values are kept as outliers (default: 1%, range: 0.1% - 10%)
- **When to use**: Recommended for INT4/NVFP4 when image quality matters
- **Tradeoff**: Slightly larger checkpoint size (outlier buffers add ~2% overhead) |

---

## Known Limitations

- **Conv2d fallback**: MXFP8 and NVFP4 require 2D weight tensors. Conv2d layers automatically fall back to FP8 quantization when using these formats.
- **SVD only for Linear layers**: SVD outlier absorption is only applied to 2D Linear weights, not Conv2d or 1D tensors.
- **PyTorch cu130+**: For maximum speedup with `comfy_kitchen` acceleration, PyTorch with CUDA 13.0+ is recommended. Current stable wheels are cu128.
- **torch.compile**: Not yet compatible with quantized models.
- **Only UNet is quantized**: VAE, text encoders, and LoRAs are not quantized by this tool.

---

## Troubleshooting

**"Model already has modelopt state!" when loading**
→ This was a bug in v0.4.0. Fixed in the latest version. Update or restart ComfyUI.

**"mat1 and mat2 must have the same dtype"**
→ Fixed in the latest version. The model now restores to its original FP16 dtype after quantization.

**"Invalid version format in requirements.txt: 2.0" on startup**
→ This is a harmless warning from ComfyUI's own requirements parser, not from this custom node. Safe to ignore.

**Quantization is slow**
→ Expected for large models (SDXL = ~60-90 seconds on RTX 5090). This is a one-time cost per model.

**No speedup during inference**
→ Without `comfy_kitchen` installed, dequantization happens on-the-fly in PyTorch. For native CUDA acceleration, ensure PyTorch cu130+ and `comfy_kitchen` are installed.

---

## How It Works (Short Version)

1. **ModelOpt calibration**: ComfyUI modules are unwrapped to standard PyTorch. ModelOpt inserts quantizers and runs calibration to learn optimal scales.
2. **Scale extraction**: We extract `amax` values from ModelOpt's quantizers to compute per-layer scales.
3. **Native quantization**: We replace layer weights with real quantized storage (FP8/INT8/MXFP8/etc.) using PyTorch operations.
4. **ModelOpt cleanup**: All ModelOpt quantizer submodules are stripped, leaving only native quantized weights.
5. **Save**: The model's state_dict (with quantized weights) is saved as `.safetensors` with metadata.
6. **Load**: The loader overlays quantized weights from the `.safetensors` file onto a fresh base model.

This approach produces **real quantized checkpoints** (50-75% smaller) that are **self-contained** and **compatible with ComfyUI's execution pipeline**.

---

## Links

- **Repository**: https://github.com/EnragedAntelope/comfy-modelopt
- **ModelOpt Docs**: https://github.com/NVIDIA/Model-Optimizer
- **Research Notes**: See [RESEARCH_NOTES.md](RESEARCH_NOTES.md) for detailed technical history
- **Issues**: https://github.com/EnragedAntelope/comfy-modelopt/issues

---

## License

MIT
