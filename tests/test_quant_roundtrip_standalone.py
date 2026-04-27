"""
Integration test: quantize -> save -> standalone load -> inference roundtrip.

Tests the core quantize/save/load pipeline using a small synthetic UNet-like model.
Uses INT8 precision for speed.

Skips gracefully if CUDA is unavailable.
"""

import importlib.util
import os
import sys
import tempfile

import torch
import torch.nn as nn

# =============================================================================
# Graceful skip if no CUDA
# =============================================================================
if not torch.cuda.is_available():
    print("SKIPPED: CUDA required")
    sys.exit(0)

# =============================================================================
# Import modules from comfy-modelopt/nodes/ using importlib to avoid
# ComfyUI dependency. These specific modules (native_quant, quant_saveload)
# do NOT use relative imports, so they load cleanly without ComfyUI.
# =============================================================================
NODES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "nodes")
)


def _load_module(module_name, file_path):
    """Load a Python module from file path without package context."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_nq = _load_module("native_quant", os.path.join(NODES_DIR, "native_quant.py"))
_qs = _load_module("quant_saveload", os.path.join(NODES_DIR, "quant_saveload.py"))

quantize_model_weights = _nq.quantize_model_weights
QuantizedLinear = _nq.QuantizedLinear
QuantizedConv2d = _nq.QuantizedConv2d
save_quantized_model = _qs.save_quantized_model
load_quantized_model = _qs.load_quantized_model


# =============================================================================
# Small UNet-like model for testing
# =============================================================================
class SmallUNet(nn.Module):
    """
    Minimal SD1.5-like UNet with Linear and Conv2d layers.

    Architecture:
      conv_in: Conv2d(4 -> D)
      time_embed: MLP (1280 -> D*4 -> D)
      input_blocks: 4x MLP blocks (D -> D*4 -> D)
      middle_block: MLP (D -> D*4 -> D)
      output_blocks: 4x MLP blocks (D -> D*4 -> D)
      conv_out: Conv2d(D -> 4)

    All Linear and Conv2d layers have >= 1024 elements so quantize_model_weights
    does not skip them.
    """

    def __init__(self, D: int = 64):
        super().__init__()
        self.D = D

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1280, D * 4),  # 327680 elements
            nn.SiLU(),
            nn.Linear(D * 4, D),  # 16384 elements
        )

        # Input projection
        self.conv_in = nn.Conv2d(4, D, kernel_size=3, padding=1)  # 2304 elements

        # Input blocks
        self.input_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(D, D * 4),    # 16384 elements
                nn.SiLU(),
                nn.Linear(D * 4, D),    # 16384 elements
            )
            for _ in range(4)
        ])

        # Middle block
        self.middle_block = nn.Sequential(
            nn.Linear(D, D * 4),        # 16384 elements
            nn.SiLU(),
            nn.Linear(D * 4, D),        # 16384 elements
        )

        # Output blocks
        self.output_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(D, D * 4),    # 16384 elements
                nn.SiLU(),
                nn.Linear(D * 4, D),    # 16384 elements
            )
            for _ in range(4)
        ])

        # Output projection
        self.conv_out = nn.Conv2d(D, 4, kernel_size=3, padding=1)  # 2304 elements

    def forward(self, latent, timestep, context, y=None):
        """
        Args:
            latent:   [B, 4, H, W] noise latent
            timestep: scalar or [B] timestep
            context:  [B, 77, 768] text embeddings
            y:        optional pooled embeddings (unused in this test model)
        """
        B = latent.shape[0]

        # Expand timestep to [B, 1280] for time_embed
        t = timestep.float()
        if t.dim() == 0:
            t = t.unsqueeze(0).expand(B)
        t = t.unsqueeze(-1).expand(B, 1280)
        t_emb = self.time_embed(t)  # [B, D]

        # Spatial: conv_in
        h = self.conv_in(latent)  # [B, D, H, W]

        # Flatten spatial to sequence
        Bc, C, H, W = h.shape
        h = h.permute(0, 2, 3, 1).reshape(Bc, H * W, C)  # [B, N, D]

        # Add time embedding
        h = h + t_emb.unsqueeze(1)

        # Input blocks
        for block in self.input_blocks:
            h = block(h)

        # Middle block
        h = self.middle_block(h)

        # Output blocks
        for block in self.output_blocks:
            h = block(h)

        # Reshape back to spatial
        h = h.reshape(Bc, H, W, C).permute(0, 3, 1, 2)  # [B, D, H, W]

        # Output projection
        return self.conv_out(h)  # [B, 4, H, W]


# =============================================================================
# Test steps
# =============================================================================
def test_quantize_model():
    """Step 1: Quantize model weights using native INT8 quantization."""
    print("\n[1/4] Quantizing model...")

    model = SmallUNet().cuda().eval()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Created SmallUNet: {param_count:,} parameters")

    # Run native quantization (INT8 for speed)
    quantized_model, metadata = quantize_model_weights(
        model, precision="int8"
    )

    # Verify quantization was applied
    quantized_count = sum(
        1 for m in quantized_model.modules()
        if isinstance(m, (QuantizedLinear, QuantizedConv2d))
    )
    print(f"  Quantized layers: {quantized_count}")
    assert quantized_count > 0, "FAIL: No layers were quantized!"
    print(f"  [OK] Quantization complete")

    return quantized_model, metadata


def test_save_model(quantized_model, metadata):
    """Step 2: Save quantized model as safetensors with metadata."""
    print("\n[2/4] Saving quantized model...")

    # Temp file for the safetensors
    fd, tmp_path = tempfile.mkstemp(suffix=".safetensors")
    os.close(fd)

    # Add minimal model config for reconstruction on load
    metadata["_model_config"] = {
        "unet_config": {
            "in_channels": 4,
            "model_channels": 128,
            "num_res_blocks": 2,
            "context_dim": 768,
        },
        "model_type": "EPS",
    }

    save_quantized_model(quantized_model, tmp_path, metadata)

    file_size = os.path.getsize(tmp_path)
    print(f"  Saved to: {tmp_path}")
    print(f"  File size: {file_size:,} bytes")
    assert file_size > 0, "FAIL: Saved file is empty!"
    print(f"  [OK] Save complete")

    return tmp_path


def _apply_quantized_weights(fresh_model, state_dict, precision):
    """
    Apply quantized weights from state_dict onto a fresh model.
    Mirrors the logic in ModelOptUNetLoader._apply_quantized_weights.
    """
    applied_count = 0

    for name, module in list(fresh_model.named_modules()):
        if not isinstance(module, (nn.Linear, nn.Conv2d)):
            continue

        # Look for quantized weight in state dict
        weight_key = f"{name}.weight_q"
        weight_q = state_dict.get(weight_key)

        if weight_q is None:
            continue

        # Get scale buffers
        weight_scale = state_dict.get(f"{name}.weight_scale")
        block_scale = state_dict.get(f"{name}.block_scale")
        tensor_scale = state_dict.get(f"{name}.tensor_scale")

        # Get bias
        bias = state_dict.get(f"{name}.bias")
        if bias is None and hasattr(module, "bias") and module.bias is not None:
            bias = module.bias

        # Locate parent and child
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1]
        parent = fresh_model.get_submodule(parent_name) if parent_name else fresh_model

        # Replace with quantized layer
        if isinstance(module, nn.Linear):
            new_layer = QuantizedLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=bias is not None,
                precision=precision,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
        elif isinstance(module, nn.Conv2d):
            new_layer = QuantizedConv2d(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=bias is not None,
                padding_mode=module.padding_mode,
                precision=precision,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )

        # Set quantized weight and scales
        new_layer.weight_q = weight_q.to(device=module.weight.device)
        if bias is not None:
            new_layer.bias = bias.to(device=module.weight.device)
        if weight_scale is not None:
            new_layer.weight_scale = weight_scale.to(device=module.weight.device)
        if block_scale is not None:
            new_layer.block_scale = block_scale.to(device=module.weight.device)
        if tensor_scale is not None:
            new_layer.tensor_scale = tensor_scale.to(device=module.weight.device)

        setattr(parent, child_name, new_layer)
        applied_count += 1

    return applied_count


def test_load_and_infer(save_path):
    """Step 3 & 4: Load quantized model, reconstruct, run inference, verify."""
    print("\n[3/4] Loading quantized model...")

    # Load safetensors
    state_dict, metadata = load_quantized_model(save_path)
    print(f"  Loaded {len(state_dict)} tensors from safetensors")
    assert len(state_dict) > 0, "FAIL: No tensors loaded!"

    precision = metadata.get("precision", "int8")
    print(f"  Precision: {precision}")

    # Reconstruct a fresh model and apply quantized weights
    model = SmallUNet().cuda().eval()
    applied = _apply_quantized_weights(model, state_dict, precision)
    print(f"  Applied quantized weights to {applied} layers")
    assert applied > 0, "FAIL: No quantized weights were applied!"

    # Run inference
    print("\n[4/4] Running inference on reconstructed model...")
    with torch.no_grad():
        latent = torch.randn(1, 4, 32, 32, device="cuda")
        timestep = torch.tensor([999.0], device="cuda")
        context = torch.randn(1, 77, 768, device="cuda")

        output = model(latent, timestep, context)

    # ----- Verify output -----
    expected_shape = (1, 4, 32, 32)
    shape_ok = output.shape == expected_shape
    no_nan = not torch.isnan(output).any()
    not_all_zero = not (output == 0).all()
    mean_val = output.mean().item()
    std_val = output.std().item()

    print(f"  Output shape: {output.shape}  [{'OK' if shape_ok else 'FAIL'}]")
    print(f"  Output mean:  {mean_val:.6f}")
    print(f"  Output std:   {std_val:.6f}")
    print(f"  No NaN:       [{'OK' if no_nan else 'FAIL'}]")
    print(f"  Not all zero: [{'OK' if not_all_zero else 'FAIL'}]")

    assert shape_ok, f"FAIL: Expected shape {expected_shape}, got {output.shape}"
    assert no_nan, "FAIL: Output contains NaN!"
    assert not_all_zero, "FAIL: Output is all zeros!"


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Quantization Roundtrip Integration Test")
    print("  quantize -> save -> standalone load -> inference")
    print("=" * 60)

    save_path = None
    exit_code = 0
    try:
        quantized_model, metadata = test_quantize_model()
        save_path = test_save_model(quantized_model, metadata)
        test_load_and_infer(save_path)

        print("\n" + "=" * 60)
        print("  ALL TESTS PASSED")
        print("=" * 60)
    except AssertionError as e:
        print(f"\nFAIL: {e}")
        exit_code = 1
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        if save_path and os.path.exists(save_path):
            os.remove(save_path)

    sys.exit(exit_code)
