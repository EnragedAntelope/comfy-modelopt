# Example Workflows

This folder will contain example ComfyUI workflow JSON files demonstrating how to use ModelOpt nodes.

## Planned Examples

1. **quantize_sdxl_basic.json** - Basic SDXL UNet quantization workflow
   - Load SDXL checkpoint
   - Quantize to INT8
   - Save quantized model

2. **quantize_with_calibration.json** - Advanced workflow with real calibration data
   - Generate images with normal workflow
   - Collect calibration samples
   - Quantize with calibration data

3. **load_quantized_model.json** - Using quantized models for inference
   - Load quantized UNet
   - Load VAE and CLIP separately
   - Generate images with quantized model

## How to Use

1. Download the JSON workflow file
2. Open ComfyUI
3. Drag and drop the JSON file onto the ComfyUI canvas
4. Adjust paths and settings as needed for your system
5. Run the workflow

## Creating Your Own

You can create custom workflows by:
1. Adding ModelOpt nodes to your existing workflow
2. Connecting them appropriately
3. Saving your workflow as JSON
4. Sharing with the community!

---

**Note**: Example workflows will be added in future updates. For now, refer to the README.md for usage instructions.
