"""
ComfyUI native quantized model save/load utilities.

Provides safetensors-based save/load for self-contained quantized checkpoints.
Also provides model reconstruction from stored metadata - enabling truly
standalone loading WITHOUT requiring the original base model checkpoint.

Architecture:
  Save: state_dict + metadata (unet_config, model_type, etc)
  Load: reconstruct BaseModel from metadata, then overlay quantized weights
"""

import json
import importlib
import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
from safetensors.torch import save_file, load_file
from safetensors import safe_open


# ==============================================================================
# Safetensors Save/Load
# ==============================================================================

def save_quantized_model(model: nn.Module, save_path: str, metadata: Dict[str, Any]) -> str:
    """
    Save a quantized model as safetensors with metadata.

    Args:
        model: The quantized diffusion model
        save_path: Path to save to (should end in .safetensors)
        metadata: Dict containing architecture info, quantization config, etc.

    Returns:
        Path to saved file
    """
    if not save_path.endswith('.safetensors'):
        save_path = save_path + '.safetensors'

    # Collect state dict
    state_dict = {}
    for name, param in model.state_dict().items():
        state_dict[name] = param

    # Convert metadata to strings for safetensors header
    metadata_strs = {}
    for k, v in metadata.items():
        if isinstance(v, (dict, list)):
            metadata_strs[k] = json.dumps(v)
        else:
            metadata_strs[k] = str(v)

    # Save with metadata in header
    save_file(state_dict, save_path, metadata=metadata_strs)

    return save_path


def load_quantized_model(load_path: str, device: torch.device = torch.device('cpu')) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Load a quantized model from safetensors.

    Returns:
        (state_dict, metadata)
    """
    # Load tensors
    result = load_file(load_path, device=str(device))

    # Load metadata from header
    metadata = {}
    with safe_open(load_path, framework='pt', device=str(device)) as f:
        header_metadata = f.metadata()
        if header_metadata:
            for k, v in header_metadata.items():
                try:
                    metadata[k] = json.loads(v)
                except (json.JSONDecodeError, TypeError):
                    metadata[k] = v

    return result, metadata


# ==============================================================================
# Model Configuration Capture & Reconstruction
# ==============================================================================
#
# To make quantized checkpoints truly standalone (no base model needed for
# loading), we capture the UNetModel's construction config during save and
# store it in the safetensors metadata. On load, we reconstruct the model
# from this config.
#
# The chain is:
#   BaseModel(model_config, model_type)
#     -> UNetModel(**unet_config, device=device, operations=ops)
#
# We store:
#   - unet_config: all kwargs for UNetModel.__init__()
#   - model_type: the ModelType enum name (EPS, V_PREDICTION, FLOW, etc.)
#   - latent_format: the class path + state for reconstruction
#   - sampling_settings: for model_sampling configuration
# ==============================================================================


def capture_model_config(model) -> Dict[str, Any]:
    """
    Capture model architecture configuration from a ComfyUI model.

    Extracts all information needed to reconstruct the model later without
    requiring the original base checkpoint.

    Args:
        model: A ComfyUI ModelPatcher with an attached base model

    Returns:
        dict: Serializable configuration for later reconstruction
    """
    base_model = model.model
    model_config = base_model.model_config

    # --- unet_config (kwargs for UNetModel.__init__) ---
    unet_config_raw = model_config.unet_config.copy()
    unet_config_serializable = {}
    for k, v in unet_config_raw.items():
        try:
            # Test JSON serializability
            json.dumps(v)
            unet_config_serializable[k] = v
        except (TypeError, OverflowError, ValueError):
            # Convert non-serializable values (e.g., torch.dtype)
            unet_config_serializable[k] = str(v)

    # --- model_type ---
    model_type_name = base_model.model_type.name

    # --- latent_format ---
    lf = model_config.latent_format
    lf_cls_path = f"{lf.__class__.__module__}.{lf.__class__.__qualname__}"
    lf_state = {}
    for k, v in lf.__dict__.items():
        if isinstance(v, (torch.Tensor,)):
            continue
        try:
            json.dumps(v)
            lf_state[k] = v
        except (TypeError, OverflowError, ValueError):
            lf_state[k] = str(v)

    # --- sampling_settings ---
    sampling_settings = getattr(model_config, 'sampling_settings', {})

    return {
        'unet_config': unet_config_serializable,
        'model_type': model_type_name,
        'latent_format_cls': lf_cls_path,
        'latent_format_state': lf_state,
        'sampling_settings': sampling_settings,
    }


def _parse_model_type(name: str):
    """Parse ModelType enum from string name."""
    import comfy.model_base
    try:
        return getattr(comfy.model_base.ModelType, name)
    except AttributeError:
        return comfy.model_base.ModelType.EPS


def _reconstruct_latent_format(cls_path: Optional[str],
                                state: Optional[Dict]) -> Optional[Any]:
    """Reconstruct a LatentFormat instance from stored class path + state."""
    if not cls_path:
        return None
    try:
        module_path, class_name = cls_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        instance = cls.__new__(cls)
        if state:
            # Restore serializable state
            instance.__dict__.update({
                k: v for k, v in state.items()
                if not k.startswith('_')
            })
        return instance
    except Exception as e:
        logging.warning(f"Failed to reconstruct latent format {cls_path}: {e}")
        return None

def _restore_dtype(unet_config):
    """Convert dtype string values back to torch.dtype in unet_config."""
    config = dict(unet_config)
    dtype_val = config.get('dtype')
    if isinstance(dtype_val, str):
        if dtype_val.startswith('torch.'):
            dtype_name = dtype_val.split('.', 1)[1]
            config['dtype'] = getattr(torch, dtype_name, torch.float16)
        elif dtype_val in ('float16', 'bfloat16', 'float32'):
            config['dtype'] = getattr(torch, dtype_val)
        else:
            config['dtype'] = torch.float16
    return config


class StoredModelConfig:
    """
    Minimal model_config wrapper for reconstructing a BaseModel from
    stored metadata.

    Satisfies the interface that BaseModel.__init__() and model_sampling()
    expect from a model_config object.
    """

    def __init__(self,
                 unet_config: Dict,
                 model_type_name: str = 'EPS',
                 latent_format_cls: Optional[str] = None,
                 latent_format_state: Optional[Dict] = None,
                 sampling_settings: Optional[Dict] = None):
        # Restore dtype from string representation if needed
        self.unet_config = _restore_dtype(unet_config)
        self.manual_cast_dtype = None
        self.custom_operations = None
        self.optimizations = {}
        self.memory_usage_factor = 1.0
        self.model_type = _parse_model_type(model_type_name)
        self.sampling_settings = sampling_settings or {}

        # Reconstruct latent format
        self.latent_format = _reconstruct_latent_format(
            latent_format_cls, latent_format_state
        )
        if self.latent_format is None:
            self.latent_format = self._guess_latent_format()

    @property
    def supported_inference_dtypes(self):
        return [torch.float16, torch.bfloat16, torch.float32]

    def set_inference_dtype(self, dtype, manual_cast_dtype):
        self.unet_config['dtype'] = dtype
        self.manual_cast_dtype = manual_cast_dtype

    def inpaint_model(self):
        return self.unet_config.get("in_channels", 4) > 4

    def _guess_latent_format(self):
        import comfy.latent_formats
        latent_channels = self.unet_config.get("in_channels", 4)
        if latent_channels == 16:
            return comfy.latent_formats.SD3()
        return comfy.latent_formats.SD15()

def reconstruct_base_model(metadata: Dict[str, Any],
                           device: torch.device,
                           dtype: torch.dtype = torch.float16) -> Any:
    """
    Reconstruct a ComfyUI BaseModel from stored metadata.

    This enables truly standalone loading - no base model checkpoint needed.

    Args:
        metadata: Dict containing unet_config, model_type, latent_format info
        device: Target device for the model
        dtype: Weight dtype for the model

    Returns:
        A ComfyUI BaseModel with the reconstructed UNetModel
    """
    import comfy.model_base
    import comfy.ops

    unet_config = metadata.get('unet_config', {})
    if not unet_config:
        raise ValueError(
            "No unet_config found in metadata. Cannot reconstruct model.\n"
            "This quantized file was saved without architecture metadata.\n"
            "You must provide a base model for loading."
        )

    model_type_name = metadata.get('model_type', 'EPS')

    # Create StoredModelConfig
    stored_config = StoredModelConfig(
        unet_config=unet_config,
        model_type_name=model_type_name,
        latent_format_cls=metadata.get('latent_format_cls'),
        latent_format_state=metadata.get('latent_format_state'),
        sampling_settings=metadata.get('sampling_settings'),
    )

    # Set inference dtype
    stored_config.set_inference_dtype(dtype, None)

    # Create BaseModel - this internally creates the UNetModel
    base_model = comfy.model_base.BaseModel(
        stored_config,
        model_type=stored_config.model_type,
        device=device,
    )
    base_model.eval()

    logging.info(f"Reconstructed BaseModel from config (type={model_type_name})")
    return base_model


def get_modelopt_metadata(model: nn.Module) -> Dict[str, Any]:
    """
    Extract metadata from a ComfyUI model for reconstruction.

    This captures the model architecture so we can reconstruct it later.
    """
    metadata = {
        'architecture': {},
        'layers': [],
    }

    # Capture model class info
    metadata['model_class'] = model.__class__.__module__ + '.' + model.__class__.__name__

    # Capture each layer's configuration
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            layer_info = {
                'name': name,
                'type': module.__class__.__name__,
            }

            if isinstance(module, nn.Linear):
                layer_info.update({
                    'in_features': module.in_features,
                    'out_features': module.out_features,
                    'bias': module.bias is not None,
                })
            elif isinstance(module, nn.Conv2d):
                layer_info.update({
                    'in_channels': module.in_channels,
                    'out_channels': module.out_channels,
                    'kernel_size': module.kernel_size,
                    'stride': module.stride,
                    'padding': module.padding,
                    'dilation': module.dilation,
                    'groups': module.groups,
                    'bias': module.bias is not None,
                    'padding_mode': module.padding_mode,
                })

            metadata['layers'].append(layer_info)

        # Capture other important modules
        elif isinstance(module, (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm2d)):
            layer_info = {
                'name': name,
                'type': module.__class__.__name__,
            }
            if hasattr(module, 'num_features'):
                layer_info['num_features'] = module.num_features
            if hasattr(module, 'num_groups'):
                layer_info['num_groups'] = module.num_groups
            if hasattr(module, 'eps'):
                layer_info['eps'] = module.eps
            metadata['layers'].append(layer_info)

    return metadata


def reconstruct_model_from_metadata(metadata: Dict[str, Any],
                                    device: torch.device = torch.device('cpu')) -> nn.Module:
    """
    Reconstruct a model architecture from metadata.

    This creates an empty model with the right structure, which can then
    be populated with quantized weights.

    Note: This is a simplified reconstruction. For complex models like SDXL UNet,
    we may need the original model class. In practice, we'll use ComfyUI's
    checkpoint loader to get the base model, then swap in quantized weights.
    """
    # For now, this returns None - the loader will use ComfyUI's native loading
    # and then replace weights. Full reconstruction requires the original class.
    return None


# ==============================================================================
# Checkpoint Conversion Helpers
# ==============================================================================

def convert_state_dict_to_quantized(state_dict: Dict[str, torch.Tensor],
                                     precision: str) -> Dict[str, torch.Tensor]:
    """
    Convert a standard state dict to quantized format.

    This is used when loading a standard checkpoint and quantizing it.
    """
    from .native_quant import quantize_fp8, quantize_int8, quantize_mxfp8, quantize_nvfp4, quantize_int4

    precision = precision.lower()
    quantized_dict = {}

    for key, tensor in state_dict.items():
        # Only quantize weight tensors (not bias, norm, etc.)
        if 'weight' not in key or tensor.dim() < 2:
            quantized_dict[key] = tensor
            continue

        try:
            if precision == 'fp8':
                qdata, scale = quantize_fp8(tensor)
                quantized_dict[key] = qdata
                quantized_dict[key + '_scale'] = scale
            elif precision == 'int8':
                qdata, scale = quantize_int8(tensor, axis=0 if tensor.dim() == 2 else None)
                quantized_dict[key] = qdata
                quantized_dict[key + '_scale'] = scale
            elif precision == 'mxfp8':
                if tensor.dim() == 2:
                    qdata, block_scale = quantize_mxfp8(tensor)
                else:
                    orig_shape = tensor.shape
                    flat = tensor.reshape(tensor.shape[0], -1)
                    qdata_flat, block_scale = quantize_mxfp8(flat)
                    qdata = qdata_flat.reshape(orig_shape)
                quantized_dict[key] = qdata
                quantized_dict[key + '_block_scale'] = block_scale
            elif precision == 'nvfp4':
                if tensor.dim() == 2:
                    qdata, tensor_scale, block_scale = quantize_nvfp4(tensor)
                else:
                    orig_shape = tensor.shape
                    flat = tensor.reshape(tensor.shape[0], -1)
                    qdata_flat, tensor_scale, block_scale = quantize_nvfp4(flat)
                    qdata = qdata_flat.reshape(orig_shape)
                quantized_dict[key] = qdata
                quantized_dict[key + '_tensor_scale'] = tensor_scale
                quantized_dict[key + '_block_scale'] = block_scale
            elif precision == 'int4':
                qdata, scale = quantize_int4(tensor, axis=0 if tensor.dim() == 2 else None)
                quantized_dict[key] = qdata
                quantized_dict[key + '_scale'] = scale
            else:
                quantized_dict[key] = tensor
        except Exception as e:
            print(f"Warning: Failed to quantize {key}: {e}")
            quantized_dict[key] = tensor

    return quantized_dict
