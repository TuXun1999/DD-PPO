"""
Model components for Diffusion Policy.
"""
from .unet_1d import ConditionalUnet1D
from .transformer import TransformerForDiffusion
from .diffusion import GaussianDiffusion1D
from .image_encoder import BaseImageEncoder, ResNetImageEncoder, build_image_encoder

from ..config import ModelConfig, DiffusionConfig


def build_backbone(config: ModelConfig, seq_length: int = 16):
    """
    Build a backbone model from configuration.

    Args:
        config: Model configuration
        seq_length: Sequence length for transformer backbone

    Returns:
        Backbone model (ConditionalUnet1D or TransformerForDiffusion)
    """
    if config.backbone == "unet1d":
        return ConditionalUnet1D.from_config(config)
    elif config.backbone == "transformer":
        return TransformerForDiffusion.from_config(config, seq_length)
    else:
        raise ValueError(f"Unknown backbone: {config.backbone}")


def build_diffusion(
    backbone,
    config: DiffusionConfig,
) -> GaussianDiffusion1D:
    """
    Build a diffusion model from configuration.

    Args:
        backbone: Backbone model for noise prediction
        config: Diffusion configuration

    Returns:
        GaussianDiffusion1D model
    """
    return GaussianDiffusion1D.from_config(backbone, config)


__all__ = [
    "ConditionalUnet1D",
    "TransformerForDiffusion",
    "GaussianDiffusion1D",
    "build_backbone",
    "build_diffusion",
    "BaseImageEncoder",
    "ResNetImageEncoder",
    "build_image_encoder",
]
