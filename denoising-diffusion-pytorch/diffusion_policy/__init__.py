"""
Diffusion Policy: PyTorch Lightning implementation for robot learning.

This package provides a modular implementation of diffusion models for
policy learning, with support for 1D conditional generation.
"""
from .config import (
    ModelConfig,
    DiffusionConfig,
    TrainingConfig,
    DataConfig,
    DiffusionPolicyConfig,
    ImageEncoderConfig,
)

from .models import (
    ConditionalUnet1D,
    TransformerForDiffusion,
    GaussianDiffusion1D,
    build_backbone,
    build_diffusion,
    BaseImageEncoder,
    ResNetImageEncoder,
    build_image_encoder,
)

from .data import (
    DiffusionPolicyDataset,
    NormalizationStats,
    load_demo_data,
    process_demos,
)

from .utils import (
    se2_norm,
    SE2_to_SE3,
    SE3_to_SE2,
    normalize_angle,
    update_pose,
)

__version__ = "1.0.0"


def __getattr__(name):
    """Lazy import for Lightning components to avoid hard dependency."""
    lightning_exports = {
        "DiffusionPolicyModule",
        "DiffusionPolicyDataModule",
        "EMACallback",
        "VisualizationCallback",
        "GradientClipCallback",
    }
    if name in lightning_exports:
        from . import lightning
        return getattr(lightning, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Config
    "ModelConfig",
    "DiffusionConfig",
    "TrainingConfig",
    "DataConfig",
    "DiffusionPolicyConfig",
    "ImageEncoderConfig",
    # Models
    "ConditionalUnet1D",
    "TransformerForDiffusion",
    "GaussianDiffusion1D",
    "build_backbone",
    "build_diffusion",
    "BaseImageEncoder",
    "ResNetImageEncoder",
    "build_image_encoder",
    # Lightning (lazy-loaded)
    "DiffusionPolicyModule",
    "DiffusionPolicyDataModule",
    "EMACallback",
    "VisualizationCallback",
    "GradientClipCallback",
    # Data
    "DiffusionPolicyDataset",
    "NormalizationStats",
    "load_demo_data",
    "process_demos",
    # Utils
    "se2_norm",
    "SE2_to_SE3",
    "SE3_to_SE2",
    "normalize_angle",
    "update_pose",
]
