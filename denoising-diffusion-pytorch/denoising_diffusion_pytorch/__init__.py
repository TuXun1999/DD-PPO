"""
Denoising Diffusion PyTorch - Legacy compatibility layer.

This module provides backward compatibility imports for the refactored
diffusion_policy package. New code should import directly from diffusion_policy.

Example (new style):
    from diffusion_policy import ConditionalUnet1D, GaussianDiffusion1D

Example (legacy style, deprecated):
    from denoising_diffusion_pytorch import ConditionalUnet1D, GaussianDiffusion1DConditional
"""
import warnings

# Import from new package structure
from diffusion_policy import (
    ConditionalUnet1D,
    TransformerForDiffusion,
    GaussianDiffusion1D,
    DiffusionPolicyDataset,
)

# Legacy class names (emit deprecation warning on first use)
class _DeprecatedAlias:
    def __init__(self, new_class, old_name, new_name):
        self._new_class = new_class
        self._old_name = old_name
        self._new_name = new_name
        self._warned = False

    def __call__(self, *args, **kwargs):
        if not self._warned:
            warnings.warn(
                f"{self._old_name} is deprecated, use {self._new_name} from diffusion_policy instead",
                DeprecationWarning,
                stacklevel=2,
            )
            self._warned = True
        return self._new_class(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._new_class, name)


GaussianDiffusion1DConditional = _DeprecatedAlias(
    GaussianDiffusion1D,
    "GaussianDiffusion1DConditional",
    "GaussianDiffusion1D",
)

Dataset1DCond = _DeprecatedAlias(
    DiffusionPolicyDataset,
    "Dataset1DCond",
    "DiffusionPolicyDataset",
)


def __getattr__(name):
    """Lazy import for legacy Trainer to avoid circular imports."""
    if name == "Trainer1DCond":
        from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d_cond import Trainer1DCond
        return Trainer1DCond
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from denoising_diffusion_pytorch.version import __version__

__all__ = [
    # New classes (preferred)
    "ConditionalUnet1D",
    "TransformerForDiffusion",
    "GaussianDiffusion1D",
    "DiffusionPolicyDataset",
    # Legacy class names (deprecated)
    "GaussianDiffusion1DConditional",
    "Dataset1DCond",
    "Trainer1DCond",
    "__version__",
]
