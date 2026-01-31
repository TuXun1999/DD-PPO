"""
Image encoder modules for vision-conditioned diffusion policy.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
)


class BaseImageEncoder(ABC):
    """Abstract base class for image encoders."""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Return the output feature dimension."""
        pass

    @property
    @abstractmethod
    def expected_input_size(self) -> Tuple[int, int]:
        """Return expected input image size (H, W)."""
        pass

    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to feature vectors.

        Args:
            images: Input images [batch, C, H, W]

        Returns:
            Feature vectors [batch, output_dim]
        """
        pass


class ResNetImageEncoder(nn.Module, BaseImageEncoder):
    """
    ResNet-based image encoder.

    Supports ResNet18, ResNet34, ResNet50, and ResNet101 backbones.
    Uses ImageNet pretrained weights by default.
    """

    SUPPORTED_VARIANTS = {
        "resnet18": (models.resnet18, ResNet18_Weights.IMAGENET1K_V1, 512),
        "resnet34": (models.resnet34, ResNet34_Weights.IMAGENET1K_V1, 512),
        "resnet50": (models.resnet50, ResNet50_Weights.IMAGENET1K_V2, 2048),
        "resnet101": (models.resnet101, ResNet101_Weights.IMAGENET1K_V2, 2048),
    }

    def __init__(
        self,
        variant: str = "resnet18",
        output_dim: int = 256,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        """
        Initialize ResNet image encoder.

        Args:
            variant: ResNet variant ("resnet18", "resnet34", "resnet50", "resnet101")
            output_dim: Output feature dimension after projection
            pretrained: Whether to use ImageNet pretrained weights
            freeze_backbone: Whether to freeze backbone weights
        """
        super().__init__()

        if variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unsupported ResNet variant: {variant}. "
                f"Supported: {list(self.SUPPORTED_VARIANTS.keys())}"
            )

        model_fn, weights, backbone_dim = self.SUPPORTED_VARIANTS[variant]

        # Load backbone
        if pretrained:
            backbone = model_fn(weights=weights)
        else:
            backbone = model_fn(weights=None)

        # Remove classification head (fc layer)
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,
        )

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Projection head: Linear + LayerNorm
        self._output_dim = output_dim
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

        self._backbone_dim = backbone_dim
        self._freeze_backbone = freeze_backbone

    @property
    def output_dim(self) -> int:
        """Return the output feature dimension."""
        return self._output_dim

    @property
    def expected_input_size(self) -> Tuple[int, int]:
        """Return expected input image size (H, W)."""
        return (224, 224)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to feature vectors.

        Args:
            images: Input images [batch, C, H, W] with C=3, H=224, W=224
                   Values should be in range [0, 1] or normalized with ImageNet stats

        Returns:
            Feature vectors [batch, output_dim]
        """
        # Extract features through backbone
        features = self.backbone(images)  # [batch, backbone_dim, 1, 1]
        features = features.flatten(1)  # [batch, backbone_dim]

        # Project to output dimension
        features = self.projection(features)  # [batch, output_dim]

        return features


def build_image_encoder(config: "ImageEncoderConfig") -> BaseImageEncoder:
    """
    Build an image encoder from configuration.

    Args:
        config: ImageEncoderConfig specifying encoder type and parameters

    Returns:
        BaseImageEncoder instance
    """
    encoder_type = config.encoder_type.lower()

    if encoder_type in ResNetImageEncoder.SUPPORTED_VARIANTS:
        return ResNetImageEncoder(
            variant=encoder_type,
            output_dim=config.output_dim,
            pretrained=config.pretrained,
            freeze_backbone=config.freeze_backbone,
        )
    else:
        raise ValueError(
            f"Unsupported encoder type: {encoder_type}. "
            f"Supported: {list(ResNetImageEncoder.SUPPORTED_VARIANTS.keys())}"
        )


__all__ = [
    "BaseImageEncoder",
    "ResNetImageEncoder",
    "build_image_encoder",
]
