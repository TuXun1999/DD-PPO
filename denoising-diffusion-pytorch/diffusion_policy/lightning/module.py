"""
PyTorch Lightning Module for Diffusion Policy.
"""
from typing import Optional, Dict, Any

import torch
import lightning.pytorch as pl
from torch.optim import Adam, AdamW

from ..config import DiffusionPolicyConfig
from ..models import build_backbone, build_diffusion, build_image_encoder
from ..data.preprocessing import NormalizationStats


class DiffusionPolicyModule(pl.LightningModule):
    """
    PyTorch Lightning Module for Diffusion Policy training.

    Wraps the diffusion model with training, validation, and sampling logic.
    """

    def __init__(
        self,
        config: DiffusionPolicyConfig,
        normalization_stats: Optional[NormalizationStats] = None,
    ):
        """
        Initialize the module.

        Args:
            config: Full configuration object
            normalization_stats: Statistics for action normalization
        """
        super().__init__()
        self.save_hyperparameters(ignore=['normalization_stats'])

        self.config = config
        self.conditioning_mode = config.model.conditioning_mode

        # Build model
        self.backbone = build_backbone(config.model, config.diffusion.seq_length)
        self.diffusion = build_diffusion(self.backbone, config.diffusion)

        # Build image encoder if using image conditioning
        self.image_encoder = None
        if self.conditioning_mode == "image_only":
            if config.model.image_encoder is None:
                raise ValueError(
                    "image_encoder config required when conditioning_mode='image_only'"
                )
            self.image_encoder = build_image_encoder(config.model.image_encoder)
            self.num_image_history = config.data.num_image_history

        # Store normalization stats as buffers
        if normalization_stats is not None:
            self.register_buffer('v_min', torch.tensor(normalization_stats.v_min))
            self.register_buffer('v_max', torch.tensor(normalization_stats.v_max))
            self.register_buffer('angular_v_min', torch.tensor(normalization_stats.angular_v_min))
            self.register_buffer('angular_v_max', torch.tensor(normalization_stats.angular_v_max))
        else:
            self.v_min = None
            self.v_max = None
            self.angular_v_min = None
            self.angular_v_max = None

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to feature vectors for conditioning.

        Args:
            images: Input images [batch, num_history, C, H, W]

        Returns:
            Encoded features [batch, num_history * encoder_output_dim]
        """
        if self.image_encoder is None:
            raise RuntimeError("Image encoder not initialized")

        batch_size, num_history, C, H, W = images.shape

        # Flatten batch and history dimensions for encoding
        images_flat = images.view(batch_size * num_history, C, H, W)

        # Encode all images
        features_flat = self.image_encoder(images_flat)  # [B*T, output_dim]

        # Reshape back and flatten history into feature dimension
        output_dim = features_flat.shape[-1]
        features = features_flat.view(batch_size, num_history * output_dim)

        return features

    def forward(
        self,
        actions: torch.Tensor,
        local_cond: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass computes training loss.

        Args:
            actions: Action tensor [batch, action_dim, seq_length]
            local_cond: Local conditioning [batch, local_cond_dim, seq_length]
            global_cond: Global conditioning [batch, global_cond_dim]
            images: Optional images [batch, num_history, C, H, W] for image_only mode

        Returns:
            Training loss
        """
        # Encode images to global conditioning if in image_only mode
        if self.conditioning_mode == "image_only" and images is not None:
            global_cond = self.encode_images(images)

        return self.diffusion(actions, local_cond, global_cond)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Training step.

        Batch format depends on conditioning_mode:
        - action_only: (actions, local_cond, global_cond)
        - image_only: (actions, images, local_cond)
        """
        if self.conditioning_mode == "image_only":
            actions, images, local_cond = batch
            loss = self(actions, local_cond=local_cond, images=images)
        else:
            actions, local_cond, global_cond = batch
            loss = self(actions, local_cond, global_cond)

        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Validation step.

        Batch format depends on conditioning_mode:
        - action_only: (actions, local_cond, global_cond)
        - image_only: (actions, images, local_cond)
        """
        if self.conditioning_mode == "image_only":
            actions, images, local_cond = batch
            loss = self(actions, local_cond=local_cond, images=images)
        else:
            actions, local_cond, global_cond = batch
            loss = self(actions, local_cond, global_cond)

        self.log('val/loss', loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        local_cond: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        unnormalize: bool = True,
    ) -> torch.Tensor:
        """
        Sample actions from the diffusion model.

        Args:
            batch_size: Number of samples to generate
            local_cond: Local conditioning [batch, local_cond_dim, seq_length]
            global_cond: Global conditioning [batch, global_cond_dim]
            images: Optional images [batch, num_history, C, H, W] for image_only mode
            unnormalize: Whether to unnormalize the output

        Returns:
            Sampled actions [batch, action_dim, seq_length]
        """
        self.diffusion.eval()

        # Encode images to global conditioning if in image_only mode
        if self.conditioning_mode == "image_only" and images is not None:
            global_cond = self.encode_images(images)

        samples = self.diffusion.sample(
            batch_size=batch_size,
            local_cond=local_cond,
            global_cond=global_cond,
        )

        if unnormalize and self.v_min is not None:
            samples = self._unnormalize_actions(samples)

        return samples

    def _unnormalize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Unnormalize actions using stored statistics."""
        min_vals = torch.stack([self.v_min, self.v_min, self.angular_v_min])
        max_vals = torch.stack([self.v_max, self.v_max, self.angular_v_max])
        min_vals = min_vals.unsqueeze(-1).unsqueeze(0).to(actions.device)
        max_vals = max_vals.unsqueeze(-1).unsqueeze(0).to(actions.device)
        return actions * (max_vals - min_vals) + min_vals

    def set_normalization_stats(self, stats: NormalizationStats) -> None:
        """Set normalization statistics after initialization."""
        self.register_buffer('v_min', torch.tensor(stats.v_min))
        self.register_buffer('v_max', torch.tensor(stats.v_max))
        self.register_buffer('angular_v_min', torch.tensor(stats.angular_v_min))
        self.register_buffer('angular_v_max', torch.tensor(stats.angular_v_max))

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        training_config = self.config.training

        # Use AdamW for transformer backbone with weight decay
        if self.config.model.backbone == "transformer":
            optimizer = AdamW(
                self.backbone.get_optim_groups(weight_decay=training_config.weight_decay),
                lr=training_config.lr,
                betas=training_config.adam_betas,
            )
        else:
            optimizer = Adam(
                self.parameters(),
                lr=training_config.lr,
                betas=training_config.adam_betas,
            )

        return optimizer

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Add normalization stats to checkpoint."""
        checkpoint['normalization_stats'] = {
            'v_min': self.v_min.item() if self.v_min is not None else None,
            'v_max': self.v_max.item() if self.v_max is not None else None,
            'angular_v_min': self.angular_v_min.item() if self.angular_v_min is not None else None,
            'angular_v_max': self.angular_v_max.item() if self.angular_v_max is not None else None,
        }

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Load normalization stats from checkpoint."""
        if 'normalization_stats' in checkpoint:
            stats = checkpoint['normalization_stats']
            if stats['v_min'] is not None:
                self.register_buffer('v_min', torch.tensor(stats['v_min']))
                self.register_buffer('v_max', torch.tensor(stats['v_max']))
                self.register_buffer('angular_v_min', torch.tensor(stats['angular_v_min']))
                self.register_buffer('angular_v_max', torch.tensor(stats['angular_v_max']))
