"""
PyTorch Lightning Callbacks for Diffusion Policy.
"""
import copy
from typing import Any, Dict, Optional

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback


class EMACallback(Callback):
    """
    Exponential Moving Average callback for model weights.

    Maintains an EMA copy of the model weights and optionally
    swaps to EMA weights during validation.
    """

    def __init__(
        self,
        decay: float = 0.995,
        update_every: int = 10,
        use_ema_for_validation: bool = True,
    ):
        """
        Initialize EMA callback.

        Args:
            decay: EMA decay rate (higher = slower update)
            update_every: Update EMA every N training steps
            use_ema_for_validation: Whether to use EMA weights for validation
        """
        super().__init__()
        self.decay = decay
        self.update_every = update_every
        self.use_ema_for_validation = use_ema_for_validation

        self.ema_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self.original_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self.step_count = 0

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Initialize EMA state dict at the start of training."""
        self.ema_state_dict = copy.deepcopy(pl_module.state_dict())

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Update EMA weights after training batch."""
        self.step_count += 1

        if self.step_count % self.update_every == 0:
            self._update_ema(pl_module)

    def _update_ema(self, pl_module: pl.LightningModule) -> None:
        """Update EMA state dict with current model weights."""
        if self.ema_state_dict is None:
            return

        model_state = pl_module.state_dict()
        for key in self.ema_state_dict:
            self.ema_state_dict[key] = (
                self.decay * self.ema_state_dict[key] +
                (1 - self.decay) * model_state[key]
            )

    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Swap to EMA weights for validation."""
        if self.use_ema_for_validation and self.ema_state_dict is not None:
            self.original_state_dict = copy.deepcopy(pl_module.state_dict())
            pl_module.load_state_dict(self.ema_state_dict)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Restore original weights after validation."""
        if self.use_ema_for_validation and self.original_state_dict is not None:
            pl_module.load_state_dict(self.original_state_dict)
            self.original_state_dict = None

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        """Save EMA state dict to checkpoint."""
        checkpoint['ema_state_dict'] = self.ema_state_dict
        checkpoint['ema_step_count'] = self.step_count

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict[str, Any],
    ) -> None:
        """Load EMA state dict from checkpoint."""
        if 'ema_state_dict' in checkpoint:
            self.ema_state_dict = checkpoint['ema_state_dict']
        if 'ema_step_count' in checkpoint:
            self.step_count = checkpoint['ema_step_count']

    def get_ema_state_dict(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get the current EMA state dict."""
        return self.ema_state_dict


class VisualizationCallback(Callback):
    """
    Callback for visualizing sampled trajectories during training.

    Logs sampled trajectories to TensorBoard/WandB at specified intervals.
    """

    def __init__(
        self,
        log_every_n_steps: int = 1000,
        num_samples: int = 5,
    ):
        """
        Initialize visualization callback.

        Args:
            log_every_n_steps: Log visualizations every N training steps
            num_samples: Number of samples to generate for visualization
        """
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.num_samples = num_samples
        self.step_count = 0

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Generate and log sample visualizations."""
        self.step_count += 1

        if self.step_count % self.log_every_n_steps == 0:
            self._log_samples(trainer, pl_module, batch)

    def _log_samples(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
    ) -> None:
        """Generate samples and log them."""
        actions, local_cond, global_cond = batch

        # Use first few samples from batch for conditioning
        num_samples = min(self.num_samples, actions.shape[0])
        local_cond_sample = local_cond[:num_samples] if local_cond is not None else None
        global_cond_sample = global_cond[:num_samples] if global_cond is not None else None

        # Generate samples
        with torch.no_grad():
            pl_module.eval()
            samples = pl_module.sample(
                batch_size=num_samples,
                local_cond=local_cond_sample,
                global_cond=global_cond_sample,
                unnormalize=False,
            )
            pl_module.train()

        # Log sample statistics
        if trainer.logger is not None:
            trainer.logger.log_metrics({
                'samples/mean': samples.mean().item(),
                'samples/std': samples.std().item(),
                'samples/min': samples.min().item(),
                'samples/max': samples.max().item(),
            }, step=self.step_count)


class GradientClipCallback(Callback):
    """
    Callback for gradient clipping with logging.

    Clips gradients and logs gradient norm statistics.
    """

    def __init__(
        self,
        max_grad_norm: float = 1.0,
        log_grad_norm: bool = True,
    ):
        """
        Initialize gradient clip callback.

        Args:
            max_grad_norm: Maximum gradient norm
            log_grad_norm: Whether to log gradient norm
        """
        super().__init__()
        self.max_grad_norm = max_grad_norm
        self.log_grad_norm = log_grad_norm

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Clip gradients and optionally log norm."""
        if self.log_grad_norm:
            grad_norm = self._compute_grad_norm(pl_module)
            pl_module.log('train/grad_norm', grad_norm, prog_bar=False)

        torch.nn.utils.clip_grad_norm_(
            pl_module.parameters(),
            self.max_grad_norm,
        )

    def _compute_grad_norm(self, pl_module: pl.LightningModule) -> float:
        """Compute total gradient norm."""
        total_norm = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
