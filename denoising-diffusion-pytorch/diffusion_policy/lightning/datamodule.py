"""
PyTorch Lightning DataModule for Diffusion Policy.
"""
from pathlib import Path
from typing import Optional

import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, random_split

from ..config import DataConfig
from ..data.dataset import DiffusionPolicyDataset
from ..data.preprocessing import (
    NormalizationStats,
    load_demo_data,
    process_demos,
    save_training_stats,
    load_training_stats,
)


class DiffusionPolicyDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for Diffusion Policy training.

    Handles loading demonstration data, processing into training format,
    and creating dataloaders.
    """

    def __init__(
        self,
        config: DataConfig,
        val_split: float = 0.1,
    ):
        """
        Initialize the DataModule.

        Args:
            config: Data configuration
            val_split: Fraction of data to use for validation
        """
        super().__init__()
        self.config = config
        self.val_split = val_split

        self.train_dataset: Optional[DiffusionPolicyDataset] = None
        self.val_dataset: Optional[DiffusionPolicyDataset] = None
        self.normalization_stats: Optional[NormalizationStats] = None

        # Path for cached training stats
        self.stats_path = Path(config.dataset_dir) / config.object_type / \
                          f"grasp_pose{config.grasp_pose}" / "training_stats.pth"

    def prepare_data(self) -> None:
        """
        Prepare data (download, process). Called only on rank 0.
        """
        if not self.stats_path.exists():
            demo_data = load_demo_data(
                self.config.dataset_dir,
                self.config.object_type,
                self.config.grasp_pose,
            )

            actions, local_cond, global_cond, stats = process_demos(
                demo_data,
                obs_length=self.config.obs_length,
                obs_dim=self.config.obs_dim,
                pred_length=self.config.pred_length,
                action_dim=self.config.action_dim,
            )

            save_training_stats(
                str(self.stats_path),
                actions,
                local_cond,
                global_cond,
                stats,
            )

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup datasets for training and validation.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'
        """
        # Load processed data
        actions, local_cond, global_cond, stats = load_training_stats(str(self.stats_path))
        self.normalization_stats = stats

        # Create full dataset
        full_dataset = DiffusionPolicyDataset(actions, local_cond, global_cond)

        # Split into train/val
        total_size = len(full_dataset)
        val_size = int(total_size * self.val_split)
        train_size = total_size - val_size

        self.train_dataset, self.val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

    def train_dataloader(self) -> DataLoader:
        """Get training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size if hasattr(self.config, 'batch_size') else 32,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        """Get validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size if hasattr(self.config, 'batch_size') else 32,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

    def get_normalization_stats(self) -> NormalizationStats:
        """Get normalization statistics."""
        if self.normalization_stats is None:
            raise RuntimeError("DataModule not set up. Call setup() first.")
        return self.normalization_stats
