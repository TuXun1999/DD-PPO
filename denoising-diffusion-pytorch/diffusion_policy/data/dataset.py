"""
Dataset classes for Diffusion Policy.
"""
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset


class DiffusionPolicyDataset(Dataset):
    """
    Dataset for diffusion policy training.

    Stores action sequences with local and global conditioning.
    """

    def __init__(
        self,
        actions: torch.Tensor,
        local_cond: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None,
    ):
        """
        Initialize the dataset.

        Args:
            actions: Action tensor [N, action_dim, seq_length]
            local_cond: Local conditioning tensor [N, local_cond_dim, seq_length]
            global_cond: Global conditioning tensor [N, global_cond_dim]
        """
        super().__init__()
        self.actions = actions.clone()
        self.local_cond = local_cond.clone() if local_cond is not None else None
        self.global_cond = global_cond.clone() if global_cond is not None else None

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get a single sample.

        Returns:
            Tuple of (actions, local_cond, global_cond)
        """
        action = self.actions[idx].clone()
        local_cond = self.local_cond[idx].clone() if self.local_cond is not None else None
        global_cond = self.global_cond[idx].clone() if self.global_cond is not None else None

        return action, local_cond, global_cond
