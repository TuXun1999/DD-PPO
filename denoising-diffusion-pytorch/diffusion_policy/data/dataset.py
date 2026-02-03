"""
Dataset classes for Diffusion Policy.
"""
from typing import Optional, Tuple, Union

import torch
from torch.utils.data import Dataset


class DiffusionPolicyDataset(Dataset):
    """
    Dataset for diffusion policy training.

    Stores action sequences with local and global conditioning.
    Optionally stores images for vision-conditioned training.
    """

    def __init__(
        self,
        actions: torch.Tensor,
        local_cond: Optional[torch.Tensor] = None,
        global_cond: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        use_images: bool = False,
    ):
        """
        Initialize the dataset.

        Args:
            actions: Action tensor [N, action_dim, seq_length]
            local_cond: Local conditioning tensor [N, local_cond_dim, seq_length]
            global_cond: Global conditioning tensor [N, global_cond_dim]
            images: Optional image tensor [N, num_history, C, H, W]
            use_images: Whether to return images in __getitem__
        """
        super().__init__()
        self.actions = actions.clone()
        self.local_cond = local_cond.clone() if local_cond is not None else None
        self.global_cond = global_cond.clone() if global_cond is not None else None
        self.images = images.clone() if images is not None else None
        self.use_images = use_images

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]],
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
    ]:
        """
        Get a single sample.

        Returns:
            If use_images=False: Tuple of (actions, local_cond, global_cond)
            If use_images=True: Tuple of (actions, images, local_cond)
        """
        action = self.actions[idx].clone()
        local_cond = self.local_cond[idx].clone() if self.local_cond is not None else None

        if self.use_images and self.images is not None:
            images = self.images[idx].clone()
            return action, images, local_cond
        else:
            global_cond = self.global_cond[idx].clone() if self.global_cond is not None else None
            return action, local_cond, global_cond
