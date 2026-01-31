"""
Data preprocessing utilities for Diffusion Policy.

This module handles loading demonstration data from pickle files
and processing it into training samples.
"""
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch


@dataclass
class NormalizationStats:
    """Statistics for normalizing action data."""
    v_min: float
    v_max: float
    angular_v_min: float
    angular_v_max: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for saving."""
        return {
            "v_min": self.v_min,
            "v_max": self.v_max,
            "angular_v_min": self.angular_v_min,
            "angular_v_max": self.angular_v_max,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "NormalizationStats":
        """Create from dictionary."""
        return cls(
            v_min=d["v_min"],
            v_max=d["v_max"],
            angular_v_min=d["angular_v_min"],
            angular_v_max=d["angular_v_max"],
        )

    def normalize(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Normalize actions to [0, 1] range.

        Args:
            actions: Action tensor [batch, action_dim, seq_length]

        Returns:
            Normalized actions
        """
        min_vals = torch.tensor([self.v_min, self.v_min, self.angular_v_min])
        max_vals = torch.tensor([self.v_max, self.v_max, self.angular_v_max])
        min_vals = min_vals.unsqueeze(-1).unsqueeze(0).to(actions.device)
        max_vals = max_vals.unsqueeze(-1).unsqueeze(0).to(actions.device)
        normalized = (actions - min_vals) / (max_vals - min_vals)
        return torch.nan_to_num(normalized)

    def unnormalize(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Unnormalize actions from [0, 1] range.

        Args:
            actions: Normalized action tensor [batch, action_dim, seq_length]

        Returns:
            Unnormalized actions
        """
        min_vals = torch.tensor([self.v_min, self.v_min, self.angular_v_min])
        max_vals = torch.tensor([self.v_max, self.v_max, self.angular_v_max])
        min_vals = min_vals.unsqueeze(-1).unsqueeze(0).to(actions.device)
        max_vals = max_vals.unsqueeze(-1).unsqueeze(0).to(actions.device)
        return actions * (max_vals - min_vals) + min_vals


def load_demo_data(
    dataset_dir: str,
    object_type: str,
    grasp_pose: int,
) -> Dict[str, Any]:
    """
    Load demonstration data from pickle file.

    Args:
        dataset_dir: Base directory for datasets
        object_type: Type of object (e.g., "banana")
        grasp_pose: Grasp pose index

    Returns:
        Dictionary containing demonstration data
    """
    grasp_pose_str = f"grasp_pose{grasp_pose}"
    filename = Path(dataset_dir) / object_type / grasp_pose_str / f"{object_type}.pkl"

    with open(filename, 'rb') as f:
        data = pickle.load(f)

    return data


def process_demos(
    demo_data: Dict[str, Any],
    obs_length: int = 2,
    obs_dim: int = 3,
    pred_length: int = 16,
    action_dim: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, NormalizationStats]:
    """
    Process demonstration data into training samples.

    Args:
        demo_data: Raw demonstration data from pickle file
        obs_length: Number of observation timesteps
        obs_dim: Dimension of each observation
        pred_length: Prediction horizon (number of actions to predict)
        action_dim: Dimension of action space

    Returns:
        Tuple of (actions, local_cond, global_cond, normalization_stats)
    """
    gripper_poses = demo_data["gripper_poses"]
    object_poses = demo_data["object_poses"]

    traj_noisy = []
    global_label = []

    for i in range(len(gripper_poses)):
        gripper_poses_one_demo = gripper_poses[i]
        object_poses_one_demo = object_poses[i]
        poses_one_demo = np.hstack((gripper_poses_one_demo, object_poses_one_demo))
        demo_length = poses_one_demo.shape[0]

        for j in range(obs_length - 1, demo_length - pred_length - 1):
            # Extract observations (both gripper and object poses)
            obs = poses_one_demo[j - obs_length + 1:j + 1, :].flatten()
            global_label.append(obs)

            # Extract actions (delta positions)
            action = poses_one_demo[j + 1:j + pred_length + 1, 0:action_dim] - \
                     poses_one_demo[j:j + pred_length, 0:action_dim]

            # Normalize angles to [-pi, pi]
            action[action < -np.pi] += 2 * np.pi
            action[action > np.pi] -= 2 * np.pi

            traj_noisy.append(action)

    # Convert to tensors
    # Shape: [N, pred_length, action_dim] -> [N, action_dim, pred_length]
    traj_noisy = np.array(traj_noisy)
    traj_noisy = np.transpose(traj_noisy, [0, 2, 1])
    global_label = np.array(global_label)

    # Local label (placeholder, not used for base diffusion)
    local_label = np.zeros((global_label.shape[0], 1, pred_length))

    # Convert to torch
    actions = torch.from_numpy(np.float32(traj_noisy))
    global_cond = torch.from_numpy(np.float32(global_label))
    local_cond = torch.from_numpy(np.float32(local_label))

    # Compute normalization statistics
    v_min = torch.min(actions[:, 0:2, :]).item()
    v_max = torch.max(actions[:, 0:2, :]).item()
    angular_v_min = torch.min(actions[:, 2, :]).item()
    angular_v_max = torch.max(actions[:, 2, :]).item()

    stats = NormalizationStats(
        v_min=v_min,
        v_max=v_max,
        angular_v_min=angular_v_min,
        angular_v_max=angular_v_max,
    )

    # Normalize actions
    actions_normalized = stats.normalize(actions)

    return actions_normalized, local_cond, global_cond, stats


def save_training_stats(
    filepath: str,
    actions: torch.Tensor,
    local_cond: torch.Tensor,
    global_cond: torch.Tensor,
    stats: NormalizationStats,
) -> None:
    """
    Save training statistics and data to file.

    Args:
        filepath: Path to save file
        actions: Normalized action tensor
        local_cond: Local conditioning tensor
        global_cond: Global conditioning tensor
        stats: Normalization statistics
    """
    data = {
        "training_sq": actions,
        "local_label": local_cond,
        "global_label": global_cond,
        **stats.to_dict(),
    }
    torch.save(data, filepath)


def load_training_stats(
    filepath: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, NormalizationStats]:
    """
    Load training statistics and data from file.

    Args:
        filepath: Path to saved file

    Returns:
        Tuple of (actions, local_cond, global_cond, normalization_stats)
    """
    data = torch.load(filepath, weights_only=False)
    stats = NormalizationStats.from_dict(data)

    return (
        data["training_sq"],
        data["local_label"],
        data["global_label"],
        stats,
    )
