"""
Data preprocessing utilities for Diffusion Policy.

This module handles loading demonstration data from pickle files
and processing it into training samples.
"""
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


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
    grasp_pose: int
) -> Dict[str, Any]:
    """Load demonstration data (supports both multi-file and legacy single-file formats)."""
    
    demo_folder = Path(dataset_dir) / object_type / f"grasp_pose{grasp_pose}"
    pkl_files = sorted(demo_folder.glob("*.pkl"))

    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {demo_folder}")

    # 1. Check legacy format using the first file
    with open(pkl_files[0], 'rb') as f:
        first_data = pickle.load(f)
    
    # If it's already a list, it's the legacy single-file format containing all trajectories
    if isinstance(first_data.get("gripper_poses"), list):
        return first_data

    # 2. Multi-file format: Load and aggregate
    result = {"gripper_poses": [], "object_poses": [], "img_data": []}
    grasp_matrix = None

    for pkl_path in pkl_files:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # Aggregate trajectory lists
        for key in result:
            if key in data:
                result[key].append(data[key])
        
        # Capture static data (overwrite is fine as it should be constant) # TODO: need to modify for multiple grasp poses
        if "grasp_pose" in data:
            grasp_matrix = data["grasp_pose"]

    # 3. Post-processing: Remove empty keys and add grasp_pose
    result = {k: v for k, v in result.items() if v}  # Filter out empty lists (e.g. no img_data)
    
    if grasp_matrix is not None:
        result["grasp_pose"] = grasp_matrix

    print(f"Loaded {len(result.get('gripper_poses', []))} trajectories from {demo_folder}")
    return result


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
            d_theta = action[:, 2]
            d_theta[d_theta < -np.pi] += 2 * np.pi
            d_theta[d_theta > np.pi] -= 2 * np.pi
            action[:, 2] = d_theta

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


def get_image_transform(image_size: Tuple[int, int] = (224, 224)) -> T.Compose:
    """
    Get image transformation pipeline for preprocessing.

    Args:
        image_size: Target image size (H, W)

    Returns:
        torchvision transform pipeline
    """
    return T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
    ])


def process_demos_with_images(
    demo_data: Dict[str, Any],
    obs_length: int = 2,
    obs_dim: int = 3,
    pred_length: int = 16,
    action_dim: int = 3,
    num_image_history: int = 1,
    image_key: str = "overhead_view",
    image_size: Tuple[int, int] = (224, 224),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], NormalizationStats]:
    """
    Process demonstration data with images into training samples.

    Args:
        demo_data: Raw demonstration data from pickle file
        obs_length: Number of observation timesteps
        obs_dim: Dimension of each observation
        pred_length: Prediction horizon (number of actions to predict)
        action_dim: Dimension of action space
        num_image_history: Number of historical image frames to use
        image_key: Key for selecting camera view ("overhead_view" or "forward_view")
        image_size: Target image size (H, W) for resizing

    Returns:
        Tuple of (actions, local_cond, global_cond, images, normalization_stats)
        images is None if no image data is available
    """
    gripper_poses = demo_data["gripper_poses"]
    object_poses = demo_data["object_poses"]
    img_data_list = demo_data.get("img_data", None)

    traj_noisy = []
    global_label = []
    image_samples = [] if img_data_list else None

    transform = get_image_transform(image_size)

    for i in range(len(gripper_poses)):
        gripper_poses_one_demo = gripper_poses[i]
        object_poses_one_demo = object_poses[i]
        poses_one_demo = np.hstack((gripper_poses_one_demo, object_poses_one_demo))
        demo_length = poses_one_demo.shape[0]

        # Get images for this trajectory if available
        traj_images = None
        if img_data_list and i < len(img_data_list):
            traj_img_data = img_data_list[i]
            if isinstance(traj_img_data, dict) and image_key in traj_img_data:
                traj_images = traj_img_data[image_key]  # Shape: (T, H, W, C)
            elif isinstance(traj_img_data, np.ndarray):
                traj_images = traj_img_data

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

            # Extract image history if available
            if traj_images is not None and image_samples is not None:
                img_history = []
                for k in range(num_image_history):
                    img_idx = j - k
                    if img_idx >= 0 and img_idx < len(traj_images):
                        img = traj_images[img_idx]
                        # Convert to PIL Image and apply transforms
                        if img.dtype == np.uint8:
                            pil_img = Image.fromarray(img)
                        else:
                            pil_img = Image.fromarray((img * 255).astype(np.uint8))
                        img_tensor = transform(pil_img)
                        img_history.append(img_tensor)
                    else:
                        # Pad with zeros if not enough history
                        img_history.append(torch.zeros(3, image_size[0], image_size[1]))

                # Stack history: [num_history, C, H, W]
                img_history = torch.stack(img_history[::-1], dim=0)  # Reverse to chronological order
                image_samples.append(img_history)

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

    # Stack images if available
    images = None
    if image_samples:
        images = torch.stack(image_samples, dim=0)  # [N, num_history, C, H, W]

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

    return actions_normalized, local_cond, global_cond, images, stats


def save_training_stats(
    filepath: str,
    actions: torch.Tensor,
    local_cond: torch.Tensor,
    global_cond: torch.Tensor,
    stats: NormalizationStats,
    images: Optional[torch.Tensor] = None,
) -> None:
    """
    Save training statistics and data to file.

    Args:
        filepath: Path to save file
        actions: Normalized action tensor
        local_cond: Local conditioning tensor
        global_cond: Global conditioning tensor
        stats: Normalization statistics
        images: Optional image tensor [N, num_history, C, H, W]
    """
    data = {
        "training_sq": actions,
        "local_label": local_cond,
        "global_label": global_cond,
        **stats.to_dict(),
    }
    if images is not None:
        data["images"] = images
    torch.save(data, filepath)


def load_training_stats(
    filepath: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], NormalizationStats]:
    """
    Load training statistics and data from file.

    Args:
        filepath: Path to saved file

    Returns:
        Tuple of (actions, local_cond, global_cond, images, normalization_stats)
        images is None if not present in the saved file
    """
    data = torch.load(filepath, weights_only=False)
    stats = NormalizationStats.from_dict(data)
    images = data.get("images", None)

    return (
        data["training_sq"],
        data["local_label"],
        data["global_label"],
        images,
        stats,
    )
