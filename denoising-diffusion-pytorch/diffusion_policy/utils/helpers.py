"""
Helper utilities for Diffusion Policy.
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


def se2_norm(array: torch.Tensor) -> torch.Tensor:
    """
    Compute the norm of an SE2 vector.

    Uses L2 norm for position and weighted angular component.

    Args:
        array: SE2 tensor [x, y, theta]

    Returns:
        Scalar norm value
    """
    return torch.sqrt(array[0]**2 + array[1]**2) + 0.6 * torch.abs(array[2])


def SE2_to_SE3(vector: np.ndarray) -> np.ndarray:
    """
    Convert an SE2 vector to SE3 matrix.

    Args:
        vector: SE2 vector [x, y, theta]

    Returns:
        4x4 SE3 transformation matrix
    """
    theta = vector[2]
    rot = R.from_euler('z', theta, degrees=False).as_matrix()
    res = np.eye(4)
    res[:3, :3] = rot
    res[:3, 3] = np.array([vector[0], vector[1], 0])
    return res


def SE3_to_SE2(matrix: np.ndarray) -> np.ndarray:
    """
    Convert an SE3 matrix to SE2 vector.

    Args:
        matrix: 4x4 SE3 transformation matrix

    Returns:
        SE2 vector [x, y, theta]
    """
    x = matrix[0, 3]
    y = matrix[1, 3]
    rot = R.from_matrix(matrix[:3, :3])
    theta = rot.as_euler('xyz', degrees=False)[2]
    return np.array([x, y, theta])


def normalize_angle(angle: float) -> float:
    """
    Normalize angle to [-pi, pi] range.

    Args:
        angle: Angle in radians

    Returns:
        Normalized angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def normalize_angles_tensor(angles: torch.Tensor) -> torch.Tensor:
    """
    Normalize angles tensor to [-pi, pi] range.

    Args:
        angles: Tensor of angles in radians

    Returns:
        Normalized angles tensor
    """
    return torch.atan2(torch.sin(angles), torch.cos(angles))


def update_pose(
    current_pose: torch.Tensor,
    action: torch.Tensor,
) -> torch.Tensor:
    """
    Update a pose given an action.

    Args:
        current_pose: Current SE2 pose [x, y, theta]
        action: Delta action [dx, dy, dtheta]

    Returns:
        New pose [x, y, theta]
    """
    new_pose = current_pose + action
    # Normalize the angle
    new_pose[2] = normalize_angles_tensor(new_pose[2])
    return new_pose
