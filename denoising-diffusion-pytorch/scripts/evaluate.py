#!/usr/bin/env python3
"""
Evaluation script for Diffusion Policy.

Usage:
    python scripts/evaluate.py --checkpoint results/model-final.ckpt
    python scripts/evaluate.py --checkpoint results/model-final.ckpt --visualize
"""
import argparse
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from diffusion_policy import (
    DiffusionPolicyConfig,
    DiffusionPolicyModule,
    load_demo_data,
    se2_norm,
    SE2_to_SE3,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Diffusion Policy")

    parser.add_argument("-c", "--checkpoint", required=True, help="Path to checkpoint")
    parser.add_argument("-o", "--object", default=None, help="Object type (auto-detect from checkpoint)")
    parser.add_argument("-g", "--grasp-pose", type=int, default=None, help="Grasp pose index")
    parser.add_argument("-d", "--dataset", default="../collected_demos", help="Dataset directory")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=50, help="Maximum steps per episode")
    parser.add_argument("--action-length", type=int, default=8, help="Number of actions to sum")
    parser.add_argument("--visualize", action="store_true", help="Visualize trajectories with Open3D")
    parser.add_argument("--device", default="cuda", help="Device to run on")

    return parser.parse_args()


def run_episode(
    model: DiffusionPolicyModule,
    initial_obs: torch.Tensor,
    local_cond: torch.Tensor,
    max_steps: int,
    action_length: int,
    obs_dim: int = 3,
    device: str = "cuda",
):
    """
    Run a single evaluation episode.

    Args:
        model: Trained diffusion policy model
        initial_obs: Initial observation [2 * obs_length * obs_dim]
        local_cond: Local conditioning [1, seq_length]
        max_steps: Maximum number of steps
        action_length: Number of actions to sum for execution
        obs_dim: Observation dimension
        device: Device to run on

    Returns:
        Dictionary with episode results
    """
    model.eval()
    trajectory = []
    obs = initial_obs.clone()

    # Extract initial gripper and object poses
    gripper_pose = obs[-2 * obs_dim:-obs_dim].clone()
    object_pose = obs[-obs_dim:].clone()

    trajectory.append({
        'gripper_pose': gripper_pose.cpu().numpy(),
        'object_pose': object_pose.cpu().numpy(),
    })

    global_cond = obs.unsqueeze(0).to(device)
    local_cond = local_cond.unsqueeze(0).to(device)

    for step in range(max_steps):
        # Sample action from model
        with torch.no_grad():
            sampled_seq = model.sample(
                batch_size=1,
                local_cond=local_cond,
                global_cond=global_cond,
                unnormalize=True,
            )

        # Get action (sum of first action_length steps)
        action = sampled_seq[0, :, :action_length].sum(dim=1).cpu()

        # Update gripper pose
        gripper_pose = gripper_pose + action

        # Update object pose (assume object moves with gripper)
        object_pose[:2] = object_pose[:2] + action[:2]
        object_pose[2] = object_pose[2] + action[2]

        # Normalize angles
        if gripper_pose[2] > np.pi:
            gripper_pose[2] -= 2 * np.pi
        elif gripper_pose[2] < -np.pi:
            gripper_pose[2] += 2 * np.pi

        if object_pose[2] > np.pi:
            object_pose[2] -= 2 * np.pi
        elif object_pose[2] < -np.pi:
            object_pose[2] += 2 * np.pi

        trajectory.append({
            'gripper_pose': gripper_pose.cpu().numpy() if torch.is_tensor(gripper_pose) else gripper_pose,
            'object_pose': object_pose.cpu().numpy() if torch.is_tensor(object_pose) else object_pose,
        })

        # Update observation
        new_obs = torch.cat([gripper_pose, object_pose])
        global_cond = torch.cat([
            global_cond[0, 2 * obs_dim:],
            new_obs,
        ]).unsqueeze(0).to(device)

        # Check termination
        distance = se2_norm(object_pose)
        if distance < 0.05:
            break

    final_distance = se2_norm(object_pose)

    return {
        'trajectory': trajectory,
        'steps': step + 1,
        'final_distance': final_distance.item() if torch.is_tensor(final_distance) else final_distance,
        'success': final_distance < 0.05,
    }


def visualize_trajectory(trajectory, title="Trajectory"):
    """Visualize trajectory using Open3D."""
    try:
        import open3d as o3d
        from scipy.spatial.transform import Rotation as R
    except ImportError:
        print("Open3D not available. Skipping visualization.")
        return

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title)

    # World frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    frame.scale(0.2, [0, 0, 0])
    vis.add_geometry(frame)

    # Trajectory frames
    for i, step in enumerate(trajectory):
        object_pose = step['object_pose']

        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = R.from_euler('xyz', [0, 0, object_pose[2]], degrees=False).as_matrix()
        pose_matrix[:3, 3] = np.array([object_pose[0], object_pose[1], 0])

        obj_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        obj_frame.scale(0.1, [0, 0, 0])
        obj_frame.transform(pose_matrix)
        vis.add_geometry(obj_frame)

    vis.run()
    vis.destroy_window()


def main():
    args = parse_args()

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    model = DiffusionPolicyModule.load_from_checkpoint(args.checkpoint)
    model.to(args.device)
    model.eval()

    # Get config from model
    config = model.config

    # Override from args if provided
    object_type = args.object or config.data.object_type
    grasp_pose = args.grasp_pose or config.data.grasp_pose
    dataset_dir = args.dataset

    # Load demo data for initial observations
    demo_data = load_demo_data(dataset_dir, object_type, grasp_pose)
    gripper_poses = demo_data["gripper_poses"]
    object_poses = demo_data["object_poses"]

    obs_length = config.data.obs_length
    obs_dim = config.data.obs_dim
    pred_length = config.diffusion.seq_length

    # Run evaluation episodes
    results = []
    print(f"\nRunning {args.num_episodes} evaluation episodes...")

    for episode in range(args.num_episodes):
        # Select random demo for initial state
        demo_idx = np.random.randint(0, len(gripper_poses))
        gripper_demo = gripper_poses[demo_idx]
        object_demo = object_poses[demo_idx]

        # Get initial observation
        initial_gripper = gripper_demo[:obs_length].flatten()
        initial_object = object_demo[:obs_length].flatten()
        initial_obs = torch.tensor(
            np.concatenate([initial_gripper, initial_object]),
            dtype=torch.float32,
        )

        # Local conditioning (placeholder)
        local_cond = torch.zeros(1, pred_length)

        # Run episode
        result = run_episode(
            model=model,
            initial_obs=initial_obs,
            local_cond=local_cond,
            max_steps=args.max_steps,
            action_length=args.action_length,
            obs_dim=obs_dim,
            device=args.device,
        )

        results.append(result)
        print(f"  Episode {episode + 1}: steps={result['steps']}, "
              f"final_distance={result['final_distance']:.4f}, "
              f"success={result['success']}")

        # Visualize if requested
        if args.visualize:
            visualize_trajectory(result['trajectory'], f"Episode {episode + 1}")

    # Print summary
    success_rate = sum(r['success'] for r in results) / len(results)
    avg_steps = sum(r['steps'] for r in results) / len(results)
    avg_distance = sum(r['final_distance'] for r in results) / len(results)

    print(f"\n=== Evaluation Summary ===")
    print(f"Success rate: {success_rate * 100:.1f}%")
    print(f"Average steps: {avg_steps:.1f}")
    print(f"Average final distance: {avg_distance:.4f}")


if __name__ == "__main__":
    main()
