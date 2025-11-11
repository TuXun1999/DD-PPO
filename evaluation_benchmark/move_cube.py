import numpy as np
import torch

import argparse
import os

# from rl.torchrl.trainers.helpers import replay_buffer

from utils import get_cfgs, MoveCubeEnv


import policy
import genesis as gs
from tqdm import tqdm
import json
from scipy.spatial.transform import Rotation as R
"""
Runs policy for X episodes and returns reward
"""
import wandb
import random
from genesis.utils.geom import xyz_to_quat
project="object-moving-se2"
config = {
    "RL algorithm": "TD3",
    "dataset": "banana",
}
# The function to fill up the replay buffer
def replay_buffer_fill(replay_buffer, training_stats_filename):
    # Store the data into the replay buffer
	# Load the normalization statistics
	training_stats = torch.load(training_stats_filename)

	actions = training_stats["actions"].cpu().numpy()  # Un-normalized actions
	local_label = training_stats["local_label"].cpu().numpy()
	global_label = training_stats["global_label"].cpu().numpy()
	next_states = training_stats["next_states"].cpu().numpy()
	rewards = training_stats["rewards"].cpu().numpy()
	done_signals = training_stats["done_signals"].cpu().numpy()
	
	N = next_states.shape[0]
	print("BUffer size: ", N)
	for i in range(N):
		state = global_label[i]
		action = actions[i]
		next_state = next_states[i].flatten()
		reward = rewards[i]
		done_bool = done_signals[i]
		replay_buffer.add(state, action[:, 0], \
					next_state, reward, done_bool)




# The function to train the policy
def train_policy(policy, replay_buffer, args, algorithm = "TD3"):
	if algorithm == "TD3":
		print("*** Pretraining the RL agent using TD3 ***")
		# Pretrain the RL agent
		with wandb.init(project=project, config=config) as run:
			for i in tqdm(range(10000)):
				# Pretrain the Q-value function
				policy.train(replay_buffer, args.batch_size, run)
	elif algorithm == "IBRL":
		# Pretrain the RL agent with Imitation Learning
		print("*** Pretraining the RL agent further with IL policy ***")
		for i in tqdm(range(50)):
			# Pretrain the IBRL policy
			policy.train_ibrl(replay_buffer, args.batch_size)

# Function to adapt the action in one grasp pose to the other grasp pose
# Policy evaluation function
def eval_policy(policy, policy_types, env, args, grasp_pose = "g1"):
	# Evaluation metrics
	evaluations = []

 
	
	if args.render_video:
		env.cam.start_recording()
	# Generate several sample starting poses
	cube_pos_list = []
	cube_quat_list = []
	rollout_num = args.rollout_num
	env_reset_num = 1
	for i in range(rollout_num):
		angle = torch.rand(size=(env_reset_num, ), device=gs.device) * 2 * torch.pi
		cube_pos = torch.hstack((\
			env.horizon_size * torch.cos(angle).view(-1,1), 
			env.horizon_size * torch.sin(angle).view(-1,1), 
			0.025 * torch.ones((env_reset_num, 1), device=gs.device))
		)
		cube_angle = torch.zeros((env_reset_num, 3), device=gs.device)
		cube_angle[:, 2] = torch.rand(size=(env_reset_num, ), device=gs.device) * np.pi * 2 - np.pi
		cube_quat = xyz_to_quat(cube_angle)
		
		# Append the initial cube_pos & cube_quat to the list
		cube_pos_list.append(cube_pos)
		cube_quat_list.append(cube_quat)
	

 	# The helper function to evaluate the policy
	for policy_to_use in policy_types:
		print(f"*** Evaluating {policy_to_use} on the object {args.env}")
		for i in range(len(cube_pos_list)):
			# Reset the env
			state, _ = env.reset(cube_pos_list[i], cube_quat_list[i])
			done = False
			truncated = False
			episode_reward = 0
			
			diffusion_model_to_use = policy_to_use.split("-")[0]
			action_sample_type = policy_to_use.split("-")[1]
   
			while not (done or truncated):
				action = policy.diffusion_policy_action(state, model = diffusion_model_to_use, type = action_sample_type)
			
				# Perform action
				next_state, reward, done, truncated, _ = env.step(action)
				
				# Extract out the results
				reward = reward.item()  # Convert to scalar
				done_bool = done and not truncated # If the task is completed, and not truncated
				done_bool = float(done_bool)
				

				# Update the statistics
				state = next_state
				episode_reward = reward


			# When the episode is completed or truncated
			print(f"Episode Num: {i+1} Episode T: {env.episode_length_buf[0].item()} Final Reward: {episode_reward:.3f} Reached: {done.squeeze(0).cpu().numpy()}")
			episode_length = env.episode_length_buf[0].item()

			evaluations.append([episode_reward, episode_length, done.squeeze(0).cpu().numpy()])
			
			# Reset episode variables
			if args.render_video:
				video_filename = f"{args.env}_{policy_to_use}_{grasp_pose}_{i}_{done.squeeze(0).cpu().numpy()}.mp4"
				env.cam.stop_recording(save_to_filename="./results/video/" + args.env + "/"+ video_filename, fps=60)

				# Re-start a new epsiode
				env.cam.start_recording()
    
			# Evaluate episode
			# if (t + 1) % args.eval_freq == 0:
			# 	evaluations.append(eval_policy(policy, args.env, args.seed))
			# 	np.save(f"./results/{file_name}", evaluations)
			# 	if args.save_model: policy.save(f"./models/{file_name}")
		print("Saving files")
		evaluation_file_name = f"{args.env}_{policy_to_use}_{grasp_pose}.npy"
		np.save(f"./results/{evaluation_file_name}", np.array(evaluations))
		print("Files saved")
	if args.render_video:
			env.cam.stop_recording() # The last incomplete trial doesn't need to be stored

def eval_policy_ppo(policy, policy_types, env, args, grasp_pose = "g1"):
	"A temporary function to evaluate the PPO finetuned model"

	# Generate several sample starting poses
	cube_pos_list = []
	cube_quat_list = []
	rollout_num = args.rollout_num
	env_reset_num = 1
	for i in range(rollout_num):
		angle = torch.rand(size=(env_reset_num, ), device=gs.device) * 2 * torch.pi
		cube_pos = torch.hstack((\
			env.horizon_size * torch.cos(angle).view(-1,1), 
			env.horizon_size * torch.sin(angle).view(-1,1), 
			0.025 * torch.ones((env_reset_num, 1), device=gs.device))
		)
		cube_angle = torch.zeros((env_reset_num, 3), device=gs.device)
		cube_angle[:, 2] = torch.rand(size=(env_reset_num, ), device=gs.device) * np.pi * 2 - np.pi
		cube_quat = xyz_to_quat(cube_angle)
		
		# Append the initial cube_pos & cube_quat to the list
		cube_pos_list.append(cube_pos)
		cube_quat_list.append(cube_quat)
	

 	# The helper function to evaluate the policy
	for policy_to_use in policy_types:
		print(f"*** Evaluating {policy_to_use} on the object {args.env}")
		for i in range(len(cube_pos_list)):
			# Reset the env
			state, _ = env.reset(cube_pos_list[i], cube_quat_list[i])
			done = False
			truncated = False
			episode_reward = 0
			
			diffusion_model_to_use = policy_to_use.split("-")[0]
			action_sample_type = policy_to_use.split("-")[1]
   
			while not (done or truncated):
				action = policy.diffusion_policy_action(state, model = diffusion_model_to_use, type = action_sample_type)
			
				# Perform action
				next_state, reward, done, truncated, _ = env.step(action)
				
				# Extract out the results
				reward = reward.item()  # Convert to scalar
				done_bool = done and not truncated # If the task is completed, and not truncated
				done_bool = float(done_bool)
				

				# Update the statistics
				state = next_state
				episode_reward = reward


			# When the episode is completed or truncated
			print(f"Episode Num: {i+1} Episode T: {env.episode_length_buf[0].item()} Final Reward: {episode_reward:.3f} Reached: {done.squeeze(0).cpu().numpy()}")
			
	policy.load_trainer_model(2)  # Load the PPO finetuned model
	print("Evaluating the PPO finetuned model")
    # The helper function to evaluate the policy
	for policy_to_use in policy_types:
		print(f"*** Evaluating {policy_to_use} on the object {args.env}")
		for i in range(len(cube_pos_list)):
			# Reset the env
			state, _ = env.reset(cube_pos_list[i], cube_quat_list[i])
			done = False
			truncated = False
			episode_reward = 0
			
			diffusion_model_to_use = policy_to_use.split("-")[0]
			action_sample_type = policy_to_use.split("-")[1]
   
			while not (done or truncated):
				action = policy.diffusion_policy_action(state, model = diffusion_model_to_use, type = action_sample_type)
			
				# Perform action
				next_state, reward, done, truncated, _ = env.step(action)
				
				# Extract out the results
				reward = reward.item()  # Convert to scalar
				done_bool = done and not truncated # If the task is completed, and not truncated
				done_bool = float(done_bool)
				

				# Update the statistics
				state = next_state
				episode_reward = reward


			# When the episode is completed or truncated
			print(f"Episode Num: {i+1} Episode T: {env.episode_length_buf[0].item()} Final Reward: {episode_reward:.3f} Reached: {done.squeeze(0).cpu().numpy()}")

		
if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", default="HalfCheetah-v2")          # OpenAI gym environment name
	parser.add_argument("-v", "--vis", action="store_true", default=False)
	parser.add_argument("--rollout_num", default=10, type=int)   # Number of rollouts
	parser.add_argument("-r", "--render_video", action="store_true", default=False)
	parser.add_argument("-d", "--dataset", default = "./trained_models") # Where to read the pretrained models
	parser.add_argument("--wandb", "-w", action="store_true", default=False)
	"""
	Stage 0: Specifications
 	"""
	args = parser.parse_args()


	print("---------------------------------------")
	print(f"Env: {args.env}")
	print("---------------------------------------")

	if not os.path.exists("./results"):
		os.makedirs("./results")


	env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
	if args.vis:
		env_cfg["visualize_target"] = True
	gs.init(logging_level="warning")
	
 
	"""
	Stage 1: Environment Setup
	"""
	## Step 1: Load the object model and the grasp pose
	obj_filename = "../object_models/" + args.env + "/textured.obj"

	# Read the two grasp poses
	grasp_pose_file = args.dataset + "/" + args.env + "/grasp_pose.json"
	with open(grasp_pose_file, 'r') as file:
		data = json.load(file)
		g1 = np.array(data["grasp_pose_1"])
		g2 = np.array(data["grasp_pose_2"])

	## Step 2: Create the environment
	env_g1 = MoveCubeEnv(
		obj_filename=obj_filename,
		grasp_pose=g1,
		num_envs=1,
		env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.vis,
    )
	
	
	## Step 3: Diffusion policy initialization
	kwargs = {
		"action_dim": 3,
		"state_dim": 3
	}
	training_stats_g1 = "../trained_models/" + args.env + "/grasp_pose1/training_stats.pth"
	training_stats_g2 = "../trained_models/" + args.env + "/grasp_pose2/training_stats.pth"
	rewarding_model_g1 = "../trained_models/" + args.env + "/grasp_pose1/results/rewarding_model.pth"
	rewarding_model_g2 = "../trained_models/" + args.env + "/grasp_pose2/results/rewarding_model.pth"
	# The pretrained weights
	result_g1 = "../trained_models/" + args.env + "/grasp_pose1/results"
	result_g2 = "../trained_models/" + args.env + "/grasp_pose2/results"
 
	kwargs["training_stats"] = training_stats_g1
	kwargs["rewarding_model_path"] = rewarding_model_g1
	kwargs["results_folder"] = result_g1
	policy_g1 = policy.DiffusionPolicyCustom(**kwargs)
 
	kwargs["training_stats"] = training_stats_g2
	kwargs["rewarding_model_path"] = rewarding_model_g2
	kwargs["results_folder"] = result_g2
	# policy_g2 = policy.DiffusionPolicyCustom(**kwargs)

	
	"""
 	Stage 2: Comparison between IL, RL, IBRL policies on g1
  	"""
	eval_policy(policy_g1, ["baseline-random", "baseline-rewarding"], env_g1, args, grasp_pose = "g1")
	

	# eval_policy_ppo(policy_g1, ["baseline-rewarding"], env_g1, args, grasp_pose = "g1")

 	# train_policy(policy_g1, replay_buffer_g1, args, algorithm="IBRL")
	# eval_policy(policy_g1, "IBRL", max_action, env_g1, args, grasp_pose = "g1", replay_buffer = None)
	
 	# """
	# Stage 3: Comparison between IL, finetuned, trained IBRL on g2
 	# """
	env_g1.close()
	
	gs.init(logging_level="warning")
	env_g2 = MoveCubeEnv(
		obj_filename=obj_filename,
		grasp_pose=g2,
		num_envs=1,
		env_cfg=env_cfg,
		obs_cfg=obs_cfg,
		reward_cfg=reward_cfg,
		command_cfg=command_cfg,
		show_viewer=args.vis,
	)
	eval_policy(policy_g1, ["baseline-random", "baseline-rewarding"], env_g2, args, grasp_pose = "g2")
	# eval_policy(policy_g2, "diffusion", max_action, env_g2, args, grasp_pose = "g2", replay_buffer = None)
	# train_policy(policy_g2, replay_buffer_g2, args, algorithm="TD3")
	# train_policy(policy_g2, replay_buffer_g2, args, algorithm="IBRL") # Finetune using IBRL policy on g2
	# eval_policy(policy_g2, "IBRL", max_action, env_g2, args, grasp_pose = "g2", replay_buffer = None)

	# eval_policy(policy_g1, "IBRL", max_action, env_g2, args, grasp_pose = "g2-finetuned", replay_buffer = replay_buffer_finetune)

