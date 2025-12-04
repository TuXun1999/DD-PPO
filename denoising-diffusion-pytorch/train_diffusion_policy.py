import torch
import os
from denoising_diffusion_pytorch import ConditionalUnet1D, \
    GaussianDiffusion1DConditionalRL, Trainer1DCondRL, Dataset1DCond,\
        TransformerForDiffusion, RewardModel, NaiveCriticModel
from denoising_diffusion_pytorch.vision.multi_image_obs_encoder import MultiImageObsEncoder
from denoising_diffusion_pytorch.vision.model_getter import get_resnet
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import scipy
import json
import pickle
import argparse

torch.cuda.set_device(0) 

"""
Global hyperparameters
"""
obs_length = 3 # Length of history of observations
obs_dim = 3 # Dimension of the observed states
seq_length = 4 # Length of sequence of action to predict
action_dim = 3 # Dimension of the predicted action
action_repeat = 5 # Number actions to predict for each state

option = 'unet1d'  # Model hierarchy

"""
Main helper functions
"""
def select_closest_sample(matrix, array):
    # The function to choose the index of row in the matrix closest to array
    assert matrix.shape[1] == array.shape[0] 
    matrix_reduce = matrix - array 
    matrix_norm = torch.norm(matrix_reduce, dim=-1)
    return torch.argmin(matrix_norm)

def se2norm(array):
    # Find the norm of a se2 vector; use l2 norm temporarily
    return torch.sqrt(array[0]**2 + array[1]**2) + 0.6 * torch.norm(array[2])

def SE2ToSE3(vector):
    '''
    The function to convert an SE2 vector to SE3 matrix ([x,y,theta] -> 4x4 matrix)
    '''
    theta = vector[2]
    rot = R.from_euler('z', theta, degrees=False).as_matrix()
    res = np.eye(4)
    res[:3, :3] = rot
    res[:3, 3] = np.array([vector[0], vector[1], 0])
    return res


def create_dataset(args):
    """
    The function to preprocess the demos & Create the training dataset
    """
    ## Step 1: Read the demos
    # Hyperparameters
    obj_type = args.object
    dataset_dir = args.dataset
    grasp_pose = "grasp_pose" + str(args.grasp_pose)
    filename = dataset_dir + "/" + obj_type + "/" + grasp_pose + "/" + obj_type + ".pkl"
    normalization_file = dataset_dir + "/" + obj_type + "/" + grasp_pose + "/" + "training_stats.pth"
    
    with open(filename, 'rb') as f:
        data  = pickle.load(f)
        
    # Create a new dataset only if we want to train diffusion policy from scratch again
    create_new_dataset = (args.train_mode == "train_ddpm")
    if not create_new_dataset:
        # If no need to create a new training dataset for diffusion policy, 
        # read the stats and return directly
        training_stats = torch.load(normalization_file)
        training_sq = training_stats["training_sq"]
        
        # Build up the training dataset
        local_label = training_stats["local_label"]
        global_label = training_stats["global_label"]
        training_dataset = Dataset1DCond(training_sq, local_label, global_label)
        
        temp_dict = dict()     # for the first time, I don't add low_dim_feature
        shape_meta = {'obs': {}}
        shape_meta['obs']['overhead_img'] = {'shape': (3, 480, 640), 'type': 'rgb'}
        # shape_meta['obs']['low_dim_feature'] = {'shape': (obs_length * obs_dim,), 'type': 'low_dim'}

        temp_dict['shape_dict'] = shape_meta
        
        return data, training_dataset, training_stats, temp_dict
        
    

    # Extract the gripper & object poses
    gripper_poses = data["gripper_poses"]
    object_poses = data["object_poses"]

    grasp_pose = data["grasp_pose"]
    img_data = data["img_data"]

    traj_noisy = []
    global_label = []
    local_label = []
    img_cond = []
    rewards = []
    next_states = []
    
    ## Step 2: Build up the training samples
    for i in range(len(gripper_poses)):
        # Select samples with a gap between them => ensure a sufficient displacement
        gripper_poses_one_demo = gripper_poses[i][::8] # time_steps x 3
        object_poses_one_demo = object_poses[i][::8] # time_steps x 3
        poses_one_demo = np.hstack((gripper_poses_one_demo, object_poses_one_demo)) # time_steps x 6
        
        overhead_image_one_demo = img_data['overhead_view'][::8]
        overhead_image_one_demo = np.transpose(overhead_image_one_demo, (0, 3, 1, 2)) # from (N, H, W, C) to (N, C, H, W)
        # print(f'overhead_image_one_demo shape: {overhead_image_one_demo.shape}')

        # _, poses_unique_idx = np.unique(poses_one_demo, axis=0, return_index=True) # Remove the duplicate elements
        # poses_one_demo = poses_one_demo[poses_unique_idx]
        demo_length = poses_one_demo.shape[0]
        
        # for j in range(obs_length-1, demo_length-seq_length-1):
        for j in range(obs_length - 1, demo_length - 2):
            # Extract out the observations
            obs_gripper = poses_one_demo[j-obs_length+1:j+1, 0:action_dim].flatten()
            assert obs_gripper.shape[0] == obs_length * action_dim, "incorrect shape: " + str(obs_gripper.shape[0])
            obs_obj = poses_one_demo[j-obs_length+1:j+1, action_dim:].flatten()
            assert obs_obj.shape[0] == obs_length * obs_dim, "incorrect shape: " + str(obs_obj.shape[0])
            
            # Extract out the observation -- Image
            img_obj = overhead_image_one_demo[j-(obs_length-1) : j+1] # [obs_length, C, H, W]
            img_cond.append(img_obj)
            
            # Find the sequence of actions
            # action = poses_one_demo[j+1:j+seq_length+1, 0:action_dim] - poses_one_demo[j:j+seq_length, 0:action_dim]
            action = poses_one_demo[j+1, 0:action_dim] - poses_one_demo[j, 0:action_dim]
            action = np.tile(action, (seq_length, 1))   # why does the action need to be the same?
            assert action.shape[0] == seq_length and action.shape[1] == action_dim, "incorrect shape: " + str(action.shape[0])
            traj_noisy.append(action)

            '''
            NOTE: Look at both gripper pose & object pose => strange RL behaviors
            Current approach: only look at object pose
            '''
            # obs = np.concatenate([obs_gripper, obs_obj], axis=-1)
            obs = obs_obj
            global_label.append(obs)
            
            
            # Calculate the rewards
            reward = se2norm(torch.from_numpy(poses_one_demo[j+1, action_dim:]))# reward
            state_score = se2norm(torch.from_numpy(obs_obj[-3:]))# state
            rewards.append(10 * (-reward + state_score))

            # Find the next-states
            next_obs_gripper = poses_one_demo[j-obs_length+2:j+2, 0:action_dim].flatten()# next state
            next_obs_obj = poses_one_demo[j-obs_length+2:j+2, action_dim:].flatten()
            
            '''
            NOTE: look at both object pose & gripper pose => strange RL behaviors
            Current approach: only look at object poses
            '''
            next_obs = next_obs_obj
            # next_obs = np.concatenate([next_obs_gripper, next_obs_obj], axis=-1)
            next_states.append(next_obs)

    ## Step 3: Post-processing
    
    # Shape convention
    # traj_noisy: N (total segment number) x D (action_dim, 3) x T (seq_length, number of actions to predict)
    # global_label: N (total sample number) x G (global observation, which is the concatenation of the previous obs_length steps of observations)

    traj_noisy = np.array(traj_noisy)
    traj_noisy = np.transpose(traj_noisy, [0, 2, 1])
    global_label = np.array(global_label)
    rewards = np.array(rewards)
    
    # The local label is not used
    local_label = np.zeros((global_label.shape[0], 1, seq_length))

    # To torch tensor
    traj_noisy = torch.from_numpy(np.float32(traj_noisy))
    global_label = torch.from_numpy(np.float32(global_label))
    local_label = torch.from_numpy(np.float32(local_label))
    rewards = torch.from_numpy(np.float32(rewards))
    img_cond = torch.from_numpy(np.float32(img_cond))
    
    v_min = torch.min(traj_noisy[:, 0:2, :])
    v_max = torch.max(traj_noisy[:, 0:2, :])
    angular_v_min = torch.min(traj_noisy[:, 2, :])
    angular_v_max = torch.max(traj_noisy[:, 2, :])

    actions = traj_noisy.clone()

    # For the diffusion model, we have to normalize the samples at first
    traj_noisy_min = torch.tensor([v_min, v_min, angular_v_min]).unsqueeze(dim=-1).unsqueeze(dim=0)
    traj_noisy_max = torch.tensor([v_max, v_max, angular_v_max]).unsqueeze(dim=-1).unsqueeze(dim=0)
    traj_noisy_normalize = (traj_noisy - traj_noisy_min) / (traj_noisy_max - traj_noisy_min)
    training_sq = torch.nan_to_num(traj_noisy_normalize)

    ## Step 4: Build up the training dataset & Store the statistics
    training_dataset = Dataset1DCond(training_sq, local_label, img_cond)  # this is just an example, but you can formulate your own Dataset and pass it into the `Trainer1D` below

    # Save the normalization statistics
    training_stats = {
        "v_min": v_min,
        "v_max": v_max,
        "angular_v_min": angular_v_min,
        "angular_v_max": angular_v_max,
        "training_sq": training_sq,
        "actions": actions,
        "local_label": local_label,
        "global_label": global_label,
    }
    
    torch.save(training_stats, normalization_file)

    # get the conditional value shape
    temp_dict = dict()     # for the first time, I don't add low_dim_feature

    shape_meta = {'obs': {}}
    N, C, H, W = overhead_image_one_demo.shape
    shape_meta['obs']['overhead_img'] = {'shape': (C, H, W), 'type': 'rgb'}
    # shape_meta['obs']['low_dim_feature'] = {'shape': (obs_length * obs_dim,), 'type': 'low_dim'}

    temp_dict['shape_dict'] = shape_meta
    
    print(temp_dict)
    
    return data, training_dataset, training_stats, temp_dict




# rewarding_model = RewardModel(
#     state_dim = obs_length * obs_dim + action_dim,
#     v_min = v_min, v_max = v_max, \
#     angular_v_max = angular_v_max, angular_v_min = angular_v_min,\
#     device="cuda:0")
# rewarding_model.load_dataset(states = global_label, actions = actions[:, :, 0], rewards = rewards)
# rewarding_model.train()
# rewarding_model.save_model(path="./results/rewarding_model.pth")
# rewarding_model.model.eval()



def train_ddpm(training_dataset, training_stats, args, temp_dict):
    """
    The main function to train a diffusion model from the existing dataset
    """
    is_wandb = args.wandb
    if is_wandb:
        import wandb
        project="object-moving-se2"
        # NOTE: the stats used when finetuning PPO
        config = {
            "RL algorithm": "PPO",
            "dataset": "banana",
            "batch_size": 32,
            "old-policy": "new",
            "epochs": 500,
        }
        wandb_logger = wandb.init(project=project, config=config)
    else:
        wandb_logger = None

    # Create Img encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rgb_model = get_resnet(name='resnet18', weights=None)
    rgb_model = rgb_model.to(device)
    obs_encoder = MultiImageObsEncoder(
        shape_meta=temp_dict['shape_dict'],
        rgb_model=rgb_model,
        resize_shape=None,
        crop_shape=(76, 76),
        random_crop=True,
        use_group_norm=True,
        share_rgb_model=False,
        imagenet_norm=True
    )
    obs_encoder.to(device)
    obs_feature_dim = obs_encoder.output_shape()[0]
    print('obs_feature_dim: ', obs_feature_dim)
    print('obs_length: ', obs_length)

    # Create the networks
    '''
    NOTE: Look at only object poses
    '''
    if option == "unet1d":
        model_baseline = ConditionalUnet1D(
            input_dim = action_dim,
            local_cond_dim = 1, # original 1
            global_cond_dim = obs_feature_dim * obs_length,  #NOTE: n_obs_steps is obs_length ?
        )
        model = ConditionalUnet1D(
            input_dim = action_dim,
            local_cond_dim = 1,
            global_cond_dim = obs_feature_dim * obs_length,
        )
    else:
        model = TransformerForDiffusion(
                input_dim = action_dim,
                output_dim = action_dim,
                horizon = seq_length,
                local_cond_dim = 1,
                global_cond_dim = 2 * obs_length * obs_dim
            )

    diffusion_baseline = GaussianDiffusion1DConditionalRL(
        model_baseline,
        seq_length = seq_length,
        timesteps = 10,
        sampling_timesteps = 8, 
        ddim_sampling_eta = 1.0,
        objective = 'pred_noise'
    )
    diffusion = GaussianDiffusion1DConditionalRL(
        model,
        seq_length = seq_length,
        timesteps = 10,
        sampling_timesteps = 8,
        ddim_sampling_eta = 1.0, 
        objective = 'pred_noise'
    )
    # Create the rewarding model (TODO: only used for PPO finetuning, which is under progress)
    v_min = training_stats["v_min"]
    v_max = training_stats["v_max"]
    angular_v_min = training_stats["angular_v_min"]
    angular_v_max = training_stats["angular_v_max"]
    rewarding_model = NaiveCriticModel(scale = 0.6, v_min = v_min, v_max = v_max, \
        angular_v_max = angular_v_max, angular_v_min = angular_v_min,\
        device="cuda:0")
    
    # Construct the trainer
    obj_type = args.object
    dataset_dir = args.dataset
    grasp_pose = "grasp_pose" + str(args.grasp_pose)
    results_folder = dataset_dir + "/" + obj_type + "/" + grasp_pose + "/results"
    trainer = Trainer1DCondRL(
        diffusion_baseline, 
        diffusion,
        dataset = training_dataset,
        train_batch_size = 32,
        train_lr = 8e-5,
        train_num_steps = 1200,         # total training steps
        PPO_train_num_steps= 2000,
        gradient_accumulate_every = 2,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                       # turn on mixed precision
        save_and_sample_every = 100000,      # Force not to save the sample result
        results_folder = results_folder,     # Read the pretrained weights
        obs_encoder = obs_encoder,
        Img_as_global_cond = True,
        
        # RL rewarding model
        rewarding_model = rewarding_model,
        
        # wandb logger
        wandb_logger = wandb_logger
    )

    # Mode (max-reward action selected by default)
    # train_ddpm: train the ddpm (without RL)
    # load_ddpm: load the trained ddpm (without RL)
    # train_ddpm_ppo: load the trained ddpm, and finetune it with PPO
    # load_ddpm_ppo: load the finetuned ddpm
    training_mode = args.train_mode
    if training_mode == "train_ddpm":
        trainer.train()
        trainer.save(1)
    elif training_mode == "load_ddpm":
        trainer.load(1)
    elif training_mode == "train_ddpm_ppo":
        trainer.load(1)
        trainer.finetune_PPO()
        trainer.save(2)
    elif training_mode == "load_ddpm_ppo":
        trainer.load(2)
    else:
        assert False, "Unknown training mode"

    return diffusion, trainer, rewarding_model




if __name__ == "__main__":
	
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--object", default = "banana")    # Type of object
    parser.add_argument("-t", "--train-mode", default = "train_ddpm") # Type of training objective
    parser.add_argument("-g", "--grasp-pose", default = 1, type = int) # Type of grasp pose
    parser.add_argument("-v", "--visualization", action="store_true", default = False) # Visualization
    parser.add_argument("-d", "--dataset", default = "./trained_models") # Where to read the demos & where to store the data
    parser.add_argument("--wandb", "-w", action ="store_true", default = False)
    
    """
	Step 0: Specifications
 	"""
    args = parser.parse_args()
    
    
    """
    Step 1: Dataset creation
    """
    demo_data, training_dataset, training_stats, temp_dict = create_dataset(args)
    
    # (Optional) Visualize one data
    if args.visualization:
        gripper_poses = demo_data["gripper_poses"]
        object_poses = demo_data["object_poses"]
        global_label = training_stats["global_label"]
        local_label = training_stats["local_label"]
        v_min = training_stats["v_min"]
        v_max = training_stats["v_max"]
        angular_v_min = training_stats["angular_v_min"]
        angular_v_max = training_stats["angular_v_max"]
        # Visualize some of the data (32: important)
        vis_select_idx = np.random.randint(0, len(gripper_poses)) # The index of the demo to evaluate
        print(vis_select_idx)
        '''
        NOTE: now, look at both object poses and gripper poses => RL behaves strangely
        Current approach: only looks at object poses
        '''
        vis_demo_start = object_poses[vis_select_idx][0]
        # vis_demo_start = np.concatenate(
        #     [gripper_poses[vis_select_idx][0], object_poses[vis_select_idx][0]])
        vis_demo_start = np.tile(vis_demo_start, (obs_length, ))

        # select_idx = select_closest_sample(global_label, vis_demo_start)# The index of the starting location in global_label
        object_pose_test = object_poses[vis_select_idx]

        # Create the object frames
        vis1= o3d.visualization.Visualizer()
        vis1.create_window()

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        frame.scale(0.2, [0, 0, 0])

        vis1.add_geometry(frame)

        for object_pose in object_pose_test[::4]:
            pose = np.eye(4)
            pose[:3, :3] = R.from_euler('xyz', [0, 0, object_pose[2]], degrees=False).as_matrix()
            pose[:3, 3] = np.array([object_pose[0], object_pose[1], 0])
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
            frame.scale(0.1, [0, 0, 0])
            frame.transform(pose)
            vis1.add_geometry(frame)

        vis1.run()
        # Close all windows
        vis1.destroy_window()
    
    
    """
    Step 2: Train the diffusion model
    """
    diffusion, trainer, rewarding_model = train_ddpm(training_dataset, training_stats, args, temp_dict)
    
    """
    Step 3: (Optional) Test the the trained model
    """
    # if args.visualization:
    #     frame_poses = []
    #     batch_size_sample = 10
    #     # the first camera frame?
    #     global_label_sample = torch.tile(global_label[select_idx], (batch_size_sample, 1)) # (2 x obs_length x obs_dim)
    #     local_label_sample = torch.tile(local_label[select_idx], (batch_size_sample, 1)).unsqueeze(1) # This is constant

    #     '''
    #     NOTE: now, only consider the object pose
    #     '''
    #     obs_pose = global_label[select_idx][-obs_dim:]
    #     # obs_pose = torch.concatenate(
    #     #     [global_label[select_idx][(obs_length - 1) * action_dim : obs_length * action_dim],
    #     #      global_label[select_idx][-obs_dim:]])
    #     steps = 0
    #     while True:
    #         steps += 1
    #         frame_poses.append(obs_pose.numpy()) # obs_length x obs_dim x 1
    #         # print(local_label_sample.shape)
    #         # print(global_label_sample.shape)
            
    #         # Sample from the diffusion model
    #         sampled_seq = diffusion.sample(batch_size = batch_size_sample, \
    #                 local_cond = local_label_sample.to(trainer.device), global_cond = global_label_sample.to(trainer.device))
            
    #         # From experiments, involving an estimated rewarding model helps to improve the performance
    #         rewards = rewarding_model(global_label_sample.to(trainer.device), sampled_seq).squeeze() # (B, )
    #         traj_recon = sampled_seq[torch.argmax(rewards), :]
    #         traj_recon = traj_recon.to(device='cpu') # DxT
    #         torch.cuda.synchronize()

    #         # Un-normalize the sampled actions
    #         traj_noisy_max_sel = torch.tensor([v_max, v_max, angular_v_max]).unsqueeze(-1)
    #         traj_noisy_min_sel = torch.tensor([v_min, v_min, angular_v_min]).unsqueeze(-1)
    #         traj_recon = traj_recon * (traj_noisy_max_sel - traj_noisy_min_sel) + traj_noisy_min_sel

    #         '''
    #         NOTE: The following code block looks at both gripper pose & object pose
    #         '''
    #         # last_gripper_pose = obs_pose[0:action_dim]
    #         # last_object_pose = obs_pose[action_dim:]
    #         # action = torch.mean(traj_recon, dim=1) # TODO: correct the dimenstion
    #         # last_gripper_pose += action
    #         # # The action is applied on the gripper, but the object needs to updated differently
    #         # last_object_pose[0:2] += action[0:2] # Update the position
    #         # last_object_pose[2] += action[2] # Update the angle
    #         # obs_gripper_pose = global_label_sample[0][action_dim : obs_length * action_dim]
    #         # obs_gripper_pose = torch.concatenate((obs_gripper_pose, last_gripper_pose))
    #         # obs_object_pose = global_label_sample[0][obs_length * action_dim + obs_dim :]
    #         # obs_object_pose = torch.concatenate((obs_object_pose, last_object_pose))
    #         # global_label_sample = torch.concatenate([obs_gripper_pose, obs_object_pose])
    #         # obs_pose = torch.concatenate([last_gripper_pose, last_object_pose])
    #         # assert obs_pose.shape[0] == 2 * obs_dim
    #         # assert global_label_sample.shape[0] == 2 * obs_dim * obs_length
            
    #         '''
    #         NOTE: The following block only looks at the object pose
    #         '''
    #         last_object_pose = obs_pose.clone()
    #         action = torch.mean(traj_recon, dim=1) # TODO: correct the dimenstion
    #         # action = torch.sum(traj_recon, dim=1)
    #         # The action is applied on the gripper, but the object needs to updated differently
    #         # last_object_pose += action
    #         last_object_pose[0:2] += action[0:2] # Update the position
    #         last_object_pose[2] += action[2] # Update the angle
            
    #         obs_object_pose = global_label_sample[0][obs_dim :]
    #         obs_object_pose = torch.concatenate((obs_object_pose, last_object_pose))
    #         global_label_sample = obs_object_pose
            
    #         obs_pose = last_object_pose.clone()
            
            

    #         # Determine whether to exit
    #         if se2norm(last_object_pose) < 0.03 or steps >= 40:
    #             print("Final object pose: ")
    #             print(last_object_pose)
    #             print("Final timesteps: ")
    #             print(steps)
    #             break

    #         # Stack global_label_sample
    #         global_label_sample = torch.tile(global_label_sample, (batch_size_sample, 1))
            

    #     # Visualize the reconstructed gripper & object poses


    #     # Create the window to display everything
    #     vis= o3d.visualization.Visualizer()
    #     vis.create_window()


    #     frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    #     frame.scale(0.2, [0, 0, 0])
    #     vis.add_geometry(frame)

    #     for item in frame_poses:
    #         camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    #         camera_frame.scale(0.1, [0, 0, 0])
    #         object_pose = SE2ToSE3(item[-3:])
    #         camera_frame.transform(object_pose)
            
    #         vis.add_geometry(camera_frame)




    #     vis.run()
    #     # Close all windows
    #     vis.destroy_window()

