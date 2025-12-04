import torch
from denoising_diffusion_pytorch import ConditionalUnet1D, \
    GaussianDiffusion1DConditionalRL, Trainer1DCondRL, Dataset1DCond,\
        TransformerForDiffusion, RewardModel, NaiveCriticModel

class DiffusionPolicyCustom(object):
    """
    The wrapped-up object to load, train & execute the diffusion policy improved with RL
    """
    def __init__(self, state_dim, action_dim, device = "cuda:0", **kwargs):
        # Dimension of state & action
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.device = device
         
        # Load the trained diffusion policy 
        self.training_stats = torch.load(kwargs["training_stats"])

        # Hyperparameters
        obs_dim = self.state_dim
        seq_length = 4
        obs_length = 3
        action_dim = self.action_dim
        training_sq = self.training_stats["training_sq"]
        local_label = self.training_stats["local_label"]
        global_label = self.training_stats["global_label"]
        
        # Construct the diffusion models
        option = 'unet1d'  # 
        if option == "unet1d":
            model_baseline = ConditionalUnet1D(
                input_dim = action_dim,
                local_cond_dim = 1,
                global_cond_dim = obs_length * obs_dim,
            )
            model = ConditionalUnet1D(
                input_dim = action_dim,
                local_cond_dim = 1,
                global_cond_dim = obs_length * obs_dim,
            )
        else:
            # model = ConditionalUnet1D(
            #     input_dim = 6,
            #     local_cond_dim = 1,
            #     global_cond_dim = 6,
            # )
            pass # TODO: the transformer

        self.diffusion_baseline = GaussianDiffusion1DConditionalRL(
            model_baseline,
            seq_length = seq_length,
            timesteps = 10,
            sampling_timesteps = 8, 
            ddim_sampling_eta = 1.0,
            objective = 'pred_noise'
        )
        self.diffusion = GaussianDiffusion1DConditionalRL(
            model,
            seq_length = seq_length,
            timesteps = 10,
            sampling_timesteps = 8,
            ddim_sampling_eta = 1.0, 
            objective = 'pred_noise'
        )
        # Construct the rewarding model
        v_min = self.training_stats["v_min"]
        v_max = self.training_stats["v_max"]
        angular_v_max = self.training_stats["angular_v_max"]
        angular_v_min = self.training_stats["angular_v_min"]
        
        # self.rewarding_model = RewardModel(\
        #     state_dim = obs_length * obs_dim + action_dim,
        #     v_min = v_min, 
        #     v_max = v_max, 
        #     angular_v_max = angular_v_max, 
        #     angular_v_min = angular_v_min,\
        #     device=self.device)
        # self.rewarding_model.load_model(path = kwargs["rewarding_model_path"])
        
        """The analytical rewarding model"""
        self.rewarding_model = NaiveCriticModel(scale = 0.36, v_min = v_min, v_max = v_max, \
            angular_v_max = angular_v_max, angular_v_min = angular_v_min,\
            device=self.device)
        # Construct the trainer
        dataset = Dataset1DCond(training_sq, local_label, global_label)
        self.trainer = Trainer1DCondRL(
            self.diffusion_baseline, 
            self.diffusion,
            dataset = dataset,
            train_batch_size = 32,
            train_lr = 8e-5,
            train_num_steps = 5000,         # total training steps
            PPO_train_num_steps= 500,
            gradient_accumulate_every = 2,    # gradient accumulation steps
            ema_decay = 0.995,                # exponential moving average decay
            amp = True,                       # turn on mixed precision
            save_and_sample_every=100000,      # Force not to save the sample result
            results_folder=kwargs["results_folder"],     # Read the pretrained weights
            
            # RL rewarding model
            rewarding_model= self.rewarding_model
        )

        # Load the previously trained model
        self.trainer.load(1)

    def load_trainer_model(self, type):
        """
        Load the diffusion model finetuned with RL
        """
        self.trainer.load(type)
    def diffusion_policy_action(self, state, model="baseline", type = "average"):
        """
        The function to sample an action from the given states & execute it
        """
        if model == "baseline":
            assert self.diffusion_baseline is not None, "Diffusion model is not initialized."
            diffusion_model = self.diffusion_baseline
        elif model == "new":
            assert self.diffusion is not None, "Diffusion model is not initialized."
            diffusion_model = self.diffusion
        else:
            assert False, "Please select at least one diffusion model to continue"
            
        
        # Hyperparameters
        seq_length = 4
        action_dim = self.action_dim

        # Extract out the normalization stats
        v_min = self.training_stats["v_min"].cpu()
        v_max = self.training_stats["v_max"].cpu()
        angular_v_max = self.training_stats["angular_v_max"].cpu()
        angular_v_min = self.training_stats["angular_v_min"].cpu()
        
        # Execute the diffusion policy on the novel state
        batch_size_sample = 32 #state.shape[0]
        
        # Obtain the state
        global_label_sample = torch.repeat_interleave(state, batch_size_sample, dim=0) #torch.tile(state.squeeze(0), (batch_size_sample, 1)).float() # (2 x obs_length x obs_dim)
        local_label_sample = torch.zeros(batch_size_sample * state.shape[0], 1, seq_length).float() # This is constant


        # Sample the sequence (s)
        sampled_seq = diffusion_model.sample(batch_size = batch_size_sample * state.shape[0], \
                local_cond = local_label_sample, global_cond = global_label_sample) # (batch_size_sample * batch_size x action_dim x seq_length)
        
        if type == "random":
            # Traditionally, average over all sampled actions
            sampled_seq = sampled_seq.view(state.shape[0], batch_size_sample, action_dim, seq_length)
            traj_recon = sampled_seq[:, 0, :, :] # (batch_size x action_dim x seq_length)
            traj_recon = traj_recon.to(device='cpu') # NxDxT
            torch.cuda.synchronize()
            
        if type == "average":
            # Traditionally, average over all sampled actions
            sampled_seq = sampled_seq.view(state.shape[0], batch_size_sample, action_dim, seq_length)
            traj_recon = torch.mean(sampled_seq, dim = 1) # (batch_size x action_dim x seq_length)
            traj_recon = traj_recon.to(device='cpu') # NxDxT
            torch.cuda.synchronize()
            
        elif type == "rewarding":
            # Rank the predicted actions with rewards
            rewards = self.rewarding_model(global_label_sample.to(self.device), sampled_seq).squeeze() # (state_num * B_sample, )
            
            # Rashape them into (state_num, batich_size_sample_per_state, ...)
            rewards = rewards.view(state.shape[0], batch_size_sample)
            sampled_seq = sampled_seq.view(state.shape[0], batch_size_sample, action_dim, seq_length)
            
            traj_recon = sampled_seq[torch.arange(state.shape[0]), torch.argmax(rewards, dim=1), :, :]
            traj_recon.squeeze(1)
            
            traj_recon = traj_recon.to(device='cpu') # state_num x D x T
            torch.cuda.synchronize()
            
            
        assert traj_recon.shape[0] == state.shape[0], traj_recon.shape
        assert traj_recon.shape[1] == action_dim, traj_recon.shape
        assert traj_recon.shape[2] == seq_length, traj_recon.shape
        # Extract the normalization statistics
        traj_noisy_max_sel = torch.zeros(size=traj_recon.shape).to(device=traj_recon.device) 
        traj_noisy_min_sel = torch.zeros(size=traj_recon.shape).to(device=traj_recon.device)
        traj_noisy_max_sel[:, 0:3, :] = torch.tensor([v_max, v_max, angular_v_max]).unsqueeze(-1).to(device=traj_recon.device)
        traj_noisy_min_sel[:, 0:3, :] = torch.tensor([v_min, v_min, angular_v_min]).unsqueeze(-1).to(device=traj_recon.device)
        traj_recon = traj_recon * (traj_noisy_max_sel - traj_noisy_min_sel) + traj_noisy_min_sel
        # Average the action within the predictions separately
        action = torch.mean(traj_recon, dim=-1)
        return action
            