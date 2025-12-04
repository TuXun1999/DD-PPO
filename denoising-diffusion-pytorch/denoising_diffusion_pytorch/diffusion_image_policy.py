from denoising_diffusion_pytorch_1d_cond import ConditionalUnet1D
from denoising_diffusion_pytorch_1d_cond_RL import GaussianDiffusion1DConditionalRL
from vision.multi_image_obs_encoder import MultiImageObsEncoder

class DiffusionImagePolicy():
    def __init__(self):      # need to add value
        super.__init__()
        
        # set action dim & action prediction horizon
        action_dim = dim_info['action']['shape']
        n_action_step = dim_info['action']['n_step']
        
        # set obs dim & latest obs steps (obs horizon)
        obs_encoder = MultiImageObsEncoder()
        obs_feature_dim = obs_encoder.output_shape()[0]
        n_obs_shape = dim_info['image']['n_step']
        
        # set input dim & global cond dim
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_shape
        
        # ======================== DD-PPO model instantiation ==========================
        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )
        
        model_baseline = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )
        
        diffusion = GaussianDiffusion1DConditionalRL(
            model,
            seq_length = seq_length,
            timesteps = 10,
            sampling_timesteps = 8, 
            ddim_sampling_eta = 1.0,
            objective = 'pred_noise'
        )
        
        diffusion_baseline = GaussianDiffusion1DConditionalRL(
            model_baseline,
            seq_length = seq_length,
            timesteps = 10,
            sampling_timesteps = 8, 
            ddim_sampling_eta = 1.0,
            objective = 'pred_noise'
        )
        
        # 这里到底要传多少参数？
        self.obs_encoder = obs_encoder
        self.diffusion = diffusion
        self.diffusion_baseline = diffusion_baseline
        self.noise_scheduler = #TODO
        
        
        
        pass
    
    # using pytorch-lightning?
    # ========== inference ==========
    
    
    # ========== training ==========
    