"""
Configuration dataclasses for Diffusion Policy.
"""
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple


@dataclass
class ImageEncoderConfig:
    """Configuration for the image encoder."""
    encoder_type: str = "resnet18"  # "resnet18", "resnet34", "resnet50", "resnet101"
    output_dim: int = 256  # Output feature dimension after projection
    pretrained: bool = True  # Use ImageNet pretrained weights
    freeze_backbone: bool = False  # Freeze backbone weights for fine-tuning


@dataclass
class ModelConfig:
    """Configuration for the diffusion model backbone."""
    backbone: str = "unet1d"  # "unet1d" or "transformer"
    input_dim: int = 3  # Action dimension
    local_cond_dim: int = 1  # Local conditioning dimension
    global_cond_dim: int = 12  # 2 * obs_length * obs_dim

    # Conditioning mode: "action_only" uses action history, "image_only" uses images
    conditioning_mode: Literal["action_only", "image_only"] = "action_only"
    image_encoder: Optional[ImageEncoderConfig] = None

    # UNet1D specific
    down_dims: List[int] = field(default_factory=lambda: [256, 512, 1024])
    diffusion_step_embed_dim: int = 256
    kernel_size: int = 3
    n_groups: int = 8
    cond_predict_scale: bool = False

    # Transformer specific
    n_layer: int = 12
    n_head: int = 12
    n_emb: int = 768
    p_drop_emb: float = 0.1
    p_drop_attn: float = 0.1
    causal_attn: bool = False
    time_as_cond: bool = True
    n_cond_layers: int = 0


@dataclass
class DiffusionConfig:
    """Configuration for the diffusion process."""
    seq_length: int = 16  # Prediction horizon
    timesteps: int = 10  # Total diffusion timesteps
    sampling_timesteps: int = 8  # DDIM sampling steps
    ddim_sampling_eta: float = 1.0  # 0.0 for deterministic, 1.0 for stochastic
    objective: str = "pred_noise"  # Only pred_noise is supported
    beta_schedule: str = "cosine"  # "linear" or "cosine"
    auto_normalize: bool = True


@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 32
    lr: float = 1e-4
    num_steps: int = 5000
    gradient_accumulate: int = 2
    max_grad_norm: float = 1.0

    # EMA settings
    ema_decay: float = 0.995
    ema_update_every: int = 10

    # Mixed precision
    use_amp: bool = True
    mixed_precision_type: str = "fp16"

    # Optimizer
    adam_betas: Tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 1e-3

    # Checkpointing
    save_every: int = 1000
    results_folder: str = "./results"

    # Logging
    use_wandb: bool = False
    wandb_project: str = "diffusion-policy"
    wandb_run_name: Optional[str] = None


@dataclass
class DataConfig:
    """Configuration for data loading."""
    dataset_dir: str = "../collected_demos"
    object_type: str = "banana"
    grasp_pose: int = 1

    # Observation settings
    obs_length: int = 2  # Number of observation steps
    obs_dim: int = 3  # Dimension of each observation

    # Action settings
    pred_length: int = 16  # Prediction horizon
    action_dim: int = 3  # Dimension of action space
    action_length: int = 8  # Number of actions to sum for execution

    # Image settings
    use_images: bool = False  # Whether to use images as conditioning
    num_image_history: int = 1  # Number of historical image frames to use

    # DataLoader settings
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class DiffusionPolicyConfig:
    """Top-level configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def __post_init__(self):
        """Compute derived values after initialization."""
        # Update seq_length to match pred_length
        self.diffusion.seq_length = self.data.pred_length
        # Update input_dim to match action_dim
        self.model.input_dim = self.data.action_dim

        # Compute global_cond_dim based on conditioning mode
        if self.model.conditioning_mode == "image_only":
            # Create default image encoder config if not provided
            if self.model.image_encoder is None:
                self.model.image_encoder = ImageEncoderConfig()
            # global_cond_dim = encoder_output_dim * num_image_history
            self.model.global_cond_dim = (
                self.model.image_encoder.output_dim * self.data.num_image_history
            )
            # Enable image usage in data config
            self.data.use_images = True
        else:
            # action_only mode: use action history as global conditioning
            self.model.global_cond_dim = 2 * self.data.obs_length * self.data.obs_dim
