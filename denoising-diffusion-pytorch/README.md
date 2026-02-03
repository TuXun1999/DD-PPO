# Diffusion Policy

A PyTorch Lightning implementation of diffusion models for robot policy learning, with support for 1D conditional generation.

## Installation

```bash
# Basic installation
pip install -e .

# With visualization support (Open3D)
pip install -e ".[visualization]"

# With logging support (WandB, TensorBoard)
pip install -e ".[logging]"

# Full installation
pip install -e ".[visualization,logging,dev]"
```

### Dependencies

- Python >= 3.8
- PyTorch >= 2.0
- PyTorch Lightning >= 2.0
- einops
- numpy
- scipy

## Quick Start

### Using the Training Script

```bash
cd denoising-diffusion-pytorch
# Train with default settings
python scripts/train.py --object banana --grasp-pose 1

# Train with custom settings
python scripts/train.py \
    --object banana \
    --grasp-pose 1 \
    --batch-size 64 \
    --lr 1e-4 \
    --max-steps 10000 \
    --wandb
```

### Using the Python API

```python
from diffusion_policy import (
    DiffusionPolicyConfig,
    DiffusionPolicyModule,
    DiffusionPolicyDataModule,
    EMACallback,
)
import lightning as L

# Create configuration
config = DiffusionPolicyConfig()
config.model.backbone = "unet1d"  # or "transformer"
config.training.batch_size = 32
config.training.lr = 1e-4

# Setup data
data_module = DiffusionPolicyDataModule(config.data)
data_module.prepare_data()
data_module.setup()

# Create model
model = DiffusionPolicyModule(
    config=config,
    normalization_stats=data_module.get_normalization_stats(),
)

# Train
trainer = L.Trainer(
    max_steps=5000,
    callbacks=[EMACallback(decay=0.995)],
)
trainer.fit(model, data_module)
```

### Using Models Directly (Without Lightning)

```python
import torch
from diffusion_policy import (
    DiffusionPolicyConfig,
    build_backbone,
    build_diffusion,
)

# Create configuration
config = DiffusionPolicyConfig()

# Build model
backbone = build_backbone(config.model, config.diffusion.seq_length)
diffusion = build_diffusion(backbone, config.diffusion)

# Training forward pass
actions = torch.randn(32, 3, 16)       # [batch, action_dim, seq_length]
local_cond = torch.randn(32, 1, 16)    # [batch, local_cond_dim, seq_length]
global_cond = torch.randn(32, 12)      # [batch, global_cond_dim]

loss = diffusion(actions, local_cond, global_cond)
loss.backward()

# Sampling
with torch.no_grad():
    samples = diffusion.sample(
        batch_size=1,
        local_cond=local_cond[:1],
        global_cond=global_cond[:1],
    )
```

## Configuration

All configuration is done via Python dataclasses. See `diffusion_policy/config.py` for full details.

### ModelConfig

```python
@dataclass
class ModelConfig:
    backbone: str = "unet1d"        # "unet1d" or "transformer"
    input_dim: int = 3              # Action dimension
    local_cond_dim: int = 1         # Local conditioning dimension
    global_cond_dim: int = 12       # Global conditioning dimension

    # UNet1D specific
    down_dims: List[int] = [256, 512, 1024]

    # Transformer specific
    n_layer: int = 12
    n_head: int = 12
    n_emb: int = 768
```

### DiffusionConfig

```python
@dataclass
class DiffusionConfig:
    seq_length: int = 16            # Prediction horizon
    timesteps: int = 10             # Diffusion timesteps
    sampling_timesteps: int = 8     # DDIM sampling steps
    ddim_sampling_eta: float = 1.0  # 0.0=deterministic, 1.0=stochastic
    objective: str = "pred_noise"   # Prediction objective
    beta_schedule: str = "cosine"   # Beta schedule type
```

### TrainingConfig

```python
@dataclass
class TrainingConfig:
    batch_size: int = 32
    lr: float = 1e-4
    num_steps: int = 5000
    gradient_accumulate: int = 2
    ema_decay: float = 0.995
    use_amp: bool = True
```

## Project Structure

```
denoising-diffusion-pytorch/
├── diffusion_policy/           # Main package
│   ├── config.py               # Configuration dataclasses
│   ├── models/
│   │   ├── unet_1d.py          # ConditionalUnet1D backbone
│   │   ├── transformer.py      # TransformerForDiffusion backbone
│   │   └── diffusion.py        # GaussianDiffusion1D
│   ├── lightning/
│   │   ├── module.py           # DiffusionPolicyModule
│   │   ├── datamodule.py       # DiffusionPolicyDataModule
│   │   └── callbacks.py        # EMACallback, etc.
│   ├── data/
│   │   ├── dataset.py          # DiffusionPolicyDataset
│   │   └── preprocessing.py    # Demo loading utilities
│   └── utils/
│       └── helpers.py          # SE2/SE3 utilities
├── scripts/
│   ├── train.py                # Training script
│   └── evaluate.py             # Evaluation script
├── configs/
│   └── default.yaml            # Default configuration
└── denoising_diffusion_pytorch/  # Legacy compatibility
```

## Data Format

The expected data format is a pickle file containing:

```python
{
    "gripper_poses": List[np.ndarray],  # List of [T, 3] arrays (x, y, theta)
    "object_poses": List[np.ndarray],   # List of [T, 3] arrays (x, y, theta)
    "grasp_pose": np.ndarray,           # Grasp pose information
}
```

Data should be organized as:
```
collected_demos/
├── banana/
│   ├── grasp_pose1/
│   │   └── banana.pkl
│   └── grasp_pose2/
│       └── banana.pkl
└── cup/
    └── grasp_pose1/
        └── cup.pkl
```

## Evaluation

```bash
# Evaluate a trained model
python scripts/evaluate.py \
    --checkpoint results/model-final.ckpt \
    --num-episodes 10

# Evaluate with visualization
python scripts/evaluate.py \
    --checkpoint results/model-final.ckpt \
    --visualize
```

## Legacy Compatibility

For backward compatibility with existing code, you can still use the old import style:

```python
# Old style (deprecated)
from denoising_diffusion_pytorch import (
    ConditionalUnet1D,
    GaussianDiffusion1DConditional,  # Deprecated, use GaussianDiffusion1D
    Dataset1DCond,                    # Deprecated, use DiffusionPolicyDataset
    Trainer1DCond,                    # Legacy trainer (still available)
)

# New style (recommended)
from diffusion_policy import (
    ConditionalUnet1D,
    GaussianDiffusion1D,
    DiffusionPolicyDataset,
    DiffusionPolicyModule,
)
```

