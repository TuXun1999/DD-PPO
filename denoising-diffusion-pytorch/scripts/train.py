#!/usr/bin/env python3
"""
Training script for Diffusion Policy.

Usage:
    python scripts/train.py --object banana --grasp-pose 1
    python scripts/train.py --object banana --grasp-pose 1 --backbone transformer
    python scripts/train.py --config configs/default.yaml
"""
import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

import torch
torch.set_float32_matmul_precision('medium')

import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from diffusion_policy import (
    DiffusionPolicyConfig,
    ModelConfig,
    DiffusionConfig,
    TrainingConfig,
    DataConfig,
    DiffusionPolicyModule,
    DiffusionPolicyDataModule,
    EMACallback,
    GradientClipCallback,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Diffusion Policy")

    # Data arguments
    parser.add_argument("-o", "--object", default="banana", help="Object type")
    parser.add_argument("-g", "--grasp-pose", type=int, default=1, help="Grasp pose index")
    parser.add_argument("-d", "--dataset", default="../collected_demos", help="Dataset directory")

    # Model arguments
    parser.add_argument("--backbone", choices=["unet1d", "transformer"], default="unet1d",
                        help="Model backbone architecture")
    parser.add_argument("--timesteps", type=int, default=10, help="Diffusion timesteps")
    parser.add_argument("--sampling-timesteps", type=int, default=8, help="DDIM sampling timesteps")

    # Training arguments
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max-steps", type=int, default=5000, help="Maximum training steps")
    parser.add_argument("--gradient-accumulate", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--ema-decay", type=float, default=0.995, help="EMA decay rate")

    # Logging arguments
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb-project", default="diffusion-policy", help="WandB project name")
    parser.add_argument("--results-folder", default=None, help="Results folder (default: auto)")

    # Hardware arguments
    parser.add_argument("--accelerator", default="auto", help="Accelerator (auto, gpu, cpu)")
    parser.add_argument("--precision", default="16-mixed", help="Training precision")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")

    return parser.parse_args()


def main():
    args = parse_args()

    # Build configuration
    model_config = ModelConfig(
        backbone=args.backbone,
    )

    diffusion_config = DiffusionConfig(
        timesteps=args.timesteps,
        sampling_timesteps=args.sampling_timesteps,
    )

    training_config = TrainingConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        num_steps=args.max_steps,
        gradient_accumulate=args.gradient_accumulate,
        ema_decay=args.ema_decay,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
    )

    data_config = DataConfig(
        dataset_dir=args.dataset,
        object_type=args.object,
        grasp_pose=args.grasp_pose,
        num_workers=args.num_workers,
    )

    config = DiffusionPolicyConfig(
        model=model_config,
        diffusion=diffusion_config,
        training=training_config,
        data=data_config,
    )

    # Results folder
    if args.results_folder is None:
        results_folder = Path(args.dataset) / args.object / f"grasp_pose{args.grasp_pose}" / "results"
    else:
        results_folder = Path(args.results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)
 
    # Setup data module
    data_module = DiffusionPolicyDataModule(config=data_config)
    data_module.prepare_data()
    data_module.setup()

    # Setup model
    model = DiffusionPolicyModule(
        config=config,
        normalization_stats=data_module.get_normalization_stats(),
    )

    # Setup callbacks
    callbacks = [
        EMACallback(decay=args.ema_decay),
        GradientClipCallback(max_grad_norm=training_config.max_grad_norm),
        ModelCheckpoint(
            dirpath=results_folder / "checkpoints",
            filename="model-{step:06d}",
            save_top_k=3,
            monitor="val/loss",
            mode="min",
            every_n_train_steps=training_config.save_every,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Setup logger
    if args.wandb:
        logger = WandbLogger(
            project=args.wandb_project,
            name=f"{args.object}_gp{args.grasp_pose}_{args.backbone}",
            save_dir=str(results_folder),
        )
    else:
        logger = TensorBoardLogger(
            save_dir=str(results_folder),
            name="logs",
        )

    # Setup trainer
    trainer = L.Trainer(
        accelerator=args.accelerator,
        precision=args.precision,
        max_steps=args.max_steps,
        accumulate_grad_batches=args.gradient_accumulate,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        val_check_interval=None,
        enable_progress_bar=True,
    )
    
    # for larger datasets, use val_check_interval instead of check_val_every_n_epoch
    # trainer = L.Trainer(
    #     accelerator=args.accelerator,
    #     precision=args.precision,
    #     max_steps=args.max_steps,
    #     accumulate_grad_batches=args.gradient_accumulate,
    #     callbacks=callbacks,
    #     logger=logger,
    #     log_every_n_steps=10,
    #     val_check_interval=500,
    #     enable_progress_bar=True,
    # )


    # Train
    print(f"Training Diffusion Policy for {args.object} (grasp_pose={args.grasp_pose})")
    print(f"Backbone: {args.backbone}")
    print(f"Results folder: {results_folder}")

    trainer.fit(model, data_module)

    # Save final model
    final_path = results_folder / "model-final.ckpt"
    trainer.save_checkpoint(final_path)
    print(f"Training complete. Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
