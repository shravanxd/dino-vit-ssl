#!/usr/bin/env python3
"""
Main entry point for DINO training.

Usage:
    # Train with default config (single GPU)
    python train.py

    # Train with custom config
    python train.py --config configs/dino_vit_small.yaml
    
    # Override specific parameters
    python train.py --epochs 200 --batch_size 128 --lr 1e-3
    
    # Resume training
    python train.py --resume outputs/exp1/checkpoints/checkpoint_epoch0050.pt
    
    # Multi-GPU training with DDP (2 GPUs)
    torchrun --nproc_per_node=2 train.py --distributed --epochs 100 --batch_size 64
    
    # Multi-GPU on specific GPUs
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py --distributed
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from training.configs import TrainingConfig
from training.trainers import DINOTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train DINO self-supervised model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Config file
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file"
    )
    
    # Training parameters
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Batch size per GPU"
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Base learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=None,
        help="Weight decay"
    )
    
    # Data parameters
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="Path to training data (CC3M)"
    )
    parser.add_argument(
        "--cc12m_data_path", type=str, default=None,
        help="Path to CC12M training data"
    )
    parser.add_argument(
        "--birds_data_path", type=str, default=None,
        help="Path to birds pretrain training data"
    )
    parser.add_argument(
        "--coco_data_path", type=str, default=None,
        help="Path to COCO unlabeled training data"
    )
    parser.add_argument(
        "--places_data_path", type=str, default=None,
        help="Path to Places365 training data"
    )
    parser.add_argument(
        "--use_cc12m", action="store_true",
        help="Include CC12M data in training (in addition to base pretrain)"
    )
    parser.add_argument(
        "--use_birds", action="store_true",
        help="Include birds data in training (in addition to base pretrain)"
    )
    parser.add_argument(
        "--use_coco", action="store_true",
        help="Include COCO data in training (in addition to base pretrain)"
    )
    parser.add_argument(
        "--use_places", action="store_true",
        help="Include Places365 data in training (in addition to base pretrain)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=None,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--num_local_crops", type=int, default=None,
        help="Number of local crops (0 to disable)"
    )
    
    # Model parameters
    parser.add_argument(
        "--backbone", type=str, choices=["vit_tiny", "vit_small", "vit_base"], default=None,
        help="Backbone architecture"
    )
    parser.add_argument(
        "--out_dim", type=int, default=None,
        help="Output dimension of projection head"
    )
    
    # Backbone parameters
    parser.add_argument(
        "--img_size", type=int, default=None,
        help="Input image size"
    )
    parser.add_argument(
        "--patch_size", type=int, default=None,
        help="Patch size for ViT"
    )
    parser.add_argument(
        "--drop_rate", type=float, default=None,
        help="Dropout rate in backbone"
    )
    parser.add_argument(
        "--attn_drop_rate", type=float, default=None,
        help="Attention dropout rate in backbone"
    )
    
    # Data augmentation parameters
    parser.add_argument(
        "--global_crop_size", type=int, default=None,
        help="Size of global crops"
    )
    parser.add_argument(
        "--local_crop_size", type=int, default=None,
        help="Size of local crops"
    )
    parser.add_argument(
        "--global_crop_scale", type=float, nargs=2, default=None,
        help="Scale range for global crops (min max)"
    )
    parser.add_argument(
        "--local_crop_scale", type=float, nargs=2, default=None,
        help="Scale range for local crops (min max)"
    )
    parser.add_argument(
        "--pin_memory", action="store_true",
        help="Pin memory for data loading"
    )
    parser.add_argument(
        "--no_pin_memory", action="store_true",
        help="Disable pin memory for data loading"
    )
    parser.add_argument(
        "--coco_sample_ratio", type=float, default=None,
        help="Fraction of COCO images to use (0-1, default 1.0)"
    )
    parser.add_argument(
        "--places_sample_ratio", type=float, default=None,
        help="Fraction of Places365 images to use (0-1, default 1.0)"
    )
    
    # DINO head parameters
    parser.add_argument(
        "--hidden_dim", type=int, default=None,
        help="Hidden dimension in projection head"
    )
    parser.add_argument(
        "--bottleneck_dim", type=int, default=None,
        help="Bottleneck dimension before output"
    )
    parser.add_argument(
        "--num_layers", type=int, default=None,
        help="Number of MLP layers in head"
    )
    parser.add_argument(
        "--norm_last_layer", action="store_true", default=None,
        help="Apply weight normalization to last layer"
    )
    parser.add_argument(
        "--no_norm_last_layer", action="store_true",
        help="Disable weight normalization on last layer"
    )
    
    # DINO temperature parameters
    parser.add_argument(
        "--teacher_temp", type=float, default=None,
        help="Final teacher temperature (default: 0.04)"
    )
    parser.add_argument(
        "--teacher_temp_warmup", type=float, default=None,
        help="Initial teacher temperature for warmup (default: 0.04)"
    )
    parser.add_argument(
        "--teacher_temp_warmup_epochs", type=int, default=None,
        help="Number of epochs for teacher temp warmup (default: 30)"
    )
    parser.add_argument(
        "--student_temp", type=float, default=None,
        help="Student temperature (default: 0.1)"
    )
    
    # DINO EMA and center parameters
    parser.add_argument(
        "--center_momentum", type=float, default=None,
        help="Momentum for center update (default: 0.9)"
    )
    parser.add_argument(
        "--ema_momentum_base", type=float, default=None,
        help="Base EMA momentum for teacher update (default: 0.996)"
    )
    parser.add_argument(
        "--ema_momentum_final", type=float, default=None,
        help="Final EMA momentum for teacher update (default: 1.0)"
    )
    
    # DINO loss weights
    parser.add_argument(
        "--l2g_weight", type=float, default=None,
        help="Weight for local-to-global loss (default: 0.5)"
    )
    parser.add_argument(
        "--reg_weight", type=float, default=None,
        help="Weight for regularization loss (default: 0.1)"
    )
    parser.add_argument(
        "--no_l2g_loss", action="store_true",
        help="Disable local-to-global loss"
    )
    parser.add_argument(
        "--no_reg_loss", action="store_true",
        help="Disable regularization loss"
    )
    
    # Scheduler parameters
    parser.add_argument(
        "--warmup_epochs", type=int, default=None,
        help="Number of warmup epochs for LR scheduler"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=None,
        help="Number of warmup steps (overrides warmup_epochs if set)"
    )
    parser.add_argument(
        "--warmup_start_lr", type=float, default=None,
        help="Starting learning rate for warmup"
    )
    parser.add_argument(
        "--min_lr", type=float, default=None,
        help="Minimum learning rate"
    )
    
    # Optimizer parameters
    parser.add_argument(
        "--optimizer", type=str, choices=["adamw", "sgd", "lars"], default=None,
        help="Optimizer name"
    )
    parser.add_argument(
        "--betas", type=float, nargs=2, default=None,
        help="Adam betas (beta1 beta2)"
    )
    parser.add_argument(
        "--momentum", type=float, default=None,
        help="SGD momentum"
    )
    
    # Gradient clipping
    parser.add_argument(
        "--gradient_clip", type=float, default=None,
        help="Max gradient norm for clipping (0 to disable, default: 3.0)"
    )
    
    # Device and precision
    parser.add_argument(
        "--device", type=str, choices=["cuda", "mps", "cpu"], default=None,
        help="Device to use for training"
    )
    parser.add_argument(
        "--no_mixed_precision", action="store_true",
        help="Disable automatic mixed precision"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=None,
        help="Accumulate gradients over N steps (effective batch = batch_size * N)"
    )
    
    # Output
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory for outputs"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default=None,
        help="Custom directory for saving checkpoints (overrides output_dir/experiment_name/checkpoints)"
    )
    parser.add_argument(
        "--experiment_name", type=str, default=None,
        help="Name for this experiment"
    )
    
    # Resuming
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--reset_scheduler", action="store_true",
        help="Reset LR scheduler when resuming (useful when changing total epochs)"
    )
    
    # Logging
    parser.add_argument(
        "--log_interval", type=int, default=None,
        help="Steps between logging"
    )
    parser.add_argument(
        "--save_interval", type=int, default=None,
        help="Epochs between checkpoints"
    )
    parser.add_argument(
        "--eval_interval", type=int, default=None,
        help="Epochs between evaluations"
    )
    parser.add_argument(
        "--eval_num_workers", type=int, default=None,
        help="Number of workers for evaluation data loading"
    )
    parser.add_argument(
        "--eval_batch_size", type=int, default=None,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--cub200_path", type=str, default=None,
        help="Path to CUB-200 evaluation dataset"
    )
    parser.add_argument(
        "--miniimagenet_path", type=str, default=None,
        help="Path to MiniImageNet evaluation dataset"
    )
    parser.add_argument(
        "--sun397_path", type=str, default=None,
        help="Path to SUN397 evaluation dataset"
    )
    parser.add_argument(
        "--prefetch_factor", type=int, default=None,
        help="Number of batches to prefetch per worker (default: 2)"
    )
    
    # Seed
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed"
    )
    
    # W&B arguments
    parser.add_argument(
        "--wandb_name", type=str, default=None,
        help="W&B run name (default: auto-generated from experiment_name + timestamp)"
    )
    parser.add_argument(
        "--wandb_project", type=str, default=None,
        help="W&B project name"
    )
    parser.add_argument(
        "--wandb_tags", type=str, nargs="+", default=None,
        help="W&B tags (space-separated)"
    )
    parser.add_argument(
        "--no_wandb", action="store_true",
        help="Disable W&B logging"
    )
    parser.add_argument(
        "--no_save_artifacts", action="store_true",
        help="Disable saving checkpoints as W&B artifacts"
    )
    
    # Distributed training arguments
    parser.add_argument(
        "--distributed", action="store_true",
        help="Enable distributed training (use with torchrun)"
    )
    parser.add_argument(
        "--dist_backend", type=str, default="nccl",
        choices=["nccl", "gloo"],
        help="Distributed backend (nccl for GPU, gloo for CPU)"
    )
    
    return parser.parse_args()


def build_config(args) -> TrainingConfig:
    """
    Build configuration from args and optional config file.
    
    Priority (highest to lowest):
    1. Command line arguments
    2. Config file
    3. Default values
    """
    # Start with defaults or config file
    if args.config:
        config = TrainingConfig.from_yaml(args.config)
        print(f"Loaded config from {args.config}")
    else:
        config = TrainingConfig()
        print("Using default configuration")
    
    # Override with command line arguments
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.lr is not None:
        config.optimizer.lr = args.lr
    if args.weight_decay is not None:
        config.optimizer.weight_decay = args.weight_decay
    if args.data_path is not None:
        config.data.data_path = args.data_path
    if args.cc12m_data_path is not None:
        config.data.cc12m_data_path = args.cc12m_data_path
    if args.birds_data_path is not None:
        config.data.birds_data_path = args.birds_data_path
    if args.coco_data_path is not None:
        config.data.coco_data_path = args.coco_data_path
    if args.places_data_path is not None:
        config.data.places_data_path = args.places_data_path
    if args.use_cc12m:
        config.data.use_cc12m = True
    if args.use_birds:
        config.data.use_birds = True
    if args.use_coco:
        config.data.use_coco = True
    if args.use_places:
        config.data.use_places = True
    if args.num_workers is not None:
        config.data.num_workers = args.num_workers
    if args.prefetch_factor is not None:
        config.data.prefetch_factor = args.prefetch_factor
    if args.num_local_crops is not None:
        config.data.num_local_crops = args.num_local_crops
    if args.backbone is not None:
        config.backbone.name = args.backbone
    if args.out_dim is not None:
        config.dino.out_dim = args.out_dim
    
    # Backbone parameters
    if args.img_size is not None:
        config.backbone.img_size = args.img_size
    if args.patch_size is not None:
        config.backbone.patch_size = args.patch_size
    if args.drop_rate is not None:
        config.backbone.drop_rate = args.drop_rate
    if args.attn_drop_rate is not None:
        config.backbone.attn_drop_rate = args.attn_drop_rate
    
    # Data augmentation parameters
    if args.global_crop_size is not None:
        config.data.global_crop_size = args.global_crop_size
    if args.local_crop_size is not None:
        config.data.local_crop_size = args.local_crop_size
    if args.global_crop_scale is not None:
        config.data.global_crop_scale = tuple(args.global_crop_scale)
    if args.local_crop_scale is not None:
        config.data.local_crop_scale = tuple(args.local_crop_scale)
    if args.pin_memory:
        config.data.pin_memory = True
    if args.no_pin_memory:
        config.data.pin_memory = False
    if args.coco_sample_ratio is not None:
        config.data.coco_sample_ratio = args.coco_sample_ratio
    if args.places_sample_ratio is not None:
        config.data.places_sample_ratio = args.places_sample_ratio
    
    # DINO head parameters
    if args.hidden_dim is not None:
        config.dino.hidden_dim = args.hidden_dim
    if args.bottleneck_dim is not None:
        config.dino.bottleneck_dim = args.bottleneck_dim
    if args.num_layers is not None:
        config.dino.num_layers = args.num_layers
    if args.norm_last_layer:
        config.dino.norm_last_layer = True
    if args.no_norm_last_layer:
        config.dino.norm_last_layer = False
    
    # DINO temperature parameters
    if args.teacher_temp is not None:
        config.dino.teacher_temp = args.teacher_temp
    if args.teacher_temp_warmup is not None:
        config.dino.teacher_temp_warmup = args.teacher_temp_warmup
    if args.teacher_temp_warmup_epochs is not None:
        config.dino.teacher_temp_warmup_epochs = args.teacher_temp_warmup_epochs
    if args.student_temp is not None:
        config.dino.student_temp = args.student_temp
    
    # DINO EMA and center parameters
    if args.center_momentum is not None:
        config.dino.center_momentum = args.center_momentum
    if args.ema_momentum_base is not None:
        config.dino.ema_momentum_base = args.ema_momentum_base
    if args.ema_momentum_final is not None:
        config.dino.ema_momentum_final = args.ema_momentum_final
    
    # DINO loss weights
    if args.l2g_weight is not None:
        config.dino.l2g_weight = args.l2g_weight
    if args.reg_weight is not None:
        config.dino.reg_weight = args.reg_weight
    if args.no_l2g_loss:
        config.dino.use_l2g_loss = False
    if args.no_reg_loss:
        config.dino.use_reg_loss = False
    
    # Scheduler parameters
    if args.warmup_epochs is not None:
        config.scheduler.warmup_epochs = args.warmup_epochs
    if args.warmup_steps is not None:
        config.scheduler.warmup_steps = args.warmup_steps
    if args.warmup_start_lr is not None:
        config.scheduler.warmup_start_lr = args.warmup_start_lr
    if args.min_lr is not None:
        config.scheduler.min_lr = args.min_lr
    
    # Optimizer parameters
    if args.optimizer is not None:
        config.optimizer.name = args.optimizer
    if args.betas is not None:
        config.optimizer.betas = tuple(args.betas)
    if args.momentum is not None:
        config.optimizer.momentum = args.momentum
    
    if args.device is not None:
        config.device = args.device
    if args.no_mixed_precision:
        config.mixed_precision = False
    if args.gradient_accumulation_steps is not None:
        config.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.gradient_clip is not None:
        config.gradient_clip = args.gradient_clip
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.experiment_name is not None:
        config.experiment_name = args.experiment_name
    if args.resume is not None:
        config.resume_from = args.resume
    if args.reset_scheduler:
        config.reset_scheduler = True
    if args.log_interval is not None:
        config.log_interval = args.log_interval
    if args.save_interval is not None:
        config.save_interval = args.save_interval
    if args.eval_interval is not None:
        config.eval_interval = args.eval_interval
    if args.eval_num_workers is not None:
        config.eval.num_workers = args.eval_num_workers
    if args.eval_batch_size is not None:
        config.eval.batch_size = args.eval_batch_size
    if args.cub200_path is not None:
        config.eval.cub200_path = args.cub200_path
    if args.miniimagenet_path is not None:
        config.eval.miniimagenet_path = args.miniimagenet_path
    if args.sun397_path is not None:
        config.eval.sun397_path = args.sun397_path
    if args.seed is not None:
        config.seed = args.seed
    
    # W&B overrides
    if args.no_wandb:
        config.wandb.enabled = False
    if args.no_save_artifacts:
        config.wandb.save_artifacts = False
    if args.wandb_name is not None:
        config.wandb.name = args.wandb_name
    if args.wandb_project is not None:
        config.wandb.project = args.wandb_project
    if args.wandb_tags is not None:
        config.wandb.tags = args.wandb_tags
    
    # Distributed training overrides
    if args.distributed:
        config.distributed.enabled = True
        config.distributed.backend = args.dist_backend
    
    # Get world_size and rank from environment (set by torchrun) - always check if running under torchrun
    if "WORLD_SIZE" in os.environ:
        config.distributed.enabled = True
        config.distributed.world_size = int(os.environ["WORLD_SIZE"])
        config.distributed.rank = int(os.environ["RANK"])
        config.distributed.local_rank = int(os.environ["LOCAL_RANK"])
    
    # Auto-generate unique W&B run name if not specified
    if config.wandb.name is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.wandb.name = f"{config.experiment_name}_{timestamp}"
    
    # Recreate derived paths after modifications
    config.checkpoint_dir = Path(config.output_dir) / config.experiment_name / "checkpoints"
    config.log_dir = Path(config.output_dir) / config.experiment_name / "logs"
    
    # Override checkpoint_dir if specified
    if args.checkpoint_dir is not None:
        config.checkpoint_dir = Path(args.checkpoint_dir)
    
    return config


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Build config
    config = build_config(args)
    
    # Determine if this is the main process (for distributed training)
    is_main = not config.distributed.enabled or config.distributed.rank == 0
    
    # Print config summary (only on main process)
    if is_main:
        print("\n" + "="*60)
        print("Training Configuration")
        print("="*60)
        print(f"  Experiment: {config.experiment_name}")
        print(f"  Epochs: {config.epochs}")
        print(f"  Backbone: {config.backbone.name}")
        print(f"  Batch size (per GPU): {config.data.batch_size}")
        print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
        if config.distributed.enabled:
            print(f"  Number of GPUs: {config.distributed.world_size}")
            effective_batch = config.data.batch_size * config.gradient_accumulation_steps * config.distributed.world_size
            print(f"  Effective batch size: {effective_batch}")
        else:
            print(f"  Effective batch size: {config.data.batch_size * config.gradient_accumulation_steps}")
        print(f"  Learning rate: {config.optimizer.lr}")
        print(f"  Data sources: pretrain" + (" + cc12m" if config.data.use_cc12m else "") + (" + birds" if config.data.use_birds else ""))
        print(f"  Data paths: {config.data.get_data_paths()}")
        print(f"  Num workers: {config.data.num_workers}")
        print(f"  Local crops: {config.data.num_local_crops}")
        print(f"  Device: {config.device}")
        print(f"  Distributed: {config.distributed.enabled}")
        if config.distributed.enabled:
            print(f"  Backend: {config.distributed.backend}")
        print(f"  Mixed precision: {config.mixed_precision}")
        print(f"  Output dir: {config.output_dir}")
        print(f"  Checkpoint dir: {config.checkpoint_dir}")
        print(f"  W&B enabled: {config.wandb.enabled}")
        if config.wandb.enabled:
            print(f"  W&B project: {config.wandb.project}")
            print(f"  W&B run name: {config.wandb.name}")
            if config.wandb.tags:
                print(f"  W&B tags: {config.wandb.tags}")
        print("="*60 + "\n")
    
    # Create output directories (only on main process)
    if is_main:
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        config.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config as YAML
        config_path = config.log_dir / "config.yaml"
        config.save_yaml(str(config_path))
        print(f"Saved config to {config_path}")
        
        # Save metadata JSON with key hyperparameters summary
        import json
        from datetime import datetime
        
        # Calculate effective batch size
        num_gpus = config.distributed.world_size if config.distributed.enabled else 1
        effective_batch = config.data.batch_size * config.gradient_accumulation_steps * num_gpus
        
        metadata = {
            "experiment": {
                "name": config.experiment_name,
                "timestamp": datetime.now().isoformat(),
                "seed": config.seed,
            },
            "training": {
                "epochs": config.epochs,
                "batch_size_per_gpu": config.data.batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "num_gpus": num_gpus,
                "effective_batch_size": effective_batch,
                "mixed_precision": config.mixed_precision,
                "gradient_clip": config.gradient_clip,
            },
            "optimizer": {
                "name": config.optimizer.name,
                "learning_rate": config.optimizer.lr,
                "weight_decay": config.optimizer.weight_decay,
                "betas": list(config.optimizer.betas),
            },
            "scheduler": {
                "name": config.scheduler.name,
                "warmup_epochs": config.scheduler.warmup_epochs,
                "min_lr": config.scheduler.min_lr,
            },
            "model": {
                "backbone": config.backbone.name,
                "img_size": config.backbone.img_size,
                "patch_size": config.backbone.patch_size,
                "drop_rate": config.backbone.drop_rate,
            },
            "dino": {
                "out_dim": config.dino.out_dim,
                "hidden_dim": config.dino.hidden_dim,
                "bottleneck_dim": config.dino.bottleneck_dim,
                "teacher_temp": config.dino.teacher_temp,
                "student_temp": config.dino.student_temp,
                "ema_momentum_base": config.dino.ema_momentum_base,
                "ema_momentum_final": config.dino.ema_momentum_final,
            },
            "augmentation": {
                "global_crop_size": config.data.global_crop_size,
                "local_crop_size": config.data.local_crop_size,
                "num_local_crops": config.data.num_local_crops,
                "global_crop_scale": list(config.data.global_crop_scale),
                "local_crop_scale": list(config.data.local_crop_scale),
            },
            "data": {
                "data_path": config.data.data_path,
                "num_workers": config.data.num_workers,
            },
            "distributed": {
                "enabled": config.distributed.enabled,
                "backend": config.distributed.backend,
                "world_size": num_gpus,
            },
        }
        
        metadata_path = config.log_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_path}")
    
    # Create trainer and train
    trainer = DINOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
