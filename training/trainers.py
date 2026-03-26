"""
Trainer classes for SSL methods.

Follows SOLID principles:
- Single Responsibility: Trainer handles training loop only
- Open/Closed: Extend via inheritance for new methods
- Liskov Substitution: All trainers follow BaseTrainer contract
- Dependency Inversion: Depends on abstractions (model, loss, etc.)
"""

import os
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Any, List
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast

# Add parent to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from model.backbones import ViTTiny, ViTSmall, ViTBase
from model.methods import DINO
from model.losses import DINOLoss

# Import evaluation module
from evaluation import run_knn_evaluation

from .configs import TrainingConfig
from .datasets import create_ssl_dataloader
from .utils import (
    CheckpointManager,
    TrainingLogger,
    EMAScheduler,
    WarmupCosineScheduler,
    TeacherTemperatureScheduler,
    create_optimizer,
)
from .wandb_logger import WandbLogger


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_world_size() -> int:
    """Get total number of processes."""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    """Get global rank of current process."""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


class BaseTrainer(ABC):
    """
    Abstract base class for SSL trainers.
    
    Defines the training interface that all SSL trainers must implement.
    Provides common functionality like device setup, checkpoint loading, etc.
    
    Follows Template Method pattern - defines skeleton of training algorithm,
    subclasses fill in specifics.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        
        # Initialize distributed training if enabled
        self._setup_distributed()
        
        # Setup device
        self.device = self._setup_device()
        
        # Will be set by subclasses
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        self.dataloader: Optional[DataLoader] = None
        self.sampler: Optional[DistributedSampler] = None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        
        # Best model tracking per dataset
        self.best_metrics: Dict[str, float] = {}
    
    def _setup_distributed(self):
        """Initialize distributed training if enabled."""
        if not self.config.distributed.enabled:
            return
        
        # Get rank and world_size from environment (set by torchrun)
        if "RANK" in os.environ:
            self.config.distributed.rank = int(os.environ["RANK"])
            self.config.distributed.world_size = int(os.environ["WORLD_SIZE"])
            self.config.distributed.local_rank = int(os.environ["LOCAL_RANK"])
        
        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.config.distributed.backend,
                rank=self.config.distributed.rank,
                world_size=self.config.distributed.world_size,
            )
        
        if is_main_process():
            print(f"Distributed training initialized:")
            print(f"  World size: {get_world_size()}")
            print(f"  Backend: {self.config.distributed.backend}")
    
    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if self.config.distributed.enabled:
            # In distributed mode, each process uses its local GPU
            local_rank = self.config.distributed.local_rank
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
            if is_main_process():
                print(f"Distributed: Using {get_world_size()} GPUs")
                for i in range(torch.cuda.device_count()):
                    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        elif self.config.device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        elif self.config.device == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple MPS")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        
        return device
    
    @abstractmethod
    def build_model(self) -> nn.Module:
        """Build the SSL model. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def build_loss(self) -> nn.Module:
        """Build the loss function. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def train_step(self, batch: Any) -> Dict[str, float]:
        """
        Execute a single training step.
        
        Args:
            batch: Batch of data from dataloader
            
        Returns:
            Dictionary of metrics (loss, etc.)
        """
        pass
    
    @abstractmethod
    def on_epoch_end(self, epoch: int):
        """
        Called at the end of each epoch.
        
        Use for EMA updates, temperature scheduling, etc.
        
        Args:
            epoch: Completed epoch number
        """
        pass
    
    def evaluate(self) -> Dict[str, float]:
        """
        Run KNN evaluation on configured datasets.
        
        Uses the evaluation module to compute KNN accuracy
        on downstream datasets like CUB-200.
        
        Returns:
            Dictionary with evaluation metrics for each dataset
        """
        if not self.config.eval.enabled:
            return {}
        
        # Only run evaluation on main process
        if not is_main_process():
            return {}
        
        eval_metrics = {}
        
        # Get backbone for feature extraction (unwrap DDP if necessary)
        model = self.model.module if self.config.distributed.enabled else self.model
        backbone = model.get_backbone()
        
        # Evaluate on each configured dataset
        for dataset_name in self.config.eval.eval_datasets:
            try:
                # Get the path for this dataset (uses custom path if set, else fallback to eval_data_root)
                dataset_path = self.config.eval.get_dataset_path(dataset_name)
                print(f"  Evaluating on {dataset_name} from {dataset_path}...")
                
                # Extract the parent dir and name for run_knn_evaluation
                from pathlib import Path
                dataset_path_obj = Path(dataset_path)
                data_root = str(dataset_path_obj.parent)
                dataset_folder = dataset_path_obj.name
                
                results = run_knn_evaluation(
                    model=backbone,
                    dataset_name=dataset_folder,
                    data_root=data_root,
                    img_size=self.config.backbone.img_size,
                    k=self.config.eval.knn_k,
                    temperature=self.config.eval.knn_temperature,
                    batch_size=self.config.eval.batch_size,
                    num_workers=self.config.eval.num_workers,
                    device=str(self.device),
                )
                
                # Add dataset prefix to metrics
                for key, value in results.items():
                    if not isinstance(value, list):  # Skip per-class list
                        eval_metrics[f'{dataset_name}/{key}'] = value
                
                print(f"    {dataset_name}: top1={results['top1']:.2f}%, top5={results['top5']:.2f}%")
                
            except FileNotFoundError as e:
                print(f"    Warning: Could not evaluate on {dataset_name}: {e}")
            except Exception as e:
                print(f"    Error evaluating on {dataset_name}: {e}")
        
        return eval_metrics
    
    def train(self):
        """
        Main training loop.
        
        Template method that orchestrates training:
        1. Setup model, optimizer, scheduler, dataloader
        2. Optionally resume from checkpoint
        3. Run training epochs
        4. Run evaluation every eval_interval epochs
        5. Save final checkpoint
        """
        # Setup
        self.setup()
        
        # Resume if specified
        if self.config.resume_from:
            self._resume_training()
        
        # Print info only on main process
        if is_main_process():
            print(f"\nStarting training for {self.config.epochs} epochs...")
            print(f"Dataset: {len(self.dataloader.dataset)} images")
            print(f"Batch size: {self.config.data.batch_size}")
            if self.config.distributed.enabled:
                print(f"Effective batch size: {self.config.data.batch_size * get_world_size()}")
            print(f"Steps per epoch: {len(self.dataloader)}")
            if self.config.eval.enabled:
                print(f"Evaluation: every {self.config.eval_interval} epochs on {self.config.eval.eval_datasets}")
            print()
        
        # Training loop
        try:
            for epoch in range(self.current_epoch, self.config.epochs):
                self.current_epoch = epoch
                
                # Train one epoch
                epoch_metrics = self._train_epoch(epoch)
                
                # Log epoch
                self.logger.log_epoch(epoch, epoch_metrics)
                
                # End of epoch processing
                self.on_epoch_end(epoch)
                
                # Run evaluation
                if self.config.eval.enabled and (epoch + 1) % self.config.eval_interval == 0:
                    eval_metrics = self.evaluate()
                    self.logger.log_epoch(epoch, eval_metrics, prefix="eval")
                    epoch_metrics.update(eval_metrics)
                    
                    # Log eval metrics to W&B
                    self.wandb_logger.log_eval_results(
                        eval_metrics,
                        step=self.global_step,
                        epoch=epoch,
                    )
                    
                    # Check and save best models for each dataset
                    self._check_and_save_best_models(eval_metrics, epoch)
                
                # Save checkpoint
                if (epoch + 1) % self.config.save_interval == 0:
                    self._save_checkpoint(epoch, epoch_metrics)
            
            # Final evaluation
            if self.config.eval.enabled:
                print("\nRunning final evaluation...")
                final_eval = self.evaluate()
                self.logger.log_epoch(self.config.epochs - 1, final_eval, prefix="final_eval")
                self.wandb_logger.log_eval_results(
                    final_eval,
                    step=self.global_step,
                    epoch=self.config.epochs - 1,
                    prefix="final",
                )
                
                # Check and save best models for final evaluation
                self._check_and_save_best_models(final_eval, self.config.epochs - 1)
                
                # Log final results as W&B summary
                for key, value in final_eval.items():
                    self.wandb_logger.log_summary(f"final/{key}", value)
                
                # Create summary table for final results
                final_results_table = []
                for dataset in self.config.eval.eval_datasets:
                    top1_key = f"{dataset}/top1"
                    top5_key = f"{dataset}/top5"
                    if top1_key in final_eval:
                        final_results_table.append({
                            "dataset": dataset,
                            "top1_accuracy": final_eval.get(top1_key, 0),
                            "top5_accuracy": final_eval.get(top5_key, 0),
                            "mean_per_class": final_eval.get(f"{dataset}/mean_per_class_acc", 0),
                        })
                
                if final_results_table:
                    self.wandb_logger.log_table(
                        "final_evaluation_results",
                        columns=["dataset", "top1_accuracy", "top5_accuracy", "mean_per_class"],
                        data=[[r["dataset"], r["top1_accuracy"], r["top5_accuracy"], r["mean_per_class"]] 
                              for r in final_results_table],
                    )
            
            # Save final checkpoint
            self._save_checkpoint(self.config.epochs - 1, epoch_metrics, is_final=True)
            
            if is_main_process():
                print("\nTraining complete!")
            
        except Exception as e:
            # Alert on error
            self.wandb_logger.alert(
                title="Training Failed",
                text=f"Training failed at epoch {self.current_epoch}: {str(e)}",
                level="ERROR",
            )
            raise
        finally:
            # Always finish W&B run
            self.wandb_logger.finish()
            
            # Cleanup distributed training
            self._cleanup_distributed()
    
    def _cleanup_distributed(self):
        """Cleanup distributed training resources."""
        if self.config.distributed.enabled and dist.is_initialized():
            dist.destroy_process_group()
    
    def setup(self):
        """Setup training components."""
        # Set seed (make it rank-dependent for different data per GPU)
        seed = self.config.seed
        if self.config.distributed.enabled:
            seed += get_rank()
        self._set_seed(seed)
        
        # Build components
        self.model = self.build_model()
        self.model = self.model.to(self.device)
        
        # Wrap model with DDP if distributed
        if self.config.distributed.enabled:
            self.model = DDP(
                self.model,
                device_ids=[self.config.distributed.local_rank],
                output_device=self.config.distributed.local_rank,
                find_unused_parameters=self.config.distributed.find_unused_parameters,
                broadcast_buffers=self.config.distributed.broadcast_buffers,
            )
            if is_main_process():
                print(f"Model wrapped with DistributedDataParallel")
        
        self.loss_fn = self.build_loss()
        
        self.optimizer = create_optimizer(
            self.model,
            name=self.config.optimizer.name,
            lr=self.config.optimizer.lr,
            weight_decay=self.config.optimizer.weight_decay,
            betas=self.config.optimizer.betas,
        )
        
        # Create dataloader first to get steps_per_epoch
        # Use get_data_paths() to support multiple data sources
        data_paths = self.config.data.get_data_paths()
        if is_main_process():
            sources = "pretrain"
            if self.config.data.use_cc12m:
                sources += " + cc12m"
            if self.config.data.use_birds:
                sources += " + birds"
            if getattr(self.config.data, "use_coco", False):
                sources += " + coco"
            if getattr(self.config.data, "use_places", False):
                sources += " + places"
            print(f"Using data sources: {sources}")
            print(f"Data paths: {data_paths}")
        
        self.dataloader, self.sampler = create_ssl_dataloader(
            data_path=data_paths,
            batch_size=self.config.data.batch_size,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
            prefetch_factor=self.config.data.prefetch_factor,
            global_crop_size=self.config.data.global_crop_size,
            local_crop_size=self.config.data.local_crop_size,
            num_local_crops=self.config.data.num_local_crops,
            distributed=self.config.distributed.enabled,
            world_size=self.config.distributed.world_size,
            rank=self.config.distributed.rank,
        )
        
        # Calculate steps for step-based scheduling
        self.steps_per_epoch = len(self.dataloader)
        total_steps = self.steps_per_epoch * self.config.epochs
        
        # Check if using step-based warmup
        warmup_steps = getattr(self.config.scheduler, 'warmup_steps', 0)
        
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_epochs=self.config.scheduler.warmup_epochs,
            total_epochs=self.config.epochs,
            warmup_start_lr=self.config.scheduler.warmup_start_lr,
            min_lr=self.config.scheduler.min_lr,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )
        
        # Store whether we're using step-based scheduling
        self.step_based_lr = warmup_steps > 0
        
        if is_main_process():
            if self.step_based_lr:
                print(f"Using step-based LR warmup: {warmup_steps} steps, total {total_steps} steps")
            else:
                print(f"Using epoch-based LR warmup: {self.config.scheduler.warmup_epochs} epochs")
        
        # Setup utilities
        self.checkpoint_manager = CheckpointManager(
            self.config.checkpoint_dir,
            max_checkpoints=5,
        )
        
        self.logger = TrainingLogger(
            self.config.log_dir,
            self.config.experiment_name,
            log_interval=self.config.log_interval,
        )
        
        # Setup W&B logging (only on main process)
        wandb_enabled = self.config.wandb.enabled and is_main_process()
        self.wandb_logger = WandbLogger(
            project=self.config.wandb.project,
            name=self.config.wandb.name or self.config.experiment_name,
            config=self.config.to_dict(),
            enabled=wandb_enabled,
            log_gradients=self.config.wandb.log_gradients,
            log_freq=self.config.wandb.log_freq,
            tags=self.config.wandb.tags,
            notes=self.config.wandb.notes,
            dir=str(self.config.log_dir),
        )
        
        # Define metrics for better W&B plotting
        self.wandb_logger.define_metrics()
        
        # Log model summary
        self.wandb_logger.log_model_summary(self.model)
        
        # Watch model for gradient logging
        if self.config.wandb.watch_model:
            self.wandb_logger.watch_model(
                self.model, 
                log="gradients", 
                log_freq=self.config.wandb.log_freq
            )
        
        # Mixed precision - use VERY conservative scaling to prevent overflow
        # ViT-Base is particularly sensitive to FP16 overflow in attention QKV
        # init_scale: Start at 2^12 (4096) - conservative for large models
        # growth_interval: Only increase scale every 2000 successful steps
        # backoff_factor: Halve on inf (default 0.5)
        # growth_factor: Only 1.5x growth instead of 2x (slower scale increase)
        if self.config.mixed_precision:
            self.scaler = GradScaler(
                init_scale=2**12,  # Start at 4096 (very conservative)
                growth_interval=2000,
                growth_factor=1.5,  # Slower growth (default is 2.0)
                backoff_factor=0.5,
            )
        else:
            self.scaler = None
        self.max_grad_scale = 2**15  # Cap scale at 32768 (lower for ViT-Base)
        
        # Count parameters
        model_for_params = self.model.module if self.config.distributed.enabled else self.model
        total_params = sum(p.numel() for p in model_for_params.parameters())
        trainable_params = sum(p.numel() for p in model_for_params.parameters() if p.requires_grad)
        if is_main_process():
            print(f"Model parameters: {total_params/1e6:.2f}M total, {trainable_params/1e6:.2f}M trainable")
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with gradient accumulation support."""
        self.model.train()
        
        # Set epoch for DistributedSampler (important for shuffling)
        if self.sampler is not None:
            self.sampler.set_epoch(epoch)
        
        epoch_metrics = {}
        accum_steps = self.config.gradient_accumulation_steps
        
        # Zero gradients at start
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(self.dataloader):
            # Move batch to device
            batch = self._to_device(batch)
            
            # Training step (returns loss and metrics, handles backward)
            step_metrics = self.train_step(batch, step, accum_steps)
            
            # Step scheduler per step if using step-based LR
            if self.step_based_lr:
                self.scheduler.step()
            
            # Accumulate metrics
            for k, v in step_metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0) + v
            
            # Log to console
            self.logger.log_step(
                step=self.global_step,
                epoch=epoch,
                metrics=step_metrics,
                batch_size=self.config.data.batch_size,
            )
            
            # Log to W&B
            lr = self.optimizer.param_groups[0]['lr']
            self.wandb_logger.log_step(
                step=self.global_step,
                epoch=epoch,
                metrics=step_metrics,
                lr=lr,
            )
            
            self.global_step += 1
        
        # Average epoch metrics
        num_steps = len(self.dataloader)
        epoch_metrics = {k: v / num_steps for k, v in epoch_metrics.items()}
        
        # Step scheduler per epoch if using epoch-based LR
        if not self.step_based_lr:
            self.scheduler.step()
        
        # Log epoch metrics to W&B
        lr = self.optimizer.param_groups[0]['lr']
        self.wandb_logger.log_epoch(
            epoch=epoch,
            metrics=epoch_metrics,
            lr=lr,
        )
        
        # Log system metrics periodically
        if (epoch + 1) % 5 == 0:
            self.wandb_logger.log_system_metrics(step=self.global_step)
        
        return epoch_metrics
    
    def _to_device(self, batch: Any) -> Any:
        """Move batch to device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        elif isinstance(batch, (list, tuple)):
            return [self._to_device(b) for b in batch]
        elif isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        return batch
    
    def _save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        is_final: bool = False,
    ):
        """Save training checkpoint (only on main process)."""
        # Only save on main process
        if not is_main_process():
            return
        
        # Unwrap model for saving (save without DDP wrapper)
        model_to_save = self.model.module if self.config.distributed.enabled else self.model
        
        checkpoint_path = self.checkpoint_manager.save(
            model=model_to_save,
            optimizer=self.optimizer,
            epoch=epoch,
            step=self.global_step,
            scheduler=self.scheduler,
            metrics=metrics,
            extra_state={'best_metrics': self.best_metrics},
        )
        
        # Save checkpoint to W&B as artifact
        if self.config.wandb.save_artifacts and checkpoint_path:
            artifact_name = f"{self.config.experiment_name}-checkpoint"
            if is_final:
                artifact_name = f"{self.config.experiment_name}-checkpoint-final"
            
            self.wandb_logger.save_artifact(
                str(checkpoint_path),
                name=artifact_name,
                artifact_type="checkpoint",
                metadata={
                    "epoch": epoch,
                    "step": self.global_step,
                    "metrics": metrics,
                },
            )
        
        if is_final:
            # Also save just the backbone for evaluation
            backbone_path = self.config.checkpoint_dir / "backbone_final.pt"
            torch.save(
                model_to_save.get_backbone_state_dict(),
                backbone_path
            )
            print(f"Saved backbone weights: {backbone_path}")
            
            # Save final backbone as W&B artifact
            if self.config.wandb.save_artifacts:
                self.wandb_logger.save_artifact(
                    str(backbone_path),
                    name=f"{self.config.experiment_name}-backbone-final",
                    artifact_type="model",
                    metadata={
                        "epoch": epoch,
                        "metrics": metrics,
                        "config": self.config.to_dict(),
                    },
                )
    
    def _save_best_model(
        self,
        dataset_name: str,
        epoch: int,
        accuracy: float,
    ):
        """
        Save best model for a specific dataset.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'cub200')
            epoch: Current epoch number
            accuracy: The top1 accuracy achieved
        """
        if not is_main_process():
            return
        
        best_model_path = self.config.checkpoint_dir / f"best_{dataset_name}.pt"
        
        # Remove previous best if it exists
        if best_model_path.exists():
            best_model_path.unlink()
        
        # Get model to save (unwrap DDP)
        model_to_save = self.model.module if self.config.distributed.enabled else self.model
        
        # Save backbone weights with metadata
        checkpoint = {
            'backbone_state_dict': model_to_save.get_backbone_state_dict(),
            'dataset': dataset_name,
            'epoch': epoch,
            'accuracy': accuracy,
        }
        torch.save(checkpoint, best_model_path)
        print(f"  [Best Model] Saved new best for {dataset_name}: {accuracy:.2f}% (epoch {epoch})")
        
        # Save to W&B as artifact
        if self.config.wandb.save_artifacts:
            self.wandb_logger.save_artifact(
                str(best_model_path),
                name=f"{self.config.experiment_name}-best-{dataset_name}",
                artifact_type="model",
                metadata={
                    "dataset": dataset_name,
                    "epoch": epoch,
                    "accuracy": accuracy,
                },
            )
    
    def _check_and_save_best_models(
        self,
        eval_metrics: Dict[str, float],
        epoch: int,
    ):
        """
        Check evaluation metrics and save best models for each dataset.
        
        Args:
            eval_metrics: Dictionary with evaluation metrics (e.g., 'cub200/top1': 50.5)
            epoch: Current epoch number
        """
        if not is_main_process():
            return
        
        for dataset_name in self.config.eval.eval_datasets:
            top1_key = f"{dataset_name}/top1"
            if top1_key not in eval_metrics:
                continue
            
            current_accuracy = eval_metrics[top1_key]
            best_accuracy = self.best_metrics.get(dataset_name, 0.0)
            
            if current_accuracy > best_accuracy:
                self.best_metrics[dataset_name] = current_accuracy
                self._save_best_model(dataset_name, epoch, current_accuracy)
    
    def _resume_training(self):
        """Resume training from checkpoint."""
        checkpoint = self.checkpoint_manager.load(
            self.config.resume_from,
            self.model,
            self.optimizer,
            self.scheduler,
            device=str(self.device),
            reset_scheduler=self.config.reset_scheduler,
        )
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint.get('step', 0)
        
        # If reset_scheduler, we need to fast-forward scheduler to current position
        # so it computes LR correctly for the new total_epochs/total_steps
        if self.config.reset_scheduler:
            if self.step_based_lr:
                # For step-based scheduling, last_epoch tracks STEPS, not epochs
                # Scale the step count proportionally to new total_steps
                old_progress = self.global_step / self.scheduler.total_steps if self.scheduler.total_steps > 0 else 0
                # Set scheduler to current step (it will compute LR based on new total_steps)
                self.scheduler.last_epoch = self.global_step
                print(f"[Reset Scheduler] Step-based: set last_epoch to step {self.global_step}, "
                      f"total_steps={self.scheduler.total_steps}, progress={old_progress:.2%}")
            else:
                # For epoch-based scheduling, last_epoch tracks epochs
                self.scheduler.last_epoch = checkpoint['epoch']
                print(f"[Reset Scheduler] Epoch-based: set last_epoch to {checkpoint['epoch']}, "
                      f"total_epochs={self.scheduler.total_epochs}")
            
            # Get the LR that will be used
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"[Reset Scheduler] LR for epoch {self.current_epoch}: {current_lr:.6f}")
        
        # Restore teacher temperature to correct value for resumed epoch
        # (teacher_temp is not saved in checkpoint, so we need to recalculate it)
        if hasattr(self, 'temp_scheduler'):
            model = self.model.module if self.config.distributed.enabled else self.model
            if hasattr(model, 'teacher_temp'):
                resumed_temp = self.temp_scheduler.get_temperature(self.current_epoch)
                model.teacher_temp = resumed_temp
                print(f"[Resume] Set teacher_temp to {resumed_temp:.4f} for epoch {self.current_epoch}")
        
        # Restore best metrics for per-dataset best model tracking
        if 'extra_state' in checkpoint and checkpoint['extra_state']:
            if 'best_metrics' in checkpoint['extra_state']:
                self.best_metrics = checkpoint['extra_state']['best_metrics']
                print(f"[Resume] Restored best_metrics: {self.best_metrics}")
        
        print(f"Resumed from epoch {checkpoint['epoch']}")
    
    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        import random
        import numpy as np
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


class DINOTrainer(BaseTrainer):
    """
    Trainer for DINO self-supervised learning.
    
    Implements the DINO training procedure:
    - Multi-crop augmentation
    - Student-teacher distillation
    - EMA teacher updates
    - Center updates
    - Temperature scheduling
    
    Example:
        >>> config = TrainingConfig()
        >>> trainer = DINOTrainer(config)
        >>> trainer.train()
    """
    
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        
        # DINO-specific schedulers
        self.ema_scheduler = EMAScheduler(
            base_momentum=config.dino.ema_momentum_base,
            final_momentum=config.dino.ema_momentum_final,
            total_epochs=config.epochs,
        )
        
        self.temp_scheduler = TeacherTemperatureScheduler(
            final_temp=config.dino.teacher_temp,
            warmup_temp=config.dino.teacher_temp_warmup,
            warmup_epochs=config.dino.teacher_temp_warmup_epochs,
        )
    
    def build_model(self) -> nn.Module:
        """Build DINO model with ViT backbone."""
        # Create backbone
        if self.config.backbone.name == "vit_tiny":
            backbone = ViTTiny(
                img_size=self.config.backbone.img_size,
                patch_size=self.config.backbone.patch_size,
                drop_rate=self.config.backbone.drop_rate,
                attn_drop_rate=self.config.backbone.attn_drop_rate,
            )
        elif self.config.backbone.name == "vit_small":
            backbone = ViTSmall(
                img_size=self.config.backbone.img_size,
                patch_size=self.config.backbone.patch_size,
                drop_rate=self.config.backbone.drop_rate,
                attn_drop_rate=self.config.backbone.attn_drop_rate,
            )
        elif self.config.backbone.name == "vit_base":
            backbone = ViTBase(
                img_size=self.config.backbone.img_size,
                patch_size=self.config.backbone.patch_size,
                drop_rate=self.config.backbone.drop_rate,
                attn_drop_rate=self.config.backbone.attn_drop_rate,
            )
        else:
            raise ValueError(f"Unknown backbone: {self.config.backbone.name}")
        
        # Create DINO model
        model = DINO(
            backbone=backbone,
            out_dim=self.config.dino.out_dim,
            hidden_dim=self.config.dino.hidden_dim,
            bottleneck_dim=self.config.dino.bottleneck_dim,
            num_layers=self.config.dino.num_layers,
            use_bn=self.config.dino.use_bn,
            norm_last_layer=self.config.dino.norm_last_layer,
            teacher_temp=self.temp_scheduler.get_temperature(0),  # Start with warmup temp
            student_temp=self.config.dino.student_temp,
            center_momentum=self.config.dino.center_momentum,
        )
        
        return model
    
    def build_loss(self) -> nn.Module:
        """Build DINO loss function."""
        return DINOLoss(
            n_global_crops=2,
            n_local_crops=self.config.data.num_local_crops,
            l2g_weight=self.config.dino.l2g_weight,
            reg_weight=self.config.dino.reg_weight,
            use_l2g=self.config.dino.use_l2g_loss,
            use_reg=self.config.dino.use_reg_loss,
        )
    
    def train_step(self, batch: List[torch.Tensor], step: int = 0, accum_steps: int = 1) -> Dict[str, float]:
        """
        Execute a single DINO training step with gradient accumulation.
        
        Args:
            batch: List of view tensors [global1, global2, local1, ...]
            step: Current step within epoch (for accumulation)
            accum_steps: Number of gradient accumulation steps
            
        Returns:
            Dictionary with loss values and metrics
        """
        is_accumulating = ((step + 1) % accum_steps != 0)
        
        # Determine context for mixed precision
        amp_context = autocast() if self.config.mixed_precision else nullcontext()
        
        # Use no_sync() during gradient accumulation to avoid redundant gradient syncs
        # Only sync gradients on the last accumulation step
        sync_context = self.model.no_sync() if (is_accumulating and self.config.distributed.enabled) else nullcontext()
        
        with sync_context:
            with amp_context:
                # Forward pass
                outputs = self.model(
                    batch,
                    n_global_crops=2,
                    return_l2g_inputs=self.config.dino.use_l2g_loss,
                    return_reg_inputs=self.config.dino.use_reg_loss,
                )
                
                # Compute loss (scale by accumulation steps)
                loss_output = self.loss_fn(outputs)
                loss = loss_output.loss / accum_steps
                
                # Check for NaN loss with diagnostic info
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n{'='*60}")
                    print(f"WARNING: NaN/Inf loss detected at step {self.global_step}")
                    print(f"{'='*60}")
                    # Print diagnostic info
                    print(f"  Loss components:")
                    if loss_output.dino_loss is not None:
                        print(f"    dino_loss: {loss_output.dino_loss.item() if not torch.isnan(loss_output.dino_loss) else 'NaN'}")
                    if loss_output.l2g_loss is not None:
                        print(f"    l2g_loss: {loss_output.l2g_loss.item() if not torch.isnan(loss_output.l2g_loss) else 'NaN'}")
                    if loss_output.reg_loss is not None:
                        print(f"    reg_loss: {loss_output.reg_loss.item() if not torch.isnan(loss_output.reg_loss) else 'NaN'}")
                    # Check outputs for NaN
                    print(f"  Output statistics:")
                    for i, out in enumerate(outputs.get('student_outputs', [])):
                        has_nan = torch.isnan(out).any().item()
                        has_inf = torch.isinf(out).any().item()
                        print(f"    student_output[{i}]: nan={has_nan}, inf={has_inf}, norm={out.norm().item():.4f}")
                    for i, out in enumerate(outputs.get('teacher_outputs', [])):
                        has_nan = torch.isnan(out).any().item()
                        has_inf = torch.isinf(out).any().item()
                        print(f"    teacher_output[{i}]: nan={has_nan}, inf={has_inf}, norm={out.norm().item():.4f}")
                    center = outputs.get('center')
                    if center is not None:
                        print(f"    center: nan={torch.isnan(center).any().item()}, norm={center.norm().item():.4f}")
                    print(f"  Training state:")
                    print(f"    LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                    print(f"    teacher_temp: {outputs.get('teacher_temp', 'N/A')}")
                    if self.config.mixed_precision and hasattr(self, 'scaler'):
                        print(f"    GradScaler scale: {self.scaler.get_scale():.1f}")
                    print(f"{'='*60}\n")
                    
                    self.optimizer.zero_grad()
                    # Don't call scaler.update() here - it requires step() to be called first
                    return {
                        'loss': 0.0,
                        'lr': self.optimizer.param_groups[0]['lr'],
                        'ema_momentum': self.ema_scheduler.get_momentum(self.current_epoch),
                        'nan_detected': 1.0,
                    }
            
            # Backward pass (gradient sync happens here on last accumulation step)
            if self.config.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
        
        # Check for NaN in gradients after backward
        has_nan_grad = False
        nan_grad_layers = []
        for name, p in self.model.named_parameters():
            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                has_nan_grad = True
                nan_grad_layers.append(name)
        
        if has_nan_grad:
            print(f"\n{'='*60}")
            print(f"WARNING: NaN/Inf gradient detected at step {self.global_step}")
            print(f"{'='*60}")
            print(f"  Layers with NaN gradients (first 10):")
            for layer_name in nan_grad_layers[:10]:
                print(f"    - {layer_name}")
            if len(nan_grad_layers) > 10:
                print(f"    ... and {len(nan_grad_layers) - 10} more")
            print(f"  Loss value: {loss_output.loss.item() if not torch.isnan(loss_output.loss) else 'NaN'}")
            if self.config.mixed_precision and hasattr(self, 'scaler'):
                current_scale = self.scaler.get_scale()
                print(f"  GradScaler scale: {current_scale:.1f}")
                # Reduce scale on NaN gradient to help recover
                # This is more aggressive than the default backoff
                new_scale = max(2**10, current_scale / 4)  # Reduce by 4x, minimum 1024
                self.scaler._scale.fill_(new_scale)
                print(f"  Reduced scale to: {new_scale:.1f}")
            print(f"{'='*60}\n")
            
            self.optimizer.zero_grad()
            # Don't call scaler.update() here - it requires step() to be called first
            return {
                'loss': loss_output.loss.item() if not torch.isnan(loss_output.loss) else 0.0,
                'lr': self.optimizer.param_groups[0]['lr'],
                'ema_momentum': self.ema_scheduler.get_momentum(self.current_epoch),
                'nan_detected': 1.0,
            }
        
        # Only update weights after accumulation
        grad_norm = 0.0
        if not is_accumulating:
            if self.config.mixed_precision:
                # Gradient clipping
                if self.config.gradient_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    # Compute gradient norm BEFORE clipping and stepping
                    total_norm = 0.0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            total_norm += p.grad.data.norm(2).item() ** 2
                    grad_norm = total_norm ** 0.5
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                
                self.scaler.step(self.optimizer)
                
                # Cap the scaler BEFORE update to prevent runaway growth
                # This must happen before update() because update() can double the scale
                if self.scaler.get_scale() > self.max_grad_scale:
                    self.scaler._scale.fill_(self.max_grad_scale)
                
                self.scaler.update()
                
                # Also cap after update (belt and suspenders)
                if self.scaler.get_scale() > self.max_grad_scale:
                    self.scaler._scale.fill_(self.max_grad_scale)
            else:
                # Compute gradient norm BEFORE clipping
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.data.norm(2).item() ** 2
                grad_norm = total_norm ** 0.5
                
                # Gradient clipping
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                
                self.optimizer.step()
            
            self.optimizer.zero_grad()
        
        # Update teacher with EMA (only after optimizer step)
        # Access underlying model if wrapped with DDP
        model = self.model.module if self.config.distributed.enabled else self.model
        if not is_accumulating:
            momentum = self.ema_scheduler.get_momentum(self.current_epoch)
            model.update_teacher(momentum)
        else:
            momentum = self.ema_scheduler.get_momentum(self.current_epoch)
        
        # Collect metrics (report unscaled loss)
        metrics = {
            'loss': loss.item() * accum_steps,  # Report unscaled loss
            'lr': self.optimizer.param_groups[0]['lr'],
            'ema_momentum': momentum,
            'teacher_temp': model.teacher_temp,
            'grad_norm': grad_norm,
        }
        
        # Add loss components if available
        if loss_output.dino_loss is not None:
            metrics['dino_loss'] = loss_output.dino_loss.item()
        if loss_output.l2g_loss is not None:
            metrics['l2g_loss'] = loss_output.l2g_loss.item()
        if loss_output.reg_loss is not None:
            metrics['reg_loss'] = loss_output.reg_loss.item()
        
        # Add extra metrics from loss
        metrics.update(loss_output.metrics)
        
        return metrics
    
    def on_epoch_end(self, epoch: int):
        """
        End of epoch processing.
        
        - Update teacher temperature (if using warmup)
        - Log epoch summary
        """
        # Access underlying model if wrapped with DDP
        model = self.model.module if self.config.distributed.enabled else self.model
        
        # Update teacher temperature if using warmup
        new_temp = self.temp_scheduler.get_temperature(epoch + 1)
        if hasattr(model, 'teacher_temp'):
            model.teacher_temp = new_temp
