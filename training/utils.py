"""
Training utilities.

Provides:
- CheckpointManager: Save and load model checkpoints
- TrainingLogger: Log metrics to console and files
- EMAScheduler: Schedule EMA momentum during training
- WarmupCosineScheduler: Learning rate scheduler with warmup

Follows Interface Segregation - each utility has a focused interface.
"""

import os
import math
import json
import time
from pathlib import Path
from typing import Dict, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


# =============================================================================
# Checkpoint Management
# =============================================================================

class CheckpointManager:
    """
    Manages saving and loading of training checkpoints.
    
    Features:
    - Saves model, optimizer, scheduler, and training state
    - Keeps track of best checkpoint
    - Supports checkpoint rotation (keep last N)
    
    Design: Single Responsibility - only handles checkpointing
    
    Args:
        checkpoint_dir: Directory to save checkpoints
        max_checkpoints: Maximum checkpoints to keep (0 = keep all)
        
    Example:
        >>> manager = CheckpointManager("checkpoints/exp1")
        >>> manager.save(model, optimizer, epoch=10, metrics={'loss': 0.5})
        >>> state = manager.load_latest()
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_checkpoints: int = 5,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        
        # Create directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Track checkpoints
        self.checkpoint_files: list = []
        self.best_metric: Optional[float] = None
        self.best_checkpoint: Optional[Path] = None
    
    def save(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        epoch: int,
        step: int = 0,
        scheduler: Optional[_LRScheduler] = None,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save a checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optimizer to save
            epoch: Current epoch
            step: Current step within epoch
            scheduler: Optional LR scheduler
            metrics: Optional metrics dict
            is_best: Whether this is the best checkpoint so far
            extra_state: Any additional state to save
            
        Returns:
            Path to saved checkpoint
        """
        # Build checkpoint state
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics or {},
            'timestamp': datetime.now().isoformat(),
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if extra_state is not None:
            checkpoint['extra_state'] = extra_state
        
        # Save checkpoint
        filename = f"checkpoint_epoch{epoch:04d}.pt"
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        
        self.checkpoint_files.append(filepath)
        print(f"Saved checkpoint: {filepath}")
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            self.best_checkpoint = best_path
            print(f"Saved best checkpoint: {best_path}")
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        return filepath
    
    def load(
        self,
        path: Union[str, Path],
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        device: str = "cpu",
        reset_scheduler: bool = False,
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            path: Path to checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            device: Device to load checkpoint to
            reset_scheduler: If True, don't load scheduler state (reset LR schedule)
            
        Returns:
            Checkpoint dict with epoch, step, metrics, etc.
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        # Handle DDP module prefix mismatch
        state_dict = checkpoint['model_state_dict']
        model_state_dict = model.state_dict()
        
        # Debug: print first few keys from both
        ckpt_keys = list(state_dict.keys())[:3]
        model_keys = list(model_state_dict.keys())[:3]
        print(f"Checkpoint keys (first 3): {ckpt_keys}")
        print(f"Model keys (first 3): {model_keys}")
        
        # Check if state_dict has 'module.' prefix but model doesn't (or vice versa)
        state_dict_has_module = any(k.startswith('module.') for k in state_dict.keys())
        model_has_module = any(k.startswith('module.') for k in model_state_dict.keys())
        
        if state_dict_has_module and not model_has_module:
            # Remove 'module.' prefix from checkpoint
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
            print("Removed 'module.' prefix from checkpoint state_dict")
        elif not state_dict_has_module and model_has_module:
            # Add 'module.' prefix to checkpoint
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
            print("Added 'module.' prefix to checkpoint state_dict")
        
        # Load model with strict=False first to see what's missing
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print(f"Strict loading failed, trying with strict=False: {e}")
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"Missing keys ({len(missing)}): {missing[:5]}...")
            if unexpected:
                print(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
        
        # Load optimizer
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler (unless reset_scheduler is True)
        if scheduler is not None and 'scheduler_state_dict' in checkpoint and not reset_scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        elif reset_scheduler:
            print("Scheduler state NOT loaded (--reset_scheduler flag set)")
        
        print(f"Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
        
        return checkpoint
    
    def load_latest(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        device: str = "cpu",
    ) -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch*.pt"))
        
        if not checkpoints:
            print("No checkpoints found")
            return None
        
        latest = checkpoints[-1]
        return self.load(latest, model, optimizer, scheduler, device)
    
    def load_best(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        device: str = "cpu",
    ) -> Optional[Dict[str, Any]]:
        """Load the best checkpoint."""
        best_path = self.checkpoint_dir / "checkpoint_best.pt"
        
        if not best_path.exists():
            print("No best checkpoint found")
            return None
        
        return self.load(best_path, model, optimizer, scheduler, device)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints if exceeding max_checkpoints."""
        if self.max_checkpoints <= 0:
            return
        
        # Get all checkpoints (excluding best)
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch*.pt"))
        
        while len(checkpoints) > self.max_checkpoints:
            oldest = checkpoints.pop(0)
            oldest.unlink()
            print(f"Removed old checkpoint: {oldest}")


# =============================================================================
# Logging
# =============================================================================

@dataclass
class MetricTracker:
    """Track a single metric with running average."""
    name: str
    total: float = 0.0
    count: int = 0
    
    def update(self, value: float, n: int = 1):
        self.total += value * n
        self.count += n
    
    @property
    def avg(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0
    
    def reset(self):
        self.total = 0.0
        self.count = 0


class TrainingLogger:
    """
    Logger for training metrics.
    
    Features:
    - Log to console with formatting
    - Log to JSON file for analysis
    - Track running averages
    - Compute throughput (samples/sec)
    
    Design: Single Responsibility - only handles logging
    
    Args:
        log_dir: Directory for log files
        experiment_name: Name of experiment
        log_interval: Steps between console logs
        
    Example:
        >>> logger = TrainingLogger("logs", "exp1")
        >>> logger.log_step(step=100, metrics={'loss': 0.5, 'lr': 1e-4})
        >>> logger.log_epoch(epoch=1, metrics={'val_acc': 0.8})
    """
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        experiment_name: str = "training",
        log_interval: int = 50,
    ):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.log_interval = log_interval
        
        # Create directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.json"
        
        # Tracking
        self.metrics: Dict[str, MetricTracker] = {}
        self.step_logs: list = []
        self.epoch_logs: list = []
        
        # Timing
        self.start_time = time.time()
        self.step_start_time = time.time()
        self.samples_processed = 0
    
    def log_step(
        self,
        step: int,
        epoch: int,
        metrics: Dict[str, float],
        batch_size: int = 1,
        print_log: bool = True,
    ):
        """
        Log metrics for a training step.
        
        Args:
            step: Global step number
            epoch: Current epoch
            metrics: Dictionary of metric values
            batch_size: Batch size for throughput calculation
            print_log: Whether to print to console
        """
        # Update running averages
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = MetricTracker(name)
            self.metrics[name].update(value, batch_size)
        
        # Track samples
        self.samples_processed += batch_size
        
        # Log to console
        if print_log and step % self.log_interval == 0:
            elapsed = time.time() - self.step_start_time
            throughput = self.samples_processed / elapsed if elapsed > 0 else 0
            
            # Format metrics
            metric_strs = [f"{k}: {v:.4f}" for k, v in metrics.items()]
            metric_str = " | ".join(metric_strs)
            
            print(f"[Epoch {epoch}][Step {step}] {metric_str} | {throughput:.1f} samples/s")
            
            # Reset timing
            self.step_start_time = time.time()
            self.samples_processed = 0
        
        # Store log
        self.step_logs.append({
            'step': step,
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': time.time() - self.start_time,
        })
    
    def log_epoch(
        self,
        epoch: int,
        metrics: Dict[str, float],
        print_log: bool = True,
        prefix: str = "",
    ):
        """
        Log metrics for a completed epoch.
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metric values (e.g., epoch averages)
            print_log: Whether to print to console
            prefix: Optional prefix for log type (e.g., "eval", "final_eval")
        """
        # Add running averages to metrics (only for training, not eval)
        if not prefix:
            for name, tracker in self.metrics.items():
                if f"avg_{name}" not in metrics:
                    metrics[f"avg_{name}"] = tracker.avg
        
        # Log to console
        if print_log:
            elapsed = time.time() - self.start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            
            # Filter out list values for display
            display_metrics = {k: v for k, v in metrics.items() if not isinstance(v, list)}
            metric_strs = [f"{k}: {v:.4f}" for k, v in display_metrics.items()]
            metric_str = " | ".join(metric_strs)
            
            header = f"Epoch {epoch}" if not prefix else f"Epoch {epoch} ({prefix})"
            
            print(f"\n{'='*60}")
            print(f"{header} [{hours}h {minutes}m]")
            print(f"{metric_str}")
            print(f"{'='*60}\n")
        
        # Store log
        log_entry = {
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': time.time() - self.start_time,
        }
        if prefix:
            log_entry['type'] = prefix
        
        self.epoch_logs.append(log_entry)
        
        # Save to file
        self._save_logs()
        
        # Reset trackers (only for training, not eval)
        if not prefix:
            for tracker in self.metrics.values():
                tracker.reset()
    
    def _save_logs(self):
        """Save logs to JSON file."""
        logs = {
            'experiment': self.experiment_name,
            'step_logs': self.step_logs,
            'epoch_logs': self.epoch_logs,
        }
        
        with open(self.log_file, 'w') as f:
            json.dump(logs, f, indent=2)
    
    def get_epoch_metrics(self) -> Dict[str, float]:
        """Get current running averages for all metrics."""
        return {name: tracker.avg for name, tracker in self.metrics.items()}


# =============================================================================
# Schedulers
# =============================================================================

class EMAScheduler:
    """
    Schedule EMA momentum during training.
    
    DINO uses momentum that increases from base to final over training.
    This helps the teacher adapt quickly early on, then stabilize.
    
    Args:
        base_momentum: Starting momentum (default: 0.996)
        final_momentum: Final momentum (default: 1.0)
        total_epochs: Total training epochs
        schedule: 'cosine' or 'linear'
        
    Example:
        >>> scheduler = EMAScheduler(0.996, 1.0, total_epochs=100)
        >>> for epoch in range(100):
        ...     momentum = scheduler.get_momentum(epoch)
        ...     model.update_teacher(momentum)
    """
    
    def __init__(
        self,
        base_momentum: float = 0.996,
        final_momentum: float = 1.0,
        total_epochs: int = 100,
        schedule: str = 'cosine',
    ):
        self.base_momentum = base_momentum
        self.final_momentum = final_momentum
        self.total_epochs = total_epochs
        self.schedule = schedule
    
    def get_momentum(self, epoch: int) -> float:
        """Get EMA momentum for given epoch."""
        if epoch >= self.total_epochs:
            return self.final_momentum
        
        progress = epoch / self.total_epochs
        
        if self.schedule == 'cosine':
            # Cosine annealing from base to final
            momentum = self.final_momentum - (self.final_momentum - self.base_momentum) * (
                1 + math.cos(math.pi * progress)
            ) / 2
        else:  # linear
            momentum = self.base_momentum + (self.final_momentum - self.base_momentum) * progress
        
        return momentum
    
    def __repr__(self) -> str:
        return (
            f"EMAScheduler(base={self.base_momentum}, "
            f"final={self.final_momentum}, epochs={self.total_epochs})"
        )


class WarmupCosineScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warmup and cosine decay.
    
    LR schedule:
    1. Warmup: Linear increase from warmup_lr to base_lr
    2. Cosine decay: Decrease from base_lr to min_lr
    
    Supports BOTH epoch-based and step-based warmup:
    - If warmup_steps > 0: use step-based warmup (recommended for DINO)
    - Otherwise: use epoch-based warmup
    
    Args:
        optimizer: Optimizer to schedule
        warmup_epochs: Number of warmup epochs (epoch-based)
        total_epochs: Total training epochs
        warmup_start_lr: LR at start of warmup
        min_lr: Minimum LR after decay
        warmup_steps: Number of warmup steps (step-based, overrides epochs if > 0)
        total_steps: Total training steps (required if warmup_steps > 0)
        last_epoch: Last epoch (for resuming)
        
    Example (epoch-based):
        >>> scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=10, total_epochs=100)
        >>> for epoch in range(100):
        ...     train_one_epoch()
        ...     scheduler.step()
    
    Example (step-based):
        >>> scheduler = WarmupCosineScheduler(
        ...     optimizer, warmup_epochs=0, total_epochs=100,
        ...     warmup_steps=1000, total_steps=39000
        ... )
        >>> for step in range(39000):
        ...     train_one_step()
        ...     scheduler.step()  # Call per step
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        warmup_start_lr: float = 1e-6,
        min_lr: float = 1e-6,
        warmup_steps: int = 0,
        total_steps: int = 0,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_start_lr = warmup_start_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        
        # Determine if using step-based scheduling
        self.step_based = warmup_steps > 0 and total_steps > 0
        
        # Store base LRs before calling super().__init__
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Calculate learning rate for current epoch/step."""
        if self.step_based:
            # Step-based scheduling
            current_step = self.last_epoch  # last_epoch tracks steps when step() is called per step
            
            if current_step < self.warmup_steps:
                # Warmup phase: linear increase
                progress = current_step / self.warmup_steps
                return [
                    self.warmup_start_lr + (base_lr - self.warmup_start_lr) * progress
                    for base_lr in self.base_lrs
                ]
            else:
                # Cosine decay phase
                decay_steps = self.total_steps - self.warmup_steps
                progress = (current_step - self.warmup_steps) / decay_steps if decay_steps > 0 else 1.0
                progress = min(progress, 1.0)
                
                return [
                    self.min_lr + (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2
                    for base_lr in self.base_lrs
                ]
        else:
            # Epoch-based scheduling (original behavior)
            if self.last_epoch < self.warmup_epochs:
                # Warmup phase: linear increase
                progress = self.last_epoch / self.warmup_epochs if self.warmup_epochs > 0 else 1.0
                return [
                    self.warmup_start_lr + (base_lr - self.warmup_start_lr) * progress
                    for base_lr in self.base_lrs
                ]
            else:
                # Cosine decay phase
                decay_epochs = self.total_epochs - self.warmup_epochs
                progress = (self.last_epoch - self.warmup_epochs) / decay_epochs if decay_epochs > 0 else 1.0
                progress = min(progress, 1.0)
                
                return [
                    self.min_lr + (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress)) / 2
                    for base_lr in self.base_lrs
                ]


class TeacherTemperatureScheduler:
    """
    Schedule teacher temperature during training.
    
    DINO can optionally warmup teacher temperature from a higher
    value to the target temperature.
    
    Args:
        final_temp: Final temperature (default: 0.04)
        warmup_temp: Temperature at start (default: 0.04)
        warmup_epochs: Epochs to warmup (default: 0, no warmup)
        
    Example:
        >>> scheduler = TeacherTemperatureScheduler(0.04, 0.07, warmup_epochs=30)
        >>> for epoch in range(100):
        ...     temp = scheduler.get_temperature(epoch)
    """
    
    def __init__(
        self,
        final_temp: float = 0.04,
        warmup_temp: float = 0.04,
        warmup_epochs: int = 0,
    ):
        self.final_temp = final_temp
        self.warmup_temp = warmup_temp
        self.warmup_epochs = warmup_epochs
    
    def get_temperature(self, epoch: int) -> float:
        """Get teacher temperature for given epoch."""
        if self.warmup_epochs == 0 or epoch >= self.warmup_epochs:
            return self.final_temp
        
        # Linear warmup (decreasing from warmup_temp to final_temp)
        progress = epoch / self.warmup_epochs
        return self.warmup_temp + (self.final_temp - self.warmup_temp) * progress


# =============================================================================
# Optimizer Factory
# =============================================================================

def create_optimizer(
    model: nn.Module,
    name: str = "adamw",
    lr: float = 5e-4,
    weight_decay: float = 0.04,
    betas: tuple = (0.9, 0.999),
    momentum: float = 0.9,
    filter_bias_and_bn: bool = True,
) -> Optimizer:
    """
    Create optimizer with proper weight decay handling.
    
    Separates parameters into:
    - Decay group: Regular parameters with weight decay
    - No decay group: Biases and normalization parameters
    
    Args:
        model: Model to optimize
        name: Optimizer name ('adamw' or 'sgd')
        lr: Learning rate
        weight_decay: Weight decay for decay group
        betas: Adam betas (for AdamW)
        momentum: Momentum (for SGD)
        filter_bias_and_bn: Disable weight decay for bias/bn params
        
    Returns:
        Configured optimizer
    """
    if filter_bias_and_bn:
        # Separate parameters
        decay_params = []
        no_decay_params = []
        
        for name_param, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # No decay for bias and normalization
            if 'bias' in name_param or 'norm' in name_param or 'bn' in name_param:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
    else:
        param_groups = [{'params': model.parameters(), 'weight_decay': weight_decay}]
    
    # Create optimizer
    if name.lower() == "adamw":
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas)
    elif name.lower() == "sgd":
        optimizer = torch.optim.SGD(param_groups, lr=lr, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer: {name}")
    
    return optimizer
