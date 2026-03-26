"""
Weights & Biases logging for training.

Provides comprehensive logging including:
- Training metrics (loss, lr, gradients)
- Evaluation metrics (KNN accuracy)
- System metrics (GPU memory, throughput)
- Model checkpoints
- Hyperparameter tracking
"""

import os
from typing import Dict, Optional, Any, List
from pathlib import Path
from dataclasses import asdict

import torch
import torch.nn as nn

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


class WandbLogger:
    """
    Weights & Biases logger for SSL training.
    
    Handles all wandb interactions including:
    - Initialization with config
    - Step-level metric logging
    - Epoch-level metric logging
    - Gradient logging
    - Model checkpointing
    - Custom charts and tables
    
    Args:
        project: W&B project name
        name: Run name (auto-generated if None)
        config: Training config dict
        enabled: Whether logging is enabled
        log_gradients: Log gradient histograms
        log_freq: Frequency of gradient logging
        save_code: Save code to W&B
        tags: List of tags for the run
        notes: Notes for the run
        
    Example:
        >>> logger = WandbLogger(
        ...     project="ssl-dino",
        ...     config=config.to_dict(),
        ...     enabled=True,
        ... )
        >>> logger.log_step({"loss": 0.5, "lr": 1e-4}, step=100)
        >>> logger.log_epoch({"val/top1": 25.0}, epoch=10)
        >>> logger.finish()
    """
    
    def __init__(
        self,
        project: str = "ssl-training",
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        enabled: bool = True,
        log_gradients: bool = True,
        log_freq: int = 100,
        save_code: bool = True,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        dir: Optional[str] = None,
    ):
        self.enabled = enabled and WANDB_AVAILABLE
        self.log_gradients = log_gradients
        self.log_freq = log_freq
        self.run = None
        
        if not self.enabled:
            if enabled and not WANDB_AVAILABLE:
                print("Warning: wandb logging requested but wandb not installed")
            return
        
        # Initialize wandb
        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
            save_code=save_code,
            tags=tags,
            notes=notes,
            dir=dir,
            reinit=True,
        )
        
        print(f"W&B Run: {self.run.name}")
        print(f"W&B URL: {self.run.url}")
    
    def log_step(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = "train",
        epoch: int = None,
        lr: float = None,
    ):
        """
        Log step-level metrics.
        
        Args:
            metrics: Dictionary of metric values
            step: Global step number
            prefix: Prefix for metric names (e.g., "train")
            epoch: Optional epoch number
            lr: Optional learning rate
        """
        if not self.enabled:
            return
        
        # Add prefix to metrics
        logged_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        logged_metrics["step"] = step
        
        if epoch is not None:
            logged_metrics["epoch"] = epoch
        if lr is not None:
            logged_metrics["lr"] = lr
        
        wandb.log(logged_metrics, step=step)
    
    def log_epoch(
        self,
        metrics: Dict[str, float],
        epoch: int,
        prefix: str = "",
        lr: float = None,
    ):
        """
        Log epoch-level metrics.
        
        Args:
            metrics: Dictionary of metric values
            epoch: Epoch number
            prefix: Optional prefix (e.g., "eval", "final")
            lr: Optional learning rate to log
        """
        if not self.enabled:
            return
        
        # Add prefix to metrics if provided
        if prefix:
            logged_metrics = {f"{prefix}/{k}": v for k, v in metrics.items() 
                           if not isinstance(v, list)}
        else:
            logged_metrics = {k: v for k, v in metrics.items() 
                           if not isinstance(v, list)}
        
        logged_metrics["epoch"] = epoch
        
        # Add learning rate if provided
        if lr is not None:
            logged_metrics["lr"] = lr
        
        wandb.log(logged_metrics)
    
    def log_gradients(
        self,
        model: nn.Module,
        step: int,
    ):
        """
        Log gradient statistics.
        
        Args:
            model: Model to log gradients for
            step: Global step number
        """
        if not self.enabled or not self.log_gradients:
            return
        
        if step % self.log_freq != 0:
            return
        
        grad_metrics = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                grad_metrics[f"gradients/{name}_norm"] = grad.norm().item()
                grad_metrics[f"gradients/{name}_mean"] = grad.mean().item()
                grad_metrics[f"gradients/{name}_std"] = grad.std().item()
        
        # Also log overall gradient norm
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.norm().item() ** 2
        total_norm = total_norm ** 0.5
        grad_metrics["gradients/total_norm"] = total_norm
        
        wandb.log(grad_metrics, step=step)
    
    def log_learning_rate(self, lr: float, step: int):
        """Log learning rate."""
        if not self.enabled:
            return
        wandb.log({"train/lr": lr}, step=step)
    
    def log_model_summary(self, model: nn.Module):
        """
        Log model architecture and parameter count.
        
        Args:
            model: Model to summarize
        """
        if not self.enabled:
            return
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        wandb.summary["model/total_params"] = total_params
        wandb.summary["model/trainable_params"] = trainable_params
        wandb.summary["model/total_params_M"] = total_params / 1e6
        wandb.summary["model/trainable_params_M"] = trainable_params / 1e6
    
    def log_system_metrics(self, step: int):
        """
        Log system metrics (GPU memory, etc.).
        
        Args:
            step: Global step number
        """
        if not self.enabled:
            return
        
        metrics = {}
        
        # GPU memory (if available)
        if torch.cuda.is_available():
            metrics["system/gpu_memory_allocated_GB"] = torch.cuda.memory_allocated() / 1e9
            metrics["system/gpu_memory_reserved_GB"] = torch.cuda.memory_reserved() / 1e9
            metrics["system/gpu_max_memory_allocated_GB"] = torch.cuda.max_memory_allocated() / 1e9
        
        # MPS memory (approximate)
        if torch.backends.mps.is_available():
            # MPS doesn't have detailed memory tracking like CUDA
            metrics["system/device"] = "mps"
        
        if metrics:
            wandb.log(metrics, step=step)
    
    def log_eval_results(
        self,
        results: Dict[str, float],
        step: int = None,
        epoch: int = None,
        prefix: str = "eval",
    ):
        """
        Log evaluation results with proper formatting.
        
        Args:
            results: Evaluation metrics dict (can contain dataset/metric keys)
            step: Optional step number
            epoch: Optional epoch number
            prefix: Prefix for metric names (e.g., "eval", "final")
        """
        if not self.enabled:
            return
        
        # Results already have dataset prefix like "cub200/top1"
        metrics = {}
        for key, value in results.items():
            if not isinstance(value, list):  # Skip lists
                metrics[f"{prefix}/{key}"] = value
        
        if epoch is not None:
            metrics["epoch"] = epoch
        
        if step is not None:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)
    
    def log_loss_components(
        self,
        loss_dict: Dict[str, float],
        step: int,
    ):
        """
        Log individual loss components.
        
        Args:
            loss_dict: Dictionary with loss components
            step: Global step number
        """
        if not self.enabled:
            return
        
        metrics = {}
        for name, value in loss_dict.items():
            metrics[f"loss/{name}"] = value
        
        wandb.log(metrics, step=step)
    
    def log_histogram(
        self,
        name: str,
        values: torch.Tensor,
        step: int,
    ):
        """
        Log histogram of values.
        
        Args:
            name: Histogram name
            values: Tensor of values
            step: Global step number
        """
        if not self.enabled:
            return
        
        wandb.log({name: wandb.Histogram(values.cpu().numpy())}, step=step)
    
    def log_image(
        self,
        name: str,
        image: torch.Tensor,
        step: int,
        caption: Optional[str] = None,
    ):
        """
        Log an image.
        
        Args:
            name: Image name
            image: Image tensor (C, H, W) or (H, W, C)
            step: Global step number
            caption: Optional caption
        """
        if not self.enabled:
            return
        
        if image.dim() == 3 and image.shape[0] in [1, 3]:
            # Convert from (C, H, W) to (H, W, C)
            image = image.permute(1, 2, 0)
        
        image_np = image.cpu().numpy()
        
        wandb.log({name: wandb.Image(image_np, caption=caption)}, step=step)
    
    def log_table(
        self,
        name: str,
        columns: List[str],
        data: List[List[Any]],
    ):
        """
        Log a table.
        
        Args:
            name: Table name
            columns: Column names
            data: Table data (list of rows)
        """
        if not self.enabled:
            return
        
        table = wandb.Table(columns=columns, data=data)
        wandb.log({name: table})
    
    def log_summary(
        self,
        key: str,
        value: Any,
    ):
        """
        Log a summary metric that appears in W&B run summary.
        
        Summary metrics are displayed prominently in W&B dashboard
        and are useful for final results comparison across runs.
        
        Args:
            key: Metric name
            value: Metric value
        """
        if not self.enabled or self.run is None:
            return
        
        self.run.summary[key] = value
    
    def save_artifact(
        self,
        file_path: str,
        name: str,
        artifact_type: str = "model",
        metadata: Optional[Dict] = None,
    ):
        """
        Save a file as W&B artifact.
        
        Args:
            file_path: Path to file to save
            name: Artifact name
            artifact_type: Type of artifact (e.g., "model", "checkpoint")
            metadata: Optional metadata dict
        """
        if not self.enabled or self.run is None:
            return
        
        try:
            # Sanitize metadata to remove NaN/Inf values
            sanitized_metadata = self._sanitize_metadata(metadata) if metadata else None
            
            artifact = wandb.Artifact(
                name=name,
                type=artifact_type,
                metadata=sanitized_metadata,
            )
            artifact.add_file(file_path)
            self.run.log_artifact(artifact)
            print(f"Saved W&B artifact: {name}")
        except Exception as e:
            # Don't block training if artifact upload fails
            print(f"[W&B] Warning: artifact upload failed (this is a known W&B issue): {e}")
    
    def _sanitize_metadata(self, metadata: Dict) -> Dict:
        """Sanitize metadata to remove NaN/Inf values that can't be JSON serialized."""
        import math
        
        def sanitize_value(v):
            if isinstance(v, float):
                if math.isnan(v) or math.isinf(v):
                    return None  # Replace NaN/Inf with None
                return v
            elif isinstance(v, dict):
                return {k: sanitize_value(val) for k, val in v.items()}
            elif isinstance(v, (list, tuple)):
                return [sanitize_value(item) for item in v]
            return v
        
        return sanitize_value(metadata)
    
    def save_checkpoint(
        self,
        checkpoint_path: str,
        metadata: Optional[Dict] = None,
    ):
        """
        Save checkpoint as W&B artifact.
        
        Args:
            checkpoint_path: Path to checkpoint file
            metadata: Optional metadata dict
        """
        if not self.enabled:
            return
        
        artifact = wandb.Artifact(
            name=f"checkpoint-{self.run.name}",
            type="model",
            metadata=metadata,
        )
        artifact.add_file(checkpoint_path)
        self.run.log_artifact(artifact)
    
    def watch_model(
        self,
        model: nn.Module,
        log: str = "gradients",
        log_freq: int = 100,
    ):
        """
        Watch model for gradient/parameter logging.
        
        Args:
            model: Model to watch
            log: What to log ("gradients", "parameters", "all")
            log_freq: Logging frequency
        """
        if not self.enabled:
            return
        
        wandb.watch(model, log=log, log_freq=log_freq)
    
    def define_metrics(self):
        """Define custom metrics for better plotting."""
        if not self.enabled:
            return
        
        # Define step metrics
        wandb.define_metric("step")
        wandb.define_metric("train/*", step_metric="step")
        wandb.define_metric("loss/*", step_metric="step")
        wandb.define_metric("gradients/*", step_metric="step")
        wandb.define_metric("system/*", step_metric="step")
        
        # Define epoch metrics
        wandb.define_metric("epoch")
        wandb.define_metric("eval/*", step_metric="epoch")
    
    def alert(
        self,
        title: str,
        text: str,
        level: str = "INFO",
    ):
        """
        Send an alert.
        
        Args:
            title: Alert title
            text: Alert text
            level: Alert level ("INFO", "WARN", "ERROR")
        """
        if not self.enabled:
            return
        
        wandb.alert(title=title, text=text, level=level)
    
    def finish(self):
        """Finish the wandb run."""
        if not self.enabled or self.run is None:
            return
        
        self.run.finish()
        print("W&B run finished")
    
    @property
    def run_name(self) -> Optional[str]:
        """Get the run name."""
        if self.run:
            return self.run.name
        return None
    
    @property
    def run_url(self) -> Optional[str]:
        """Get the run URL."""
        if self.run:
            return self.run.url
        return None
