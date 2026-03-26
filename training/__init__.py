"""
Training module for SSL pretraining.

This module provides:
- Configs: Type-safe configuration dataclasses
- Datasets: Data loading with multi-crop augmentation
- Trainers: Training loops for different SSL methods
- Utils: Logging, checkpointing, schedulers

Architecture follows SOLID principles:
- Single Responsibility: Each class has one job
- Open/Closed: Extend via inheritance/composition, not modification
- Liskov Substitution: Base classes define contracts
- Interface Segregation: Small, focused interfaces
- Dependency Inversion: Depend on abstractions, not concretions
"""

from .configs import (
    TrainingConfig,
    DINOConfig,
    OptimizerConfig,
    SchedulerConfig,
    DataConfig,
    BackboneConfig,
)
from .datasets import SSLDataset, MultiCropCollator, create_ssl_dataloader
from .trainers import BaseTrainer, DINOTrainer
from .utils import (
    CheckpointManager,
    TrainingLogger,
    EMAScheduler,
    WarmupCosineScheduler,
    TeacherTemperatureScheduler,
    create_optimizer,
)

__all__ = [
    # Configs
    "TrainingConfig",
    "DINOConfig", 
    "OptimizerConfig",
    "SchedulerConfig",
    "DataConfig",
    "BackboneConfig",
    # Datasets
    "SSLDataset",
    "MultiCropCollator",
    "create_ssl_dataloader",
    # Trainers
    "BaseTrainer",
    "DINOTrainer",
    # Utils
    "CheckpointManager",
    "TrainingLogger",
    "EMAScheduler",
    "WarmupCosineScheduler",
    "TeacherTemperatureScheduler",
    "create_optimizer",
]
