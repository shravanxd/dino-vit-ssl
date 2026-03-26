"""
Preprocessing module for SSL Capstone Project.

This module provides:
- Image resizing utilities (offline preprocessing)
- Normalization classes (runtime)
- Augmentation pipelines (runtime - SSL and Eval)
- Data verification utilities

Architecture follows SOLID principles with:
- Abstract base classes for extensibility
- Registry pattern for easy plug-in of new transforms
- Separation of offline vs runtime operations
"""

from .base import (
    BaseTransform,
    BaseResizer,
    BaseNormalizer,
    BaseAugmentation,
    TransformRegistry,
)
from .resizers import ImageResizer, ResizeMode
from .normalizers import ImageNetNormalizer, CustomNormalizer, StatsComputer
from .augmentations import (
    DINOAugmentation,
    SimCLRAugmentation,
    BYOLAugmentation,
    EvalTransform,
)

__all__ = [
    # Base classes
    "BaseTransform",
    "BaseResizer",
    "BaseNormalizer",
    "BaseAugmentation",
    "TransformRegistry",
    # Resizers
    "ImageResizer",
    "ResizeMode",
    # Normalizers
    "ImageNetNormalizer",
    "CustomNormalizer",
    "StatsComputer",
    # Augmentations
    "DINOAugmentation",
    "SimCLRAugmentation",
    "BYOLAugmentation",
    "EvalTransform",
]
