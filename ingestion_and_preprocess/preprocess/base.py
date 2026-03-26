"""
Base classes for preprocessing module.

Follows SOLID principles:
- Single Responsibility: Each class has one job
- Open/Closed: Extend via inheritance, don't modify base
- Liskov Substitution: Subclasses are interchangeable
- Interface Segregation: Small, focused interfaces
- Dependency Inversion: Depend on abstractions
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from PIL import Image


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TransformConfig:
    """Configuration for transforms."""
    target_size: Tuple[int, int] = (96, 96)
    normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    jpeg_quality: int = 95


@dataclass
class ProcessingResult:
    """Result of a preprocessing operation."""
    success: bool
    input_path: Optional[Path] = None
    output_path: Optional[Path] = None
    original_size: Optional[Tuple[int, int]] = None
    new_size: Optional[Tuple[int, int]] = None
    error_message: Optional[str] = None


@dataclass
class DatasetStats:
    """Statistics computed from a dataset."""
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    num_images: int
    num_channels: int = 3


# =============================================================================
# Abstract Base Classes
# =============================================================================

class BaseTransform(ABC):
    """
    Abstract base class for all transforms.
    
    Transforms can be applied to PIL Images or torch Tensors.
    Follows the callable pattern for compatibility with torchvision.Compose.
    """
    
    @abstractmethod
    def __call__(self, image: Union[Image.Image, torch.Tensor]) -> Union[Image.Image, torch.Tensor]:
        """Apply the transform to an image."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class BaseResizer(ABC):
    """
    Abstract base class for image resizers.
    
    Resizers handle the offline preprocessing of resizing images
    to a target size and saving them to disk.
    """
    
    def __init__(self, target_size: Tuple[int, int] = (96, 96)):
        self.target_size = target_size
    
    @abstractmethod
    def resize(self, image: Image.Image) -> Image.Image:
        """Resize a single image."""
        pass
    
    @abstractmethod
    def resize_and_save(
        self, 
        input_path: Path, 
        output_path: Path,
        quality: int = 95
    ) -> ProcessingResult:
        """Resize an image and save to disk."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(target_size={self.target_size})"


class BaseNormalizer(ABC):
    """
    Abstract base class for normalizers.
    
    Normalizers transform pixel values to have zero mean and unit variance.
    Can be used as a transform in a pipeline.
    """
    
    def __init__(
        self, 
        mean: Tuple[float, float, float], 
        std: Tuple[float, float, float]
    ):
        self.mean = mean
        self.std = std
    
    @abstractmethod
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize a tensor."""
        pass
    
    @abstractmethod
    def denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reverse normalization (useful for visualization)."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class BaseAugmentation(ABC):
    """
    Abstract base class for augmentation pipelines.
    
    Augmentations are composed transforms used during training.
    They return one or more augmented views of an image.
    """
    
    @abstractmethod
    def __call__(self, image: Image.Image) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Apply augmentation to an image.
        
        Args:
            image: PIL Image
            
        Returns:
            Single tensor or list of tensors (for multi-view methods like DINO)
        """
        pass
    
    @abstractmethod
    def get_num_views(self) -> int:
        """Return the number of views this augmentation produces."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_views={self.get_num_views()})"


class BaseStatsComputer(ABC):
    """
    Abstract base class for computing dataset statistics.
    
    Computes mean and std from a dataset for normalization.
    """
    
    @abstractmethod
    def compute(self, image_paths: List[Path]) -> DatasetStats:
        """Compute statistics from a list of image paths."""
        pass
    
    @abstractmethod
    def compute_from_directory(self, directory: Path) -> DatasetStats:
        """Compute statistics from all images in a directory."""
        pass


# =============================================================================
# Registry Pattern
# =============================================================================

class TransformRegistry:
    """
    Registry for transform classes.
    
    Allows dynamic registration and retrieval of transforms by name.
    Useful for config-driven transform selection.
    
    Usage:
        @TransformRegistry.register("my_transform")
        class MyTransform(BaseTransform):
            ...
        
        transform = TransformRegistry.get("my_transform")()
    """
    
    _transforms: Dict[str, Type[BaseTransform]] = {}
    _resizers: Dict[str, Type[BaseResizer]] = {}
    _normalizers: Dict[str, Type[BaseNormalizer]] = {}
    _augmentations: Dict[str, Type[BaseAugmentation]] = {}
    
    @classmethod
    def register(cls, name: str, category: str = "transform"):
        """
        Decorator to register a transform class.
        
        Args:
            name: Unique name for the transform
            category: One of "transform", "resizer", "normalizer", "augmentation"
        """
        def decorator(transform_cls: Type):
            registry = cls._get_registry(category)
            if name in registry:
                raise ValueError(f"{category} '{name}' is already registered")
            registry[name] = transform_cls
            return transform_cls
        return decorator
    
    @classmethod
    def get(cls, name: str, category: str = "transform") -> Type:
        """
        Get a registered transform class by name.
        
        Args:
            name: Name of the transform
            category: One of "transform", "resizer", "normalizer", "augmentation"
            
        Returns:
            The transform class (not instantiated)
        """
        registry = cls._get_registry(category)
        if name not in registry:
            available = list(registry.keys())
            raise ValueError(
                f"Unknown {category} '{name}'. Available: {available}"
            )
        return registry[name]
    
    @classmethod
    def list_available(cls, category: str = "transform") -> List[str]:
        """List all registered transforms in a category."""
        return list(cls._get_registry(category).keys())
    
    @classmethod
    def _get_registry(cls, category: str) -> Dict[str, Type]:
        """Get the appropriate registry dict for a category."""
        registries = {
            "transform": cls._transforms,
            "resizer": cls._resizers,
            "normalizer": cls._normalizers,
            "augmentation": cls._augmentations,
        }
        if category not in registries:
            raise ValueError(
                f"Unknown category '{category}'. "
                f"Must be one of: {list(registries.keys())}"
            )
        return registries[category]


# =============================================================================
# Utility Functions
# =============================================================================

def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Searches upward from this file for common project markers.
    """
    current = Path(__file__).resolve().parent
    markers = ['requirements.txt', '.git', 'setup.py', 'pyproject.toml']
    
    for _ in range(10):  # Max 10 levels up
        for marker in markers:
            if (current / marker).exists():
                return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    
    # Fallback: return grandparent of this file
    return Path(__file__).resolve().parent.parent.parent


def is_valid_image(path: Path) -> bool:
    """Check if a file is a valid image that can be loaded."""
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def get_image_info(path: Path) -> Optional[Dict[str, Any]]:
    """
    Get information about an image file.
    
    Returns:
        Dict with size, mode, format, or None if invalid
    """
    try:
        with Image.open(path) as img:
            return {
                "size": img.size,
                "mode": img.mode,
                "format": img.format,
            }
    except Exception:
        return None
