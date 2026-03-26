"""
Normalization classes for image preprocessing.

Provides:
- ImageNetNormalizer: Standard ImageNet normalization
- CustomNormalizer: User-defined mean/std
- StatsComputer: Compute mean/std from dataset

Used at runtime after ToTensor() to normalize pixel values.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Union
import json

import torch
from torch import Tensor
from PIL import Image
from tqdm import tqdm

from .base import (
    BaseNormalizer,
    BaseStatsComputer,
    DatasetStats,
    TransformRegistry,
)


# =============================================================================
# ImageNet Statistics (Standard)
# =============================================================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# =============================================================================
# Normalizer Classes
# =============================================================================

@TransformRegistry.register("imagenet_normalizer", category="normalizer")
class ImageNetNormalizer(BaseNormalizer):
    """
    Normalizer using ImageNet statistics.
    
    This is the standard normalization used by most pretrained models.
    
    Example:
        normalizer = ImageNetNormalizer()
        tensor = transforms.ToTensor()(image)  # [0, 1]
        normalized = normalizer(tensor)  # ~[-2, +2]
    """
    
    def __init__(self):
        super().__init__(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        # Pre-compute tensors for efficiency
        self._mean_tensor: Optional[Tensor] = None
        self._std_tensor: Optional[Tensor] = None
    
    def _ensure_tensors(self, device: torch.device) -> None:
        """Create mean/std tensors on the correct device."""
        if self._mean_tensor is None or self._mean_tensor.device != device:
            self._mean_tensor = torch.tensor(self.mean, device=device).view(3, 1, 1)
            self._std_tensor = torch.tensor(self.std, device=device).view(3, 1, 1)
    
    def __call__(self, tensor: Tensor) -> Tensor:
        """
        Normalize a tensor.
        
        Args:
            tensor: [C, H, W] or [B, C, H, W] tensor with values in [0, 1]
            
        Returns:
            Normalized tensor with approximately zero mean and unit variance
        """
        self._ensure_tensors(tensor.device)
        return (tensor - self._mean_tensor) / self._std_tensor
    
    def denormalize(self, tensor: Tensor) -> Tensor:
        """
        Reverse normalization (for visualization).
        
        Args:
            tensor: Normalized tensor
            
        Returns:
            Tensor with values in [0, 1]
        """
        self._ensure_tensors(tensor.device)
        return tensor * self._std_tensor + self._mean_tensor


@TransformRegistry.register("custom_normalizer", category="normalizer")
class CustomNormalizer(BaseNormalizer):
    """
    Normalizer with user-defined mean and std.
    
    Use this when you've computed statistics from your own dataset.
    
    Args:
        mean: Tuple of (R, G, B) mean values
        std: Tuple of (R, G, B) std values
        
    Example:
        # Load stats computed from your dataset
        normalizer = CustomNormalizer(
            mean=(0.512, 0.478, 0.421),
            std=(0.245, 0.238, 0.251)
        )
    """
    
    def __init__(
        self,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
    ):
        super().__init__(mean=mean, std=std)
        self._mean_tensor: Optional[Tensor] = None
        self._std_tensor: Optional[Tensor] = None
    
    def _ensure_tensors(self, device: torch.device) -> None:
        """Create mean/std tensors on the correct device."""
        if self._mean_tensor is None or self._mean_tensor.device != device:
            self._mean_tensor = torch.tensor(self.mean, device=device).view(3, 1, 1)
            self._std_tensor = torch.tensor(self.std, device=device).view(3, 1, 1)
    
    def __call__(self, tensor: Tensor) -> Tensor:
        """Normalize a tensor."""
        self._ensure_tensors(tensor.device)
        return (tensor - self._mean_tensor) / self._std_tensor
    
    def denormalize(self, tensor: Tensor) -> Tensor:
        """Reverse normalization."""
        self._ensure_tensors(tensor.device)
        return tensor * self._std_tensor + self._mean_tensor
    
    @classmethod
    def from_stats(cls, stats: DatasetStats) -> "CustomNormalizer":
        """Create normalizer from computed DatasetStats."""
        return cls(mean=stats.mean, std=stats.std)
    
    @classmethod
    def from_json(cls, path: Path) -> "CustomNormalizer":
        """Load normalizer from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            mean=tuple(data["mean"]),
            std=tuple(data["std"]),
        )
    
    def to_json(self, path: Path) -> None:
        """Save normalizer config to JSON file."""
        data = {
            "mean": list(self.mean),
            "std": list(self.std),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# =============================================================================
# Statistics Computer
# =============================================================================

@TransformRegistry.register("stats_computer", category="normalizer")
class StatsComputer(BaseStatsComputer):
    """
    Computes mean and std from a dataset of images.
    
    Uses Welford's online algorithm for numerical stability.
    
    Example:
        computer = StatsComputer()
        stats = computer.compute_from_directory(Path("data/pretrain/train"))
        print(f"Mean: {stats.mean}, Std: {stats.std}")
        
        # Save for later use
        normalizer = CustomNormalizer.from_stats(stats)
        normalizer.to_json(Path("configs/normalization_stats.json"))
    """
    
    def __init__(self, batch_size: int = 100):
        """
        Args:
            batch_size: Number of images to process at once (for progress reporting)
        """
        self.batch_size = batch_size
    
    def compute(self, image_paths: List[Path], show_progress: bool = True) -> DatasetStats:
        """
        Compute mean and std from a list of image paths.
        
        Uses Welford's online algorithm for numerical stability with large datasets.
        
        Args:
            image_paths: List of paths to images
            show_progress: Show progress bar
            
        Returns:
            DatasetStats with computed mean, std, and count
        """
        n = 0
        mean = torch.zeros(3)
        M2 = torch.zeros(3)  # Sum of squared differences
        
        iterator = image_paths
        if show_progress:
            iterator = tqdm(image_paths, desc="Computing statistics")
        
        for path in iterator:
            try:
                # Load and convert to tensor
                with Image.open(path) as img:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    
                    # Convert to tensor [C, H, W] with values [0, 1]
                    tensor = torch.tensor(
                        list(img.getdata()),
                        dtype=torch.float32
                    ).view(img.size[1], img.size[0], 3).permute(2, 0, 1) / 255.0
                    
                    # Compute mean over spatial dimensions for this image
                    img_mean = tensor.mean(dim=(1, 2))  # [3]
                    
                    # Welford's online algorithm
                    n += 1
                    delta = img_mean - mean
                    mean += delta / n
                    delta2 = img_mean - mean
                    M2 += delta * delta2
                    
            except Exception as e:
                continue  # Skip corrupted images
        
        if n < 2:
            raise ValueError("Need at least 2 valid images to compute statistics")
        
        # Compute std from M2
        variance = M2 / (n - 1)
        std = torch.sqrt(variance)
        
        return DatasetStats(
            mean=tuple(mean.tolist()),
            std=tuple(std.tolist()),
            num_images=n,
        )
    
    def compute_from_directory(
        self,
        directory: Path,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
        show_progress: bool = True,
    ) -> DatasetStats:
        """
        Compute statistics from all images in a directory.
        
        Args:
            directory: Directory containing images
            extensions: Valid image extensions
            show_progress: Show progress bar
            
        Returns:
            DatasetStats
        """
        directory = Path(directory)
        
        # Collect all image files
        image_paths = []
        for ext in extensions:
            image_paths.extend(directory.rglob(f"*{ext}"))
            image_paths.extend(directory.rglob(f"*{ext.upper()}"))
        
        image_paths = sorted(set(image_paths))
        
        if not image_paths:
            raise ValueError(f"No images found in {directory}")
        
        print(f"Found {len(image_paths)} images in {directory}")
        return self.compute(image_paths, show_progress=show_progress)
    
    def compute_and_save(
        self,
        directory: Path,
        output_path: Path,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
        show_progress: bool = True,
    ) -> DatasetStats:
        """
        Compute statistics and save to JSON file.
        
        Args:
            directory: Directory containing images
            output_path: Path to save JSON file
            extensions: Valid image extensions
            show_progress: Show progress bar
            
        Returns:
            DatasetStats
        """
        stats = self.compute_from_directory(
            directory, 
            extensions=extensions,
            show_progress=show_progress
        )
        
        # Save to JSON
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "mean": list(stats.mean),
            "std": list(stats.std),
            "num_images": stats.num_images,
            "source_directory": str(directory),
        }
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"Statistics saved to {output_path}")
        return stats


# =============================================================================
# Convenience Functions
# =============================================================================

def get_imagenet_normalizer() -> ImageNetNormalizer:
    """Get the standard ImageNet normalizer."""
    return ImageNetNormalizer()


def get_normalizer(
    mean: Optional[Tuple[float, float, float]] = None,
    std: Optional[Tuple[float, float, float]] = None,
) -> BaseNormalizer:
    """
    Get a normalizer with specified or default (ImageNet) stats.
    
    Args:
        mean: Custom mean, or None for ImageNet
        std: Custom std, or None for ImageNet
        
    Returns:
        Normalizer instance
    """
    if mean is None and std is None:
        return ImageNetNormalizer()
    elif mean is not None and std is not None:
        return CustomNormalizer(mean=mean, std=std)
    else:
        raise ValueError("Must provide both mean and std, or neither")


def compute_dataset_stats(directory: Path, show_progress: bool = True) -> DatasetStats:
    """
    Convenience function to compute statistics from a directory.
    
    Args:
        directory: Directory containing images
        show_progress: Show progress bar
        
    Returns:
        DatasetStats
    """
    computer = StatsComputer()
    return computer.compute_from_directory(directory, show_progress=show_progress)
