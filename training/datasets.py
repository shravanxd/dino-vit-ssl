"""
Dataset classes for SSL training.

Follows Single Responsibility Principle:
- SSLDataset: Handles loading images from disk
- MultiCropCollator: Handles batching multi-crop views

Follows Open/Closed Principle:
- Augmentation is injected, not hardcoded
"""

import os
from pathlib import Path
from typing import List, Callable, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from PIL import Image

import sys
sys.path.append(str(Path(__file__).parent.parent))

from ingestion_and_preprocess.preprocess.augmentations import DINOAugmentation


class SSLDataset(Dataset):
    """
    Dataset for self-supervised learning.
    
    Loads images from a directory and applies augmentation to create
    multiple views of each image.
    
    Design Principles:
    - Single Responsibility: Only handles loading images
    - Dependency Inversion: Augmentation is injected via transform
    
    Args:
        data_path: Path to directory containing images (or list of paths)
        transform: Augmentation function that returns multiple views
        extensions: Allowed image file extensions
        
    Example:
        >>> aug = DINOAugmentation(global_crop_size=96, num_local_crops=6)
        >>> dataset = SSLDataset("data/pretrain/train", transform=aug)
        >>> views = dataset[0]  # Returns list of tensors
        
        # Multiple paths
        >>> dataset = SSLDataset(["data/pretrain/train", "data/pretrain_laion/train"], transform=aug)
    """
    
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    def __init__(
        self,
        data_path: Union[str, Path, List[Union[str, Path]]],
        transform: Optional[Callable] = None,
        extensions: Optional[set] = None,
    ):
        self.transform = transform
        self.extensions = extensions or self.SUPPORTED_EXTENSIONS
        
        # Handle single path or list of paths
        if isinstance(data_path, (str, Path)):
            self.data_paths = [Path(data_path)]
        else:
            self.data_paths = [Path(p) for p in data_path]
        
        # Discover all image files from all paths
        self.image_paths = []
        for path in self.data_paths:
            if path.exists():
                images = self._discover_images_in_path(path)
                self.image_paths.extend(images)
                print(f"SSLDataset: Found {len(images):,} images in {path}")
            else:
                print(f"SSLDataset: WARNING - Path does not exist: {path}")
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.data_paths}")
        
        print(f"SSLDataset: Total {len(self.image_paths):,} images from {len(self.data_paths)} source(s)")
    
    def _discover_images_in_path(self, data_path: Path) -> List[Path]:
        """Discover all image files in a single data directory."""
        extensions_lower = {ext.lower() for ext in self.extensions}
        data_path_str = str(data_path)
        
        # Fast scan using os.listdir (returns strings, no stat calls)
        try:
            filenames = os.listdir(data_path)
            # Filter by extension - use string operations (faster than Path)
            image_paths = [
                Path(data_path_str, f) 
                for f in filenames 
                if '.' in f and f[f.rfind('.'):].lower() in extensions_lower
            ]
            
            if image_paths:
                return image_paths
        except Exception as e:
            print(f"Warning: Fast scan failed for {data_path}: {e}")
        
        # Fallback: search subdirectories
        image_paths = []
        for root, dirs, files in os.walk(data_path):
            for filename in files:
                if '.' in filename and filename[filename.rfind('.'):].lower() in extensions_lower:
                    image_paths.append(Path(root, filename))
        
        return image_paths
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Union[List[Tensor], Tensor]:
        """
        Load image and apply augmentation.
        
        Args:
            idx: Image index
            
        Returns:
            If transform returns list: List of view tensors
            If transform returns tensor: Single tensor
        """
        img_path = self.image_paths[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}")
        
        # Apply transform
        if self.transform is not None:
            return self.transform(image)
        
        return image
    
    def __repr__(self) -> str:
        paths_str = ", ".join(str(p) for p in self.data_paths)
        return (
            f"SSLDataset(\n"
            f"  paths=[{paths_str}],\n"
            f"  num_images={len(self)},\n"
            f"  transform={self.transform.__class__.__name__ if self.transform else None}\n"
            f")"
        )


class MultiCropCollator:
    """
    Collate function for multi-crop batches.
    
    Stacks each view separately across the batch, producing a list
    of tensors where each tensor contains all samples for that view.
    
    Input: List of [view1, view2, ..., viewN] from each sample
    Output: [batch_view1, batch_view2, ..., batch_viewN]
    
    This is more efficient than a nested structure and works directly
    with the DINO forward pass.
    
    Example:
        >>> collator = MultiCropCollator()
        >>> batch = [sample1_views, sample2_views, ...]  # Each is [v1, v2, ...]
        >>> collated = collator(batch)  # [batch_v1, batch_v2, ...]
    """
    
    def __call__(self, batch: List[List[Tensor]]) -> List[Tensor]:
        """
        Collate batch of multi-crop samples.
        
        Args:
            batch: List of samples, each sample is a list of view tensors
            
        Returns:
            List of batched view tensors
        """
        # batch = [[v1, v2, ...], [v1, v2, ...], ...]
        # We want [[batch_v1], [batch_v2], ...]
        
        num_views = len(batch[0])
        
        # Stack each view across the batch
        collated = []
        for view_idx in range(num_views):
            view_batch = torch.stack([sample[view_idx] for sample in batch])
            collated.append(view_batch)
        
        return collated
    
    def __repr__(self) -> str:
        return "MultiCropCollator()"


def create_ssl_dataloader(
    data_path: Union[str, Path, List[Union[str, Path]]],
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    global_crop_size: int = 96,
    local_crop_size: int = 48,
    num_local_crops: int = 6,
    global_crop_scale: Tuple[float, float] = (0.4, 1.0),
    local_crop_scale: Tuple[float, float] = (0.05, 0.4),
    drop_last: bool = True,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    """
    Factory function to create SSL DataLoader.
    
    Creates the complete data pipeline:
    1. SSLDataset with image discovery
    2. DINOAugmentation with multi-crop
    3. DataLoader with MultiCropCollator
    4. DistributedSampler if distributed training
    
    Args:
        data_path: Path to training images (single path or list of paths)
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers
        pin_memory: Pin memory for GPU transfer
        global_crop_size: Size of global crops
        local_crop_size: Size of local crops
        num_local_crops: Number of local crops
        global_crop_scale: Scale range for global crops
        local_crop_scale: Scale range for local crops
        drop_last: Drop last incomplete batch
        distributed: Whether to use distributed training
        world_size: Total number of processes
        rank: Current process rank
        
    Returns:
        Tuple of (DataLoader, DistributedSampler or None)
        
    Example:
        # Single source
        loader, sampler = create_ssl_dataloader("data/pretrain/train")
        
        # Multiple sources (CC3M + LAION)
        loader, sampler = create_ssl_dataloader([
            "data/pretrain/train",
            "data/pretrain_laion/train"
        ])
    """
    # Create augmentation
    transform = DINOAugmentation(
        global_crop_size=global_crop_size,
        local_crop_size=local_crop_size,
        num_local_crops=num_local_crops,
        global_crop_scale=global_crop_scale,
        local_crop_scale=local_crop_scale,
    )
    
    # Create dataset
    dataset = SSLDataset(
        data_path=data_path,
        transform=transform,
    )
    
    # Create sampler for distributed training
    sampler = None
    shuffle = True
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=drop_last,
        )
        shuffle = False  # Sampler handles shuffling
    
    # Create dataloader
    # prefetch_factor means each worker queues N batches ahead
    # This gives more buffer time for workers to prepare data
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=MultiCropCollator(),
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    
    return dataloader, sampler


# For backwards compatibility
def get_ssl_dataloader(*args, **kwargs):
    """Alias for create_ssl_dataloader."""
    return create_ssl_dataloader(*args, **kwargs)
