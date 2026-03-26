"""
Datasets for evaluation.

Provides labeled datasets for downstream evaluation tasks.
"""

from typing import Optional, Callable, Union, Tuple
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms


class LabeledDataset(Dataset):
    """
    Dataset for labeled evaluation data.
    
    Loads images and labels from a directory structure with CSV labels.
    Automatically filters out samples where the image file doesn't exist.
    
    Expected structure:
        data_root/
        ├── train/          # Image directory
        ├── val/            # Image directory
        ├── train_labels.csv  # image_id,label
        └── val_labels.csv    # image_id,label
    
    Args:
        image_dir: Directory containing images
        labels_file: CSV file with image_id,label columns
        transform: Image transforms to apply
        
    Example:
        >>> dataset = LabeledDataset(
        ...     image_dir="data/eval_public/cub200/train",
        ...     labels_file="data/eval_public/cub200/train_labels.csv",
        ...     transform=eval_transform,
        ... )
        >>> image, label = dataset[0]
    """
    
    def __init__(
        self,
        image_dir: Union[str, Path],
        labels_file: Union[str, Path],
        transform: Optional[Callable] = None,
    ):
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        # Load labels from CSV and filter to only existing files
        all_samples = []
        with open(labels_file, 'r') as f:
            # Skip header
            header = f.readline()
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    image_id = parts[0]
                    label = int(parts[1])
                    all_samples.append((image_id, label))
        
        # Filter to only samples where the image exists
        self.samples = []
        missing_count = 0
        for image_id, label in all_samples:
            image_path = self._find_image_path(image_id)
            if image_path is not None:
                self.samples.append((image_id, label, image_path))
            else:
                missing_count += 1
        
        # Get unique labels for num_classes (from ALL labels, not just found ones)
        self.all_labels = [s[1] for s in all_samples]
        self.num_classes = len(set(self.all_labels))
        
        # Labels from found samples
        self.labels = [s[1] for s in self.samples]
        
        print(f"Loaded {len(self.samples)}/{len(all_samples)} samples "
              f"({missing_count} missing), {self.num_classes} classes from {labels_file}")
    
    def _find_image_path(self, image_id: str) -> Optional[Path]:
        """Find the image file path, trying different extensions."""
        # First try direct path (image_id might already include extension)
        image_path = self.image_dir / image_id
        if image_path.exists():
            return image_path
        
        # Try adding different extensions
        for ext in ['.jpg', '.jpeg', '.png', '.JPEG', '.PNG', '.JPG']:
            candidate = self.image_dir / f"{image_id}{ext}"
            if candidate.exists():
                return candidate
        
        return None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        image_id, label, image_path = self.samples[idx]
        
        image = Image.open(image_path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label


class TestDataset(Dataset):
    """
    Dataset for unlabeled test images (for submission generation).
    
    Loads images listed in a CSV file (e.g., test_images.csv) without labels.
    Returns (image, image_id) tuples.
    
    Expected structure:
        data_root/
        ├── test/              # Image directory
        └── test_images.csv    # filename column listing image filenames
    
    Args:
        image_dir: Directory containing test images
        images_csv: CSV file listing test image filenames (single 'filename' column)
        transform: Image transforms to apply
        
    Example:
        >>> dataset = TestDataset(
        ...     image_dir="data/eval_public/cub200/test",
        ...     images_csv="data/eval_public/cub200/test_images.csv",
        ...     transform=eval_transform,
        ... )
        >>> image, image_id = dataset[0]
    """
    
    def __init__(
        self,
        image_dir: Union[str, Path],
        images_csv: Union[str, Path],
        transform: Optional[Callable] = None,
    ):
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        # Load image filenames from CSV
        self.samples = []
        with open(images_csv, 'r') as f:
            # Skip header
            header = f.readline()
            for line in f:
                filename = line.strip()
                if filename:
                    image_path = self._find_image_path(filename)
                    if image_path is not None:
                        self.samples.append((filename, image_path))
                    else:
                        print(f"  Warning: test image not found: {filename}")
        
        print(f"Loaded {len(self.samples)} test images from {images_csv}")
    
    def _find_image_path(self, filename: str) -> Optional[Path]:
        """Find the image file path."""
        image_path = self.image_dir / filename
        if image_path.exists():
            return image_path
        
        # Try without extension and add common ones
        base = Path(filename).stem
        for ext in ['.jpg', '.jpeg', '.png', '.JPEG', '.PNG', '.JPG']:
            candidate = self.image_dir / f"{base}{ext}"
            if candidate.exists():
                return candidate
        
        return None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[Tensor, str]:
        filename, image_path = self.samples[idx]
        
        image = Image.open(image_path).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, filename


def get_eval_transform(img_size: int = 96) -> transforms.Compose:
    """
    Get standard evaluation transform.
    
    Simple resize + normalize (no augmentation for eval).
    
    Args:
        img_size: Target image size
        
    Returns:
        Composed transform
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def create_eval_dataloaders(
    dataset_name: str,
    data_root: Union[str, Path],
    img_size: int = 96,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Create train and val dataloaders for KNN evaluation.
    
    Args:
        dataset_name: Name of dataset (e.g., 'cub200', 'miniimagenet')
        data_root: Root directory for eval datasets
        img_size: Image size for evaluation
        batch_size: Batch size for feature extraction
        num_workers: Number of dataloader workers
        
    Returns:
        Tuple of (train_dataloader, val_dataloader, num_classes)
        
    Example:
        >>> train_loader, val_loader, num_classes = create_eval_dataloaders(
        ...     dataset_name="cub200",
        ...     data_root="data/eval_public",
        ...     img_size=96,
        ... )
    """
    data_root = Path(data_root) / dataset_name
    transform = get_eval_transform(img_size)
    
    train_dataset = LabeledDataset(
        image_dir=data_root / 'train',
        labels_file=data_root / 'train_labels.csv',
        transform=transform,
    )
    
    val_dataset = LabeledDataset(
        image_dir=data_root / 'val',
        labels_file=data_root / 'val_labels.csv',
        transform=transform,
    )
    
    # Determine pin_memory based on device
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffle for feature extraction
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return train_loader, val_loader, train_dataset.num_classes
