"""
Augmentation pipelines for SSL training and evaluation.

Provides:
- DINOAugmentation: Multi-crop strategy for DINO
- SimCLRAugmentation: Two-view augmentation for SimCLR
- BYOLAugmentation: Asymmetric augmentation for BYOL
- EvalTransform: Minimal transforms for evaluation

All augmentation classes follow the BaseAugmentation interface
and can be used interchangeably via the registry.
"""

from typing import List, Optional, Tuple

import torch
from torch import Tensor
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import functional as F

from .base import BaseAugmentation, TransformRegistry
from .normalizers import IMAGENET_MEAN, IMAGENET_STD


# =============================================================================
# Helper Transforms
# =============================================================================

class GaussianBlur:
    """
    Gaussian blur augmentation.
    
    Args:
        p: Probability of applying blur
        kernel_size: Size of blur kernel (will be adjusted to be odd)
        sigma: Range of sigma values (min, max)
    """
    
    def __init__(
        self,
        p: float = 0.5,
        kernel_size: int = 23,
        sigma: Tuple[float, float] = (0.1, 2.0),
    ):
        self.p = p
        # Ensure kernel size is odd
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.sigma = sigma
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if torch.rand(1).item() < self.p:
            sigma = torch.empty(1).uniform_(self.sigma[0], self.sigma[1]).item()
            return F.gaussian_blur(img, self.kernel_size, [sigma, sigma])
        return img


class Solarization:
    """
    Solarization augmentation (invert pixels above threshold).
    
    Args:
        p: Probability of applying solarization
        threshold: Pixel value threshold (0-255)
    """
    
    def __init__(self, p: float = 0.2, threshold: int = 128):
        self.p = p
        self.threshold = threshold
    
    def __call__(self, img: Image.Image) -> Image.Image:
        if torch.rand(1).item() < self.p:
            return F.solarize(img, self.threshold)
        return img


# =============================================================================
# SSL Augmentation Pipelines
# =============================================================================

@TransformRegistry.register("dino", category="augmentation")
class DINOAugmentation(BaseAugmentation):
    """
    DINO-style multi-crop augmentation.
    
    Produces multiple views of an image:
    - 2 global views (large crops covering most of image)
    - N local views (small crops, optional)
    
    The teacher sees global views, student sees all views.
    
    Args:
        global_crop_size: Size of global crops
        local_crop_size: Size of local crops
        global_crop_scale: Scale range for global crops
        local_crop_scale: Scale range for local crops
        num_local_crops: Number of local crops (0 to disable)
        normalize_mean: Normalization mean
        normalize_std: Normalization std
    
    Example:
        aug = DINOAugmentation(
            global_crop_size=96,
            local_crop_size=48,
            num_local_crops=6
        )
        views = aug(image)  # Returns [global1, global2, local1, ..., local6]
    """
    
    def __init__(
        self,
        global_crop_size: int = 96,
        local_crop_size: int = 48,
        global_crop_scale: Tuple[float, float] = (0.4, 1.0),
        local_crop_scale: Tuple[float, float] = (0.05, 0.4),
        num_local_crops: int = 0,
        normalize_mean: Tuple[float, float, float] = IMAGENET_MEAN,
        normalize_std: Tuple[float, float, float] = IMAGENET_STD,
    ):
        self.global_crop_size = global_crop_size
        self.local_crop_size = local_crop_size
        self.global_crop_scale = global_crop_scale
        self.local_crop_scale = local_crop_scale
        self.num_local_crops = num_local_crops
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        
        # Build transforms
        self._build_transforms()
    
    def _build_transforms(self):
        """Build the transform pipelines."""
        # Color jitter (same as DINO paper)
        color_jitter = T.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.2,
            hue=0.1,
        )
        
        # Normalization
        normalize = T.Normalize(mean=self.normalize_mean, std=self.normalize_std)
        
        # Global transform 1 (with blur, no solarization)
        self.global_transform_1 = T.Compose([
            T.RandomResizedCrop(
                self.global_crop_size,
                scale=self.global_crop_scale,
                interpolation=T.InterpolationMode.BICUBIC,
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),  # Always blur for view 1
            T.ToTensor(),
            normalize,
        ])
        
        # Global transform 2 (with blur + solarization)
        self.global_transform_2 = T.Compose([
            T.RandomResizedCrop(
                self.global_crop_size,
                scale=self.global_crop_scale,
                interpolation=T.InterpolationMode.BICUBIC,
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),  # Less blur for view 2
            Solarization(p=0.2),  # Add solarization
            T.ToTensor(),
            normalize,
        ])
        
        # Local transform (smaller crops, no solarization)
        if self.num_local_crops > 0:
            self.local_transform = T.Compose([
                T.RandomResizedCrop(
                    self.local_crop_size,
                    scale=self.local_crop_scale,
                    interpolation=T.InterpolationMode.BICUBIC,
                ),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomApply([color_jitter], p=0.8),
                T.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.5),
                T.ToTensor(),
                normalize,
            ])
    
    def __call__(self, image: Image.Image) -> List[Tensor]:
        """
        Apply augmentation to produce multiple views.
        
        Args:
            image: PIL Image
            
        Returns:
            List of tensors: [global1, global2, local1, ..., localN]
        """
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        views = []
        
        # Global views
        views.append(self.global_transform_1(image))
        views.append(self.global_transform_2(image))
        
        # Local views
        for _ in range(self.num_local_crops):
            views.append(self.local_transform(image))
        
        return views
    
    def get_num_views(self) -> int:
        """Return total number of views (2 global + N local)."""
        return 2 + self.num_local_crops
    
    def __repr__(self) -> str:
        return (
            f"DINOAugmentation("
            f"global_size={self.global_crop_size}, "
            f"local_size={self.local_crop_size}, "
            f"num_local={self.num_local_crops})"
        )


@TransformRegistry.register("simclr", category="augmentation")
class SimCLRAugmentation(BaseAugmentation):
    """
    SimCLR-style two-view augmentation.
    
    Produces exactly 2 augmented views of the same image.
    Both views use the same transform pipeline with independent randomness.
    
    Args:
        crop_size: Size of random crop
        crop_scale: Scale range for random resize crop
        normalize_mean: Normalization mean
        normalize_std: Normalization std
    """
    
    def __init__(
        self,
        crop_size: int = 96,
        crop_scale: Tuple[float, float] = (0.2, 1.0),
        normalize_mean: Tuple[float, float, float] = IMAGENET_MEAN,
        normalize_std: Tuple[float, float, float] = IMAGENET_STD,
    ):
        self.crop_size = crop_size
        self.crop_scale = crop_scale
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        
        self._build_transforms()
    
    def _build_transforms(self):
        """Build the SimCLR transform pipeline."""
        color_jitter = T.ColorJitter(
            brightness=0.8,
            contrast=0.8,
            saturation=0.8,
            hue=0.2,
        )
        
        self.transform = T.Compose([
            T.RandomResizedCrop(
                self.crop_size,
                scale=self.crop_scale,
                interpolation=T.InterpolationMode.BICUBIC,
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.5, kernel_size=int(0.1 * self.crop_size) | 1),
            T.ToTensor(),
            T.Normalize(mean=self.normalize_mean, std=self.normalize_std),
        ])
    
    def __call__(self, image: Image.Image) -> List[Tensor]:
        """
        Apply augmentation to produce two views.
        
        Args:
            image: PIL Image
            
        Returns:
            List of 2 tensors [view1, view2]
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        return [self.transform(image), self.transform(image)]
    
    def get_num_views(self) -> int:
        return 2


@TransformRegistry.register("byol", category="augmentation")
class BYOLAugmentation(BaseAugmentation):
    """
    BYOL-style asymmetric augmentation.
    
    Produces 2 views with slightly different augmentation pipelines.
    View 1 has stronger augmentation, view 2 has weaker augmentation.
    
    Args:
        crop_size: Size of random crop
        normalize_mean: Normalization mean
        normalize_std: Normalization std
    """
    
    def __init__(
        self,
        crop_size: int = 96,
        normalize_mean: Tuple[float, float, float] = IMAGENET_MEAN,
        normalize_std: Tuple[float, float, float] = IMAGENET_STD,
    ):
        self.crop_size = crop_size
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        
        self._build_transforms()
    
    def _build_transforms(self):
        """Build asymmetric transform pipelines."""
        color_jitter = T.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.2,
            hue=0.1,
        )
        
        normalize = T.Normalize(mean=self.normalize_mean, std=self.normalize_std)
        
        # View 1: Stronger augmentation
        self.transform_1 = T.Compose([
            T.RandomResizedCrop(
                self.crop_size,
                scale=(0.2, 1.0),
                interpolation=T.InterpolationMode.BICUBIC,
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            T.ToTensor(),
            normalize,
        ])
        
        # View 2: Weaker augmentation
        self.transform_2 = T.Compose([
            T.RandomResizedCrop(
                self.crop_size,
                scale=(0.2, 1.0),
                interpolation=T.InterpolationMode.BICUBIC,
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            T.ToTensor(),
            normalize,
        ])
    
    def __call__(self, image: Image.Image) -> List[Tensor]:
        """Apply asymmetric augmentation."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        return [self.transform_1(image), self.transform_2(image)]
    
    def get_num_views(self) -> int:
        return 2


# =============================================================================
# Evaluation Transforms
# =============================================================================

@TransformRegistry.register("eval", category="augmentation")
class EvalTransform(BaseAugmentation):
    """
    Minimal transforms for evaluation.
    
    For images that are already 96x96 (preprocessed offline),
    this just converts to tensor and normalizes.
    
    For train set: Optional horizontal flip for slight augmentation
    For test set: No augmentation, deterministic
    
    Args:
        crop_size: Expected image size (for verification)
        is_train: Whether this is for train set (enables flip)
        normalize_mean: Normalization mean
        normalize_std: Normalization std
    """
    
    def __init__(
        self,
        crop_size: int = 96,
        is_train: bool = False,
        normalize_mean: Tuple[float, float, float] = IMAGENET_MEAN,
        normalize_std: Tuple[float, float, float] = IMAGENET_STD,
    ):
        self.crop_size = crop_size
        self.is_train = is_train
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        
        self._build_transforms()
    
    def _build_transforms(self):
        """Build eval transform pipeline."""
        transforms = []
        
        if self.is_train:
            transforms.append(T.RandomHorizontalFlip(p=0.5))
        
        transforms.extend([
            T.ToTensor(),
            T.Normalize(mean=self.normalize_mean, std=self.normalize_std),
        ])
        
        self.transform = T.Compose(transforms)
    
    def __call__(self, image: Image.Image) -> Tensor:
        """
        Apply eval transform.
        
        Args:
            image: PIL Image (should be 96x96)
            
        Returns:
            Single tensor [C, H, W]
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        return self.transform(image)
    
    def get_num_views(self) -> int:
        return 1
    
    def __repr__(self) -> str:
        return f"EvalTransform(is_train={self.is_train})"


# =============================================================================
# Factory Functions
# =============================================================================

def get_ssl_augmentation(
    method: str,
    crop_size: int = 96,
    **kwargs
) -> BaseAugmentation:
    """
    Get SSL augmentation by method name.
    
    Args:
        method: One of "dino", "simclr", "byol"
        crop_size: Crop size
        **kwargs: Additional arguments for the augmentation
        
    Returns:
        Augmentation instance
    """
    method = method.lower()
    
    if method == "dino":
        return DINOAugmentation(global_crop_size=crop_size, **kwargs)
    elif method == "simclr":
        return SimCLRAugmentation(crop_size=crop_size, **kwargs)
    elif method == "byol":
        return BYOLAugmentation(crop_size=crop_size, **kwargs)
    else:
        raise ValueError(f"Unknown SSL method: {method}. Choose from: dino, simclr, byol")


def get_eval_transform(
    is_train: bool = False,
    crop_size: int = 96,
    normalize_mean: Tuple[float, float, float] = IMAGENET_MEAN,
    normalize_std: Tuple[float, float, float] = IMAGENET_STD,
) -> EvalTransform:
    """
    Get evaluation transform.
    
    Args:
        is_train: Whether for train set (enables flip)
        crop_size: Expected image size
        normalize_mean: Normalization mean
        normalize_std: Normalization std
        
    Returns:
        EvalTransform instance
    """
    return EvalTransform(
        crop_size=crop_size,
        is_train=is_train,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )
