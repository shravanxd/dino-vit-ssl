"""
Image resizer classes for offline preprocessing.

Provides various resizing strategies:
- Direct resize (stretch to target size)
- Resize with aspect ratio preservation + center crop
- Resize shortest side + center crop

Used for offline preprocessing of eval datasets to 96x96.
"""

from enum import Enum
from pathlib import Path
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from PIL import Image

from .base import (
    BaseResizer,
    ProcessingResult,
    TransformRegistry,
    is_valid_image,
)


class ResizeMode(Enum):
    """Resize modes for image preprocessing."""
    DIRECT = "direct"  # Stretch to exact size (may distort)
    SHORTEST_SIDE = "shortest_side"  # Resize shortest side, then center crop
    LONGEST_SIDE = "longest_side"  # Resize longest side, then pad or crop


@TransformRegistry.register("image_resizer", category="resizer")
class ImageResizer(BaseResizer):
    """
    Resizes images to a target size using various strategies.
    
    This is the main class for offline preprocessing of eval datasets.
    
    Args:
        target_size: Target (width, height) tuple, default (96, 96)
        mode: Resize mode (DIRECT, SHORTEST_SIDE, LONGEST_SIDE)
        resample: PIL resampling filter (LANCZOS recommended for downscaling)
    
    Example:
        resizer = ImageResizer(target_size=(96, 96), mode=ResizeMode.DIRECT)
        
        # Resize single image
        resized_img = resizer.resize(img)
        
        # Resize and save
        result = resizer.resize_and_save(input_path, output_path)
        
        # Batch resize directory
        results = resizer.resize_directory(input_dir, output_dir)
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (96, 96),
        mode: ResizeMode = ResizeMode.DIRECT,
        resample: int = Image.Resampling.LANCZOS,
    ):
        super().__init__(target_size)
        self.mode = mode
        self.resample = resample
    
    def resize(self, image: Image.Image) -> Image.Image:
        """
        Resize a single image to target size.
        
        Args:
            image: PIL Image (any size)
            
        Returns:
            PIL Image resized to target_size
        """
        # Ensure RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        if self.mode == ResizeMode.DIRECT:
            return self._resize_direct(image)
        elif self.mode == ResizeMode.SHORTEST_SIDE:
            return self._resize_shortest_side(image)
        elif self.mode == ResizeMode.LONGEST_SIDE:
            return self._resize_longest_side(image)
        else:
            raise ValueError(f"Unknown resize mode: {self.mode}")
    
    def _resize_direct(self, image: Image.Image) -> Image.Image:
        """Directly resize to target size (may distort aspect ratio)."""
        return image.resize(self.target_size, resample=self.resample)
    
    def _resize_shortest_side(self, image: Image.Image) -> Image.Image:
        """
        Resize so shortest side matches target, then center crop.
        
        Preserves aspect ratio, no distortion.
        """
        width, height = image.size
        target_w, target_h = self.target_size
        
        # Calculate scale to make shortest side match target
        scale = max(target_w / width, target_h / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize
        image = image.resize((new_width, new_height), resample=self.resample)
        
        # Center crop to exact target size
        left = (new_width - target_w) // 2
        top = (new_height - target_h) // 2
        right = left + target_w
        bottom = top + target_h
        
        return image.crop((left, top, right, bottom))
    
    def _resize_longest_side(self, image: Image.Image) -> Image.Image:
        """
        Resize so longest side matches target, then center crop.
        
        May lose some content from edges.
        """
        width, height = image.size
        target_w, target_h = self.target_size
        
        # Calculate scale to make longest side match target
        scale = min(target_w / width, target_h / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize
        image = image.resize((new_width, new_height), resample=self.resample)
        
        # Center crop to exact target size (will crop if needed)
        left = (new_width - target_w) // 2
        top = (new_height - target_h) // 2
        right = left + target_w
        bottom = top + target_h
        
        # Handle case where resized image is smaller than target
        if new_width < target_w or new_height < target_h:
            # Create new image with target size and paste resized in center
            result = Image.new("RGB", self.target_size, (0, 0, 0))
            paste_x = (target_w - new_width) // 2
            paste_y = (target_h - new_height) // 2
            result.paste(image, (paste_x, paste_y))
            return result
        
        return image.crop((left, top, right, bottom))
    
    def resize_and_save(
        self,
        input_path: Path,
        output_path: Path,
        quality: int = 95,
    ) -> ProcessingResult:
        """
        Resize an image and save to disk.
        
        Args:
            input_path: Path to input image
            output_path: Path to save resized image
            quality: JPEG quality (1-100)
            
        Returns:
            ProcessingResult with success status and details
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        try:
            # Load image
            with Image.open(input_path) as img:
                original_size = img.size
                
                # Resize
                resized = self.resize(img)
                
                # Ensure output directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save
                resized.save(output_path, "JPEG", quality=quality)
                
                return ProcessingResult(
                    success=True,
                    input_path=input_path,
                    output_path=output_path,
                    original_size=original_size,
                    new_size=resized.size,
                )
                
        except Exception as e:
            return ProcessingResult(
                success=False,
                input_path=input_path,
                output_path=output_path,
                error_message=str(e),
            )
    
    def resize_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        quality: int = 95,
        num_workers: int = 4,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
        overwrite: bool = False,
        show_progress: bool = True,
    ) -> dict:
        """
        Resize all images in a directory.
        
        Args:
            input_dir: Directory containing images
            output_dir: Directory to save resized images
            quality: JPEG quality
            num_workers: Number of parallel workers
            extensions: Valid image extensions
            overwrite: Whether to overwrite existing files
            show_progress: Show progress bar
            
        Returns:
            Dict with statistics: total, success, failed, skipped
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        # Collect all image files
        image_files = []
        for ext in extensions:
            image_files.extend(input_dir.rglob(f"*{ext}"))
            image_files.extend(input_dir.rglob(f"*{ext.upper()}"))
        
        # Remove duplicates and sort
        image_files = sorted(set(image_files))
        
        stats = {
            "total": len(image_files),
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "errors": [],
        }
        
        def process_single(img_path: Path) -> ProcessingResult:
            # Calculate output path (preserve directory structure)
            rel_path = img_path.relative_to(input_dir)
            out_path = output_dir / rel_path.with_suffix(".jpg")
            
            # Skip if exists and not overwriting
            if out_path.exists() and not overwrite:
                return ProcessingResult(
                    success=True,
                    input_path=img_path,
                    output_path=out_path,
                    error_message="skipped",
                )
            
            return self.resize_and_save(img_path, out_path, quality)
        
        # Process in parallel
        iterator = image_files
        if show_progress:
            iterator = tqdm(image_files, desc="Resizing images")
        
        if num_workers > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = {
                    executor.submit(process_single, f): f 
                    for f in image_files
                }
                
                for future in tqdm(
                    as_completed(futures), 
                    total=len(futures),
                    desc="Resizing images",
                    disable=not show_progress
                ):
                    result = future.result()
                    if result.error_message == "skipped":
                        stats["skipped"] += 1
                    elif result.success:
                        stats["success"] += 1
                    else:
                        stats["failed"] += 1
                        stats["errors"].append(
                            (result.input_path, result.error_message)
                        )
        else:
            # Single-threaded for debugging
            for img_path in iterator:
                result = process_single(img_path)
                if result.error_message == "skipped":
                    stats["skipped"] += 1
                elif result.success:
                    stats["success"] += 1
                else:
                    stats["failed"] += 1
                    stats["errors"].append(
                        (result.input_path, result.error_message)
                    )
        
        return stats
    
    def resize_inplace(
        self,
        directory: Path,
        quality: int = 95,
        num_workers: int = 4,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
        show_progress: bool = True,
    ) -> dict:
        """
        Resize all images in a directory IN PLACE (overwrites originals).
        
        ⚠️ WARNING: This modifies original files!
        
        Args:
            directory: Directory containing images
            quality: JPEG quality
            num_workers: Number of parallel workers
            extensions: Valid image extensions
            show_progress: Show progress bar
            
        Returns:
            Dict with statistics
        """
        return self.resize_directory(
            input_dir=directory,
            output_dir=directory,
            quality=quality,
            num_workers=num_workers,
            extensions=extensions,
            overwrite=True,
            show_progress=show_progress,
        )
    
    def __repr__(self) -> str:
        return (
            f"ImageResizer(target_size={self.target_size}, "
            f"mode={self.mode.value})"
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def resize_image(
    image: Image.Image,
    target_size: Tuple[int, int] = (96, 96),
    mode: ResizeMode = ResizeMode.DIRECT,
) -> Image.Image:
    """
    Convenience function to resize a single image.
    
    Args:
        image: PIL Image
        target_size: Target (width, height)
        mode: Resize mode
        
    Returns:
        Resized PIL Image
    """
    resizer = ImageResizer(target_size=target_size, mode=mode)
    return resizer.resize(image)


def resize_file(
    input_path: Path,
    output_path: Path,
    target_size: Tuple[int, int] = (96, 96),
    mode: ResizeMode = ResizeMode.DIRECT,
    quality: int = 95,
) -> ProcessingResult:
    """
    Convenience function to resize and save a single image file.
    
    Args:
        input_path: Path to input image
        output_path: Path to save resized image
        target_size: Target (width, height)
        mode: Resize mode
        quality: JPEG quality
        
    Returns:
        ProcessingResult
    """
    resizer = ImageResizer(target_size=target_size, mode=mode)
    return resizer.resize_and_save(input_path, output_path, quality)
