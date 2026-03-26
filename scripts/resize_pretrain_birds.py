#!/usr/bin/env python3
"""
Resize images in pretrain_birds dataset to 96x96.

Usage:
    python scripts/resize_pretrain_birds.py
    
    # Custom target size
    python scripts/resize_pretrain_birds.py --size 96
    
    # Check sizes without resizing
    python scripts/resize_pretrain_birds.py --check-only
"""

import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image
from tqdm import tqdm


def get_image_size(img_path: Path) -> tuple:
    """Get image dimensions."""
    try:
        with Image.open(img_path) as img:
            return img.size
    except Exception as e:
        return None


def resize_image_if_needed(img_path: Path, target_size: int, quality: int = 95) -> dict:
    """
    Resize image in-place if it's not already the target size.
    
    Returns dict with status info.
    """
    result = {
        "path": str(img_path),
        "action": None,
        "original_size": None,
        "error": None,
    }
    
    try:
        with Image.open(img_path) as img:
            original_size = img.size
            result["original_size"] = original_size
            
            # Check if already correct size
            if original_size == (target_size, target_size):
                result["action"] = "skipped"
                return result
            
            # Resize
            img_resized = img.convert("RGB").resize(
                (target_size, target_size), 
                Image.Resampling.LANCZOS
            )
            
            # Determine output format based on extension
            ext = img_path.suffix.lower()
            if ext in ['.jpg', '.jpeg']:
                img_resized.save(img_path, "JPEG", quality=quality)
            elif ext == '.png':
                img_resized.save(img_path, "PNG")
            elif ext == '.webp':
                img_resized.save(img_path, "WEBP", quality=quality)
            else:
                # Default to JPEG
                img_resized.save(img_path, "JPEG", quality=quality)
            
            result["action"] = "resized"
            
    except Exception as e:
        result["action"] = "error"
        result["error"] = str(e)
    
    return result


def find_images(directory: Path) -> list:
    """Find all images in directory."""
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    images = []
    
    for root, _, files in os.walk(directory):
        for f in files:
            if Path(f).suffix.lower() in extensions:
                images.append(Path(root) / f)
    
    return images


def check_sizes(directory: Path) -> dict:
    """Check the distribution of image sizes in directory."""
    images = find_images(directory)
    sizes = {}
    
    print(f"Checking {len(images):,} images...")
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(get_image_size, img): img for img in images}
        
        for future in tqdm(as_completed(futures), total=len(images)):
            size = future.result()
            if size:
                sizes[size] = sizes.get(size, 0) + 1
    
    return sizes


def main():
    parser = argparse.ArgumentParser(description="Resize pretrain_birds images to target size")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/pretrain_birds/train"),
        help="Path to birds training data (default: data/pretrain_birds/train)"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=96,
        help="Target image size (default: 96)"
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality (default: 95)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check sizes, don't resize"
    )
    
    args = parser.parse_args()
    
    # Handle relative paths
    if not args.data_dir.is_absolute():
        project_root = Path(__file__).parent.parent
        args.data_dir = project_root / args.data_dir
    
    print("=" * 60)
    print("PRETRAIN BIRDS IMAGE RESIZER")
    print("=" * 60)
    print(f"Directory: {args.data_dir}")
    print(f"Target size: {args.size}x{args.size}")
    
    if not args.data_dir.exists():
        print(f"\n❌ Directory not found: {args.data_dir}")
        print("Please create the directory and add images first.")
        return
    
    # Find all images
    images = find_images(args.data_dir)
    print(f"Found {len(images):,} images")
    
    if len(images) == 0:
        print("No images found!")
        return
    
    # Check sizes first
    if args.check_only:
        print("\n📊 Checking image sizes...")
        sizes = check_sizes(args.data_dir)
        
        print("\nSize distribution:")
        for size, count in sorted(sizes.items(), key=lambda x: -x[1])[:20]:
            target_match = "✓" if size == (args.size, args.size) else ""
            print(f"  {size[0]}x{size[1]}: {count:,} images {target_match}")
        
        target_count = sizes.get((args.size, args.size), 0)
        need_resize = len(images) - target_count
        print(f"\nAlready {args.size}x{args.size}: {target_count:,}")
        print(f"Need resizing: {need_resize:,}")
        return
    
    # Resize images
    print(f"\n🔄 Resizing images to {args.size}x{args.size}...")
    
    stats = {"resized": 0, "skipped": 0, "error": 0}
    errors = []
    
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(resize_image_if_needed, img, args.size, args.quality): img 
            for img in images
        }
        
        for future in tqdm(as_completed(futures), total=len(images)):
            result = future.result()
            stats[result["action"]] += 1
            if result["action"] == "error":
                errors.append(result)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✓ Resized: {stats['resized']:,}")
    print(f"⏭ Skipped (already correct size): {stats['skipped']:,}")
    print(f"✗ Errors: {stats['error']:,}")
    
    if errors:
        print(f"\nFirst 5 errors:")
        for e in errors[:5]:
            print(f"  {e['path']}: {e['error']}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
