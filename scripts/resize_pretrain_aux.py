#!/usr/bin/env python3
"""
Resize images in an arbitrary pretrain dataset folder (e.g., COCO, Places).

Usage:
    python scripts/resize_pretrain_aux.py --data_dir data/auxiliary/places365/train --size 96
    python scripts/resize_pretrain_aux.py --data_dir data/auxiliary/coco_unlabeled/train --size 96

This mirrors the behavior of scripts/resize_pretrain_birds.py but works for any
flat or nested image directory.
"""

import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image
from tqdm import tqdm


def resize_image_if_needed(img_path: Path, target_size: int, quality: int = 95) -> dict:
    """Resize image in-place if it's not already the target size.

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
                Image.Resampling.LANCZOS,
            )

            # Determine output format based on extension
            ext = img_path.suffix.lower()
            if ext in [".jpg", ".jpeg"]:
                img_resized.save(img_path, "JPEG", quality=quality)
            elif ext == ".png":
                img_resized.save(img_path, "PNG")
            elif ext == ".webp":
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
    """Find all images in directory (recursively)."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = []

    for root, _, files in os.walk(directory):
        for f in files:
            if Path(f).suffix.lower() in extensions:
                images.append(Path(root) / f)

    return images


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resize auxiliary pretrain images to target size",
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Path to training images directory (e.g., data/auxiliary/places365/train)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=96,
        help="Target image size (default: 96)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG/WebP quality (default: 95)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )

    args = parser.parse_args()

    # Normalize data_dir to project root if relative
    data_dir = args.data_dir
    if not data_dir.is_absolute():
        project_root = Path(__file__).parent.parent
        data_dir = project_root / data_dir

    print("=" * 60)
    print("AUXILIARY PRETRAIN IMAGE RESIZER")
    print("=" * 60)
    print(f"Directory: {data_dir}")
    print(f"Target size: {args.size}x{args.size}")

    if not data_dir.exists():
        print(f"\n❌ Directory not found: {data_dir}")
        print("Please create the directory and add images first.")
        return

    images = find_images(data_dir)
    print(f"Found {len(images):,} images")

    if len(images) == 0:
        print("No images found!")
        return

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

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✓ Resized: {stats['resized']:,}")
    print(f"⏭ Skipped (already correct size): {stats['skipped']:,}")
    print(f"✗ Errors: {stats['error']:,}")

    if errors:
        print("\nFirst 5 errors:")
        for e in errors[:5]:
            print(f"  {e['path']}: {e['error']}")

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
