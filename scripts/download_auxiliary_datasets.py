#!/usr/bin/env python3
"""
Download auxiliary datasets for DINO pretraining:
- COCO Unlabeled 2017 (~123K images)
- Places365 Standard (sample ~200K images)
"""

import argparse
import subprocess
import random
import shutil
from pathlib import Path
from typing import Optional


def download_coco_unlabeled(output_dir: str) -> str:
    """Download COCO Unlabeled 2017 dataset and flatten into a train/ folder."""
    print("=" * 60)
    print("Downloading COCO Unlabeled 2017...")
    print("=" * 60)

    output_path = Path(output_dir) / "coco_unlabeled"
    output_path.mkdir(parents=True, exist_ok=True)

    zip_path = output_path / "unlabeled2017.zip"
    train_path = output_path / "train"

    # Download if not exists
    if not zip_path.exists() and not train_path.exists():
        print("Downloading COCO unlabeled2017.zip (~19GB)...")
        subprocess.run([
            "wget", "-c",
            "http://images.cocodataset.org/zips/unlabeled2017.zip",
            "-O", str(zip_path),
        ], check=True)

    # Extract with -j to flatten directory structure
    if not train_path.exists():
        print("Extracting images (flattening directory)...")
        train_path.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "unzip", "-j", str(zip_path),
            "-d", str(train_path),
        ], check=True)

        # Remove zip to save space
        print("Removing zip file to save space...")
        zip_path.unlink()

    # Count images
    num_images = len(list(train_path.glob("*.jpg")))
    print(f"COCO Unlabeled ready: {num_images} images at {train_path}")

    return str(train_path)


def download_places365(output_dir: str, num_samples: int = 200000) -> Optional[str]:
    """Download Places365 Standard and extract only a sampled subset into train/.

    Storage-efficient: we never keep the full extracted dataset, only the
    sampled subset plus the tar during extraction. Sampling is deterministic
    via a fixed RNG seed, so the same subset is chosen across machines.
    """
    print("=" * 60)
    print(f"Preparing Places365 Standard (sampling {num_samples} images)...")
    print("=" * 60)

    output_path = Path(output_dir) / "places365"
    output_path.mkdir(parents=True, exist_ok=True)

    train_path = output_path / "train"
    tar_path = output_path / "places365_standard.tar"

    # If we've already created the sampled train/ folder, just reuse it
    if train_path.exists() and any(train_path.iterdir()):
        num_images = len(list(train_path.glob("*")))
        print(f"Places365 already prepared with {num_images} images at {train_path}")
        return str(train_path)

    # Ensure tar is present (download or manual)
    if not tar_path.exists():
        print("Attempting wget download of Places365 archive (may require manual download)...")
        try:
            subprocess.run([
                "wget", "-c",
                "http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar",
                "-O", str(tar_path),
            ], check=True, timeout=7200)
        except Exception as e:
            print(f"wget failed: {e}")
            print("\nManual download instructions:")
            print("  1. Go to: http://places2.csail.mit.edu/download.html")
            print("  2. Download 'Small images (256 x 256)' d> 'Train images'")
            print(f"  3. Save as: {tar_path}")
            print("  4. Re-run this script")
            return None

    # List members in the tar and deterministically sample a subset of images
    print("Listing members in Places365 tar (this may take a while)...")
    list_proc = subprocess.run(
        ["tar", "-tf", str(tar_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    all_files = [line.strip() for line in list_proc.stdout.splitlines() if line.strip()]

    # Filter to image files inside train/ directories
    image_exts = (".jpg", ".jpeg", ".png", ".JPEG")
    image_files = [f for f in all_files if "/train/" in f and f.lower().endswith(image_exts)]
    print(f"Found {len(image_files)} train images in tar")

    if not image_files:
        print("No train images found inside tar; please verify the archive.")
        return None

    # Deterministic sampling
    if len(image_files) > num_samples:
        print(f"Sampling {num_samples} random images from tar entries...")
        random.seed(42)
        sampled_files = random.sample(image_files, num_samples)
    else:
        sampled_files = image_files

    # Write sampled file list to a temp file for tar -T
    sample_list_path = output_path / "places365_sample_list.txt"
    with sample_list_path.open("w") as f:
        for path in sampled_files:
            f.write(path + "\n")

    # Extract only the sampled files into a temporary directory
    temp_extract_root = output_path / "_tmp_extract"
    if temp_extract_root.exists():
        shutil.rmtree(temp_extract_root)
    temp_extract_root.mkdir(parents=True, exist_ok=True)

    print("Extracting sampled files from tar (storage-efficient extraction)...")
    subprocess.run(
        [
            "tar",
            "-xf",
            str(tar_path),
            "-C",
            str(temp_extract_root),
            "-T",
            str(sample_list_path),
        ],
        check=True,
    )

    # Now flatten into train/ directory
    train_path.mkdir(parents=True, exist_ok=True)
    print(f"Copying {len(sampled_files)} sampled images into flat train/ directory...")
    for i, rel_path in enumerate(sampled_files):
        if i % 10000 == 0:
            print(f"  Progress: {i}/{len(sampled_files)}")
        src_path = temp_extract_root / rel_path
        if not src_path.exists():
            # Should not normally happen, but guard anyway
            continue
        new_name = f"places_{i:06d}_{Path(rel_path).name}"
        dest_path = train_path / new_name
        if not dest_path.exists():
            shutil.copy2(src_path, dest_path)

    # Cleanup: temporary extraction and sample list, and tar to save space
    print("Cleaning up temporary files and archive to save space...")
    if temp_extract_root.exists():
        shutil.rmtree(temp_extract_root)
    if sample_list_path.exists():
        sample_list_path.unlink()
    if tar_path.exists():
        try:
            tar_path.unlink()
        except Exception as e:
            print(f"Warning: failed to delete tar archive: {e}")

    num_images = len(list(train_path.glob("*")))
    print(f"Places365 ready: {num_images} images at {train_path}")

    return str(train_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download auxiliary datasets (COCO, Places365)")
    parser.add_argument("--output_dir", type=str, default="data/auxiliary",
                        help="Root output directory for datasets")
    parser.add_argument("--coco", action="store_true",
                        help="Download COCO Unlabeled 2017")
    parser.add_argument("--places", action="store_true",
                        help="Download Places365 and sample subset")
    parser.add_argument("--places_samples", type=int, default=200000,
                        help="Number of Places365 images to sample")
    parser.add_argument("--all", action="store_true",
                        help="Download all supported datasets")

    args = parser.parse_args()

    if not any([args.coco, args.places, args.all]):
        print("No dataset selected. Use --coco, --places, or --all.")
        parser.print_help()
        return

    print("=" * 60)
    print("Auxiliary Dataset Downloader")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}\n")

    results: dict[str, str | None] = {}

    if args.coco or args.all:
        results["coco"] = download_coco_unlabeled(args.output_dir)

    if args.places or args.all:
        results["places"] = download_places365(args.output_dir, args.places_samples)

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print("\nDataset paths (use these in train.py args):")
    if results.get("coco"):
        print(f"  --use_coco --coco_data_path {results['coco']}")
    if results.get("places"):
        print(f"  --use_places --places_data_path {results['places']}")


if __name__ == "__main__":
    main()
