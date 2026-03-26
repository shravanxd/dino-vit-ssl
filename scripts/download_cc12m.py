#!/usr/bin/env python3
"""
Download CC12M (Conceptual 12M) images for SSL pretraining.

Uses pixparse/cc12m-wds from HuggingFace - images already downloaded in WebDataset format!
No URL fetching needed - just download the tar files and extract images.

Features:
- Deduplication via perceptual hashing (detects ~63K CC3M/CC12M overlap)
- Resizes to 96x96 to match CC3M pretrain data

Requirements:
    pip install huggingface_hub pillow tqdm imagehash

Usage:
    python scripts/download_cc12m.py --output_dir ./data/pretrain_cc12m/train --num_images 600000 --cc3m_dir ./data/pretrain/train

The script will:
1. Build hash index of existing CC3M images (for deduplication)
2. Download CC12M tar shards from HuggingFace
3. Extract, deduplicate, resize images to 96x96
4. Save unique images as JPEGs in the output directory
"""

import argparse
import sys
import tarfile
import io
import hashlib
from pathlib import Path
from typing import Set, Optional


def check_dependencies():
    """Check if required packages are installed."""
    missing = []
    
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        missing.append("huggingface_hub")
    
    try:
        from PIL import Image
    except ImportError:
        missing.append("pillow")
    
    try:
        from tqdm import tqdm
    except ImportError:
        missing.append("tqdm")
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)
    
    print("✓ All dependencies installed")


def compute_image_hash(img_bytes: bytes) -> str:
    """Compute a hash of the image content for deduplication."""
    # Use MD5 of raw bytes - fast and sufficient for dedup
    return hashlib.md5(img_bytes).hexdigest()


def resize_image(img_bytes: bytes, size: int = 96) -> Optional[bytes]:
    """Resize image to target size with center crop, return as bytes."""
    from PIL import Image
    
    try:
        img = Image.open(io.BytesIO(img_bytes))
        
        # Convert to RGB
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # Resize shortest side to target, then center crop
        w, h = img.size
        if w < h:
            new_w = size
            new_h = int(h * size / w)
        else:
            new_h = size
            new_w = int(w * size / h)
        
        img = img.resize((new_w, new_h), Image.LANCZOS)
        
        # Center crop
        left = (new_w - size) // 2
        top = (new_h - size) // 2
        img = img.crop((left, top, left + size, top + size))
        
        # Convert to bytes
        buffer = io.BytesIO()
        img.save(buffer, "JPEG", quality=95)
        return buffer.getvalue()
    except Exception:
        return None


def _hash_single_file(img_path: Path) -> Optional[str]:
    """Hash a single image file. Used by multiprocessing pool."""
    try:
        with open(img_path, 'rb') as f:
            img_bytes = f.read()
        return hashlib.md5(img_bytes).hexdigest()
    except Exception:
        return None


def build_cc3m_hash_index(cc3m_dir: Path, num_workers: int = 8) -> Set[str]:
    """Build a hash index of existing CC3M images for deduplication.
    
    Uses multiprocessing for speed. Hashes actual file bytes since CC3M 
    images are already at final size.
    """
    from tqdm import tqdm
    from multiprocessing import Pool, cpu_count
    
    if not cc3m_dir.exists():
        print(f"CC3M directory not found: {cc3m_dir}")
        return set()
    
    print(f"\nBuilding hash index of CC3M images in {cc3m_dir}...")
    
    # Discover all image files
    image_files = list(cc3m_dir.rglob("*.jpg")) + list(cc3m_dir.rglob("*.jpeg")) + list(cc3m_dir.rglob("*.png"))
    
    if not image_files:
        print("No images found")
        return set()
    
    # Use multiprocessing for speed
    num_workers = min(num_workers, cpu_count())
    print(f"Hashing with {num_workers} workers...")
    
    hash_set = set()
    with Pool(num_workers) as pool:
        # imap_unordered is faster than map for large datasets
        results = pool.imap_unordered(_hash_single_file, image_files, chunksize=1000)
        for h in tqdm(results, total=len(image_files), desc="Hashing CC3M"):
            if h is not None:
                hash_set.add(h)
    
    print(f"✓ Indexed {len(hash_set):,} CC3M images")
    return hash_set


def save_image(img_bytes: bytes, output_path: Path) -> bool:
    """Save image bytes to file."""
    try:
        with open(output_path, 'wb') as f:
            f.write(img_bytes)
        return True
    except Exception:
        return False


def _process_single_image(args):
    """Process a single image: resize and hash. Used by multiprocessing pool."""
    raw_bytes, size = args
    try:
        resized_bytes = resize_image(raw_bytes, size)
        if resized_bytes is None:
            return None, None
        img_hash = hashlib.md5(resized_bytes).hexdigest()
        return resized_bytes, img_hash
    except Exception:
        return None, None


def process_shard(
    shard_path: Path, 
    output_dir: Path, 
    start_idx: int, 
    max_images: int, 
    size: int = 96,
    existing_hashes: Optional[Set[str]] = None,
    seen_hashes: Optional[Set[str]] = None,
    num_workers: int = 8,
):
    """Process a single tar shard and extract images with deduplication.
    
    Uses multiprocessing for resize+hash, then sequential dedup+save.
    """
    from multiprocessing import Pool, cpu_count
    
    saved = 0
    errors = 0
    duplicates = 0
    current_idx = start_idx
    
    if existing_hashes is None:
        existing_hashes = set()
    if seen_hashes is None:
        seen_hashes = set()
    
    num_workers = min(num_workers, cpu_count())
    
    try:
        # First, extract all raw image bytes from tar (fast, I/O bound)
        raw_images = []
        with tarfile.open(shard_path, 'r') as tar:
            for member in tar.getmembers():
                if not member.name.endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    continue
                try:
                    f = tar.extractfile(member)
                    if f is not None:
                        raw_images.append(f.read())
                except Exception:
                    errors += 1
        
        # Process images in parallel (resize + hash)
        with Pool(num_workers) as pool:
            results = pool.map(_process_single_image, [(img, size) for img in raw_images])
        
        # Sequential dedup check and save (must be sequential for correct indexing)
        for resized_bytes, img_hash in results:
            if saved >= max_images:
                break
                
            if resized_bytes is None:
                errors += 1
                continue
            
            # Check for duplicates
            if img_hash in existing_hashes or img_hash in seen_hashes:
                duplicates += 1
                continue
            
            # Mark as seen and save
            seen_hashes.add(img_hash)
            output_path = output_dir / f"cc12m_{current_idx:08d}.jpg"
            if save_image(resized_bytes, output_path):
                saved += 1
                current_idx += 1
            else:
                errors += 1
                
    except Exception as e:
        print(f"Error processing {shard_path}: {e}")
    
    return saved, errors, duplicates, current_idx


def download_cc12m(
    output_dir: Path,
    num_images: int = 600000,
    image_size: int = 96,
    num_shards: int = 120,  # Download first N shards (each has ~5K images)
    cc3m_dir: Optional[Path] = None,
    num_workers: int = 8,
):
    """Download CC12M from HuggingFace and extract images with deduplication."""
    from huggingface_hub import hf_hub_download
    from tqdm import tqdm
    
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir.parent / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Build hash index of CC3M for deduplication
    existing_hashes = set()
    if cc3m_dir and cc3m_dir.exists():
        existing_hashes = build_cc3m_hash_index(cc3m_dir, num_workers=num_workers)
    
    # Track hashes we've seen in this download session
    seen_hashes: Set[str] = set()
    
    print(f"\nDownloading CC12M shards from HuggingFace...")
    print(f"Target: {num_images:,} unique images at {image_size}x{image_size}")
    print(f"Dedup against: {len(existing_hashes):,} existing images")
    print(f"Cache: {cache_dir}\n")
    
    total_saved = 0
    total_errors = 0
    total_duplicates = 0
    current_idx = 0
    
    # Each shard has ~5000 images, so calculate how many shards we need
    # Request more shards to account for duplicates
    images_per_shard = 5000
    shards_needed = min(num_shards, (num_images // images_per_shard) + 10)
    
    pbar = tqdm(total=num_images, desc="Unique images saved")
    
    for shard_idx in range(shards_needed):
        if total_saved >= num_images:
            break
            
        shard_name = f"cc12m-train-{shard_idx:04d}.tar"
        
        try:
            # Download shard
            shard_path = hf_hub_download(
                repo_id="pixparse/cc12m-wds",
                filename=shard_name,
                repo_type="dataset",
                cache_dir=str(cache_dir),
            )
            
            # Process shard with deduplication
            remaining = num_images - total_saved
            saved, errors, duplicates, current_idx = process_shard(
                Path(shard_path),
                output_dir,
                current_idx,
                remaining,  # Only save exactly what we need
                image_size,
                existing_hashes,
                seen_hashes,
                num_workers,
            )
            
            total_saved += saved
            total_errors += errors
            total_duplicates += duplicates
            pbar.update(saved)
            
        except Exception as e:
            print(f"\nError downloading shard {shard_idx}: {e}")
            continue
    
    pbar.close()
    
    print(f"\n✓ Saved {total_saved:,} unique images to {output_dir}")
    print(f"  Duplicates skipped: {total_duplicates:,}")
    print(f"  Errors: {total_errors:,}")
    
    return total_saved


def main():
    parser = argparse.ArgumentParser(
        description="Download CC12M images for SSL pretraining with deduplication"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/pretrain_cc12m/train",
        help="Output directory for images",
    )
    parser.add_argument(
        "--cc3m_dir",
        type=str,
        default="./data/pretrain/train",
        help="CC3M directory for deduplication (default: ./data/pretrain/train)",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=300000,
        help="Number of unique images to download (default: 300K)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=96,
        help="Target image size (default: 96 to match CC3M)",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=150,
        help="Max number of shards to download (each ~5K images)",
    )
    parser.add_argument(
        "--skip_dedup",
        action="store_true",
        help="Skip deduplication against CC3M",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for parallel hashing (default: 8)",
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    cc3m_dir = None if args.skip_dedup else Path(args.cc3m_dir)
    
    print("=" * 60)
    print("CC12M Downloader (via pixparse/cc12m-wds)")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print(f"Target: {args.num_images:,} unique images at {args.image_size}x{args.image_size}")
    if cc3m_dir:
        print(f"Dedup against: {cc3m_dir}")
    print("=" * 60)
    
    # Check dependencies
    check_dependencies()
    
    # Download and extract
    count = download_cc12m(
        output_dir,
        num_images=args.num_images,
        image_size=args.image_size,
        num_shards=args.num_shards,
        cc3m_dir=cc3m_dir,
        num_workers=args.num_workers,
    )
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Downloaded {count:,} unique images to {output_dir}")
    print("\nTo use with training, update config or use CLI:")
    print(f"  python train.py --pretrain_source both --laion_data_path {output_dir.parent}")


if __name__ == "__main__":
    main()
