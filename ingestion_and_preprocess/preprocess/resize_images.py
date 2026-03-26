"""
CLI script for offline image resizing.

Resizes eval dataset images to 96x96 for consistent preprocessing.

Usage:
    # Resize all eval datasets
    python -m ingestion_and_preprocess.preprocess.resize_images
    
    # Resize specific dataset
    python -m ingestion_and_preprocess.preprocess.resize_images --dataset cub200
    
    # Custom output directory
    python -m ingestion_and_preprocess.preprocess.resize_images --output_dir ./data/processed
    
    # Resize in place (overwrite originals)
    python -m ingestion_and_preprocess.preprocess.resize_images --inplace
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

from .base import get_project_root
from .resizers import ImageResizer, ResizeMode


# =============================================================================
# Dataset Configuration
# =============================================================================

def get_eval_datasets(data_dir: Path) -> Dict[str, Dict]:
    """
    Get configuration for eval datasets.
    
    Returns dict mapping dataset name to config with:
    - input_dir: Where images currently are
    - subfolders: List of subfolders to process (train, val, test)
    """
    eval_dir = data_dir / "eval_public"
    
    return {
        "cub200": {
            "input_dir": eval_dir / "cub200",
            "subfolders": ["train", "val", "test"],
            "description": "CUB-200 bird classification",
        },
        "miniimagenet": {
            "input_dir": eval_dir / "miniimagenet",
            "subfolders": ["train", "val", "test"],
            "description": "Mini-ImageNet 100-class subset",
        },
        "sun397": {
            "input_dir": eval_dir / "sun397",
            "subfolders": ["train", "test"],
            "description": "SUN397 scene classification",
        },
    }


# =============================================================================
# Main Functions
# =============================================================================

def resize_dataset(
    dataset_name: str,
    input_dir: Path,
    output_dir: Optional[Path],
    subfolders: List[str],
    target_size: tuple = (96, 96),
    mode: ResizeMode = ResizeMode.DIRECT,
    quality: int = 95,
    num_workers: int = 4,
    inplace: bool = False,
) -> Dict:
    """
    Resize all images in a dataset.
    
    Args:
        dataset_name: Name for logging
        input_dir: Input directory containing subfolders
        output_dir: Output directory (None for inplace)
        subfolders: List of subfolders to process
        target_size: Target image size
        mode: Resize mode
        quality: JPEG quality
        num_workers: Number of parallel workers
        inplace: Whether to overwrite originals
        
    Returns:
        Dict with statistics
    """
    resizer = ImageResizer(target_size=target_size, mode=mode)
    
    total_stats = {
        "dataset": dataset_name,
        "total": 0,
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "errors": [],
    }
    
    for subfolder in subfolders:
        subfolder_input = input_dir / subfolder
        
        if not subfolder_input.exists():
            print(f"  ⚠ Subfolder not found: {subfolder_input}")
            continue
        
        print(f"\n  Processing {subfolder}/...")
        
        if inplace:
            stats = resizer.resize_inplace(
                directory=subfolder_input,
                quality=quality,
                num_workers=num_workers,
            )
        else:
            subfolder_output = output_dir / subfolder
            stats = resizer.resize_directory(
                input_dir=subfolder_input,
                output_dir=subfolder_output,
                quality=quality,
                num_workers=num_workers,
            )
        
        # Aggregate stats
        total_stats["total"] += stats["total"]
        total_stats["success"] += stats["success"]
        total_stats["failed"] += stats["failed"]
        total_stats["skipped"] += stats["skipped"]
        total_stats["errors"].extend(stats["errors"])
        
        print(f"    ✓ {stats['success']}/{stats['total']} images resized")
        if stats["failed"] > 0:
            print(f"    ✗ {stats['failed']} failed")
    
    return total_stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Resize eval dataset images to 96x96",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Resize all eval datasets (creates new folders with _96 suffix)
  python -m ingestion_and_preprocess.preprocess.resize_images
  
  # Resize specific dataset
  python -m ingestion_and_preprocess.preprocess.resize_images --dataset cub200
  
  # Resize in place (overwrite originals) - USE WITH CAUTION
  python -m ingestion_and_preprocess.preprocess.resize_images --inplace
  
  # Custom target size
  python -m ingestion_and_preprocess.preprocess.resize_images --size 64
        """
    )
    
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help="Data directory (default: project_root/data)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["cub200", "miniimagenet", "sun397", "all"],
        default="all",
        help="Dataset to resize (default: all)"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=96,
        help="Target image size (default: 96)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["direct", "shortest_side", "longest_side"],
        default="direct",
        help="Resize mode (default: direct)"
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality 1-100 (default: 95)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Resize in place (overwrite originals)"
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="_96",
        help="Suffix for output directory (default: _96)"
    )
    
    args = parser.parse_args()
    
    # Get project root and data directory
    project_root = get_project_root()
    data_dir = args.data_dir or project_root / "data"
    
    print("=" * 60)
    print("OFFLINE IMAGE RESIZING")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Data directory: {data_dir}")
    print(f"Target size: {args.size}x{args.size}")
    print(f"Resize mode: {args.mode}")
    print(f"JPEG quality: {args.quality}")
    print(f"Inplace: {args.inplace}")
    
    # Get resize mode
    mode_map = {
        "direct": ResizeMode.DIRECT,
        "shortest_side": ResizeMode.SHORTEST_SIDE,
        "longest_side": ResizeMode.LONGEST_SIDE,
    }
    resize_mode = mode_map[args.mode]
    
    # Get datasets to process
    all_datasets = get_eval_datasets(data_dir)
    
    if args.dataset == "all":
        datasets_to_process = all_datasets
    else:
        datasets_to_process = {args.dataset: all_datasets[args.dataset]}
    
    # Process each dataset
    all_stats = []
    
    for name, config in datasets_to_process.items():
        print(f"\n{'=' * 60}")
        print(f"RESIZING: {name.upper()}")
        print(f"{'=' * 60}")
        print(f"Description: {config['description']}")
        print(f"Input: {config['input_dir']}")
        
        if not config['input_dir'].exists():
            print(f"⚠ Dataset directory not found: {config['input_dir']}")
            print("  Skipping...")
            continue
        
        # Determine output directory
        if args.inplace:
            output_dir = config['input_dir']
            print(f"Output: IN PLACE (overwriting)")
        else:
            output_dir = config['input_dir'].parent / f"{name}{args.output_suffix}"
            print(f"Output: {output_dir}")
        
        stats = resize_dataset(
            dataset_name=name,
            input_dir=config['input_dir'],
            output_dir=output_dir,
            subfolders=config['subfolders'],
            target_size=(args.size, args.size),
            mode=resize_mode,
            quality=args.quality,
            num_workers=args.workers,
            inplace=args.inplace,
        )
        
        all_stats.append(stats)
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    
    total_all = sum(s["total"] for s in all_stats)
    success_all = sum(s["success"] for s in all_stats)
    failed_all = sum(s["failed"] for s in all_stats)
    
    for stats in all_stats:
        print(f"\n{stats['dataset']}:")
        print(f"  Total: {stats['total']}")
        print(f"  Success: {stats['success']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Skipped: {stats['skipped']}")
    
    print(f"\n{'=' * 60}")
    print(f"TOTAL: {success_all}/{total_all} images resized successfully")
    if failed_all > 0:
        print(f"FAILED: {failed_all} images")
    print("=" * 60)


if __name__ == "__main__":
    main()
