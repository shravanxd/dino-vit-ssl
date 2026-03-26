"""
CLI script for verifying image data.

Checks that all images:
- Load correctly (not corrupted)
- Are the expected size (96x96)
- Are RGB format

Usage:
    # Verify all datasets
    python -m ingestion_and_preprocess.preprocess.verify_data
    
    # Verify specific dataset
    python -m ingestion_and_preprocess.preprocess.verify_data --dataset pretrain
    
    # Save report to file
    python -m ingestion_and_preprocess.preprocess.verify_data --report ./verification_report.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict

from PIL import Image
from tqdm import tqdm

from .base import get_project_root, is_valid_image, get_image_info


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ImageVerificationResult:
    """Result of verifying a single image."""
    path: str
    valid: bool
    size: Optional[Tuple[int, int]] = None
    mode: Optional[str] = None
    error: Optional[str] = None


@dataclass
class DatasetVerificationReport:
    """Verification report for a dataset."""
    dataset_name: str
    directory: str
    total_images: int
    valid_images: int
    invalid_images: int
    wrong_size: int
    wrong_mode: int
    corrupted: int
    expected_size: Tuple[int, int]
    errors: List[Dict]


# =============================================================================
# Verification Functions
# =============================================================================

def verify_single_image(
    path: Path,
    expected_size: Tuple[int, int] = (96, 96),
) -> ImageVerificationResult:
    """
    Verify a single image file.
    
    Args:
        path: Path to image
        expected_size: Expected (width, height)
        
    Returns:
        ImageVerificationResult
    """
    try:
        with Image.open(path) as img:
            # Force load to check for corruption
            img.load()
            
            return ImageVerificationResult(
                path=str(path),
                valid=True,
                size=img.size,
                mode=img.mode,
            )
            
    except Exception as e:
        return ImageVerificationResult(
            path=str(path),
            valid=False,
            error=str(e),
        )


def verify_directory(
    directory: Path,
    expected_size: Tuple[int, int] = (96, 96),
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    num_workers: int = 4,
    show_progress: bool = True,
) -> DatasetVerificationReport:
    """
    Verify all images in a directory.
    
    Args:
        directory: Directory to verify
        expected_size: Expected image size
        extensions: Valid image extensions
        num_workers: Number of parallel workers
        show_progress: Show progress bar
        
    Returns:
        DatasetVerificationReport
    """
    directory = Path(directory)
    
    # Collect all image files
    image_files = []
    for ext in extensions:
        image_files.extend(directory.rglob(f"*{ext}"))
        image_files.extend(directory.rglob(f"*{ext.upper()}"))
    
    image_files = sorted(set(image_files))
    
    # Initialize counters
    valid_count = 0
    invalid_count = 0
    wrong_size_count = 0
    wrong_mode_count = 0
    corrupted_count = 0
    errors = []
    
    # Verify in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(verify_single_image, f, expected_size): f
            for f in image_files
        }
        
        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(
                iterator,
                total=len(futures),
                desc=f"Verifying {directory.name}",
            )
        
        for future in iterator:
            result = future.result()
            
            if not result.valid:
                invalid_count += 1
                corrupted_count += 1
                errors.append({
                    "path": result.path,
                    "type": "corrupted",
                    "error": result.error,
                })
            else:
                # Check size
                if result.size != expected_size:
                    wrong_size_count += 1
                    errors.append({
                        "path": result.path,
                        "type": "wrong_size",
                        "actual": result.size,
                        "expected": expected_size,
                    })
                
                # Check mode
                if result.mode != "RGB":
                    wrong_mode_count += 1
                    errors.append({
                        "path": result.path,
                        "type": "wrong_mode",
                        "actual": result.mode,
                        "expected": "RGB",
                    })
                
                # Count as valid only if size and mode are correct
                if result.size == expected_size and result.mode == "RGB":
                    valid_count += 1
                else:
                    invalid_count += 1
    
    return DatasetVerificationReport(
        dataset_name=directory.name,
        directory=str(directory),
        total_images=len(image_files),
        valid_images=valid_count,
        invalid_images=invalid_count,
        wrong_size=wrong_size_count,
        wrong_mode=wrong_mode_count,
        corrupted=corrupted_count,
        expected_size=expected_size,
        errors=errors[:100],  # Limit errors to first 100
    )


def create_valid_images_list(
    directory: Path,
    output_path: Path,
    expected_size: Tuple[int, int] = (96, 96),
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png"),
    num_workers: int = 4,
    show_progress: bool = True,
) -> int:
    """
    Create a text file listing all valid image paths.
    
    This is useful for fast loading during training.
    
    Args:
        directory: Directory to scan
        output_path: Path to save the list
        expected_size: Expected image size
        extensions: Valid image extensions
        num_workers: Number of parallel workers
        show_progress: Show progress bar
        
    Returns:
        Number of valid images
    """
    directory = Path(directory)
    output_path = Path(output_path)
    
    # Collect all image files
    image_files = []
    for ext in extensions:
        image_files.extend(directory.rglob(f"*{ext}"))
        image_files.extend(directory.rglob(f"*{ext.upper()}"))
    
    image_files = sorted(set(image_files))
    
    valid_paths = []
    
    # Verify in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(verify_single_image, f, expected_size): f
            for f in image_files
        }
        
        iterator = as_completed(futures)
        if show_progress:
            iterator = tqdm(
                iterator,
                total=len(futures),
                desc="Creating valid images list",
            )
        
        for future in iterator:
            result = future.result()
            
            if result.valid and result.size == expected_size and result.mode == "RGB":
                valid_paths.append(result.path)
    
    # Sort and save
    valid_paths.sort()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for path in valid_paths:
            f.write(f"{path}\n")
    
    return len(valid_paths)


# =============================================================================
# Main CLI
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify image data for SSL training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify all datasets
  python -m ingestion_and_preprocess.preprocess.verify_data
  
  # Verify pretrain only
  python -m ingestion_and_preprocess.preprocess.verify_data --dataset pretrain
  
  # Create valid images list for pretrain
  python -m ingestion_and_preprocess.preprocess.verify_data --dataset pretrain --create_list
  
  # Save detailed report
  python -m ingestion_and_preprocess.preprocess.verify_data --report report.json
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
        choices=["pretrain", "cub200", "miniimagenet", "sun397", "all"],
        default="all",
        help="Dataset to verify (default: all)"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=96,
        help="Expected image size (default: 96)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Path to save JSON report"
    )
    parser.add_argument(
        "--create_list",
        action="store_true",
        help="Create valid_images.txt for each dataset"
    )
    
    args = parser.parse_args()
    
    # Get project root and data directory
    project_root = get_project_root()
    data_dir = args.data_dir or project_root / "data"
    
    expected_size = (args.size, args.size)
    
    print("=" * 60)
    print("DATA VERIFICATION")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Data directory: {data_dir}")
    print(f"Expected size: {expected_size}")
    
    # Define datasets
    datasets = {
        "pretrain": data_dir / "pretrain" / "train",
        "cub200": data_dir / "eval_public" / "cub200",
        "miniimagenet": data_dir / "eval_public" / "miniimagenet",
        "sun397": data_dir / "eval_public" / "sun397",
    }
    
    if args.dataset != "all":
        datasets = {args.dataset: datasets[args.dataset]}
    
    # Verify each dataset
    all_reports = []
    
    for name, directory in datasets.items():
        print(f"\n{'=' * 60}")
        print(f"VERIFYING: {name.upper()}")
        print("=" * 60)
        
        if not directory.exists():
            print(f"⚠ Directory not found: {directory}")
            print("  Skipping...")
            continue
        
        print(f"Directory: {directory}")
        
        # For eval datasets, verify each subfolder
        if name != "pretrain":
            subfolders = ["train", "val", "test"]
            for subfolder in subfolders:
                subfolder_dir = directory / subfolder
                if subfolder_dir.exists():
                    report = verify_directory(
                        directory=subfolder_dir,
                        expected_size=expected_size,
                        num_workers=args.workers,
                    )
                    all_reports.append(report)
                    
                    print(f"\n  {subfolder}/:")
                    print(f"    Total: {report.total_images}")
                    print(f"    Valid: {report.valid_images}")
                    if report.wrong_size > 0:
                        print(f"    Wrong size: {report.wrong_size}")
                    if report.wrong_mode > 0:
                        print(f"    Wrong mode: {report.wrong_mode}")
                    if report.corrupted > 0:
                        print(f"    Corrupted: {report.corrupted}")
        else:
            report = verify_directory(
                directory=directory,
                expected_size=expected_size,
                num_workers=args.workers,
            )
            all_reports.append(report)
            
            print(f"\n  Results:")
            print(f"    Total: {report.total_images}")
            print(f"    Valid: {report.valid_images}")
            if report.wrong_size > 0:
                print(f"    Wrong size: {report.wrong_size}")
            if report.wrong_mode > 0:
                print(f"    Wrong mode: {report.wrong_mode}")
            if report.corrupted > 0:
                print(f"    Corrupted: {report.corrupted}")
        
        # Create valid images list if requested
        if args.create_list:
            if name == "pretrain":
                list_path = directory.parent / "valid_images.txt"
            else:
                list_path = directory / "valid_images.txt"
            
            print(f"\n  Creating valid images list: {list_path}")
            count = create_valid_images_list(
                directory=directory,
                output_path=list_path,
                expected_size=expected_size,
                num_workers=args.workers,
            )
            print(f"    ✓ {count} valid images listed")
    
    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    
    total_images = sum(r.total_images for r in all_reports)
    total_valid = sum(r.valid_images for r in all_reports)
    total_invalid = sum(r.invalid_images for r in all_reports)
    
    print(f"Total images checked: {total_images}")
    print(f"Valid (96x96 RGB): {total_valid}")
    print(f"Invalid: {total_invalid}")
    
    if total_invalid > 0:
        print(f"\n⚠ {total_invalid} images need attention!")
    else:
        print(f"\n✓ All images are valid!")
    
    # Save report if requested
    if args.report:
        report_data = {
            "summary": {
                "total_images": total_images,
                "valid_images": total_valid,
                "invalid_images": total_invalid,
                "expected_size": expected_size,
            },
            "datasets": [asdict(r) for r in all_reports],
        }
        
        with open(args.report, "w") as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\nReport saved to: {args.report}")


if __name__ == "__main__":
    main()
