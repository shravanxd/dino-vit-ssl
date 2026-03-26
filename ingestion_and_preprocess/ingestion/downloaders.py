"""
Concrete downloader implementations.

Contains:
- PretrainDownloader: Downloads CC3M pretrain data from HuggingFace (~500K images)
- LAIONDownloader: Downloads LAION-Aesthetics data (~600K images)
- TestsetDownloader: Downloads eval testsets (CUB-200, Mini-ImageNet, SUN397)

Usage:
    # Download CC3M (original pretrain data)
    python -m ingestion_and_preprocess.ingestion.download_data --pretrain_only --pretrain_source cc3m
    
    # Download LAION (additional pretrain data)  
    python -m ingestion_and_preprocess.ingestion.download_data --pretrain_only --pretrain_source laion
    
    # Download both
    python -m ingestion_and_preprocess.ingestion.download_data --pretrain_only --pretrain_source both

Then at training time, select which source(s) to use:
    python train.py --pretrain_source cc3m    # Use only CC3M
    python train.py --pretrain_source laion   # Use only LAION
    python train.py --pretrain_source both    # Use both combined
"""

import os
import sys
import zipfile
import subprocess
import shutil
from pathlib import Path
from typing import Optional, List, Literal
from tqdm import tqdm
from enum import Enum

try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

from .base import (
    BaseDownloader, 
    DownloaderRegistry, 
    DownloadResult, 
    DownloadStatus,
    DatasetInfo
)


class PretrainSource(Enum):
    """Available pretraining data sources."""
    CC3M = "cc3m"           # Original 500K CC3M subset
    LAION = "laion"         # LAION-Aesthetics (~600K images)
    BOTH = "both"           # Combined CC3M + LAION


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Searches upward from this file for a directory containing
    known project markers (e.g., 'requirements.txt', '.git', 'src/').
    
    Returns:
        Path to project root (always resolved to absolute for reliability)
    """
    current = Path(__file__).resolve().parent
    
    # Search upward for project markers
    markers = ['requirements.txt', '.git', 'src', 'testsets']
    
    for _ in range(10):  # Max 10 levels up
        # Check if any marker exists at this level
        if any((current / marker).exists() for marker in markers):
            return current
        
        parent = current.parent
        if parent == current:  # Reached filesystem root
            break
        current = parent
    
    # Fallback: assume 3 levels up from this file
    return Path(__file__).resolve().parent.parent.parent


@DownloaderRegistry.register("pretrain")
class PretrainDownloader(BaseDownloader):
    """
    Downloads pretrain dataset from HuggingFace.
    
    Source: tsbpp/fall2025_deeplearning
    Content: ~500k unlabeled images at 96x96 resolution (CC3M subset)
    """
    
    REPO_ID = "tsbpp/fall2025_deeplearning"
    ZIP_FILES = [
        "cc3m_96px_part1.zip",
        "cc3m_96px_part2.zip",
        "cc3m_96px_part3.zip",
        "cc3m_96px_part4.zip",
        "cc3m_96px_part5.zip",
    ]
    
    def __init__(self, output_dir: Path):
        super().__init__(output_dir)
        self.pretrain_dir = self.output_dir / "pretrain"
    
    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="Pretrain (CC3M Subset)",
            description="Unlabeled images for self-supervised learning",
            size="~2.6 GB",
            num_images=500000,
        )
    
    def download(self, keep_zip: bool = False) -> DownloadResult:
        """Download and extract pretrain data from HuggingFace."""
        
        if not HF_AVAILABLE:
            return DownloadResult(
                status=DownloadStatus.FAILED,
                message="huggingface_hub not installed. Run: pip install huggingface_hub"
            )
        
        self._print_header(f"DOWNLOADING {self.info.name.upper()}")
        self._print_info()
        print(f"Output: {self.pretrain_dir}")
        print(f"Files: {len(self.ZIP_FILES)} zip archives\n")
        
        self.pretrain_dir.mkdir(parents=True, exist_ok=True)
        
        successful = 0
        for i, zip_file in enumerate(self.ZIP_FILES, 1):
            print(f"\n[{i}/{len(self.ZIP_FILES)}] {zip_file}")
            
            try:
                # Download
                print("    Downloading...")
                local_path = hf_hub_download(
                    repo_id=self.REPO_ID,
                    filename=zip_file,
                    repo_type="dataset",
                    local_dir=str(self.pretrain_dir),
                    local_dir_use_symlinks=False,
                )
                
                # Extract
                print("    Extracting...")
                zip_path = Path(local_path)
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    for file in tqdm(zf.namelist(), desc="    Progress", unit="files", leave=False):
                        zf.extract(file, self.pretrain_dir)
                
                # Cleanup
                if not keep_zip:
                    zip_path.unlink()
                    print(f"    Cleaned up")
                
                successful += 1
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                continue
        
        # Verify
        image_count = self._count_images(self.pretrain_dir)
        
        if successful == len(self.ZIP_FILES):
            return DownloadResult(
                status=DownloadStatus.COMPLETED,
                message=f"Downloaded {image_count:,} images",
                output_path=self.pretrain_dir,
                file_count=image_count
            )
        else:
            return DownloadResult(
                status=DownloadStatus.FAILED,
                message=f"Only {successful}/{len(self.ZIP_FILES)} files downloaded",
                output_path=self.pretrain_dir,
                file_count=image_count
            )
    
    def verify(self) -> DownloadResult:
        """Verify pretrain data exists and count images."""
        
        if not self.pretrain_dir.exists():
            return DownloadResult(
                status=DownloadStatus.NOT_STARTED,
                message=f"Directory not found: {self.pretrain_dir}"
            )
        
        image_count = self._count_images(self.pretrain_dir)
        
        if image_count > 0:
            return DownloadResult(
                status=DownloadStatus.COMPLETED,
                message=f"Found {image_count:,} images",
                output_path=self.pretrain_dir,
                file_count=image_count
            )
        else:
            return DownloadResult(
                status=DownloadStatus.FAILED,
                message="No images found in pretrain directory"
            )


@DownloaderRegistry.register("laion")
class LAIONDownloader(BaseDownloader):
    """
    Downloads LAION-Aesthetics dataset using img2dataset.
    
    NOTE: This requires:
    1. Python 3.11 or 3.12 (img2dataset doesn't work with Python 3.13)
    2. pip install img2dataset huggingface_hub
    3. Accepting LAION terms on HuggingFace (for some datasets)
    
    For easiest setup, run this on Vast.ai with the download_laion.py script.
    """
    
    # HuggingFace dataset with URL metadata (requires img2dataset to fetch)
    LAION_REPO = "laion/laion2B-en-aesthetic"
    
    def __init__(self, output_dir: Path, num_images: int = 600000, image_size: int = 96):
        super().__init__(output_dir)
        self.laion_dir = self.output_dir / "pretrain_laion"
        self.num_images = num_images
        self.image_size = image_size
    
    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name="LAION-Aesthetics",
            description="High-quality web images filtered by aesthetic score",
            size=f"~{self.num_images // 1000}K images at {self.image_size}x{self.image_size}",
            num_images=self.num_images,
        )
    
    def download(self) -> DownloadResult:
        """
        Placeholder - LAION download is complex and best done via script on Vast.ai.
        See scripts/download_laion.py for a working implementation.
        """
        return DownloadResult(
            status=DownloadStatus.FAILED,
            message="LAION download requires Python 3.11/3.12 and img2dataset. "
                    "Use scripts/download_laion.py on Vast.ai instead."
        )
    
    def verify(self) -> DownloadResult:
        """Verify LAION data exists and count images."""
        if not self.laion_dir.exists():
            return DownloadResult(
                status=DownloadStatus.NOT_STARTED,
                message=f"Directory not found: {self.laion_dir}"
            )
        
        image_count = self._count_images(self.laion_dir)
        
        if image_count > 0:
            return DownloadResult(
                status=DownloadStatus.COMPLETED,
                message=f"Found {image_count:,} LAION images",
                output_path=self.laion_dir,
                file_count=image_count
            )
        else:
            return DownloadResult(
                status=DownloadStatus.FAILED,
                message="No images found in LAION directory"
            )


@DownloaderRegistry.register("pretrain_combined")
class CombinedPretrainDownloader(BaseDownloader):
    """
    Combined pretraining data manager.
    
    Supports downloading and organizing:
    - CC3M only (original 500K)
    - LAION only (~168K)
    - Both combined (~668K)
    
    Creates a unified pretrain/train directory with selected sources.
    """
    
    def __init__(
        self, 
        output_dir: Path, 
        source: PretrainSource = PretrainSource.BOTH,
        laion_num_images: int = 168000,
        image_size: int = 96,
    ):
        """
        Initialize combined pretrain downloader.
        
        Args:
            output_dir: Base output directory
            source: Which data source(s) to use (cc3m, laion, or both)
            laion_num_images: Number of LAION images to download
            image_size: Target image size for LAION (CC3M is pre-resized to 96)
        """
        super().__init__(output_dir)
        self.source = source if isinstance(source, PretrainSource) else PretrainSource(source)
        self.laion_num_images = laion_num_images
        self.image_size = image_size
        
        # Output directory for combined data
        self.combined_dir = self.output_dir / "pretrain" / "train"
        
        # Individual downloaders
        self.cc3m_downloader = PretrainDownloader(self.output_dir)
        self.laion_downloader = LAIONDownloader(
            self.output_dir, 
            num_images=laion_num_images,
            image_size=image_size
        )
    
    @property
    def info(self) -> DatasetInfo:
        if self.source == PretrainSource.CC3M:
            return DatasetInfo(
                name="Pretrain (CC3M)",
                description="CC3M subset for self-supervised learning",
                size="~2.6 GB (~500K images)",
                num_images=500000,
            )
        elif self.source == PretrainSource.LAION:
            return DatasetInfo(
                name="Pretrain (LAION-Aesthetics)",
                description="LAION-Aesthetics for self-supervised learning",
                size=f"~{self.laion_num_images // 1000}K images",
                num_images=self.laion_num_images,
            )
        else:  # BOTH
            total = 500000 + self.laion_num_images
            return DatasetInfo(
                name="Pretrain (CC3M + LAION)",
                description="Combined CC3M and LAION-Aesthetics",
                size=f"~{total // 1000}K images",
                num_images=total,
            )
    
    def download(self, keep_zip: bool = False, keep_metadata: bool = True) -> DownloadResult:
        """
        Download pretraining data based on selected source.
        
        Args:
            keep_zip: Keep CC3M zip files after extraction
            keep_metadata: Keep LAION parquet metadata
        """
        self._print_header(f"DOWNLOADING {self.info.name.upper()}")
        self._print_info()
        print(f"Source: {self.source.value}")
        print(f"Output: {self.combined_dir}\n")
        
        results = []
        
        # Download CC3M if needed
        if self.source in [PretrainSource.CC3M, PretrainSource.BOTH]:
            print("\n" + "-"*40)
            print("Step 1: Downloading CC3M...")
            print("-"*40)
            cc3m_result = self.cc3m_downloader.download(keep_zip=keep_zip)
            results.append(("CC3M", cc3m_result))
            
            if cc3m_result.status == DownloadStatus.FAILED:
                return DownloadResult(
                    status=DownloadStatus.FAILED,
                    message=f"CC3M download failed: {cc3m_result.message}"
                )
        
        # Download LAION if needed
        if self.source in [PretrainSource.LAION, PretrainSource.BOTH]:
            print("\n" + "-"*40)
            print("Step 2: Downloading LAION-Aesthetics...")
            print("-"*40)
            laion_result = self.laion_downloader.download(keep_metadata=keep_metadata)
            results.append(("LAION", laion_result))
            
            if laion_result.status == DownloadStatus.FAILED:
                return DownloadResult(
                    status=DownloadStatus.FAILED,
                    message=f"LAION download failed: {laion_result.message}"
                )
        
        # Create combined directory structure if using both
        if self.source == PretrainSource.BOTH:
            print("\n" + "-"*40)
            print("Step 3: Creating combined dataset...")
            print("-"*40)
            self._create_combined_dataset()
        
        return self.verify()
    
    def _create_combined_dataset(self):
        """
        Create symlinks or copy images to combined directory.
        
        Uses symlinks when possible to save disk space.
        """
        self.combined_dir.mkdir(parents=True, exist_ok=True)
        
        # For CC3M, the images are already in pretrain/train
        cc3m_train = self.output_dir / "pretrain" / "train"
        
        # For LAION, images are in pretrain_laion/train
        laion_train = self.output_dir / "pretrain_laion" / "train"
        
        if laion_train.exists():
            print(f"Copying LAION images to combined directory...")
            # Copy LAION images to the CC3M directory
            laion_images = list(laion_train.rglob("*.jpg")) + list(laion_train.rglob("*.jpeg"))
            
            for i, img in enumerate(tqdm(laion_images, desc="Copying", unit="files")):
                # Create unique filename with laion_ prefix
                new_name = f"laion_{img.stem}{img.suffix}"
                dest = self.combined_dir / new_name
                
                if not dest.exists():
                    try:
                        # Try symlink first (saves space)
                        dest.symlink_to(img.resolve())
                    except OSError:
                        # Fall back to copy if symlinks not supported
                        shutil.copy2(img, dest)
            
            print(f"Combined {len(laion_images):,} LAION images with CC3M data")
    
    def verify(self) -> DownloadResult:
        """Verify combined pretrain data."""
        
        total_images = 0
        sources = []
        
        # Check CC3M
        if self.source in [PretrainSource.CC3M, PretrainSource.BOTH]:
            cc3m_result = self.cc3m_downloader.verify()
            if cc3m_result.status == DownloadStatus.COMPLETED:
                total_images += cc3m_result.file_count
                sources.append(f"CC3M: {cc3m_result.file_count:,}")
        
        # Check LAION
        if self.source in [PretrainSource.LAION, PretrainSource.BOTH]:
            laion_result = self.laion_downloader.verify()
            if laion_result.status == DownloadStatus.COMPLETED:
                total_images += laion_result.file_count
                sources.append(f"LAION: {laion_result.file_count:,}")
        
        if total_images > 0:
            source_str = ", ".join(sources)
            return DownloadResult(
                status=DownloadStatus.COMPLETED,
                message=f"Found {total_images:,} images ({source_str})",
                output_path=self.combined_dir,
                file_count=total_images
            )
        else:
            return DownloadResult(
                status=DownloadStatus.FAILED,
                message="No pretrain images found"
            )


class TestsetDownloader(BaseDownloader):
    """
    Downloads an eval testset by running its prepare script.
    
    This is a factory-style class that wraps the existing prepare scripts
    in testsets/.
    """
    
    # Configuration for each testset
    TESTSETS = {
        "cub200": {
            "script_path": "testsets/testset_1/prepare_cub200_for_kaggle.py",
            "name": "CUB-200",
            "description": "Bird classification (200 species)",
            "size": "~1.1 GB",
            "classes": 200,
        },
        "miniimagenet": {
            "script_path": "testsets/testset_2/prepare_miniimagenet_for_kaggle.py",
            "name": "Mini-ImageNet",
            "description": "100-class subset of ImageNet",
            "size": "~3 GB",
            "classes": 100,
        },
        "sun397": {
            "script_path": "testsets/testset_3/prepare_sun397_for_kaggle.py",
            "name": "SUN397",
            "description": "Scene recognition (397 categories)",
            "size": "~37 GB",
            "classes": 397,
        },
    }
    
    def __init__(self, output_dir: Path, testset_name: str, project_root: Optional[Path] = None):
        """
        Initialize testset downloader.
        
        Args:
            output_dir: Base output directory
            testset_name: One of 'cub200', 'miniimagenet', 'sun397'
            project_root: Project root directory (auto-detected if None)
        """
        super().__init__(output_dir)
        
        if testset_name not in self.TESTSETS:
            raise ValueError(f"Unknown testset: {testset_name}. Available: {list(self.TESTSETS.keys())}")
        
        self.testset_name = testset_name
        self.config = self.TESTSETS[testset_name]
        
        # Resolve project root
        if project_root is None:
            self.project_root = get_project_root()
        else:
            # If provided as relative, resolve relative to cwd
            self.project_root = Path(project_root).resolve()
        
        self.script_path = self.project_root / self.config["script_path"]
        self.testset_dir = self.output_dir / "eval_public" / testset_name
    
    @property
    def info(self) -> DatasetInfo:
        return DatasetInfo(
            name=self.config["name"],
            description=self.config["description"],
            size=self.config["size"],
            num_classes=self.config["classes"],
        )
    
    @classmethod
    def available_testsets(cls) -> List[str]:
        """Return list of available testset names."""
        return list(cls.TESTSETS.keys())
    
    def download(self, **kwargs) -> DownloadResult:
        """Download testset by running its prepare script."""
        
        self._print_header(f"DOWNLOADING {self.info.name.upper()}")
        self._print_info()
        print(f"Script: {self.script_path}")
        print(f"Output: {self.testset_dir}")
        
        # Check script exists
        if not self.script_path.exists():
            return DownloadResult(
                status=DownloadStatus.FAILED,
                message=f"Script not found: {self.script_path}"
            )
        
        # Create output directory
        self.testset_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up raw data directory (alongside the script)
        raw_data_dir = self.script_path.parent / "raw_data"
        
        # Build command
        cmd = [
            sys.executable,
            str(self.script_path),
            "--download_dir", str(raw_data_dir),
            "--output_dir", str(self.testset_dir),
        ]
        
        print(f"\nRunning: {' '.join(cmd)}\n")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.script_path.parent),
                capture_output=False,
            )
            
            if result.returncode == 0:
                verification = self.verify()
                return DownloadResult(
                    status=DownloadStatus.COMPLETED,
                    message=f"Downloaded successfully. {verification.message}",
                    output_path=self.testset_dir,
                    file_count=verification.file_count
                )
            else:
                return DownloadResult(
                    status=DownloadStatus.FAILED,
                    message=f"Script exited with code {result.returncode}"
                )
                
        except Exception as e:
            return DownloadResult(
                status=DownloadStatus.FAILED,
                message=f"Error running script: {e}",
                error=e
            )
    
    def verify(self) -> DownloadResult:
        """Verify testset data exists and count images."""
        
        if not self.testset_dir.exists():
            return DownloadResult(
                status=DownloadStatus.NOT_STARTED,
                message=f"Directory not found: {self.testset_dir}"
            )
        
        # Count images in each split
        splits = {}
        total = 0
        for split in ["train", "val", "test"]:
            split_dir = self.testset_dir / split
            count = self._count_images(split_dir)
            splits[split] = count
            total += count
        
        # Check for CSV files
        csv_files = list(self.testset_dir.glob("*.csv"))
        
        if total > 0:
            split_str = ", ".join(f"{k}: {v:,}" for k, v in splits.items() if v > 0)
            return DownloadResult(
                status=DownloadStatus.COMPLETED,
                message=f"Found {total:,} images ({split_str})",
                output_path=self.testset_dir,
                file_count=total
            )
        else:
            return DownloadResult(
                status=DownloadStatus.FAILED,
                message="No images found"
            )


# Register testset downloaders dynamically
for testset_name in TestsetDownloader.TESTSETS:
    # Create a factory function for each testset
    def make_factory(name):
        def factory(output_dir: Path, **kwargs):
            return TestsetDownloader(output_dir, name, **kwargs)
        return factory
    
    DownloaderRegistry._downloaders[testset_name] = make_factory(testset_name)
