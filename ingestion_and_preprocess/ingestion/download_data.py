"""
Main entry point for data download.

Usage (run from project root):
    python -m ingestion_and_preprocess.ingestion.download_data --output_dir ./data
    python -m ingestion_and_preprocess.ingestion.download_data --output_dir ./data --pretrain_only
    python -m ingestion_and_preprocess.ingestion.download_data --output_dir ./data --pretrain_only --pretrain_source cc3m
    python -m ingestion_and_preprocess.ingestion.download_data --output_dir ./data --pretrain_only --pretrain_source laion
    python -m ingestion_and_preprocess.ingestion.download_data --output_dir ./data --pretrain_only --pretrain_source both
    python -m ingestion_and_preprocess.ingestion.download_data --output_dir ./data --eval_only
    python -m ingestion_and_preprocess.ingestion.download_data --output_dir ./data --testset cub200
    python -m ingestion_and_preprocess.ingestion.download_data --output_dir ./data --verify_only

All paths are relative to the current working directory.
"""

import argparse
from pathlib import Path
from typing import List, Optional, Union

from .base import DownloaderRegistry, DownloadStatus, DownloadResult
from .downloaders import (
    PretrainDownloader, 
    LAIONDownloader,
    CombinedPretrainDownloader,
    TestsetDownloader, 
    PretrainSource,
    get_project_root
)


class DataDownloadManager:
    """
    Orchestrates downloading of all datasets.
    
    Responsibilities:
    - Coordinate multiple downloaders
    - Handle command-line arguments
    - Provide summary of all downloads
    
    All paths are resolved relative to cwd, making it portable across systems.
    """
    
    def __init__(self, output_dir: Union[str, Path], project_root: Optional[Path] = None):
        """
        Initialize download manager.
        
        Args:
            output_dir: Output directory (relative to cwd or absolute)
            project_root: Project root for finding scripts (auto-detected if None)
        """
        # Resolve output_dir relative to cwd
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Project root for finding testset scripts
        self.project_root = project_root or get_project_root()
        
        self.results: dict[str, DownloadResult] = {}
    
    def download_pretrain(
        self, 
        source: PretrainSource = PretrainSource.CC3M,
        keep_zip: bool = False,
        laion_num_images: int = 600000,
    ) -> DownloadResult:
        """
        Download pretrain data from specified source(s).
        
        Args:
            source: Which source to download (cc3m, laion, or both)
            keep_zip: Keep CC3M zip files after extraction
            laion_num_images: Number of LAION images to download
        """
        if source == PretrainSource.CC3M:
            # Original CC3M only
            downloader = PretrainDownloader(self.output_dir)
            result = downloader.download(keep_zip=keep_zip)
            self.results["pretrain_cc3m"] = result
        elif source == PretrainSource.LAION:
            # LAION only
            downloader = LAIONDownloader(
                self.output_dir, 
                num_images=laion_num_images
            )
            result = downloader.download()
            self.results["pretrain_laion"] = result
        else:
            # Both sources
            downloader = CombinedPretrainDownloader(
                self.output_dir,
                source=PretrainSource.BOTH,
                laion_num_images=laion_num_images,
            )
            result = downloader.download(keep_zip=keep_zip)
            self.results["pretrain_combined"] = result
        
        return result
    
    def download_testset(self, name: str) -> DownloadResult:
        """Download a specific testset."""
        downloader = TestsetDownloader(self.output_dir, name, project_root=self.project_root)
        result = downloader.download()
        self.results[name] = result
        return result
    
    def download_all_testsets(self) -> dict[str, DownloadResult]:
        """Download all eval testsets."""
        results = {}
        for name in TestsetDownloader.available_testsets():
            results[name] = self.download_testset(name)
        return results
    
    def download_all(
        self, 
        keep_zip: bool = False,
        pretrain_source: PretrainSource = PretrainSource.CC3M,
        laion_num_images: int = 600000,
    ) -> dict[str, DownloadResult]:
        """Download everything (pretrain + all testsets)."""
        self.download_pretrain(
            source=pretrain_source,
            keep_zip=keep_zip,
            laion_num_images=laion_num_images,
        )
        self.download_all_testsets()
        return self.results
    
    def verify_all(self) -> dict[str, DownloadResult]:
        """Verify all datasets."""
        
        # Verify CC3M pretrain
        pretrain_cc3m = PretrainDownloader(self.output_dir)
        cc3m_result = pretrain_cc3m.verify()
        if cc3m_result.file_count > 0:
            self.results["pretrain_cc3m"] = cc3m_result
        
        # Verify LAION pretrain
        pretrain_laion = LAIONDownloader(self.output_dir)
        laion_result = pretrain_laion.verify()
        if laion_result.file_count > 0:
            self.results["pretrain_laion"] = laion_result
        
        # Verify testsets
        for name in TestsetDownloader.available_testsets():
            testset = TestsetDownloader(self.output_dir, name, project_root=self.project_root)
            self.results[name] = testset.verify()
        
        return self.results
    
    def print_summary(self):
        """Print summary of all download results."""
        
        print(f"\n{'='*60}")
        print("DOWNLOAD SUMMARY")
        print(f"{'='*60}\n")
        
        print(f"Output directory: {self.output_dir}\n")
        
        for name, result in self.results.items():
            status_icon = {
                DownloadStatus.COMPLETED: "✅",
                DownloadStatus.FAILED: "❌",
                DownloadStatus.NOT_STARTED: "⚪",
                DownloadStatus.IN_PROGRESS: "🔄",
                DownloadStatus.SKIPPED: "⏭️",
            }.get(result.status, "❓")
            
            print(f"{status_icon} {name.upper()}")
            print(f"   Status: {result.status.value}")
            print(f"   {result.message}")
            if result.output_path:
                print(f"   Path: {result.output_path}")
            print()


def parse_args():
    """Parse command-line arguments."""
    
    parser = argparse.ArgumentParser(
        description="Download SSL Capstone datasets (pretrain + eval testsets)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download everything with CC3M only (default)
  python -m ingestion_and_preprocess.ingestion.download_data --output_dir ./data
  
  # Download only CC3M pretrain data
  python -m ingestion_and_preprocess.ingestion.download_data --pretrain_only --pretrain_source cc3m
  
  # Download only LAION pretrain data
  python -m ingestion_and_preprocess.ingestion.download_data --pretrain_only --pretrain_source laion
  
  # Download both CC3M + LAION (recommended for best performance)
  python -m ingestion_and_preprocess.ingestion.download_data --pretrain_only --pretrain_source both
  
  # Download with custom LAION size
  python -m ingestion_and_preprocess.ingestion.download_data --pretrain_only --pretrain_source laion --laion_num_images 300000
  
  # Download eval testsets only
  python -m ingestion_and_preprocess.ingestion.download_data --eval_only
  
  # Verify existing data
  python -m ingestion_and_preprocess.ingestion.download_data --verify_only
        """
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Output directory for downloaded data (default: ./data)"
    )
    
    # Pretrain source selection
    parser.add_argument(
        "--pretrain_source",
        type=str,
        choices=["cc3m", "laion", "both"],
        default="cc3m",
        help="Pretrain data source: 'cc3m' (500K, default), 'laion' (600K LAION-Aesthetics), 'both' (1.1M combined)"
    )
    parser.add_argument(
        "--laion_num_images",
        type=int,
        default=600000,
        help="Number of LAION images to download (default: 600000)"
    )
    
    # Download mode flags
    parser.add_argument(
        "--pretrain_only",
        action="store_true",
        help="Download only pretrain data"
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Download only eval testsets"
    )
    parser.add_argument(
        "--testset",
        type=str,
        choices=TestsetDownloader.available_testsets(),
        help="Download only a specific testset"
    )
    
    # Other options
    parser.add_argument(
        "--keep_zip",
        action="store_true",
        help="Keep zip files after extraction"
    )
    parser.add_argument(
        "--verify_only",
        action="store_true",
        help="Only verify existing data, don't download"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    
    args = parse_args()
    
    print("\n" + "="*60)
    print("SSL CAPSTONE - DATA DOWNLOAD")
    print("="*60)
    
    # Get project root for reference
    project_root = get_project_root()
    
    manager = DataDownloadManager(args.output_dir, project_root=project_root)
    print(f"Working directory: {Path.cwd()}")
    print(f"Project root: {project_root}")
    print(f"Output directory: {manager.output_dir}")
    
    # Parse pretrain source
    pretrain_source = PretrainSource(args.pretrain_source)
    
    # Verify only mode
    if args.verify_only:
        manager.verify_all()
        manager.print_summary()
        return
    
    # Download based on flags
    if args.testset:
        # Specific testset only
        manager.download_testset(args.testset)
    elif args.pretrain_only:
        # Pretrain only with specified source
        print(f"\nPretrain source: {pretrain_source.value}")
        if pretrain_source in [PretrainSource.LAION, PretrainSource.BOTH]:
            print(f"LAION images: {args.laion_num_images:,}")
        manager.download_pretrain(
            source=pretrain_source,
            keep_zip=args.keep_zip,
            laion_num_images=args.laion_num_images,
        )
    elif args.eval_only:
        # All testsets only
        manager.download_all_testsets()
    else:
        # Download everything
        manager.download_all(
            keep_zip=args.keep_zip,
            pretrain_source=pretrain_source,
            laion_num_images=args.laion_num_images,
        )
    
    manager.print_summary()
    
    print("="*60)
    print("COMPLETE")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
