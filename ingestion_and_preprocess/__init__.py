"""
Ingestion and Preprocessing Package

This package provides tools for downloading and preparing datasets
for the SSL Capstone project.

Structure:
    ingestion_and_preprocess/
    ├── __init__.py
    ├── ingestion/          # Data downloading
    │   ├── base.py         # Abstract base classes
    │   ├── downloaders.py  # Concrete downloader implementations
    │   └── download_data.py # CLI entry point
    └── preprocessing/      # Data preprocessing (future)

Usage:
    # From command line
    python -m ingestion_and_preprocess.ingestion.download_data --output_dir ./data
    
    # Or import directly
    from ingestion_and_preprocess.ingestion import DataDownloadManager, PretrainDownloader
"""

from .ingestion import (
    # Base classes
    BaseDownloader,
    DownloaderRegistry,
    DownloadResult,
    DownloadStatus,
    DatasetInfo,
    # Concrete downloaders
    PretrainDownloader,
    TestsetDownloader,
    # Manager
    DataDownloadManager,
    # Utilities
    get_project_root,
)

__all__ = [
    # Base classes
    "BaseDownloader",
    "DownloaderRegistry", 
    "DownloadResult",
    "DownloadStatus",
    "DatasetInfo",
    # Concrete downloaders
    "PretrainDownloader",
    "TestsetDownloader",
    # Manager
    "DataDownloadManager",
    # Utilities
    "get_project_root",
]
