"""
Ingestion subpackage - handles data downloading.
"""

from .base import (
    BaseDownloader,
    DownloaderRegistry,
    DownloadResult,
    DownloadStatus,
    DatasetInfo,
)

from .downloaders import (
    PretrainDownloader,
    LAIONDownloader,
    CombinedPretrainDownloader,
    TestsetDownloader,
    PretrainSource,
    get_project_root,
)

from .download_data import (
    DataDownloadManager,
)

__all__ = [
    "BaseDownloader",
    "DownloaderRegistry", 
    "DownloadResult",
    "DownloadStatus",
    "DatasetInfo",
    "PretrainDownloader",
    "LAIONDownloader",
    "CombinedPretrainDownloader",
    "TestsetDownloader",
    "PretrainSource",
    "DataDownloadManager",
    "get_project_root",
]
