"""
Base classes and interfaces for data downloaders.

Follows SOLID principles:
- Single Responsibility: Each downloader handles one data source
- Open/Closed: Easy to add new downloaders without modifying existing code
- Liskov Substitution: All downloaders are interchangeable via base class
- Interface Segregation: Clean, minimal interfaces
- Dependency Inversion: Depend on abstractions, not concretions
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict
from enum import Enum


class DownloadStatus(Enum):
    """Status of a download operation."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DownloadResult:
    """Result of a download operation."""
    status: DownloadStatus
    message: str
    output_path: Optional[Path] = None
    file_count: int = 0
    error: Optional[Exception] = None


@dataclass
class DatasetInfo:
    """Information about a dataset."""
    name: str
    description: str
    size: str
    num_classes: Optional[int] = None
    num_images: Optional[int] = None


class BaseDownloader(ABC):
    """
    Abstract base class for all data downloaders.
    
    Each downloader is responsible for:
    1. Downloading data from a source
    2. Extracting/processing the data
    3. Organizing it into the expected structure
    4. Verifying the download
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize downloader.
        
        Args:
            output_dir: Base directory for downloaded data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    @abstractmethod
    def info(self) -> DatasetInfo:
        """Return information about this dataset."""
        pass
    
    @abstractmethod
    def download(self, **kwargs) -> DownloadResult:
        """
        Download and prepare the dataset.
        
        Returns:
            DownloadResult with status and details
        """
        pass
    
    @abstractmethod
    def verify(self) -> DownloadResult:
        """
        Verify that the dataset is correctly downloaded.
        
        Returns:
            DownloadResult with verification status
        """
        pass
    
    def _count_images(self, directory: Path) -> int:
        """Count images in a directory recursively."""
        if not directory.exists():
            return 0
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'}
        return sum(
            1 for f in directory.rglob("*") 
            if f.suffix.lower() in image_extensions
        )
    
    def _print_header(self, title: str):
        """Print a formatted header."""
        print(f"\n{'='*60}")
        print(title)
        print(f"{'='*60}")
    
    def _print_info(self):
        """Print dataset info."""
        info = self.info
        print(f"Dataset: {info.name}")
        print(f"Description: {info.description}")
        print(f"Size: {info.size}")
        if info.num_classes:
            print(f"Classes: {info.num_classes}")


class DownloaderRegistry:
    """
    Registry for dataset downloaders.
    
    Allows dynamic registration and retrieval of downloaders.
    """
    
    _downloaders: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a downloader class."""
        def decorator(downloader_cls: type):
            cls._downloaders[name] = downloader_cls
            return downloader_cls
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Optional[type]:
        """Get a downloader class by name."""
        return cls._downloaders.get(name)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered downloader names."""
        return list(cls._downloaders.keys())
    
    @classmethod
    def create(cls, name: str, output_dir: Path, **kwargs) -> Optional[BaseDownloader]:
        """Create a downloader instance by name."""
        downloader_cls = cls.get(name)
        if downloader_cls:
            return downloader_cls(output_dir, **kwargs)
        return None
