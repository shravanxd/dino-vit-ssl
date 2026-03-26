"""
Projection heads for SSL methods.

Available heads:
- DINOHead: Multi-layer projection head for DINO
"""

from .dino_head import DINOHead

__all__ = [
    "DINOHead",
]
