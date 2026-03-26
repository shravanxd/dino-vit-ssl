"""
SSL method implementations.

Available methods:
- DINO: Self-distillation with no labels
"""

from .base import BaseSSLMethod, MethodRegistry
from .dino import DINO

__all__ = [
    "BaseSSLMethod",
    "MethodRegistry",
    "DINO",
]
