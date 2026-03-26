"""
Backbone architectures for SSL pretraining.

Available backbones:
- ViTTiny: Vision Transformer Tiny (~5.7M params)
- ViTSmall: Vision Transformer Small (~22M params)
- ViTBase: Vision Transformer Base (~86M params)
"""

from .base import BaseBackbone, BackboneRegistry
from .vit import ViTTiny, ViTSmall, ViTBase

__all__ = [
    "BaseBackbone",
    "BackboneRegistry",
    "ViTTiny",
    "ViTSmall",
    "ViTBase",
]
