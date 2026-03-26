"""
Abstract base class for backbone architectures.

Provides:
- BaseBackbone: Abstract interface all backbones must implement
- BackboneRegistry: Factory pattern for backbone instantiation
"""

from abc import ABC, abstractmethod
from typing import Dict, Type, Optional, Tuple
import torch
import torch.nn as nn


class BackboneRegistry:
    """
    Registry for backbone architectures.
    
    Allows dynamic registration and instantiation of backbones by name.
    
    Usage:
        @BackboneRegistry.register("vit_small")
        class ViTSmall(BaseBackbone):
            ...
        
        # Later:
        backbone = BackboneRegistry.create("vit_small", img_size=96)
    """
    
    _backbones: Dict[str, Type["BaseBackbone"]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a backbone class."""
        def decorator(backbone_cls: Type["BaseBackbone"]):
            cls._backbones[name.lower()] = backbone_cls
            return backbone_cls
        return decorator
    
    @classmethod
    def create(cls, name: str, **kwargs) -> "BaseBackbone":
        """Create a backbone instance by name."""
        name_lower = name.lower()
        if name_lower not in cls._backbones:
            available = list(cls._backbones.keys())
            raise ValueError(f"Unknown backbone: {name}. Available: {available}")
        return cls._backbones[name_lower](**kwargs)
    
    @classmethod
    def available(cls) -> list:
        """List available backbone names."""
        return list(cls._backbones.keys())


class BaseBackbone(nn.Module, ABC):
    """
    Abstract base class for all backbone architectures.
    
    All backbones must:
    1. Accept images and return feature embeddings
    2. Report their embedding dimension
    3. Report their parameter count
    4. Support random initialization (no pretrained weights)
    
    Attributes:
        embed_dim: Dimension of output feature embeddings
    """
    
    def __init__(self):
        super().__init__()
        self._embed_dim: int = 0
    
    @property
    def embed_dim(self) -> int:
        """Dimension of the output embedding vector."""
        return self._embed_dim
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the backbone.
        
        Args:
            x: Input images of shape (B, C, H, W)
            
        Returns:
            Feature embeddings of shape (B, embed_dim)
        """
        pass
    
    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """
        Count the number of parameters.
        
        Args:
            trainable_only: If True, count only trainable parameters
            
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def get_num_parameters_millions(self, trainable_only: bool = False) -> float:
        """Get parameter count in millions."""
        return self.get_num_parameters(trainable_only) / 1e6
    
    def check_parameter_budget(self, max_params: int = 100_000_000) -> Tuple[bool, int]:
        """
        Check if model is under the parameter budget.
        
        Args:
            max_params: Maximum allowed parameters (default: 100M)
            
        Returns:
            Tuple of (is_under_budget, actual_param_count)
        """
        num_params = self.get_num_parameters()
        return num_params < max_params, num_params
    
    def freeze(self) -> None:
        """Freeze all parameters (for evaluation)."""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self) -> None:
        """Unfreeze all parameters (for training)."""
        for param in self.parameters():
            param.requires_grad = True
    
    @abstractmethod
    def get_intermediate_layers(
        self, 
        x: torch.Tensor, 
        n: int = 1
    ) -> list:
        """
        Get outputs from the last n transformer layers.
        
        Useful for methods that use multi-layer features.
        
        Args:
            x: Input images of shape (B, C, H, W)
            n: Number of last layers to return
            
        Returns:
            List of tensors from the last n layers
        """
        pass
    
    def __repr__(self) -> str:
        params_m = self.get_num_parameters_millions()
        return f"{self.__class__.__name__}(embed_dim={self.embed_dim}, params={params_m:.2f}M)"
