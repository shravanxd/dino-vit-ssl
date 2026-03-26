"""
Abstract base class for SSL methods.

Provides:
- BaseSSLMethod: Abstract interface all SSL methods must implement
- MethodRegistry: Factory pattern for method instantiation
"""

from abc import ABC, abstractmethod
from typing import Dict, Type, Optional, Tuple, Any
import torch
import torch.nn as nn

from model.backbones.base import BaseBackbone


class MethodRegistry:
    """
    Registry for SSL methods.
    
    Allows dynamic registration and instantiation of methods by name.
    
    Usage:
        @MethodRegistry.register("dino")
        class DINO(BaseSSLMethod):
            ...
        
        # Later:
        method = MethodRegistry.create("dino", backbone=vit)
    """
    
    _methods: Dict[str, Type["BaseSSLMethod"]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register an SSL method class."""
        def decorator(method_cls: Type["BaseSSLMethod"]):
            cls._methods[name.lower()] = method_cls
            return method_cls
        return decorator
    
    @classmethod
    def create(cls, name: str, **kwargs) -> "BaseSSLMethod":
        """Create an SSL method instance by name."""
        name_lower = name.lower()
        if name_lower not in cls._methods:
            available = list(cls._methods.keys())
            raise ValueError(f"Unknown method: {name}. Available: {available}")
        return cls._methods[name_lower](**kwargs)
    
    @classmethod
    def available(cls) -> list:
        """List available method names."""
        return list(cls._methods.keys())


class BaseSSLMethod(nn.Module, ABC):
    """
    Abstract base class for all SSL methods.
    
    All SSL methods must:
    1. Define a forward pass that processes augmented views
    2. Provide access to the backbone for feature extraction
    3. Report their total parameter count
    4. Support checkpointing (save/load state)
    
    The actual loss computation is handled separately in the training loop,
    allowing flexibility in how methods are trained.
    
    Attributes:
        backbone: The encoder backbone (ViT, ResNet, etc.)
    """
    
    def __init__(self, backbone: BaseBackbone):
        super().__init__()
        self._backbone = backbone
    
    @property
    def backbone(self) -> BaseBackbone:
        """Get the backbone encoder."""
        return self._backbone
    
    @property
    def embed_dim(self) -> int:
        """Get the backbone embedding dimension."""
        return self._backbone.embed_dim
    
    @abstractmethod
    def forward(self, views: list) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the SSL method.
        
        Args:
            views: List of augmented views of the same images.
                   For DINO: [global_1, global_2, local_1, ..., local_n]
                   Each view has shape (B, C, H, W)
        
        Returns:
            Dictionary containing outputs needed for loss computation.
            The exact contents depend on the method.
        """
        pass
    
    @abstractmethod
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features using the backbone only.
        
        This is used during evaluation (KNN, linear probe) where
        we only need the backbone features, not the projection head.
        
        Args:
            x: Input images (B, C, H, W)
            
        Returns:
            Feature embeddings (B, embed_dim)
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
        
        Note: For methods with teacher networks (like DINO), only the
        student/trainable parameters count toward the budget.
        
        Args:
            max_params: Maximum allowed parameters (default: 100M)
            
        Returns:
            Tuple of (is_under_budget, actual_param_count)
        """
        # Count trainable params (teacher params don't count)
        num_params = self.get_num_parameters(trainable_only=True)
        return num_params < max_params, num_params
    
    def freeze_backbone(self) -> None:
        """Freeze the backbone for linear evaluation."""
        self._backbone.freeze()
    
    def unfreeze_backbone(self) -> None:
        """Unfreeze the backbone for training."""
        self._backbone.unfreeze()
    
    def get_backbone_state_dict(self) -> Dict[str, Any]:
        """
        Get only the backbone state dict for saving.
        
        This is what you submit for evaluation - just the encoder.
        """
        return self._backbone.state_dict()
    
    def load_backbone_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load backbone weights from a state dict."""
        self._backbone.load_state_dict(state_dict)
    
    @abstractmethod
    def __repr__(self) -> str:
        """String representation of the method."""
        pass
