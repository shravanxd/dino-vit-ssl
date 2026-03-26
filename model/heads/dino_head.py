"""
DINO projection head.

The DINO head is a multi-layer MLP that projects backbone features
into a space suitable for the self-distillation loss.

Architecture:
    Input (embed_dim) → MLP layers → Bottleneck → L2 Norm → Output (out_dim)
    
The head includes:
- Multiple hidden layers with GELU activation
- A bottleneck layer before the output
- L2 normalization of the BOTTLENECK (not the output!)
- Optional weight normalization on the last layer

CRITICAL: Following the original DINO paper, L2 normalization is applied
to the bottleneck features BEFORE the final projection, NOT after.
Normalizing after the projection kills the signal and causes collapse.

Reference:
    "Emerging Properties in Self-Supervised Vision Transformers"
    https://arxiv.org/abs/2104.14294
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class DINOHead(nn.Module):
    """
    DINO projection head.
    
    Projects backbone features to a high-dimensional space for
    self-distillation. Used by both student and teacher networks.
    
    Architecture:
        Linear → GELU → Linear → GELU → ... → Linear (bottleneck) → Linear (output)
        + L2 normalization
        + Weight normalization on last layer
    
    Args:
        in_dim: Input dimension (backbone embed_dim)
        out_dim: Output dimension (number of prototypes, e.g., 65536)
        hidden_dim: Hidden layer dimension (default: 2048)
        bottleneck_dim: Bottleneck dimension before output (default: 256)
        num_layers: Number of hidden layers (default: 3)
        use_bn: Whether to use batch normalization (default: False)
        norm_last_layer: Whether to apply weight norm to last layer (default: True)
        
    Example:
        >>> head = DINOHead(in_dim=384, out_dim=65536)
        >>> x = torch.randn(32, 384)  # Batch of backbone features
        >>> out = head(x)  # (32, 65536) - NOT normalized (allows variance)
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        num_layers: int = 3,
        use_bn: bool = False,
        norm_last_layer: bool = True,
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        
        # Build MLP layers
        layers = self._build_mlp(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            num_layers=num_layers,
            use_bn=use_bn,
        )
        self.mlp = nn.Sequential(*layers)
        
        # Final projection to output dimension
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        
        # Apply weight normalization to last layer
        if norm_last_layer:
            self.last_layer = nn.utils.weight_norm(self.last_layer)
            # Initialize with fixed norm
            self.last_layer.weight_g.data.fill_(1)
            self.last_layer.weight_g.requires_grad = False
    
    def _build_mlp(
        self,
        in_dim: int,
        hidden_dim: int,
        bottleneck_dim: int,
        num_layers: int,
        use_bn: bool,
    ) -> List[nn.Module]:
        """
        Build the MLP layers.
        
        Creates a sequence of Linear → (BN) → GELU layers,
        ending with a bottleneck layer.
        """
        layers = []
        
        # First layer: in_dim → hidden_dim
        layers.append(nn.Linear(in_dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        
        # Hidden layers: hidden_dim → hidden_dim
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        
        # Bottleneck layer: hidden_dim → bottleneck_dim
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
        
        return layers
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the DINO head.
        
        Args:
            x: Input features from backbone (B, in_dim)
            
        Returns:
            Output features (B, out_dim) - NOT L2 normalized
            
        Note:
            Following the original DINO implementation, we L2 normalize
            the bottleneck features BEFORE the last layer projection,
            not after. This is critical - normalizing after projection
            kills the signal and causes collapse.
        """
        # MLP projection to bottleneck
        x = self.mlp(x)
        
        # L2 normalize the bottleneck (BEFORE last layer, not after!)
        # This is critical for DINO to work properly
        x = F.normalize(x, dim=-1, p=2)
        
        # Final projection to output dimension
        # Output is NOT normalized - this allows variance in magnitudes
        x = self.last_layer(x)
        
        return x
    
    def get_num_parameters(self) -> int:
        """Count total parameters in the head."""
        return sum(p.numel() for p in self.parameters())
    
    def __repr__(self) -> str:
        params = self.get_num_parameters() / 1e6
        return (
            f"{self.__class__.__name__}("
            f"in_dim={self.in_dim}, "
            f"out_dim={self.out_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"bottleneck_dim={self.bottleneck_dim}, "
            f"params={params:.2f}M)"
        )
