"""
Vision Transformer (ViT) implementations.

Contains:
- ViTTiny: ~5.7M parameters (embed_dim=192, heads=3, layers=12)
- ViTSmall: ~22M parameters (embed_dim=384, heads=6, layers=12)
- ViTBase: ~86M parameters (embed_dim=768, heads=12, layers=12)

Both designed for 96x96 input images with patch size 8 (giving 12x12=144 patches).

References:
- "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
- DINO paper uses ViT-S/16 and ViT-B/16
"""

import math
from typing import Optional, List, Tuple
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseBackbone, BackboneRegistry


class PatchEmbed(nn.Module):
    """
    Convert image into patch embeddings.
    
    Splits image into non-overlapping patches and projects each patch
    to the embedding dimension using a convolution.
    
    Supports variable input sizes - num_patches is computed dynamically.
    """
    
    def __init__(
        self,
        img_size: int = 96,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 384,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Use conv2d as efficient patch projection
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) - supports variable H, W
        Returns:
            (B, num_patches, embed_dim) where num_patches = (H/P) * (W/P)
        """
        # (B, C, H, W) -> (B, embed_dim, H/P, W/P)
        x = self.proj(x)
        # (B, embed_dim, H/P, W/P) -> (B, embed_dim, num_patches)
        x = x.flatten(2)
        # (B, embed_dim, num_patches) -> (B, num_patches, embed_dim)
        x = x.transpose(1, 2)
        return x


class Attention(nn.Module):
    """
    Multi-head self-attention.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Combined QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) where N = num_patches + 1 (CLS token)
        Returns:
            (B, N, D)
        """
        B, N, D = x.shape
        
        # (B, N, 3*D) -> (B, N, 3, heads, head_dim) -> (3, B, heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention: (B, heads, N, head_dim) @ (B, heads, head_dim, N) -> (B, heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # (B, heads, N, N) @ (B, heads, N, head_dim) -> (B, heads, N, head_dim)
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MLP(nn.Module):
    """
    Feed-forward MLP block with GELU activation.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        out_features = out_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer block: LayerNorm -> Attention -> LayerNorm -> MLP
    
    Uses pre-norm architecture (LayerNorm before attention/MLP).
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=drop,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(BaseBackbone):
    """
    Vision Transformer base implementation.
    
    Architecture:
    1. Patch embedding (split image into patches, project to embed_dim)
    2. Add CLS token and positional embeddings
    3. N transformer blocks
    4. LayerNorm
    5. Return CLS token as image representation
    
    Args:
        img_size: Input image size (assumes square)
        patch_size: Size of each patch
        in_channels: Number of input channels
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim = embed_dim * mlp_ratio
        qkv_bias: Include bias in QKV projection
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
    """
    
    def __init__(
        self,
        img_size: int = 96,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()
        
        self._embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embedding (learnable)
        # +1 for CLS token
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
            )
            for _ in range(depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with truncated normal and zeros."""
        # Position embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # CLS token
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Apply to all modules
        self.apply(self._init_module_weights)
    
    def _init_module_weights(self, m):
        """Initialize individual module weights."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            # For patch embedding
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def _interpolate_pos_embed(
        self,
        num_patches: int,
    ) -> torch.Tensor:
        """
        Interpolate positional embeddings for different input sizes.
        
        This allows the model to handle different crop sizes (e.g., local crops)
        by interpolating the learned positional embeddings.
        
        Args:
            num_patches: Number of patches in current input
            
        Returns:
            Interpolated positional embedding (1, 1+num_patches, embed_dim)
        """
        if num_patches == self.num_patches:
            # Same size, no interpolation needed
            return self.pos_embed
        
        # Get trained positional embeddings (excluding CLS token)
        pos_embed = self.pos_embed[:, 1:, :]  # (1, num_patches, embed_dim)
        cls_pos_embed = self.pos_embed[:, :1, :]  # (1, 1, embed_dim)
        
        # Get spatial dimensions
        trained_size = int(self.num_patches ** 0.5)  # e.g., 12 for 144 patches
        new_size = int(num_patches ** 0.5)  # e.g., 6 for 36 patches
        
        # Reshape to spatial grid for interpolation
        # (1, trained_size^2, embed_dim) -> (1, embed_dim, trained_size, trained_size)
        pos_embed = pos_embed.reshape(1, trained_size, trained_size, -1).permute(0, 3, 1, 2)
        
        # Interpolate to new size
        pos_embed = F.interpolate(
            pos_embed,
            size=(new_size, new_size),
            mode='bicubic',
            align_corners=False,
        )
        
        # Reshape back: (1, embed_dim, new_size, new_size) -> (1, new_size^2, embed_dim)
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, num_patches, -1)
        
        # Concatenate CLS positional embedding
        return torch.cat([cls_pos_embed, pos_embed], dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning CLS token embedding.
        
        Supports variable input sizes via positional embedding interpolation.
        
        Args:
            x: Input images (B, C, H, W)
            
        Returns:
            CLS token embeddings (B, embed_dim)
        """
        B, C, H, W = x.shape
        
        # Patch embedding: (B, C, H, W) -> (B, num_patches, embed_dim)
        x = self.patch_embed(x)
        num_patches = x.shape[1]
        
        # Prepend CLS token: (B, num_patches, D) -> (B, 1 + num_patches, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding (interpolate if needed)
        pos_embed = self._interpolate_pos_embed(num_patches)
        x = x + pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final norm
        x = self.norm(x)
        
        # Return CLS token
        return x[:, 0]
    
    def forward_features(
        self, 
        x: torch.Tensor,
        return_patch_tokens: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass returning CLS token and optionally patch tokens.
        
        Supports variable input sizes via positional embedding interpolation.
        
        Args:
            x: Input images (B, C, H, W)
            return_patch_tokens: If True, also return patch tokens
            
        Returns:
            Tuple of:
                - CLS token embedding (B, embed_dim)
                - Optional patch tokens (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        
        x = self.patch_embed(x)
        num_patches = x.shape[1]
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding (interpolate if needed)
        pos_embed = self._interpolate_pos_embed(num_patches)
        x = x + pos_embed
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        if return_patch_tokens:
            return x[:, 0], x[:, 1:]  # CLS, patch tokens
        return x[:, 0], None
    
    def get_intermediate_layers(
        self, 
        x: torch.Tensor, 
        n: int = 1
    ) -> List[torch.Tensor]:
        """
        Get outputs from the last n transformer layers.
        
        Args:
            x: Input images (B, C, H, W)
            n: Number of last layers to return
            
        Returns:
            List of CLS token outputs from last n layers
        """
        B = x.shape[0]
        
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        outputs = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i >= len(self.blocks) - n:
                outputs.append(self.norm(x)[:, 0])
        
        return outputs


@BackboneRegistry.register("vit_tiny")
class ViTTiny(VisionTransformer):
    """
    Vision Transformer Tiny.
    
    Configuration:
    - embed_dim: 192
    - depth: 12 layers
    - heads: 3
    - params: ~5.7M
    
    Smallest ViT variant, useful for fast experiments and debugging.
    Suitable for 96x96 images with patch_size=8.
    """
    
    def __init__(
        self,
        img_size: int = 96,
        patch_size: int = 8,
        in_channels: int = 3,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=192,
            depth=12,
            num_heads=3,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
        )


@BackboneRegistry.register("vit_small")
class ViTSmall(VisionTransformer):
    """
    Vision Transformer Small.
    
    Configuration:
    - embed_dim: 384
    - depth: 12 layers
    - heads: 6
    - params: ~22M
    
    Suitable for 96x96 images with patch_size=8.
    """
    
    def __init__(
        self,
        img_size: int = 96,
        patch_size: int = 8,
        in_channels: int = 3,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
        )


@BackboneRegistry.register("vit_base")
class ViTBase(VisionTransformer):
    """
    Vision Transformer Base.
    
    Configuration:
    - embed_dim: 768
    - depth: 12 layers
    - heads: 12
    - params: ~86M
    
    Suitable for 96x96 images with patch_size=8.
    Still under 100M parameter limit.
    """
    
    def __init__(
        self,
        img_size: int = 96,
        patch_size: int = 8,
        in_channels: int = 3,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
        )

