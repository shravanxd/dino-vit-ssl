"""
DINO: Self-Distillation with No Labels.

DINO is a self-supervised method that uses a student-teacher framework
where the teacher is an exponential moving average (EMA) of the student.

Key components:
1. Student network: backbone + projection head (receives gradients)
2. Teacher network: EMA copy of student (no gradients)
3. Multi-crop strategy: 2 global views + N local views
4. Centering: prevents collapse by centering teacher outputs
5. Sharpening: different temperatures for student/teacher

Loss Components (in dino_loss.py):
A. Global-to-global distillation - CLS token distillation
B. Local-to-global alignment (L2G) - patch tokens → teacher CLS
C. Feature regularization - variance/covariance penalty

Reference:
    "Emerging Properties in Self-Supervised Vision Transformers"
    Caron et al., 2021
    https://arxiv.org/abs/2104.14294
"""

import copy
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.backbones.base import BaseBackbone
from model.heads.dino_head import DINOHead
from .base import BaseSSLMethod, MethodRegistry


class TeacherNetwork(nn.Module):
    """
    Teacher network for DINO.
    
    The teacher is an EMA copy of the student. It does not receive
    gradients and is updated via exponential moving average.
    
    Includes centering mechanism to prevent mode collapse.
    """
    
    def __init__(
        self,
        backbone: BaseBackbone,
        head: DINOHead,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.center_momentum = center_momentum
        
        # Center: running mean of teacher outputs (prevents collapse)
        # Registered as buffer so it's saved with state_dict but not a parameter
        self.register_buffer("center", torch.zeros(1, head.out_dim))
        
        # Teacher should not have gradients
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through teacher network.
        
        Args:
            x: Input images (B, C, H, W)
            
        Returns:
            Teacher output after centering (B, out_dim)
        """
        features = self.backbone(x)
        output = self.head(features)
        return output
    
    @torch.no_grad()
    def update_center(self, teacher_output: torch.Tensor) -> None:
        """
        Update the center using exponential moving average.
        
        In distributed training, we need to compute the center across ALL GPUs,
        not just the local batch. This is critical for DINO to work properly.
        
        Args:
            teacher_output: Batch of teacher outputs (B, out_dim)
        """
        # Check for NaN in teacher output - skip update if detected
        if torch.isnan(teacher_output).any() or torch.isinf(teacher_output).any():
            return
        
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        
        # In distributed training, sync the batch_center across all GPUs
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(batch_center)
            batch_center = batch_center / torch.distributed.get_world_size()
        
        # Check for NaN after all_reduce (can happen with distributed sync)
        if torch.isnan(batch_center).any() or torch.isinf(batch_center).any():
            return
        
        # Update center with EMA (in-place to work with DDP buffers)
        self.center.mul_(self.center_momentum).add_(batch_center * (1 - self.center_momentum))


class StudentNetwork(nn.Module):
    """
    Student network for DINO.
    
    The student receives gradients and learns from the teacher's
    output distribution via cross-entropy loss.
    """
    
    def __init__(
        self,
        backbone: BaseBackbone,
        head: DINOHead,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(
        self, 
        x: torch.Tensor,
        return_features: bool = False,
        return_patch_tokens: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through student network.
        
        Args:
            x: Input images (B, C, H, W)
            return_features: If True, also return backbone CLS features
            return_patch_tokens: If True, also return projected patch tokens
            
        Returns:
            Tuple of:
                - Head output (B, out_dim)
                - Optional backbone CLS features (B, embed_dim)
                - Optional projected patch tokens (B, num_patches, out_dim)
        """
        if return_patch_tokens:
            cls_feat, patch_tokens = self.backbone.forward_features(x, return_patch_tokens=True)
            output = self.head(cls_feat)
            
            # Project patch tokens through head
            if patch_tokens is not None:
                B, N, D = patch_tokens.shape
                patch_tokens_flat = patch_tokens.reshape(B * N, D)
                patch_outputs = self.head(patch_tokens_flat)
                patch_outputs = patch_outputs.reshape(B, N, -1)
            else:
                patch_outputs = None
            
            if return_features:
                return output, cls_feat, patch_outputs
            return output, None, patch_outputs
        else:
            features = self.backbone(x)
            output = self.head(features)
            if return_features:
                return output, features, None
            return output, None, None


@MethodRegistry.register("dino")
class DINO(BaseSSLMethod):
    """
    DINO: Self-Distillation with No Labels.
    
    A self-supervised method using student-teacher distillation where:
    - Student learns to match teacher's output distribution
    - Teacher is EMA of student (no gradients)
    - Multi-crop augmentation (global + local views)
    - Centering prevents collapse
    
    Args:
        backbone: The encoder backbone (e.g., ViTSmall)
        out_dim: Output dimension of projection head (default: 65536)
        hidden_dim: Hidden dimension in projection head (default: 2048)
        bottleneck_dim: Bottleneck dimension (default: 256)
        num_layers: Number of MLP layers in head (default: 3)
        use_bn: Use batch norm in head (default: False)
        norm_last_layer: Weight normalize last layer (default: True)
        teacher_temp: Teacher softmax temperature (default: 0.04)
        student_temp: Student softmax temperature (default: 0.1)
        center_momentum: Momentum for center update (default: 0.9)
        
    Example:
        >>> backbone = ViTSmall(img_size=96, patch_size=8)
        >>> dino = DINO(backbone=backbone)
        >>> 
        >>> # Training forward pass
        >>> views = [global_1, global_2, local_1, local_2, ...]
        >>> outputs = dino(views)
        >>> loss = dino_loss(outputs)
        >>> 
        >>> # Feature extraction (evaluation)
        >>> features = dino.get_features(images)
    """
    
    def __init__(
        self,
        backbone: BaseBackbone,
        out_dim: int = 65536,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        num_layers: int = 3,
        use_bn: bool = False,
        norm_last_layer: bool = True,
        teacher_temp: float = 0.04,
        student_temp: float = 0.1,
        center_momentum: float = 0.9,
    ):
        super().__init__(backbone)
        
        # Store config
        self.out_dim = out_dim
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        
        # Store head config for creating teacher
        head_config = {
            'in_dim': backbone.embed_dim,
            'out_dim': out_dim,
            'hidden_dim': hidden_dim,
            'bottleneck_dim': bottleneck_dim,
            'num_layers': num_layers,
            'use_bn': use_bn,
            'norm_last_layer': norm_last_layer,
        }
        
        # Create student head
        student_head = DINOHead(**head_config)
        
        # Student network
        self.student = StudentNetwork(backbone, student_head)
        
        # Create teacher with fresh instances (avoid deepcopy issues with weight_norm)
        teacher_backbone = self._create_teacher_backbone(backbone)
        teacher_head = DINOHead(**head_config)
        self.teacher = TeacherNetwork(
            teacher_backbone,
            teacher_head,
            center_momentum=center_momentum,
        )
        
        # Copy student weights to teacher
        self._copy_student_to_teacher()
        
        # Disable gradients for teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
    
    def _create_teacher_backbone(self, student_backbone: BaseBackbone) -> BaseBackbone:
        """
        Create a fresh teacher backbone with the same architecture.
        
        Avoids deepcopy issues by creating new instance with same config.
        """
        # Get backbone class and recreate
        backbone_cls = student_backbone.__class__
        
        # Extract config from student backbone
        config = {
            'img_size': student_backbone.patch_embed.img_size,
            'patch_size': student_backbone.patch_size,
        }
        
        return backbone_cls(**config)
    
    @torch.no_grad()
    def _copy_student_to_teacher(self) -> None:
        """Copy student weights to teacher."""
        for teacher_param, student_param in zip(
            self.teacher.parameters(),
            self.student.parameters()
        ):
            teacher_param.data.copy_(student_param.data)
    
    @property
    def backbone(self) -> BaseBackbone:
        """Get the student backbone (used for evaluation)."""
        return self.student.backbone
    
    def forward(
        self, 
        views: List[torch.Tensor],
        n_global_crops: int = 2,
        return_l2g_inputs: bool = True,
        return_reg_inputs: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for DINO training.
        
        Processes multiple views through student and teacher networks.
        Only global crops go through teacher, all crops go through student.
        
        Args:
            views: List of augmented views [global_1, global_2, local_1, ...]
                   Global crops: larger crops (e.g., 224x224 or 96x96)
                   Local crops: smaller crops (e.g., 96x96 or 48x48)
            n_global_crops: Number of global crops (default: 2)
            return_l2g_inputs: If True, return patch tokens for L2G loss
            return_reg_inputs: If True, return student features for regularization
            
        Returns:
            Dictionary with:
                - 'student_outputs': List of student outputs for all views
                - 'teacher_outputs': List of teacher outputs for global views
                - 'teacher_temp': Teacher temperature
                - 'student_temp': Student temperature
                - 'center': Current center vector
                - 'student_patch_tokens': (optional) List of patch tokens for local crops
                - 'teacher_cls_features': (optional) List of teacher CLS features
                - 'student_features': (optional) Concatenated student features for regularization
        """
        # Separate global and local crops
        global_crops = views[:n_global_crops]
        local_crops = views[n_global_crops:]
        all_crops = views
        
        # Teacher forward (only global crops, no gradient)
        teacher_outputs = []
        teacher_cls_features = []
        with torch.no_grad():
            for crop in global_crops:
                features = self.teacher.backbone(crop)
                output = self.teacher.head(features)
                teacher_outputs.append(output)
                if return_l2g_inputs:
                    teacher_cls_features.append(output)  # Already has head projection
            
            # Stack and update center
            teacher_stacked = torch.cat(teacher_outputs, dim=0)
            self.teacher.update_center(teacher_stacked)
        
        # Student forward (all crops)
        student_outputs = []
        student_features_list = []
        student_patch_tokens = []
        
        for i, crop in enumerate(all_crops):
            is_local = i >= n_global_crops
            
            # For local crops, get patch tokens for L2G loss
            if is_local and return_l2g_inputs:
                output, features, patch_tokens = self.student(
                    crop, 
                    return_features=return_reg_inputs,
                    return_patch_tokens=True
                )
                if patch_tokens is not None:
                    student_patch_tokens.append(patch_tokens)
            else:
                output, features, _ = self.student(
                    crop, 
                    return_features=return_reg_inputs,
                    return_patch_tokens=False
                )
            
            student_outputs.append(output)
            
            if return_reg_inputs and features is not None:
                student_features_list.append(features)
        
        # Build output dictionary
        result = {
            'student_outputs': student_outputs,
            'teacher_outputs': teacher_outputs,
            'center': self.teacher.center,
            'teacher_temp': self.teacher_temp,
            'student_temp': self.student_temp,
        }
        
        # Add optional outputs for extended loss
        if return_l2g_inputs and len(student_patch_tokens) > 0:
            result['student_patch_tokens'] = student_patch_tokens
            result['teacher_cls_features'] = teacher_outputs  # Use head outputs
        
        if return_reg_inputs and len(student_features_list) > 0:
            # Concatenate all student features for regularization
            result['student_features'] = torch.cat(student_features_list, dim=0)
        
        return result
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features using the student backbone.
        
        Used for evaluation (KNN, linear probe).
        
        Args:
            x: Input images (B, C, H, W)
            
        Returns:
            Feature embeddings (B, embed_dim)
        """
        return self.student.backbone(x)
    
    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        """
        Update teacher weights using EMA of student weights.
        
        teacher = momentum * teacher + (1 - momentum) * student
        
        Args:
            momentum: EMA momentum (typically 0.996 → 1.0 over training)
        """
        for teacher_param, student_param in zip(
            self.teacher.parameters(), 
            self.student.parameters()
        ):
            teacher_param.data.mul_(momentum).add_(
                student_param.data, alpha=1 - momentum
            )
    
    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """
        Count parameters.
        
        Note: Teacher parameters don't count toward budget since
        they're just an EMA copy of student.
        """
        if trainable_only:
            return sum(p.numel() for p in self.student.parameters() if p.requires_grad)
        # For total, only count student (teacher is a copy)
        return sum(p.numel() for p in self.student.parameters())
    
    def get_backbone_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get student backbone state dict for evaluation."""
        return self.student.backbone.state_dict()
    
    def get_backbone(self) -> nn.Module:
        """
        Get student backbone for feature extraction.
        
        Returns the backbone module that can be used directly
        for extracting features during evaluation.
        
        Returns:
            The student's backbone network (e.g., ViT)
        """
        return self.student.backbone
    
    def __repr__(self) -> str:
        student_params = sum(p.numel() for p in self.student.parameters()) / 1e6
        return (
            f"DINO(\n"
            f"  student_params={student_params:.2f}M,\n"
            f"  out_dim={self.out_dim},\n"
            f"  teacher_temp={self.teacher_temp},\n"
            f"  student_temp={self.student_temp},\n"
            f"  backbone={self.student.backbone.__class__.__name__},\n"
            f")"
        )
