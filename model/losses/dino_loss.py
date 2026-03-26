"""
DINO Loss: Complete loss implementation for DINO training.

Combines three loss components:
A. Global-to-global distillation (DINO loss) - CLS token distillation
B. Local-to-global alignment (L2G) - Local patch tokens → teacher CLS
C. Feature regularization - Variance/covariance to prevent collapse

Reference:
    "Emerging Properties in Self-Supervised Vision Transformers"
    Caron et al., 2021
    https://arxiv.org/abs/2104.14294
    
    "DINOv2: Learning Robust Visual Features without Supervision"
    Oquab et al., 2023
"""

import math
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossOutput:
    """
    Structured output from DINO loss computation.
    
    Attributes:
        loss: Total combined loss
        dino_loss: Global-to-global distillation loss (A)
        l2g_loss: Local-to-global alignment loss (B)
        reg_loss: Feature regularization loss (C)
        n_loss_terms: Number of cross-entropy terms
        metrics: Additional metrics for logging
    """
    loss: torch.Tensor
    n_loss_terms: int = 0
    dino_loss: Optional[torch.Tensor] = None
    l2g_loss: Optional[torch.Tensor] = None
    reg_loss: Optional[torch.Tensor] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        parts = [f"loss={self.loss.item():.4f}"]
        if self.dino_loss is not None:
            parts.append(f"dino={self.dino_loss.item():.4f}")
        if self.l2g_loss is not None:
            parts.append(f"l2g={self.l2g_loss.item():.4f}")
        if self.reg_loss is not None:
            parts.append(f"reg={self.reg_loss.item():.4f}")
        return f"LossOutput({', '.join(parts)})"


class BaseLoss(nn.Module, ABC):
    """Abstract base class for SSL loss functions."""
    
    @abstractmethod
    def forward(self, outputs: Dict[str, torch.Tensor]) -> LossOutput:
        pass


class DINOLoss(BaseLoss):
    """
    Complete DINO loss with three components.
    
    A. Global-to-global distillation (DINO loss):
       - Student global CLS vs Teacher global CLS
       - Cross-entropy with centering and temperature sharpening
       
    B. Local-to-global alignment (L2G):
       - Student local patch tokens → Teacher global CLS distribution
       - Encourages local patches to capture global semantics
       
    C. Feature regularization:
       - Variance loss to maintain feature diversity
       - Covariance loss to decorrelate features
       - Prevents representation collapse
    
    Args:
        n_global_crops: Number of global crops (default: 2)
        n_local_crops: Number of local crops (default: 6)
        l2g_weight: Weight for local-to-global loss (default: 0.5)
        reg_weight: Weight for regularization loss (default: 0.1)
        use_l2g: Whether to use L2G loss (default: True)
        use_reg: Whether to use regularization (default: True)
        variance_target: Target variance for features (default: 1.0)
        covariance_weight: Weight for covariance penalty (default: 0.04)
    """
    
    def __init__(
        self,
        n_global_crops: int = 2,
        n_local_crops: int = 6,
        l2g_weight: float = 0.5,
        reg_weight: float = 0.1,
        use_l2g: bool = True,
        use_reg: bool = True,
        variance_target: float = 1.0,
        covariance_weight: float = 0.04,
    ):
        super().__init__()
        self.n_global_crops = n_global_crops
        self.n_local_crops = n_local_crops
        self.l2g_weight = l2g_weight
        self.reg_weight = reg_weight
        self.use_l2g = use_l2g
        self.use_reg = use_reg
        self.variance_target = variance_target
        self.covariance_weight = covariance_weight
    
    def forward(self, outputs: Dict[str, torch.Tensor]) -> LossOutput:
        """
        Compute combined DINO loss.
        
        Args:
            outputs: Dictionary containing:
                - 'student_outputs': List of student CLS outputs for all views
                - 'teacher_outputs': List of teacher CLS outputs for global views
                - 'student_patch_tokens': (Optional) List of student patch tokens for L2G
                - 'teacher_cls_features': (Optional) Teacher CLS features for L2G
                - 'student_features': (Optional) Student features for regularization
                - 'center': Current center vector
                - 'teacher_temp': Teacher temperature
                - 'student_temp': Student temperature
                
        Returns:
            LossOutput with all loss components
        """
        # A. Global-to-global distillation loss
        dino_loss, n_terms, dino_metrics = self._compute_dino_loss(
            student_outputs=outputs['student_outputs'],
            teacher_outputs=outputs['teacher_outputs'],
            center=outputs['center'],
            teacher_temp=outputs['teacher_temp'],
            student_temp=outputs['student_temp'],
        )
        
        total_loss = dino_loss
        metrics = dino_metrics
        
        # B. Local-to-global alignment loss
        l2g_loss = None
        if self.use_l2g and 'student_patch_tokens' in outputs and 'teacher_cls_features' in outputs:
            l2g_loss, l2g_metrics = self._compute_l2g_loss(
                student_patch_tokens=outputs['student_patch_tokens'],
                teacher_cls_features=outputs['teacher_cls_features'],
                center=outputs['center'],
                teacher_temp=outputs['teacher_temp'],
                student_temp=outputs['student_temp'],
            )
            total_loss = total_loss + self.l2g_weight * l2g_loss
            metrics.update(l2g_metrics)
        
        # C. Feature regularization loss
        reg_loss = None
        if self.use_reg and 'student_features' in outputs:
            reg_loss, reg_metrics = self._compute_regularization_loss(
                student_features=outputs['student_features'],
            )
            total_loss = total_loss + self.reg_weight * reg_loss
            metrics.update(reg_metrics)
        
        return LossOutput(
            loss=total_loss,
            dino_loss=dino_loss,
            l2g_loss=l2g_loss,
            reg_loss=reg_loss,
            n_loss_terms=n_terms,
            metrics=metrics,
        )
    
    def _compute_dino_loss(
        self,
        student_outputs: List[torch.Tensor],
        teacher_outputs: List[torch.Tensor],
        center: torch.Tensor,
        teacher_temp: float,
        student_temp: float,
    ) -> Tuple[torch.Tensor, int, Dict[str, float]]:
        """
        A. Global-to-global distillation loss.
        
        Cross-entropy between student and teacher CLS token distributions.
        Teacher: softmax((output - center) / teacher_temp)
        Student: log_softmax(output / student_temp)
        """
        total_loss = 0.0
        n_loss_terms = 0
        teacher_entropies = []
        student_entropies = []
        
        # Diagnostic metrics for collapse detection
        teacher_output_norms = []
        student_output_norms = []
        teacher_center_dists = []
        
        for t_idx, teacher_out in enumerate(teacher_outputs):
            # Track teacher output statistics
            teacher_output_norms.append(teacher_out.norm(dim=-1).mean().item())
            teacher_center_dists.append((teacher_out - center).norm(dim=-1).mean().item())
            
            # Teacher probabilities (sharpened + centered, detached)
            teacher_probs = self._compute_teacher_probs(
                teacher_out.detach(), center, teacher_temp
            )
            teacher_entropies.append(self._entropy(teacher_probs).mean().item())
            
            for s_idx, student_out in enumerate(student_outputs):
                # Skip same view for global crops
                if s_idx < self.n_global_crops and s_idx == t_idx:
                    continue
                
                # Track student output statistics (only once per student view)
                if t_idx == 0:
                    student_output_norms.append(student_out.norm(dim=-1).mean().item())
                
                # Student log probabilities (sharpened) - with numerical stability
                # Cast to float32 for stability in mixed precision
                student_out_fp32 = student_out.float()
                student_scaled = student_out_fp32 / student_temp
                student_scaled = student_scaled - student_scaled.max(dim=-1, keepdim=True).values
                student_log_probs = F.log_softmax(student_scaled, dim=-1)
                
                if t_idx == 0:
                    student_probs = F.softmax(student_scaled, dim=-1)
                    student_entropies.append(self._entropy(student_probs).mean().item())
                
                # Cross-entropy (teacher_probs is already fp32 from _compute_teacher_probs)
                loss = -torch.sum(teacher_probs.float() * student_log_probs, dim=-1).mean()
                total_loss += loss
                n_loss_terms += 1
        
        if n_loss_terms > 0:
            total_loss = total_loss / n_loss_terms
        
        # Compute max entropy for reference (log(out_dim) = collapse indicator)
        out_dim = teacher_outputs[0].shape[-1]
        max_entropy = math.log(out_dim)
        
        avg_teacher_entropy = sum(teacher_entropies) / len(teacher_entropies) if teacher_entropies else 0
        avg_student_entropy = sum(student_entropies) / len(student_entropies) if student_entropies else 0
        
        metrics = {
            # Core metrics
            'dino/teacher_entropy': avg_teacher_entropy,
            'dino/student_entropy': avg_student_entropy,
            'dino/n_terms': n_loss_terms,
            
            # Collapse indicators (entropy ratio: 1.0 = collapsed, <0.8 = healthy)
            'dino/teacher_entropy_ratio': avg_teacher_entropy / max_entropy if max_entropy > 0 else 0,
            'dino/student_entropy_ratio': avg_student_entropy / max_entropy if max_entropy > 0 else 0,
            'dino/max_entropy': max_entropy,
            
            # Output statistics (should vary, not be constant)
            'dino/teacher_output_norm': sum(teacher_output_norms) / len(teacher_output_norms) if teacher_output_norms else 0,
            'dino/student_output_norm': sum(student_output_norms) / len(student_output_norms) if student_output_norms else 0,
            'dino/teacher_center_dist': sum(teacher_center_dists) / len(teacher_center_dists) if teacher_center_dists else 0,
            
            # Center statistics
            'dino/center_norm': center.norm().item(),
        }
        
        return total_loss, n_loss_terms, metrics
    
    def _compute_l2g_loss(
        self,
        student_patch_tokens: List[torch.Tensor],
        teacher_cls_features: List[torch.Tensor],
        center: torch.Tensor,
        teacher_temp: float,
        student_temp: float,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        B. Local-to-global alignment loss.
        
        Aligns student local patch tokens to teacher global CLS distribution.
        Pushes local patches to capture global semantics.
        
        Args:
            student_patch_tokens: List of (B, N_patches, out_dim) for local crops
            teacher_cls_features: List of (B, out_dim) teacher CLS outputs
        """
        total_loss = 0.0
        n_terms = 0
        
        # Get teacher target distribution (average over global views)
        # Keep in float32 throughout for numerical stability
        teacher_probs_list = []
        for teacher_cls in teacher_cls_features:
            teacher_probs = self._compute_teacher_probs(
                teacher_cls.detach(), center, teacher_temp
            )
            teacher_probs_list.append(teacher_probs.float())  # Keep in float32
        
        # Average teacher distribution (in float32)
        teacher_probs_avg = torch.stack(teacher_probs_list).mean(dim=0)  # (B, out_dim)
        
        # Check for NaN in teacher_probs_avg
        if torch.isnan(teacher_probs_avg).any():
            print(f"[L2G DEBUG] NaN in teacher_probs_avg!")
            for i, tp in enumerate(teacher_probs_list):
                if torch.isnan(tp).any():
                    print(f"  teacher_probs[{i}] has NaN, min={tp.min().item():.6f}, max={tp.max().item():.6f}")
        
        # For each local crop's patch tokens
        for patch_idx, patch_tokens in enumerate(student_patch_tokens):
            # patch_tokens: (B, N_patches, out_dim)
            B, N, D = patch_tokens.shape
            
            # Check for NaN in input patch tokens
            if torch.isnan(patch_tokens).any():
                print(f"[L2G DEBUG] NaN in patch_tokens[{patch_idx}]!")
                continue
            
            # Reshape to (B*N, D) for batch processing
            patch_tokens_flat = patch_tokens.reshape(B * N, D)
            
            # Student log probs for each patch (with numerical stability)
            # Cast to float32 for stability in mixed precision
            patch_tokens_fp32 = patch_tokens_flat.float()
            student_scaled = patch_tokens_fp32 / student_temp
            student_scaled = student_scaled - student_scaled.max(dim=-1, keepdim=True).values
            student_log_probs = F.log_softmax(student_scaled, dim=-1)
            
            # Check for NaN/Inf in student_log_probs
            if torch.isnan(student_log_probs).any() or torch.isinf(student_log_probs).any():
                print(f"[L2G DEBUG] NaN/Inf in student_log_probs[{patch_idx}]!")
                print(f"  patch_tokens norm: {patch_tokens_flat.norm(dim=-1).mean().item():.4f}")
                print(f"  student_scaled min: {student_scaled.min().item():.4f}, max: {student_scaled.max().item():.4f}")
                continue
            
            # Expand teacher target for all patches: (B, D) -> (B*N, D)
            # teacher_probs_avg is already float32
            teacher_target = teacher_probs_avg.unsqueeze(1).expand(B, N, D).reshape(B * N, D)
            
            # Cross-entropy loss (ensure both in float32)
            loss = -torch.sum(teacher_target.float() * student_log_probs, dim=-1).mean()
            
            # Check for NaN in loss
            if torch.isnan(loss):
                print(f"[L2G DEBUG] NaN in loss for patch[{patch_idx}]!")
                print(f"  teacher_target: min={teacher_target.min().item():.6f}, max={teacher_target.max().item():.6f}, sum={teacher_target.sum(dim=-1).mean().item():.6f}")
                print(f"  student_log_probs: min={student_log_probs.min().item():.4f}, max={student_log_probs.max().item():.4f}")
                # Check for -inf in log_probs multiplied by non-zero teacher_target
                product = teacher_target * student_log_probs
                print(f"  product has inf: {torch.isinf(product).any().item()}, nan: {torch.isnan(product).any().item()}")
                continue
            
            total_loss += loss
            n_terms += 1
        
        if n_terms > 0:
            total_loss = total_loss / n_terms
        else:
            # All terms had NaN, return zero loss to avoid crashing
            total_loss = torch.tensor(0.0, device=center.device, dtype=torch.float32)
        
        metrics = {
            'l2g/n_terms': n_terms,
            'l2g/loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
        }
        
        return total_loss, metrics
    
    def _compute_regularization_loss(
        self,
        student_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        C. Feature regularization loss (VICReg-style).
        
        Combines:
        - Variance loss: Keep feature variance close to target
        - Covariance loss: Decorrelate different feature dimensions
        
        Args:
            student_features: (B, D) student backbone features
        """
        # Center features
        features = student_features - student_features.mean(dim=0)
        
        # Variance loss: penalize if variance is below target
        std = torch.sqrt(features.var(dim=0) + 1e-4)
        variance_loss = F.relu(self.variance_target - std).mean()
        
        # Covariance loss: penalize off-diagonal correlations
        B, D = features.shape
        cov = (features.T @ features) / (B - 1)  # (D, D)
        
        # Zero out diagonal (we only want to penalize off-diagonal)
        off_diag_mask = ~torch.eye(D, dtype=torch.bool, device=cov.device)
        cov_loss = cov[off_diag_mask].pow(2).mean()
        
        total_reg_loss = variance_loss + self.covariance_weight * cov_loss
        
        metrics = {
            'reg/variance_loss': variance_loss.item(),
            'reg/covariance_loss': cov_loss.item(),
            'reg/feature_std': std.mean().item(),
        }
        
        return total_reg_loss, metrics
    
    def _compute_teacher_probs(
        self,
        teacher_out: torch.Tensor,
        center: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """Compute centered and sharpened teacher probabilities with numerical stability."""
        # Cast to float32 for numerical stability (critical for AMP/mixed precision)
        # fp16 can overflow during exp() even with max subtraction
        orig_dtype = teacher_out.dtype
        teacher_out = teacher_out.float()
        center = center.float()
        
        centered = teacher_out - center
        scaled = centered / temperature
        
        # Numerical stability: subtract max before softmax to prevent overflow
        # This doesn't change the result since softmax is shift-invariant
        scaled = scaled - scaled.max(dim=-1, keepdim=True).values
        
        probs = F.softmax(scaled, dim=-1)
        
        # Cast back to original dtype
        return probs.to(orig_dtype)
    
    def _entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute entropy: -sum(p * log(p)) with numerical stability."""
        # Cast to float32 for numerical stability (fp16 log can produce -inf)
        probs_fp32 = probs.float()
        eps = 1e-8
        # Clamp to avoid log(0)
        probs_clamped = probs_fp32.clamp(min=eps)
        return -torch.sum(probs_fp32 * torch.log(probs_clamped), dim=-1)
    
    def __repr__(self) -> str:
        return (
            f"DINOLoss(\n"
            f"  n_global_crops={self.n_global_crops},\n"
            f"  n_local_crops={self.n_local_crops},\n"
            f"  use_l2g={self.use_l2g}, l2g_weight={self.l2g_weight},\n"
            f"  use_reg={self.use_reg}, reg_weight={self.reg_weight},\n"
            f")"
        )


class EMAMomentumScheduler:
    """
    Scheduler for EMA momentum that ramps from start to end.
    
    DINO typically uses momentum starting at 0.996 and ramping to 1.0.
    
    Args:
        base_momentum: Starting momentum (default: 0.996)
        final_momentum: Final momentum (default: 1.0)
        total_steps: Total training steps
        schedule: 'cosine' or 'linear' (default: 'cosine')
    """
    
    def __init__(
        self,
        base_momentum: float = 0.996,
        final_momentum: float = 1.0,
        total_steps: int = 100000,
        schedule: str = 'cosine',
    ):
        self.base_momentum = base_momentum
        self.final_momentum = final_momentum
        self.total_steps = total_steps
        self.schedule = schedule
    
    def get_momentum(self, step: int) -> float:
        """Get momentum value for current step."""
        if step >= self.total_steps:
            return self.final_momentum
        
        progress = step / self.total_steps
        
        if self.schedule == 'cosine':
            # Cosine schedule from base to final
            return self.final_momentum - (self.final_momentum - self.base_momentum) * (
                1 + math.cos(math.pi * progress)
            ) / 2
        else:  # linear
            return self.base_momentum + (self.final_momentum - self.base_momentum) * progress
    
    def __repr__(self) -> str:
        return (
            f"EMAMomentumScheduler(base={self.base_momentum}, "
            f"final={self.final_momentum}, schedule={self.schedule})"
        )
