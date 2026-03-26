"""
Loss functions for SSL methods.

Available losses:
- DINOLoss: Complete DINO loss (global-to-global + L2G + regularization)
- EMAMomentumScheduler: Schedule for EMA momentum during training
- LossOutput: Structured output from loss computation
"""

from .dino_loss import DINOLoss, EMAMomentumScheduler, LossOutput

__all__ = [
    "DINOLoss",
    "EMAMomentumScheduler",
    "LossOutput",
]
