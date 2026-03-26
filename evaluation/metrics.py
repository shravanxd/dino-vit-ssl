"""
Metrics for evaluation.

Provides accuracy computation functions for classification tasks.
"""

import torch
from torch import Tensor


def compute_accuracy(
    predictions: Tensor,
    targets: Tensor,
    topk: tuple = (1, 5),
) -> dict:
    """
    Compute top-k accuracy.
    
    Args:
        predictions: (N, num_classes) logits or probabilities
        targets: (N,) ground truth labels
        topk: Tuple of k values to compute accuracy for
        
    Returns:
        Dictionary with top-k accuracies as percentages
        
    Example:
        >>> logits = model(images)  # (batch, num_classes)
        >>> acc = compute_accuracy(logits, labels, topk=(1, 5))
        >>> print(acc)  # {'top1': 75.0, 'top5': 92.5}
    """
    maxk = max(topk)
    batch_size = targets.size(0)
    
    # Get top-k predictions
    _, pred = predictions.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()  # (maxk, N)
    
    # Check correctness
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    
    results = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        results[f'top{k}'] = (correct_k / batch_size * 100).item()
    
    return results


def compute_per_class_accuracy(
    predictions: Tensor,
    targets: Tensor,
    num_classes: int,
) -> dict:
    """
    Compute per-class accuracy (balanced accuracy).
    
    Args:
        predictions: (N,) predicted labels
        targets: (N,) ground truth labels
        num_classes: Number of classes
        
    Returns:
        Dictionary with mean per-class accuracy and per-class breakdown
        
    Example:
        >>> preds = model(images).argmax(dim=1)
        >>> acc = compute_per_class_accuracy(preds, labels, num_classes=200)
        >>> print(acc['mean_per_class_acc'])  # 45.2
    """
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)
    
    for c in range(num_classes):
        mask = targets == c
        if mask.sum() > 0:
            class_correct[c] = (predictions[mask] == targets[mask]).sum().float()
            class_total[c] = mask.sum().float()
    
    # Per-class accuracy
    per_class_acc = class_correct / (class_total + 1e-8) * 100
    
    # Mean class accuracy (balanced - each class weighted equally)
    valid_classes = class_total > 0
    mean_acc = per_class_acc[valid_classes].mean().item()
    
    return {
        'mean_per_class_acc': mean_acc,
        'per_class_acc': per_class_acc.tolist(),
    }
