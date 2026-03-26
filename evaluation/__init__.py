"""
Evaluation utilities for self-supervised learning.

This module provides tools to evaluate learned representations
on downstream tasks without fine-tuning.

Main components:
- KNNEvaluator: K-nearest neighbors classification
- LabeledDataset: Dataset for labeled eval data
- Metrics: Accuracy computation functions

Example:
    >>> from evaluation import run_knn_evaluation
    >>> 
    >>> results = run_knn_evaluation(
    ...     model=backbone,
    ...     dataset_name="cub200",
    ...     data_root="data/eval_public",
    ...     device="cuda",
    ... )
    >>> print(f"Top-1: {results['top1']:.2f}%")
"""

from .metrics import compute_accuracy, compute_per_class_accuracy
from .datasets import LabeledDataset, create_eval_dataloaders, get_eval_transform
from .knn import KNNEvaluator, run_knn_evaluation

__all__ = [
    # Evaluators
    "KNNEvaluator",
    "run_knn_evaluation",
    # Datasets
    "LabeledDataset",
    "create_eval_dataloaders",
    "get_eval_transform",
    # Metrics
    "compute_accuracy",
    "compute_per_class_accuracy",
]
