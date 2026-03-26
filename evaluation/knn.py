"""
K-Nearest Neighbors evaluation for self-supervised learning.

Provides KNN classification to evaluate representation quality
without any training.
"""

from typing import Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from .metrics import compute_accuracy, compute_per_class_accuracy
from .datasets import create_eval_dataloaders


class KNNEvaluator:
    """
    K-Nearest Neighbors evaluator for self-supervised representations.
    
    Performs weighted KNN classification on extracted features to
    evaluate representation quality without any training.
    
    Algorithm:
        1. Extract features from train set using backbone
        2. Extract features from val set using backbone
        3. For each val sample, find K nearest neighbors in train set
        4. Predict class via weighted voting (weight = softmax(sim/τ))
        5. Compute top-1 and top-5 accuracy
    
    Args:
        k: Number of neighbors (default: 20)
        temperature: Temperature for distance weighting (default: 0.07)
        device: Device to run evaluation on
        
    Example:
        >>> evaluator = KNNEvaluator(k=20, temperature=0.07, device='cuda')
        >>> results = evaluator.evaluate_model(
        ...     model=backbone,
        ...     train_dataloader=train_loader,
        ...     val_dataloader=val_loader,
        ... )
        >>> print(f"Top-1: {results['top1']:.2f}%")
    """
    
    def __init__(
        self,
        k: int = 20,
        temperature: float = 0.07,
        device: str = 'cpu',
    ):
        self.k = k
        self.temperature = temperature
        self.device = device
    
    @torch.no_grad()
    def extract_features(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        use_cls_token: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """
        Extract features from a model.
        
        Args:
            model: Feature extractor (backbone)
            dataloader: DataLoader with (image, label) pairs
            use_cls_token: Use CLS token (True) or global avg pool (False)
            
        Returns:
            Tuple of (features, labels) tensors
            - features: (N, embed_dim) normalized feature vectors
            - labels: (N,) integer labels
        """
        model.eval()
        features_list = []
        labels_list = []
        
        for images, labels in dataloader:
            images = images.to(self.device)
            
            # Get features from backbone
            if hasattr(model, 'forward_features'):
                # ViT-style: returns (cls_token, patch_tokens)
                cls_token, patch_tokens = model.forward_features(images)
                if use_cls_token:
                    features = cls_token
                else:
                    # Global average pool of patch tokens
                    features = patch_tokens.mean(dim=1)
            else:
                # Standard forward (e.g., ResNet)
                features = model(images)
            
            # L2 normalize features
            features = F.normalize(features, dim=1)
            
            features_list.append(features.cpu())
            labels_list.append(labels)
        
        features = torch.cat(features_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        
        return features, labels
    
    def evaluate(
        self,
        train_features: Tensor,
        train_labels: Tensor,
        test_features: Tensor,
        test_labels: Tensor,
        num_classes: Optional[int] = None,
    ) -> dict:
        """
        Perform KNN classification.
        
        Args:
            train_features: (N, D) normalized training features
            train_labels: (N,) training labels
            test_features: (M, D) normalized test features
            test_labels: (M,) test labels
            num_classes: Number of classes (inferred if None)
            
        Returns:
            Dictionary with evaluation metrics:
            - top1: Top-1 accuracy (%)
            - top5: Top-5 accuracy (%)
            - mean_per_class_acc: Mean per-class accuracy (%)
        """
        if num_classes is None:
            num_classes = max(train_labels.max(), test_labels.max()).item() + 1
        
        # Move to device
        train_features = train_features.to(self.device)
        train_labels = train_labels.to(self.device)
        test_features = test_features.to(self.device)
        test_labels = test_labels.to(self.device)
        
        # Process in batches to save memory
        batch_size = 256
        num_test = test_features.size(0)
        all_predictions = []
        all_probs = []
        
        for i in range(0, num_test, batch_size):
            batch_features = test_features[i:i + batch_size]
            
            # Compute cosine similarity (features are normalized)
            similarity = batch_features @ train_features.t()  # (batch, N)
            
            # Get top-k neighbors
            topk_sim, topk_idx = similarity.topk(self.k, dim=1)  # (batch, k)
            topk_labels = train_labels[topk_idx]  # (batch, k)
            
            # Weighted voting with temperature
            weights = F.softmax(topk_sim / self.temperature, dim=1)  # (batch, k)
            
            # Accumulate votes per class
            votes = torch.zeros(batch_features.size(0), num_classes, device=self.device)
            for j in range(self.k):
                votes.scatter_add_(
                    1,
                    topk_labels[:, j:j+1],
                    weights[:, j:j+1],
                )
            
            all_probs.append(votes.cpu())
            all_predictions.append(votes.argmax(dim=1).cpu())
        
        predictions = torch.cat(all_predictions, dim=0)
        probs = torch.cat(all_probs, dim=0)
        test_labels_cpu = test_labels.cpu()
        
        # Compute metrics
        accuracy = compute_accuracy(probs, test_labels_cpu, topk=(1, 5))
        per_class = compute_per_class_accuracy(predictions, test_labels_cpu, num_classes)
        
        return {
            **accuracy,
            **per_class,
        }
    
    @torch.no_grad()
    def evaluate_model(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        num_classes: Optional[int] = None,
    ) -> dict:
        """
        Full KNN evaluation pipeline.
        
        This is the main entry point for evaluation. It:
        1. Extracts features from train set
        2. Extracts features from val set
        3. Runs KNN classification
        4. Returns accuracy metrics
        
        Args:
            model: Feature extractor (backbone)
            train_dataloader: DataLoader for training set
            val_dataloader: DataLoader for validation set
            num_classes: Number of classes (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Extract features
        print("    Extracting train features...")
        train_features, train_labels = self.extract_features(model, train_dataloader)
        print(f"    Train: {train_features.shape[0]} samples, {train_features.shape[1]}-dim")
        
        print("    Extracting val features...")
        val_features, val_labels = self.extract_features(model, val_dataloader)
        print(f"    Val: {val_features.shape[0]} samples")
        
        # Run KNN
        print(f"    Running KNN (k={self.k})...")
        results = self.evaluate(
            train_features,
            train_labels,
            val_features,
            val_labels,
            num_classes,
        )
        
        # Remove per-class list for cleaner output
        results.pop('per_class_acc', None)
        
        return results


def run_knn_evaluation(
    model: nn.Module,
    dataset_name: str,
    data_root: str,
    img_size: int = 96,
    k: int = 20,
    temperature: float = 0.07,
    batch_size: int = 64,
    num_workers: int = 4,
    device: str = 'cpu',
) -> dict:
    """
    Convenience function to run KNN evaluation on a dataset.
    
    Args:
        model: Backbone model for feature extraction
        dataset_name: Name of dataset (e.g., 'cub200')
        data_root: Root directory for eval datasets
        img_size: Image size for evaluation
        k: Number of KNN neighbors
        temperature: Temperature for weighting
        batch_size: Batch size for feature extraction
        num_workers: Dataloader workers
        device: Device to run on
        
    Returns:
        Dictionary with evaluation metrics
        
    Example:
        >>> results = run_knn_evaluation(
        ...     model=backbone,
        ...     dataset_name="cub200",
        ...     data_root="data/eval_public",
        ...     device="cuda",
        ... )
        >>> print(f"CUB-200 Top-1: {results['top1']:.2f}%")
    """
    # Create dataloaders
    train_loader, val_loader, num_classes = create_eval_dataloaders(
        dataset_name=dataset_name,
        data_root=data_root,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    
    # Create evaluator
    evaluator = KNNEvaluator(
        k=k,
        temperature=temperature,
        device=device,
    )
    
    # Run evaluation
    results = evaluator.evaluate_model(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        num_classes=num_classes,
    )
    
    return results
