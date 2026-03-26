"""
Tuned KNN Evaluation Script for Competition
============================================

This script evaluates a checkpoint with hyperparameter tuning and feature tricks
to maximize KNN accuracy on downstream tasks.

Features:
1. Hyperparameter sweep (k, temperature)
2. CLS + patch token combination
3. Multi-k ensemble
4. Feature normalization variants

Usage:
    python scripts/evaluate_checkpoint_tuned.py \
        --checkpoint outputs/dino_vit_base/checkpoint_epoch0159.pt \
        --dataset cub200 \
        --data_root data/eval_public \
        --device cuda
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from itertools import product

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model.backbones import ViTTiny, ViTSmall, ViTBase
from evaluation.datasets import create_eval_dataloaders, LabeledDataset, get_eval_transform


def create_eval_dataloaders_from_path(
    dataset_path: str,
    img_size: int = 96,
    batch_size: int = 64,
    num_workers: int = 4,
):
    """Create dataloaders from a direct dataset path."""
    from pathlib import Path
    from torch.utils.data import DataLoader
    
    data_root = Path(dataset_path)
    transform = get_eval_transform(img_size)
    
    train_dataset = LabeledDataset(
        image_dir=data_root / 'train',
        labels_file=data_root / 'train_labels.csv',
        transform=transform,
    )
    
    val_dataset = LabeledDataset(
        image_dir=data_root / 'val',
        labels_file=data_root / 'val_labels.csv',
        transform=transform,
    )
    
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    return train_loader, val_loader, train_dataset.num_classes


class TunedKNNEvaluator:
    """KNN evaluator with hyperparameter tuning and feature tricks."""
    
    def __init__(self, device='cuda'):
        self.device = device
    
    @torch.no_grad()
    def extract_features(
        self,
        model,
        dataloader,
        feature_type='cls',  # 'cls', 'patch_mean', 'concat', 'sum'
    ):
        """
        Extract features with different strategies.
        
        Args:
            model: ViT backbone
            dataloader: DataLoader
            feature_type: How to combine CLS and patch tokens
                - 'cls': Just CLS token (default DINO)
                - 'patch_mean': Mean of patch tokens
                - 'concat': Concatenate CLS + patch mean (2x dim)
                - 'sum': Average of CLS + patch mean
        """
        model.eval()
        features_list = []
        labels_list = []
        
        # Determine if we need patch tokens
        need_patch_tokens = feature_type in ['patch_mean', 'concat', 'sum']
        
        for images, labels in dataloader:
            images = images.to(self.device)
            
            # Get CLS and optionally patch tokens
            cls_token, patch_tokens = model.forward_features(images, return_patch_tokens=need_patch_tokens)
            
            # Combine based on strategy
            if feature_type == 'cls':
                features = cls_token
            elif feature_type == 'patch_mean':
                features = patch_tokens.mean(dim=1)
            elif feature_type == 'concat':
                patch_mean = patch_tokens.mean(dim=1)
                features = torch.cat([cls_token, patch_mean], dim=1)
            elif feature_type == 'sum':
                patch_mean = patch_tokens.mean(dim=1)
                features = (cls_token + patch_mean) / 2
            else:
                raise ValueError(f"Unknown feature_type: {feature_type}")
            
            # L2 normalize
            features = F.normalize(features, dim=1)
            features_list.append(features.cpu())
            labels_list.append(labels)
        
        return torch.cat(features_list), torch.cat(labels_list)
    
    def knn_evaluate(
        self,
        train_features,
        train_labels,
        test_features,
        test_labels,
        k=20,
        temperature=0.07,
        num_classes=None,
    ):
        """Run KNN classification with given hyperparameters."""
        if num_classes is None:
            num_classes = max(train_labels.max(), test_labels.max()).item() + 1
        
        train_features = train_features.to(self.device)
        train_labels = train_labels.to(self.device)
        test_features = test_features.to(self.device)
        test_labels = test_labels.to(self.device)
        
        batch_size = 256
        num_test = test_features.size(0)
        correct = 0
        
        for i in range(0, num_test, batch_size):
            batch_features = test_features[i:i + batch_size]
            batch_labels = test_labels[i:i + batch_size]
            
            # Cosine similarity
            similarity = batch_features @ train_features.t()
            
            # Top-k neighbors
            topk_sim, topk_idx = similarity.topk(k, dim=1)
            topk_labels = train_labels[topk_idx]
            
            # Weighted voting
            weights = F.softmax(topk_sim / temperature, dim=1)
            
            # Accumulate votes
            votes = torch.zeros(batch_features.size(0), num_classes, device=self.device)
            for j in range(k):
                votes.scatter_add_(1, topk_labels[:, j:j+1], weights[:, j:j+1])
            
            predictions = votes.argmax(dim=1)
            correct += (predictions == batch_labels).sum().item()
        
        accuracy = 100.0 * correct / num_test
        return accuracy
    
    def ensemble_knn(
        self,
        train_features,
        train_labels,
        test_features,
        test_labels,
        k_values=[5, 10, 20, 50],
        temperature=0.07,
        num_classes=None,
    ):
        """Ensemble KNN with multiple k values - average probabilities."""
        if num_classes is None:
            num_classes = max(train_labels.max(), test_labels.max()).item() + 1
        
        train_features = train_features.to(self.device)
        train_labels = train_labels.to(self.device)
        test_features = test_features.to(self.device)
        test_labels = test_labels.to(self.device)
        
        batch_size = 256
        num_test = test_features.size(0)
        all_votes = []
        
        for k in k_values:
            batch_votes = []
            for i in range(0, num_test, batch_size):
                batch_features = test_features[i:i + batch_size]
                
                similarity = batch_features @ train_features.t()
                topk_sim, topk_idx = similarity.topk(k, dim=1)
                topk_labels = train_labels[topk_idx]
                
                weights = F.softmax(topk_sim / temperature, dim=1)
                
                votes = torch.zeros(batch_features.size(0), num_classes, device=self.device)
                for j in range(k):
                    votes.scatter_add_(1, topk_labels[:, j:j+1], weights[:, j:j+1])
                
                batch_votes.append(votes.cpu())
            
            all_votes.append(torch.cat(batch_votes, dim=0))
        
        # Average votes across all k values
        ensemble_votes = torch.stack(all_votes).mean(dim=0)
        predictions = ensemble_votes.argmax(dim=1)
        accuracy = 100.0 * (predictions == test_labels.cpu()).sum().item() / num_test
        
        return accuracy


def load_checkpoint(checkpoint_path, device='cuda'):
    """Load model from checkpoint."""
    import math
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Infer model config from state dict
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('student_backbone', checkpoint))
    
    # Clean state dict keys (handle DDP wrapper)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Try to extract backbone from DINO checkpoint structure
    if not any(k.startswith('patch_embed') or k.startswith('blocks') for k in state_dict.keys()):
        # Try different prefixes used in DINO checkpoints
        for prefix in ['_backbone.', 'student.backbone.', 'teacher.backbone.', 'backbone.']:
            backbone_state = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
            if backbone_state and len(backbone_state) > 10:
                state_dict = backbone_state
                print(f"  Extracted backbone weights with prefix '{prefix}'")
                break
    
    # Infer embed_dim from pos_embed or patch_embed weight
    if 'pos_embed' in state_dict:
        embed_dim = state_dict['pos_embed'].shape[-1]
    elif 'patch_embed.proj.weight' in state_dict:
        embed_dim = state_dict['patch_embed.proj.weight'].shape[0]
    else:
        raise ValueError("Could not infer embed_dim from checkpoint")
    
    # Infer patch_size from patch_embed.proj.weight
    if 'patch_embed.proj.weight' in state_dict:
        patch_size = state_dict['patch_embed.proj.weight'].shape[-1]
    else:
        patch_size = 8  # Default
    
    # Infer img_size from pos_embed
    if 'pos_embed' in state_dict:
        num_tokens = state_dict['pos_embed'].shape[1]  # includes cls token
        num_patches = num_tokens - 1
        grid_size = int(math.sqrt(num_patches))
        img_size = grid_size * patch_size
    else:
        img_size = 96  # Default
    
    # Infer depth from number of transformer blocks
    depth = len([k for k in state_dict.keys() if k.startswith('blocks.') and k.endswith('.norm1.weight')])
    
    # Select the appropriate model class based on embed_dim
    if embed_dim == 192:
        print(f"Detected ViT-Tiny: embed_dim={embed_dim}, depth={depth}, patch_size={patch_size}, img_size={img_size}")
        model = ViTTiny(img_size=img_size, patch_size=patch_size)
    elif embed_dim == 384:
        print(f"Detected ViT-Small: embed_dim={embed_dim}, depth={depth}, patch_size={patch_size}, img_size={img_size}")
        model = ViTSmall(img_size=img_size, patch_size=patch_size)
    elif embed_dim == 768:
        print(f"Detected ViT-Base: embed_dim={embed_dim}, depth={depth}, patch_size={patch_size}, img_size={img_size}")
        model = ViTBase(img_size=img_size, patch_size=patch_size)
    else:
        raise ValueError(f"Unknown embed_dim: {embed_dim}. Expected 192, 384, or 768.")
    
    # Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Warning: Missing keys: {missing[:5]}..." if len(missing) > 5 else f"  Warning: Missing keys: {missing}")
    if unexpected:
        print(f"  Warning: Unexpected keys: {unexpected[:5]}..." if len(unexpected) > 5 else f"  Warning: Unexpected keys: {unexpected}")
    
    model = model.to(device)
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Loaded model with {param_count:,} parameters")
    
    return model, img_size


def main():
    parser = argparse.ArgumentParser(description='Tuned KNN Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset', type=str, nargs='+', 
                        default=['cub200', 'miniimagenet', 'sun397'],
                        help='Dataset(s) to evaluate on')
    parser.add_argument('--data_root', type=str, default='data/eval_public',
                        help='Root directory for eval datasets (default structure)')
    parser.add_argument('--cub200_path', type=str, default='data/eval_public/cub200',
                        help='Path to CUB-200 dataset directory')
    parser.add_argument('--miniimagenet_path', type=str, default='data/eval_public/miniimagenet',
                        help='Path to MiniImageNet dataset directory')
    parser.add_argument('--sun397_path', type=str, default='data/eval_public/sun397',
                        help='Path to SUN397 dataset directory')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--quick', action='store_true', help='Quick mode: fewer hyperparams')
    
    args = parser.parse_args()
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Build dataset path mapping
    dataset_paths = {
        'cub200': args.cub200_path,
        'miniimagenet': args.miniimagenet_path,
        'sun397': args.sun397_path,
    }
    
    print("\nDataset paths:")
    for name, path in dataset_paths.items():
        if name in args.dataset:
            print(f"  {name}: {path}")
    
    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, img_size = load_checkpoint(args.checkpoint, device)
    print(f"  Using img_size={img_size} for evaluation")
    
    # Initialize evaluator
    evaluator = TunedKNNEvaluator(device=device)
    
    # Define hyperparameter grid
    if args.quick:
        feature_types = ['cls', 'sum']
        k_values = [10, 20]
        temperatures = [0.07]
    else:
        feature_types = ['cls', 'patch_mean', 'sum', 'concat']
        k_values = [5, 10, 15, 20, 50, 100]
        temperatures = [0.01, 0.05, 0.07, 0.1, 0.2]
    
    # Get list of datasets to evaluate
    datasets = args.dataset  # This is now a list
    
    # Store results for all datasets
    all_dataset_results = {}
    
    for dataset_name in datasets:
        print("\n" + "#"*70)
        print(f"# DATASET: {dataset_name.upper()}")
        print("#"*70)
        
        # Get the path for this dataset
        dataset_path = dataset_paths.get(dataset_name)
        if not dataset_path:
            print(f"  ⚠ Unknown dataset: {dataset_name}")
            continue
        
        # Create dataloaders
        print(f"\nLoading {dataset_name} from: {dataset_path}")
        try:
            train_loader, val_loader, num_classes = create_eval_dataloaders_from_path(
                dataset_path=dataset_path,
                img_size=img_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            print(f"  Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Classes: {num_classes}")
        except Exception as e:
            print(f"  ⚠ Failed to load {dataset_name}: {e}")
            continue
        
        print("\n" + "="*70)
        print("HYPERPARAMETER SWEEP")
        print("="*70)
        print(f"\n{'Feature':<12}{'k':<8}{'Temp':<8}{'Accuracy':<12}{'Status'}")
        print("-" * 50)
        
        best_acc = 0
        best_config = {}
        results = []
        result_count = 0
        
        for feature_type in feature_types:
            # Extract features once per feature type
            print(f"\n[Extracting {feature_type} features...]")
            train_features, train_labels = evaluator.extract_features(model, train_loader, feature_type)
            val_features, val_labels = evaluator.extract_features(model, val_loader, feature_type)
            print(f"[Feature dim: {train_features.shape[1]}]\n")
            
            for k, temp in product(k_values, temperatures):
                acc = evaluator.knn_evaluate(
                    train_features, train_labels,
                    val_features, val_labels,
                    k=k, temperature=temp, num_classes=num_classes,
                )
                
                result_count += 1
                results.append({
                    'feature_type': feature_type,
                    'k': k,
                    'temperature': temp,
                    'accuracy': acc,
                })
                
                # Check if new best
                is_new_best = acc > best_acc
                if is_new_best:
                    best_acc = acc
                    best_config = {'feature_type': feature_type, 'k': k, 'temperature': temp}
                    status = "⭐ NEW BEST!"
                else:
                    status = ""
                
                print(f"{feature_type:<12}{k:<8}{temp:<8.2f}{acc:<12.2f}%{status}")
        
        # Try ensemble
        print("\n" + "-"*50)
        print("ENSEMBLE KNN (averaging multiple k values: [5,10,20,50])")
        print("-"*50)
        print(f"{'Feature':<12}{'k':<10}{'Temp':<8}{'Accuracy':<12}{'Status'}")
        
        for feature_type in feature_types:
            train_features, train_labels = evaluator.extract_features(model, train_loader, feature_type)
            val_features, val_labels = evaluator.extract_features(model, val_loader, feature_type)
            
            ensemble_acc = evaluator.ensemble_knn(
                train_features, train_labels,
                val_features, val_labels,
                k_values=[5, 10, 20, 50],
                temperature=0.07,
                num_classes=num_classes,
            )
            
            results.append({
                'feature_type': feature_type,
                'k': 'ensemble',
                'temperature': 0.07,
                'accuracy': ensemble_acc,
                'ensemble': True,
            })
            
            # Check if new best
            is_new_best = ensemble_acc > best_acc
            if is_new_best:
                best_acc = ensemble_acc
                best_config = {'feature_type': feature_type, 'ensemble': True, 'k_values': [5, 10, 20, 50]}
                status = "⭐ NEW BEST!"
            else:
                status = ""
            
            print(f"{feature_type:<12}{'ensemble':<10}{0.07:<8.2f}{ensemble_acc:<12.2f}%{status}")
        
        # Print ALL results sorted for this dataset
        print("\n" + "="*70)
        print(f"ALL RESULTS FOR {dataset_name.upper()} (SORTED BY ACCURACY)")
        print("="*70)
        
        sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
        
        print(f"\n{'Rank':<6}{'Feature Type':<15}{'k':<10}{'Temp':<10}{'Accuracy':<10}")
        print("-" * 55)
        for i, r in enumerate(sorted_results, 1):
            k_str = str(r['k']) if not r.get('ensemble') else 'ensemble'
            print(f"{i:<6}{r['feature_type']:<15}{k_str:<10}{r['temperature']:<10.2f}{r['accuracy']:<10.2f}%")
        
        # Get default accuracy for comparison
        default_acc = next((r['accuracy'] for r in results if r['feature_type'] == 'cls' and r['k'] == 20 and r['temperature'] == 0.07), None)
        
        # Store results for this dataset
        all_dataset_results[dataset_name] = {
            'best_acc': best_acc,
            'best_config': best_config,
            'default_acc': default_acc,
            'all_results': sorted_results,
        }
        
        # Summary for this dataset
        print("\n" + "="*70)
        print(f"BEST CONFIG FOR {dataset_name.upper()}")
        print("="*70)
        print(f"\n  Feature Type: {best_config.get('feature_type', 'N/A')}")
        if best_config.get('ensemble'):
            print(f"  Ensemble k values: {best_config.get('k_values')}")
        else:
            print(f"  k: {best_config.get('k')}")
            print(f"  Temperature: {best_config.get('temperature')}")
        print(f"\n  *** BEST ACCURACY: {best_acc:.2f}% ***")
        
        if default_acc:
            improvement = best_acc - default_acc
            print(f"\n  Default (cls, k=20, T=0.07): {default_acc:.2f}%")
            print(f"  Improvement from tuning: +{improvement:.2f}%")
    
    # =========================================
    # FINAL SUMMARY ACROSS ALL DATASETS
    # =========================================
    print("\n" + "#"*70)
    print("#" + " "*25 + "FINAL SUMMARY" + " "*30 + "#")
    print("#"*70)
    
    print(f"\n{'Dataset':<15}{'Default Acc':<15}{'Best Acc':<15}{'Improvement':<15}{'Best Config'}")
    print("-" * 90)
    
    for dataset_name, data in all_dataset_results.items():
        default = data['default_acc'] or 0
        best = data['best_acc']
        improvement = best - default if default else 0
        config = data['best_config']
        
        if config.get('ensemble'):
            config_str = f"{config['feature_type']}, ensemble"
        else:
            config_str = f"{config.get('feature_type')}, k={config.get('k')}, T={config.get('temperature')}"
        
        print(f"{dataset_name:<15}{default:<15.2f}{best:<15.2f}+{improvement:<14.2f}{config_str}")
    
    # Average improvement
    if all_dataset_results:
        avg_default = sum(d['default_acc'] or 0 for d in all_dataset_results.values()) / len(all_dataset_results)
        avg_best = sum(d['best_acc'] for d in all_dataset_results.values()) / len(all_dataset_results)
        avg_improvement = avg_best - avg_default
        
        print("-" * 90)
        print(f"{'AVERAGE':<15}{avg_default:<15.2f}{avg_best:<15.2f}+{avg_improvement:<14.2f}")
    
    print("\n" + "#"*70)
    print("# Use the best config for each dataset in your final submission!")
    print("#"*70 + "\n")


if __name__ == "__main__":
    main()
