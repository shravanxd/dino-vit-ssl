#!/usr/bin/env python3
"""
Evaluate a trained checkpoint on all datasets.

Evaluates on:
1. Validation set (with accuracy metrics) for CUB-200, Mini-ImageNet, SUN397
2. Test set (generates submission CSVs) for all 3 datasets

Usage:
    python scripts/evaluate_checkpoint.py \
        --checkpoint outputs/dino_vit_small/checkpoints/backbone_final.pt \
        --data_root data/eval_public \
        --output_dir submissions \
        --device cuda

    # Or with a full checkpoint (not just backbone)
    python scripts/evaluate_checkpoint.py \
        --checkpoint outputs/dino_vit_small/checkpoints/checkpoint_epoch0100.pt \
        --data_root data/eval_public \
        --output_dir submissions \
        --device cuda
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model.backbones.vit import ViTSmall, ViTBase, ViTTiny
from evaluation.knn import run_knn_evaluation, KNNEvaluator
from evaluation.datasets import get_eval_transform


# ============================================================================
#                          DATASET CLASSES
# ============================================================================

class EvalDataset(Dataset):
    """Dataset for evaluation with labels."""
    
    def __init__(self, image_dir: Path, labels_file: Path, transform, img_size: int = 96):
        self.image_dir = image_dir
        self.transform = transform
        self.samples = []
        
        # Load labels from CSV
        with open(labels_file, 'r') as f:
            header = f.readline()  # Skip header
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    filename = parts[0]
                    label = int(parts[1])
                    img_path = self._find_image(filename)
                    if img_path:
                        self.samples.append((img_path, label, filename))
        
        print(f"  Loaded {len(self.samples)} samples from {labels_file.name}")
    
    def _find_image(self, filename: str) -> Optional[Path]:
        """Find image file with various extensions."""
        path = self.image_dir / filename
        if path.exists():
            return path
        for ext in ['.jpg', '.jpeg', '.png', '.JPEG']:
            candidate = self.image_dir / f"{filename}{ext}"
            if candidate.exists():
                return candidate
        return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, filename = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, label, filename


class TestDataset(Dataset):
    """Dataset for test set (no labels)."""
    
    def __init__(self, image_dir: Path, images_file: Path, transform, img_size: int = 96):
        self.image_dir = image_dir
        self.transform = transform
        self.samples = []
        
        # Load image list from CSV
        with open(images_file, 'r') as f:
            header = f.readline()  # Skip header
            for line in f:
                filename = line.strip()
                if filename:
                    img_path = self._find_image(filename)
                    if img_path:
                        self.samples.append((img_path, filename))
        
        print(f"  Loaded {len(self.samples)} test images from {images_file.name}")
    
    def _find_image(self, filename: str) -> Optional[Path]:
        """Find image file with various extensions."""
        path = self.image_dir / filename
        if path.exists():
            return path
        for ext in ['.jpg', '.jpeg', '.png', '.JPEG']:
            candidate = self.image_dir / f"{filename}{ext}"
            if candidate.exists():
                return candidate
        return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, filename = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, filename


def get_eval_transform(img_size: int = 96):
    """Standard evaluation transform."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ============================================================================
#                          MODEL LOADING
# ============================================================================

def load_backbone(checkpoint_path: str, device: str = 'cuda') -> nn.Module:
    """Load backbone from checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Default values
    backbone_name = 'vit_small'
    img_size = 96
    patch_size = 8
    
    # Determine backbone type and load state dict
    if isinstance(checkpoint, dict):
        # Get config if available
        config = checkpoint.get('config', {})
        if isinstance(config, dict):
            backbone_name = config.get('backbone', {}).get('name', 'vit_small')
            img_size = config.get('backbone', {}).get('img_size', 96)
            patch_size = config.get('backbone', {}).get('patch_size', 8)
        
        # Try different key patterns for the backbone state dict
        if 'backbone' in checkpoint:
            # Direct backbone state dict
            state_dict = checkpoint['backbone']
        elif 'model_state_dict' in checkpoint:
            # Full DINO checkpoint with model_state_dict
            full_state = checkpoint['model_state_dict']
            state_dict = {}
            # Try _backbone. prefix first (DINO method wrapper)
            for k, v in full_state.items():
                if k.startswith('_backbone.'):
                    state_dict[k[len('_backbone.'):]] = v
            # If not found, try other prefixes
            if not state_dict:
                for prefix in ['student.backbone.', 'teacher.backbone.', 'backbone.']:
                    for k, v in full_state.items():
                        if k.startswith(prefix):
                            state_dict[k[len(prefix):]] = v
                    if state_dict:
                        break
        elif 'student' in checkpoint:
            # Older DINO checkpoint format
            full_state = checkpoint['student']
            state_dict = {}
            prefix = 'backbone.'
            for k, v in full_state.items():
                if k.startswith(prefix):
                    state_dict[k[len(prefix):]] = v
        else:
            # Try to find backbone keys with various prefixes
            state_dict = {}
            for prefix in ['_backbone.', 'student.backbone.', 'teacher.backbone.', 'backbone.']:
                for k, v in checkpoint.items():
                    if k.startswith(prefix):
                        state_dict[k[len(prefix):]] = v
                if state_dict and len(state_dict) > 10:  # Found meaningful keys
                    break
            
            if not state_dict:
                # Last resort: assume it's just the state dict
                state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Infer backbone type from state dict dimensions
    if 'pos_embed' in state_dict:
        embed_dim = state_dict['pos_embed'].shape[-1]
        if embed_dim == 192:
            backbone_name = 'vit_tiny'
        elif embed_dim == 384:
            backbone_name = 'vit_small'
        elif embed_dim == 768:
            backbone_name = 'vit_base'
    
    # Infer patch_size and img_size from checkpoint
    if 'patch_embed.proj.weight' in state_dict:
        patch_size = state_dict['patch_embed.proj.weight'].shape[-1]
    if 'pos_embed' in state_dict:
        import math
        num_tokens = state_dict['pos_embed'].shape[1]  # includes cls token
        num_patches = num_tokens - 1
        grid_size = int(math.sqrt(num_patches))
        img_size = grid_size * patch_size
    
    print(f"  Backbone: {backbone_name}, img_size: {img_size}, patch_size: {patch_size}")
    print(f"  State dict keys: {len(state_dict)}")
    
    # Create backbone
    if backbone_name == 'vit_tiny':
        backbone = ViTTiny(img_size=img_size, patch_size=patch_size)
    elif backbone_name == 'vit_small':
        backbone = ViTSmall(img_size=img_size, patch_size=patch_size)
    elif backbone_name == 'vit_base':
        backbone = ViTBase(img_size=img_size, patch_size=patch_size)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    
    # Load weights
    backbone.load_state_dict(state_dict, strict=True)
    backbone = backbone.to(device)
    backbone.eval()
    
    print(f"  ✓ Loaded {backbone_name} with {sum(p.numel() for p in backbone.parameters()):,} params")
    
    return backbone, img_size


# ============================================================================
#                          FEATURE EXTRACTION
# ============================================================================

@torch.no_grad()
def extract_features(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda',
) -> Tuple[torch.Tensor, List, List]:
    """Extract features from dataloader."""
    model.eval()
    features_list = []
    labels_list = []
    filenames_list = []
    
    for batch in tqdm(dataloader, desc="Extracting features"):
        if len(batch) == 3:
            images, labels, filenames = batch
            labels_list.extend(labels.tolist() if isinstance(labels, torch.Tensor) else labels)
        else:
            images, filenames = batch
        
        images = images.to(device)
        
        # Get features from backbone
        cls_token, _ = model.forward_features(images)
        features = F.normalize(cls_token, dim=1)
        
        features_list.append(features.cpu())
        filenames_list.extend(filenames)
    
    features = torch.cat(features_list, dim=0)
    return features, labels_list, filenames_list


# ============================================================================
#                          KNN CLASSIFIER
# ============================================================================

def knn_predict(
    train_features: torch.Tensor,
    train_labels: List[int],
    test_features: torch.Tensor,
    k: int = 20,
    temperature: float = 0.07,
) -> Tuple[List[int], torch.Tensor]:
    """KNN prediction with weighted voting."""
    train_labels_t = torch.tensor(train_labels)
    num_classes = train_labels_t.max().item() + 1
    
    predictions = []
    all_probs = []
    
    batch_size = 256
    for i in range(0, len(test_features), batch_size):
        batch = test_features[i:i + batch_size]
        
        # Cosine similarity (features are normalized)
        similarity = batch @ train_features.t()
        
        # Get top-k neighbors
        topk_sim, topk_idx = similarity.topk(k, dim=1)
        topk_labels = train_labels_t[topk_idx]
        
        # Weighted voting
        weights = F.softmax(topk_sim / temperature, dim=1)
        
        # Accumulate votes
        votes = torch.zeros(batch.size(0), num_classes)
        for j in range(k):
            votes.scatter_add_(1, topk_labels[:, j:j+1], weights[:, j:j+1])
        
        all_probs.append(votes)
        predictions.extend(votes.argmax(dim=1).tolist())
    
    probs = torch.cat(all_probs, dim=0)
    return predictions, probs


def compute_accuracy(predictions: List[int], labels: List[int]) -> Dict[str, float]:
    """Compute accuracy metrics."""
    predictions = torch.tensor(predictions)
    labels = torch.tensor(labels)
    
    correct = (predictions == labels).sum().item()
    total = len(labels)
    top1 = 100.0 * correct / total
    
    return {'top1': top1, 'correct': correct, 'total': total}


# ============================================================================
#                          MAIN EVALUATION
# ============================================================================

def evaluate_dataset(
    backbone: nn.Module,
    dataset_name: str,
    data_root: Path,
    output_dir: Path,
    img_size: int = 96,
    k: int = 20,
    temperature: float = 0.07,
    batch_size: int = 64,
    num_workers: int = 4,
    device: str = 'cuda',
) -> Dict[str, float]:
    """Evaluate on a single dataset (val + test)."""
    
    dataset_dir = data_root / dataset_name
    
    if not dataset_dir.exists():
        print(f"  ⚠ Dataset not found: {dataset_dir}")
        return {}
    
    results = {}
    
    # Check what files exist
    has_train = (dataset_dir / 'train_labels.csv').exists()
    has_val = (dataset_dir / 'val_labels.csv').exists()
    has_test = (dataset_dir / 'test_images.csv').exists()
    
    if not has_train:
        print(f"  ⚠ No train_labels.csv found for {dataset_name}")
        return {}
    
    # =========================================
    # VAL SET: Use existing evaluation function
    # =========================================
    if has_val:
        print(f"\n  Running KNN evaluation on val set...")
        val_results = run_knn_evaluation(
            model=backbone,
            dataset_name=dataset_name,
            data_root=str(data_root),
            img_size=img_size,
            k=k,
            temperature=temperature,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )
        results['val_top1'] = val_results['top1']
        if 'top5' in val_results:
            results['val_top5'] = val_results['top5']
        print(f"  ✓ Val Accuracy: {val_results['top1']:.2f}%")
    
    # =========================================
    # TEST SET: Generate submission CSV
    # =========================================
    if has_test:
        print(f"\n  Generating test predictions...")
        transform = get_eval_transform(img_size)
        
        # Load train set for KNN
        train_dataset = EvalDataset(
            dataset_dir / 'train',
            dataset_dir / 'train_labels.csv',
            transform,
            img_size,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
        
        # Extract train features
        train_features, train_labels, _ = extract_features(backbone, train_loader, device)
        
        # Load test set
        test_dataset = TestDataset(
            dataset_dir / 'test',
            dataset_dir / 'test_images.csv',
            transform,
            img_size,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
        
        test_features, _, test_filenames = extract_features(backbone, test_loader, device)
        print(f"  Test: {len(test_filenames)} samples")
        
        # KNN prediction
        test_preds, _ = knn_predict(train_features, train_labels, test_features, k, temperature)
        
        # Create submission CSV
        submission_df = pd.DataFrame({
            'id': test_filenames,
            'class_id': test_preds,
        })
        
        output_dir.mkdir(parents=True, exist_ok=True)
        submission_path = output_dir / f"{dataset_name}_submission.csv"
        submission_df.to_csv(submission_path, index=False)
        
        results['test_samples'] = len(test_filenames)
        print(f"  ✓ Saved submission: {submission_path} ({len(test_filenames)} predictions)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate checkpoint on all datasets')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--data_root', type=str, default='data/eval_public',
                        help='Root directory for evaluation datasets')
    parser.add_argument('--output_dir', type=str, default='submissions',
                        help='Output directory for submission CSVs')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['cub200', 'miniimagenet', 'sun397'],
                        help='Datasets to evaluate on')
    parser.add_argument('--k', type=int, default=20,
                        help='Number of KNN neighbors')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='KNN temperature')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print("=" * 60)
    print("Checkpoint Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Data root: {args.data_root}")
    print(f"Datasets: {args.datasets}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    # Load backbone
    backbone, img_size = load_backbone(args.checkpoint, args.device)
    
    # Evaluate on each dataset
    all_results = {}
    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Evaluating: {dataset_name.upper()}")
        print("=" * 60)
        
        results = evaluate_dataset(
            backbone=backbone,
            dataset_name=dataset_name,
            data_root=Path(args.data_root),
            output_dir=Path(args.output_dir),
            img_size=img_size,
            k=args.k,
            temperature=args.temperature,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
        )
        
        all_results[dataset_name] = results
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    
    for dataset_name, results in all_results.items():
        if 'val_top1' in results:
            print(f"{dataset_name:15s} Val: {results['val_top1']:6.2f}%")
        if 'test_samples' in results:
            print(f"{dataset_name:15s} Test: {results['test_samples']} samples → {args.output_dir}/{dataset_name}_submission.csv")
    
    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
