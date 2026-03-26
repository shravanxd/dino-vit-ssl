"""Linear probe evaluation with hyperparameter tuning.

Given a pretrained DINO checkpoint, this script:
- Loads the ViT backbone (tiny/small/base)
- Freezes the backbone
- Trains a single-layer linear classifier (one per dataset)
- Tunes hyperparameters (optimizer, learning rate, weight decay)
- Evaluates on validation split
- Optionally runs on multiple datasets (cub200, miniimagenet, sun397)

Usage:
    python scripts/evaluate_checkpoint_linear.py \
        --checkpoint outputs/dino_vit_base/checkpoints/checkpoint_epoch0159.pt \
        --datasets cub200 miniimagenet sun397 \
        --batch_size 512 --num_workers 32
"""

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Make project importable when run as a script
import sys
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.backbones import ViTTiny, ViTSmall, ViTBase
from evaluation.datasets import LabeledDataset, TestDataset, get_eval_transform


@dataclass
class OptimConfig:
    name: str
    lr: float
    weight_decay: float


class LinearHead(nn.Module):
    """Simple single-layer linear classifier."""

    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def _collect_candidate_dicts(obj: object) -> List[Dict[str, torch.Tensor]]:
    """Recursively collect dict-like objects that may hold state dicts."""
    stack: List[object] = [obj]
    seen: set[int] = set()
    candidates: List[Dict[str, torch.Tensor]] = []

    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))

        if isinstance(current, (dict, OrderedDict)):
            # ensure keys are strings to avoid confusion with metadata values
            if all(isinstance(k, str) for k in current.keys()):
                candidates.append(current)  # type: ignore[arg-type]
            for value in current.values():
                if isinstance(value, (dict, OrderedDict)):
                    stack.append(value)
        elif hasattr(current, "state_dict") and callable(getattr(current, "state_dict")):
            try:
                stack.append(current.state_dict())  # type: ignore[no-untyped-call]
            except Exception:
                continue

    return candidates


def _strip_known_prefixes(state_dict: Dict[str, torch.Tensor], prefixes: Iterable[str]) -> Dict[str, torch.Tensor]:
    for prefix in prefixes:
        if any(key.startswith(prefix) for key in state_dict.keys()):
            stripped = {
                key[len(prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            if stripped:
                return stripped
    return state_dict


def _normalize_vit_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Attempt to normalize keys so they start with ViT components."""
    prefixes = [
        "module.",
        "model.",
        "backbone.",
        "student.",
        "teacher.",
        "_backbone.",
        "student.backbone.",
        "teacher.backbone.",
        "encoder.",
    ]

    normalized = _strip_known_prefixes(state_dict, prefixes)

    # If keys still contain long prefixes, trim everything before vital token names
    if not any(k.startswith("patch_embed") or k.startswith("blocks") or k.startswith("pos_embed") for k in normalized.keys()):
        trimmed: Dict[str, torch.Tensor] = {}
        for key, value in normalized.items():
            if "patch_embed" in key:
                trimmed_key = key[key.index("patch_embed"):]
            elif "blocks" in key:
                trimmed_key = key[key.index("blocks"):]
            elif "pos_embed" in key:
                trimmed_key = key[key.index("pos_embed"):]
            elif "cls_token" in key:
                trimmed_key = key[key.index("cls_token"):]
            else:
                continue
            trimmed[trimmed_key] = value
        if trimmed:
            normalized = trimmed

    return normalized


def _looks_like_vit_state(state_dict: Dict[str, torch.Tensor]) -> bool:
    if not state_dict:
        return False
    vit_keys = 0
    tensor_like = 0
    for key, value in state_dict.items():
        if not isinstance(key, str):
            continue
        if any(token in key for token in ("patch_embed", "blocks", "pos_embed", "cls_token")):
            vit_keys += 1
        if hasattr(value, "shape"):
            tensor_like += 1
        if vit_keys >= 3 and tensor_like >= 3:
            return True
    return False


def _extract_backbone_state_dict(ckpt: object) -> Dict[str, torch.Tensor]:
    """Extract a ViT-like state dict from an arbitrary checkpoint payload."""
    if isinstance(ckpt, (dict, OrderedDict)):
        # Common direct keys first
        for key in ["model_state_dict", "state_dict", "student_backbone", "teacher_backbone", "backbone"]:
            if key in ckpt and isinstance(ckpt[key], (dict, OrderedDict)):
                candidate = _normalize_vit_state_dict(dict(ckpt[key]))
                if _looks_like_vit_state(candidate):
                    return candidate

    # Fallback: collect every nested dict
    for candidate in _collect_candidate_dicts(ckpt):
        normalized = _normalize_vit_state_dict(candidate.copy())
        if _looks_like_vit_state(normalized):
            return normalized

    raise ValueError(
        "Could not locate a ViT backbone in checkpoint. "
        "Expected keys like 'patch_embed' or 'pos_embed'."
    )


def load_checkpoint_backbone(checkpoint_path: str, device: str = "cuda") -> Tuple[nn.Module, int, int]:
    """Load ViT backbone from checkpoint, handling multiple save formats."""
    import math

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = _extract_backbone_state_dict(ckpt)
    state_dict = _normalize_vit_state_dict(state_dict)

    # Infer embed dim
    if "pos_embed" in state_dict:
        embed_dim = state_dict["pos_embed"].shape[-1]
    elif "patch_embed.proj.weight" in state_dict:
        embed_dim = state_dict["patch_embed.proj.weight"].shape[0]
    else:
        sample_keys = list(state_dict.keys())[:5]
        raise ValueError(
            "Could not infer embed_dim from checkpoint; sample keys: "
            f"{sample_keys}"
        )

    # Infer patch size
    if "patch_embed.proj.weight" in state_dict:
        patch_size = state_dict["patch_embed.proj.weight"].shape[-1]
    else:
        patch_size = 8

    # Infer image size
    if "pos_embed" in state_dict:
        num_tokens = state_dict["pos_embed"].shape[1]
        num_patches = num_tokens - 1
        grid = int(math.sqrt(max(num_patches, 1)))
        img_size = grid * patch_size
    else:
        img_size = 96

    # Infer depth (for logging only)
    depth = len([k for k in state_dict.keys() if k.startswith("blocks.") and k.endswith(".norm1.weight")])

    if embed_dim == 192:
        print(f"Detected ViT-Tiny: dim={embed_dim}, depth={depth}, patch={patch_size}, img={img_size}")
        backbone = ViTTiny(img_size=img_size, patch_size=patch_size)
    elif embed_dim == 384:
        print(f"Detected ViT-Small: dim={embed_dim}, depth={depth}, patch={patch_size}, img={img_size}")
        backbone = ViTSmall(img_size=img_size, patch_size=patch_size)
    elif embed_dim == 768:
        print(f"Detected ViT-Base: dim={embed_dim}, depth={depth}, patch={patch_size}, img={img_size}")
        backbone = ViTBase(img_size=img_size, patch_size=patch_size)
    else:
        raise ValueError(f"Unknown embed_dim {embed_dim}; expected 192/384/768")

    missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys (first 10): {missing[:10]}")
    if unexpected:
        print(f"  Unexpected keys (first 10): {unexpected[:10]}")

    backbone.to(device)
    backbone.eval()

    # Freeze backbone
    for p in backbone.parameters():
        p.requires_grad = False

    return backbone, img_size, embed_dim


@torch.no_grad()
def extract_backbone_features(
    backbone: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract CLS-token features and labels for a labeled dataset."""
    feats = []
    labels = []
    backbone.eval()

    for images, y in dataloader:
        images = images.to(device)
        cls_token, _ = backbone.forward_features(images, return_patch_tokens=True)
        feats.append(cls_token.cpu())
        labels.append(y)

    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def create_eval_loaders_from_dir(
    dataset_dir: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, int]:
    """Create train/val loaders and num_classes from competition-style folder."""
    root = Path(dataset_dir)
    transform = get_eval_transform(img_size)

    train_ds = LabeledDataset(
        image_dir=root / "train",
        labels_file=root / "train_labels.csv",
        transform=transform,
    )
    val_ds = LabeledDataset(
        image_dir=root / "val",
        labels_file=root / "val_labels.csv",
        transform=transform,
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, train_ds.num_classes


def create_test_loader_from_dir(
    dataset_dir: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    """Create test loader for submission generation.
    
    Note: Test loader runs only on main process (no distributed sampler).
    """
    root = Path(dataset_dir)
    transform = get_eval_transform(img_size)

    test_ds = TestDataset(
        image_dir=root / "test",
        images_csv=root / "test_images.csv",
        transform=transform,
    )

    pin_memory = torch.cuda.is_available()

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return test_loader


def train_linear_probe(

    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    val_feats: torch.Tensor,
    val_labels: torch.Tensor,
    in_dim: int,
    num_classes: int,
    device: torch.device,
    optim_cfg: OptimConfig,
    epochs: int = 100,
) -> Tuple[LinearHead, Dict[str, float]]:
    """Train a linear head with given optimizer config using cached features; return best model and metrics."""
    head = LinearHead(in_dim, num_classes).to(device)

    if optim_cfg.name.lower() == "sgd":
        optimizer = torch.optim.SGD(head.parameters(), lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay, momentum=0.9)
    elif optim_cfg.name.lower() == "rmsprop":
        optimizer = torch.optim.RMSprop(head.parameters(), lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay, momentum=0.9)
    elif optim_cfg.name.lower() == "adam":
        optimizer = torch.optim.Adam(head.parameters(), lr=optim_cfg.lr, weight_decay=optim_cfg.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer {optim_cfg.name}")

    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        head.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Shuffle train indices each epoch
        perm = torch.randperm(train_feats.size(0))
        train_feats_shuf = train_feats[perm].to(device)
        train_labels_shuf = train_labels[perm].to(device)

        # Mini-batch training
        batch_size = 512 if train_feats_shuf.size(0) > 512 else train_feats_shuf.size(0)
        for i in range(0, train_feats_shuf.size(0), batch_size):
            feats_batch = train_feats_shuf[i:i+batch_size]
            labels_batch = train_labels_shuf[i:i+batch_size]

            logits = head(feats_batch)
            loss = criterion(logits, labels_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * feats_batch.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels_batch).sum().item()
            total_samples += feats_batch.size(0)

        train_loss = total_loss / max(total_samples, 1)
        train_acc = 100.0 * total_correct / max(total_samples, 1)

        # Validation
        head.eval()
        with torch.no_grad():
            val_feats_device = val_feats.to(device)
            val_labels_device = val_labels.to(device)
            logits = head(val_feats_device)
            loss = criterion(logits, val_labels_device)
            preds = logits.argmax(dim=1)
            val_correct = (preds == val_labels_device).sum().item()
            val_samples = val_labels_device.size(0)
            val_loss = loss.item()
            val_acc = 100.0 * val_correct / max(val_samples, 1)

        print(
            f"    [optim={optim_cfg.name}, lr={optim_cfg.lr}, wd={optim_cfg.weight_decay}] "
            f"Epoch {epoch+1}/{epochs} | train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = head.state_dict().copy()

    if best_state is not None:
        head.load_state_dict(best_state)

    metrics = {"best_val_acc": best_acc}
    return head, metrics


def run_linear_probe_for_dataset(
    dataset_name: str,
    dataset_path: str,
    backbone: nn.Module,
    img_size: int,
    embed_dim: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    quick: bool = False,
) -> Optional[Dict[str, any]]:
    print("\n" + "#" * 70)
    print(f"# LINEAR PROBE: {dataset_name.upper()}")
    print("#" * 70)


    train_loader, val_loader, num_classes = create_eval_loaders_from_dir(
        dataset_dir=dataset_path,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    print(f"  Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}, Classes: {num_classes}")

    # === FEATURE CACHING ===
    print("  Extracting and caching backbone features for train set...")
    train_feats, train_labels = extract_backbone_features(
        backbone=backbone,
        dataloader=train_loader,
        device=device,
    )
    print(f"    Cached train features: {train_feats.shape}, labels: {train_labels.shape}")

    print("  Extracting and caching backbone features for val set...")
    val_feats, val_labels = extract_backbone_features(
        backbone=backbone,
        dataloader=val_loader,
        device=device,
    )
    print(f"    Cached val features: {val_feats.shape}, labels: {val_labels.shape}")

    # Hyperparameter grid
    if quick:
        optims = [
            OptimConfig("adam", lr=1e-3, weight_decay=0.0),
            OptimConfig("adam", lr=5e-4, weight_decay=1e-4),
            OptimConfig("adam", lr=1e-3, weight_decay=1e-4),
            OptimConfig("adam", lr=5e-4, weight_decay=0.0),
        ]
        epochs = 30
    else:
        optims = [
            OptimConfig("sgd", lr=0.1, weight_decay=0.0),
            OptimConfig("sgd", lr=0.05, weight_decay=1e-4),
            OptimConfig("rmsprop", lr=0.01, weight_decay=1e-4),
            OptimConfig("adam", lr=1e-3, weight_decay=0.0),
            OptimConfig("adam", lr=5e-4, weight_decay=1e-4),
        ]
        epochs = 30

    best_overall_acc = 0.0
    best_overall_cfg = None


    for cfg in optims:
        print(f"\n  >>> Optimizer {cfg.name}, lr={cfg.lr}, wd={cfg.weight_decay}")
        head, metrics = train_linear_probe(
            train_feats=train_feats,
            train_labels=train_labels,
            val_feats=val_feats,
            val_labels=val_labels,
            in_dim=embed_dim,
            num_classes=num_classes,
            device=device,
            optim_cfg=cfg,
            epochs=epochs,
        )

        val_acc = metrics["best_val_acc"]
        print(f"  Result for {cfg.name} (lr={cfg.lr}, wd={cfg.weight_decay}): best_val_acc={val_acc:.2f}%")

        if val_acc > best_overall_acc:
            best_overall_acc = val_acc
            best_overall_cfg = (cfg, head.state_dict())

    if best_overall_cfg is None:
        print("  No configuration improved validation accuracy; skipping.")
        return None

    cfg, state_dict = best_overall_cfg
    print(
        f"\nBest config for {dataset_name}: optimizer={cfg.name}, lr={cfg.lr}, "
        f"wd={cfg.weight_decay}, best_val_acc={best_overall_acc:.2f}%"
    )

    # Generate submission CSV
    print(f"\n  >>> Generating submission for {dataset_name}...")
    final_head = LinearHead(embed_dim, num_classes).to(device)
    final_head.load_state_dict(state_dict)

    test_loader = create_test_loader_from_dir(
        dataset_dir=dataset_path,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    final_head.eval()
    predictions = []
    image_ids = []

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(device)
            feats, _ = backbone.forward_features(images, return_patch_tokens=True)
            logits = final_head(feats)
            preds = logits.argmax(dim=1).cpu().tolist()
            predictions.extend(preds)
            image_ids.extend(ids)

    # Write submission CSV
    submission_dir = Path("submissions")
    submission_dir.mkdir(exist_ok=True)
    submission_path = submission_dir / f"{dataset_name}_submission.csv"

    with open(submission_path, 'w') as f:
        f.write("id,class_id\n")
        for img_id, pred in zip(image_ids, predictions):
            f.write(f"{img_id},{pred}\n")

    print(f"  Saved submission to: {submission_path}")
    print(f"  Total predictions: {len(predictions)}")

    return {
        "dataset": dataset_name,
        "best_optimizer": cfg.name,
        "best_lr": cfg.lr,
        "best_wd": cfg.weight_decay,
        "best_val_acc": best_overall_acc,
        "num_predictions": len(predictions),
        "submission_path": str(submission_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linear probe evaluation with optimizer tuning")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--datasets", nargs="+", default=["cub200", "miniimagenet", "sun397"], help="Datasets to run")
    parser.add_argument("--cub200_path", type=str, default="data/eval_public/cub200")
    parser.add_argument("--miniimagenet_path", type=str, default="data/eval_public/miniimagenet")
    parser.add_argument("--sun397_path", type=str, default="data/eval_public/sun397")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--quick", action="store_true", help="Smaller hyperparameter grid and fewer epochs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    print(f"\nLoading checkpoint: {args.checkpoint}")
    backbone, img_size, embed_dim = load_checkpoint_backbone(args.checkpoint, device=str(device))
    print(f"  Using img_size={img_size}, embed_dim={embed_dim}")

    dataset_paths = {
        "cub200": args.cub200_path,
        "miniimagenet": args.miniimagenet_path,
        "sun397": args.sun397_path,
    }

    all_results = []

    for name in args.datasets:
        path = dataset_paths.get(name)
        if path is None:
            print(f"\n[WARN] Unknown dataset name: {name}; skipping")
            continue
        result = run_linear_probe_for_dataset(
            dataset_name=name,
            dataset_path=path,
            backbone=backbone,
            img_size=img_size,
            embed_dim=embed_dim,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            quick=args.quick,
        )
        if result is not None:
            all_results.append(result)

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    for r in all_results:
        print(f"\n  {r['dataset'].upper()}:")
        print(f"    Best Config: {r['best_optimizer']} (lr={r['best_lr']}, wd={r['best_wd']})")
        print(f"    Validation Accuracy: {r['best_val_acc']:.2f}%")
        print(f"    Submission: {r['submission_path']} ({r['num_predictions']} predictions)")

    print("\n" + "=" * 70)
    print("All submissions saved to: submissions/")
    print("=" * 70)


if __name__ == "__main__":
    main()
