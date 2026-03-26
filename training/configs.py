"""
Configuration dataclasses for training.

Uses dataclasses for type-safe, validated configurations.
Follows Single Responsibility Principle - each config handles one aspect.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from pathlib import Path


@dataclass
class DataConfig:
    """
    Data loading configuration.
    
    Attributes:
        data_path: Path to training images (CC3M - ~500K images, always used)
        cc12m_data_path: Path to CC12M training data
        birds_data_path: Path to birds pretrain data (~200K images)
        coco_data_path: Path to COCO unlabeled data
        places_data_path: Path to Places365 data
        use_cc12m: Whether to include CC12M data in training
        use_birds: Whether to include birds data in training
        use_coco: Whether to include COCO data in training
        use_places: Whether to include Places365 data in training
        batch_size: Batch size per GPU
        num_workers: DataLoader workers
        pin_memory: Pin memory for faster GPU transfer
        global_crop_size: Size of global crops
        local_crop_size: Size of local crops
        num_local_crops: Number of local crops (0 to disable)
        global_crop_scale: Scale range for global crops
        local_crop_scale: Scale range for local crops
    """
    data_path: str = "data/pretrain/train"  # CC3M - always used as base
    cc12m_data_path: str = "data/pretrain_cc12m/train"  # Path to CC12M data
    birds_data_path: str = "data/pretrain_birds/train"  # Path to birds pretrain data
    coco_data_path: str = "data/pretrain_coco/train"  # Path to COCO unlabeled data
    places_data_path: str = "data/pretrain_places/train"  # Path to Places365 data
    use_cc12m: bool = False  # Flag to include CC12M data
    use_birds: bool = False  # Flag to include birds data
    use_coco: bool = False  # Flag to include COCO data
    use_places: bool = False  # Flag to include Places365 data
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    # Sampling ratios for auxiliary datasets (1.0 = use all images)
    coco_sample_ratio: float = 1.0
    places_sample_ratio: float = 1.0
    
    # Augmentation
    global_crop_size: int = 96
    local_crop_size: int = 48
    num_local_crops: int = 6
    global_crop_scale: Tuple[float, float] = (0.4, 1.0)
    local_crop_scale: Tuple[float, float] = (0.05, 0.4)
    
    def get_data_paths(self) -> List[str]:
        """Get list of data paths based on enabled auxiliary dataset flags.
        
        Note: Sampling ratios are handled at dataset construction time, this
        helper only returns which roots should be included.
        """
        paths = [self.data_path]  # Always include base pretrain (CC3M)
        
        if self.use_cc12m:
            paths.append(self.cc12m_data_path)
        
        if self.use_birds:
            paths.append(self.birds_data_path)
        
        if self.use_coco:
            paths.append(self.coco_data_path)
        
        if self.use_places:
            paths.append(self.places_data_path)
        
        return paths
    
    def __post_init__(self):
        """Validate config values."""
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.num_workers >= 0, "num_workers must be non-negative"
        assert self.global_crop_size > 0, "global_crop_size must be positive"
        assert self.local_crop_size > 0, "local_crop_size must be positive"
        assert self.num_local_crops >= 0, "num_local_crops must be non-negative"


@dataclass
class OptimizerConfig:
    """
    Optimizer configuration.
    
    Attributes:
        name: Optimizer name ('adamw', 'sgd')
        lr: Base learning rate
        weight_decay: Weight decay coefficient
        betas: Adam betas (for AdamW)
        momentum: Momentum (for SGD)
        use_lars: Use LARS optimizer wrapper
    """
    name: str = "adamw"
    lr: float = 5e-4
    weight_decay: float = 0.04
    betas: Tuple[float, float] = (0.9, 0.999)
    momentum: float = 0.9  # For SGD
    use_lars: bool = False
    
    # Layer-wise learning rate decay (for ViT)
    layer_decay: Optional[float] = None  # e.g., 0.65
    
    def __post_init__(self):
        """Validate config values."""
        assert self.name in ["adamw", "sgd"], f"Unknown optimizer: {self.name}"
        assert self.lr > 0, "lr must be positive"
        assert 0 <= self.weight_decay < 1, "weight_decay must be in [0, 1)"


@dataclass
class SchedulerConfig:
    """
    Learning rate scheduler configuration.
    
    Attributes:
        name: Scheduler name ('cosine', 'linear', 'constant')
        warmup_epochs: Number of warmup epochs (used if warmup_steps=0)
        warmup_steps: Number of warmup steps (overrides warmup_epochs if > 0)
        min_lr: Minimum learning rate (for cosine)
        warmup_start_lr: Starting LR for warmup
    """
    name: str = "cosine"
    warmup_epochs: int = 10
    warmup_steps: int = 0  # If > 0, use step-based warmup instead of epoch-based
    min_lr: float = 1e-6
    warmup_start_lr: float = 1e-6
    
    def __post_init__(self):
        """Validate config values."""
        assert self.name in ["cosine", "linear", "constant"], f"Unknown scheduler: {self.name}"
        assert self.warmup_epochs >= 0 or self.warmup_steps >= 0, "warmup must be non-negative"


@dataclass  
class DINOConfig:
    """
    DINO method-specific configuration.
    
    Attributes:
        out_dim: Output dimension of projection head
        hidden_dim: Hidden dimension in projection head
        bottleneck_dim: Bottleneck dimension in head
        num_layers: Number of MLP layers in head
        use_bn: Use batch normalization in head
        norm_last_layer: Weight normalize last layer
        teacher_temp: Teacher softmax temperature
        student_temp: Student softmax temperature  
        teacher_temp_warmup_epochs: Epochs to warmup teacher temp
        teacher_temp_warmup: Starting teacher temperature
        center_momentum: Momentum for center update
        ema_momentum_base: Base EMA momentum for teacher
        ema_momentum_final: Final EMA momentum for teacher
    """
    # Head architecture
    out_dim: int = 65536
    hidden_dim: int = 2048
    bottleneck_dim: int = 256
    num_layers: int = 3
    use_bn: bool = False
    norm_last_layer: bool = True
    
    # Temperatures
    teacher_temp: float = 0.04
    student_temp: float = 0.1
    teacher_temp_warmup_epochs: int = 30
    teacher_temp_warmup: float = 0.04  # Can start higher (e.g., 0.07)
    
    # Center and EMA
    center_momentum: float = 0.9
    ema_momentum_base: float = 0.9  # Start low so teacher adapts fast early on
    ema_momentum_final: float = 1.0
    
    # Loss weights
    use_l2g_loss: bool = True
    l2g_weight: float = 0.5
    use_reg_loss: bool = True  
    reg_weight: float = 0.1
    
    def __post_init__(self):
        """Validate config values."""
        assert self.out_dim > 0, "out_dim must be positive"
        assert 0 < self.teacher_temp < 1, "teacher_temp should be in (0, 1)"
        assert 0 < self.student_temp < 1, "student_temp should be in (0, 1)"
        assert self.teacher_temp < self.student_temp, "teacher_temp should be < student_temp"
        assert 0.9 <= self.ema_momentum_base <= 1.0, "ema_momentum_base should be in [0.9, 1.0]"


@dataclass
class BackboneConfig:
    """
    Backbone configuration.
    
    Attributes:
        name: Backbone name ('vit_small', 'vit_base')
        img_size: Input image size
        patch_size: Patch size for ViT
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
    """
    name: str = "vit_small"
    img_size: int = 96
    patch_size: int = 8
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    
    def __post_init__(self):
        """Validate config values."""
        assert self.name in ["vit_tiny", "vit_small", "vit_base"], f"Unknown backbone: {self.name}"
        assert self.img_size % self.patch_size == 0, "img_size must be divisible by patch_size"


@dataclass
class EvalConfig:
    """
    Evaluation configuration for KNN evaluation.
    
    Attributes:
        enabled: Whether to run evaluation during training
        eval_datasets: List of dataset names to evaluate on
        eval_data_root: Root directory for eval datasets (fallback)
        cub200_path: Custom path to CUB-200 dataset
        miniimagenet_path: Custom path to MiniImageNet dataset
        sun397_path: Custom path to SUN397 dataset
        knn_k: Number of neighbors for KNN
        knn_temperature: Temperature for KNN distance weighting
        batch_size: Batch size for feature extraction
        num_workers: DataLoader workers for eval
    """
    enabled: bool = True
    eval_datasets: List[str] = field(default_factory=lambda: ["cub200", "miniimagenet", "sun397"])
    eval_data_root: str = "data/eval_public"
    cub200_path: Optional[str] = None
    miniimagenet_path: Optional[str] = None
    sun397_path: Optional[str] = None
    knn_k: int = 20
    knn_temperature: float = 0.07
    batch_size: int = 64
    num_workers: int = 4
    
    def get_dataset_path(self, dataset_name: str) -> str:
        """Get the path for a specific dataset."""
        custom_paths = {
            'cub200': self.cub200_path,
            'miniimagenet': self.miniimagenet_path,
            'sun397': self.sun397_path,
        }
        custom = custom_paths.get(dataset_name)
        if custom:
            return custom
        return f"{self.eval_data_root}/{dataset_name}"
    
    def __post_init__(self):
        """Validate config values."""
        assert self.knn_k > 0, "knn_k must be positive"
        assert self.knn_temperature > 0, "knn_temperature must be positive"


@dataclass
class WandbConfig:
    """
    Weights & Biases logging configuration.
    
    Attributes:
        enabled: Whether to use wandb logging
        project: W&B project name
        name: Run name (auto-generated if None)
        tags: List of tags for the run
        notes: Notes for the run
        log_gradients: Log gradient histograms
        log_freq: Frequency of detailed logging (steps)
        watch_model: Use wandb.watch for gradient tracking
        save_artifacts: Save checkpoints as W&B artifacts
    """
    enabled: bool = True
    project: str = "ssl-dino"
    name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    log_gradients: bool = True
    log_freq: int = 100
    watch_model: bool = True
    save_artifacts: bool = True
    
    def __post_init__(self):
        """Validate config values."""
        assert self.log_freq > 0, "log_freq must be positive"


@dataclass
class DistributedConfig:
    """
    Distributed training configuration for multi-GPU (DDP).
    
    Attributes:
        enabled: Whether to use distributed training
        backend: Distributed backend ('nccl' for GPU, 'gloo' for CPU)
        world_size: Total number of processes (set automatically by torchrun)
        rank: Global rank of this process (set automatically)
        local_rank: Local rank on this node (set automatically)
        find_unused_parameters: DDP flag for unused parameters
        broadcast_buffers: DDP flag to broadcast buffers
    """
    enabled: bool = False
    backend: str = "nccl"  # 'nccl' for GPU, 'gloo' for CPU
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    find_unused_parameters: bool = False
    broadcast_buffers: bool = True
    
    def __post_init__(self):
        """Validate config values."""
        assert self.backend in ["nccl", "gloo"], f"Unknown backend: {self.backend}"
        assert self.world_size >= 1, "world_size must be >= 1"


@dataclass
class TrainingConfig:
    """
    Complete training configuration.
    
    Combines all sub-configs into a single configuration object.
    This is the main config used by the trainer.
    
    Attributes:
        epochs: Total training epochs
        seed: Random seed
        device: Device to use ('cuda', 'cpu', 'mps')
        mixed_precision: Use automatic mixed precision
        gradient_clip: Max gradient norm (0 to disable)
        log_interval: Steps between logging
        save_interval: Epochs between checkpoints
        eval_interval: Epochs between evaluation
        output_dir: Directory for outputs (checkpoints, logs)
        experiment_name: Name for this experiment
        resume_from: Path to checkpoint to resume from
        reset_scheduler: Reset LR scheduler when resuming (don't load scheduler state)
    """
    # Training
    epochs: int = 100
    seed: int = 42
    device: str = "cuda"
    mixed_precision: bool = True
    gradient_clip: float = 3.0
    gradient_accumulation_steps: int = 1  # Accumulate gradients over N steps
    
    # Logging and checkpointing
    log_interval: int = 50
    save_interval: int = 10
    eval_interval: int = 10
    output_dir: str = "outputs"
    experiment_name: str = "dino_vit_small"
    resume_from: Optional[str] = None
    reset_scheduler: bool = False  # Reset LR scheduler when resuming
    
    # Sub-configs
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    dino: DINOConfig = field(default_factory=DINOConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    
    def __post_init__(self):
        """Validate and setup config."""
        assert self.epochs > 0, "epochs must be positive"
        assert self.gradient_clip >= 0, "gradient_clip must be non-negative"
        
        # Create output directory path
        self.checkpoint_dir = Path(self.output_dir) / self.experiment_name / "checkpoints"
        self.log_dir = Path(self.output_dir) / self.experiment_name / "logs"
    
    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        """Load config from YAML file."""
        import yaml
        
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "TrainingConfig":
        """Create config from dictionary."""
        # Extract sub-configs
        data_config = DataConfig(**config_dict.pop('data', {}))
        optimizer_config = OptimizerConfig(**config_dict.pop('optimizer', {}))
        scheduler_config = SchedulerConfig(**config_dict.pop('scheduler', {}))
        dino_config = DINOConfig(**config_dict.pop('dino', {}))
        backbone_config = BackboneConfig(**config_dict.pop('backbone', {}))
        eval_config = EvalConfig(**config_dict.pop('eval', {}))
        wandb_config = WandbConfig(**config_dict.pop('wandb', {}))
        distributed_config = DistributedConfig(**config_dict.pop('distributed', {}))
        
        return cls(
            data=data_config,
            optimizer=optimizer_config,
            scheduler=scheduler_config,
            dino=dino_config,
            backbone=backbone_config,
            eval=eval_config,
            wandb=wandb_config,
            distributed=distributed_config,
            **config_dict
        )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        from dataclasses import asdict
        return asdict(self)
    
    def save_yaml(self, path: str) -> None:
        """Save config to YAML file."""
        import yaml
        from dataclasses import asdict
        
        # Convert to dict, excluding Path objects
        config_dict = {}
        for key, value in asdict(self).items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def __repr__(self) -> str:
        return (
            f"TrainingConfig(\n"
            f"  epochs={self.epochs},\n"
            f"  device={self.device},\n"
            f"  backbone={self.backbone.name},\n"
            f"  batch_size={self.data.batch_size},\n"
            f"  lr={self.optimizer.lr},\n"
            f"  experiment={self.experiment_name}\n"
            f")"
        )
