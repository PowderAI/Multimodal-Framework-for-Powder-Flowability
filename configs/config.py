
from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class TrainingConfig:
    
    image_size: int = 224
    batch_size: int = 128
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    patience: int = 7
    num_workers: int = 12
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class DataConfig:
    
    csv_path: str = 'Scaled_data.csv'
    train_csv: str = 'train_data.csv'
    val_csv: str = 'val_data.csv'
    test_csv: str = 'test_data.csv'
    magnif_col: str = 'magnification_scaled'
    
@dataclass
class ModelConfig:
    
    backbones: list = None
    tabular_dim: int = 1
    
    def __post_init__(self):
        if self.backbones is None:
            self.backbones = [
                "regnety_400mf", 
                "shufflenetv2", 
                "efficientnet_b0", 
                "resnet18", 
                "mobilenet_v3_large"
            ]

@dataclass
class PathConfig:
    
    model_save_dir: Path = Path("saved_models")
    results_dir: Path = Path("results")
    
    def __post_init__(self):
        self.model_save_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)

@dataclass
class WandbConfig:
    
    project: str = "SEM_multimodal"
    enabled: bool = True
