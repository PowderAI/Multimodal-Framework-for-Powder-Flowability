
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Tuple

from data.dataset import SEMDataset

def get_transforms(image_size: int = 224) -> transforms.Compose:

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def create_dataloaders(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    batch_size: int,
    num_workers: int,
    image_size: int = 224,
    magnif_col: str = "magnification_scaled"
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    
    transform = get_transforms(image_size)
    
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)
    
    train_loader = DataLoader(
        SEMDataset(df_train, transform, magnif_col),
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        SEMDataset(df_val, transform, magnif_col),
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        SEMDataset(df_test, transform, magnif_col),
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader
