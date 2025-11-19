
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Dict, Optional

class SEMDataset(Dataset):
    
    def __init__(self, dataframe: pd.DataFrame, transform, magnif_col: str = "magnification_scaled"):

        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.magnif_col = magnif_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        
        # Load and transform image
        image = Image.open(row['image_path']).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Prepare tabular features
        tabular = np.array([row[self.magnif_col]], dtype=np.float32)
        
        # Prepare targets
        aor, hr, ci = row['AOR'], row['HR'], row['CI']
        flow_rate = row['Flow_rate']
        flow_rate_available = 0 if pd.isna(flow_rate) else 1
        flow_rate_value = 0.0 if pd.isna(flow_rate) else float(flow_rate)
        
        return {
            'image': image,
            'tabular': torch.tensor(tabular, dtype=torch.float32),
            'target': torch.tensor([aor, hr, ci], dtype=torch.float32),
            'flow_rate_available': torch.tensor(flow_rate_available, dtype=torch.long),
            'flow_rate_value': torch.tensor(flow_rate_value, dtype=torch.float32),
            'idx': idx
        }
