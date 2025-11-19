
import torch
import torch.nn as nn
from models.backbone import ImageBackbone

class MultiModalModel(nn.Module):

    
    def __init__(self, tabular_dim: int, backbone_name: str):
        super().__init__()
        
        # Image feature extractor
        self.backbone = ImageBackbone(backbone_name)
        img_feat_dim = self.backbone.out_dim
        
        # Tabular feature processor
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_dim, 32), 
            nn.ReLU(),
            nn.Linear(32, 16), 
            nn.ReLU()
        )
        
        self.shared_features = nn.Sequential(
            nn.Linear(img_feat_dim + 16, 256), 
            nn.ReLU(),
            nn.Linear(256, 128), 
            nn.ReLU()
        )
        
        
        self.regressor = nn.Linear(128, 3)  
        self.flow_classifier = nn.Linear(128, 2)  
        self.flow_regressor = nn.Linear(128, 1)  
    def forward(self, image: torch.Tensor, tabular: torch.Tensor):

        # Extract features
        img_feat = self.backbone(image)
        tab_feat = self.tabular_mlp(tabular)
        
        # Fuse features
        combined = torch.cat([img_feat, tab_feat], dim=1)
        shared = self.shared_features(combined)

        aor_hr_ci = self.regressor(shared)
        flow_class = self.flow_classifier(shared)
        flow_reg = self.flow_regressor(shared)
        
        return aor_hr_ci, flow_class, flow_reg, shared
