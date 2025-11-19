
import torch
import torch.nn as nn
from torchvision.models import (
    mobilenet_v3_large, shufflenet_v2_x1_0, efficientnet_b0,
    regnet_y_400mf, resnet18, MobileNet_V3_Large_Weights,
    ShuffleNet_V2_X1_0_Weights, EfficientNet_B0_Weights,
    RegNet_Y_400MF_Weights, ResNet18_Weights
)

class ImageBackbone(nn.Module):
    
    SUPPORTED_BACKBONES = {
        "mobilenet_v3_large", "shufflenetv2", "efficientnet_b0",
        "regnety_400mf", "resnet18"
    }
    
    def __init__(self, name: str):
        super().__init__()
        name = name.lower()
        
        if name not in self.SUPPORTED_BACKBONES:
            raise ValueError(f"Unsupported backbone: {name}. Choose from {self.SUPPORTED_BACKBONES}")
        
        self.name = name
        self._build_backbone()
    
    def _build_backbone(self):
        if self.name == "mobilenet_v3_large":
            m = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
            self.features = m.features
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.out_dim = 960
            
        elif self.name == "shufflenetv2":
            m = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
            self.features = nn.Sequential(*list(m.children())[:-1])
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.out_dim = 1024
            
        elif self.name == "efficientnet_b0":
            m = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.feature_extractor = m.features
            self.out_dim = 1280
            
        elif self.name == "regnety_400mf":
            m = regnet_y_400mf(weights=RegNet_Y_400MF_Weights.IMAGENET1K_V2)
            self.features = nn.Sequential(m.stem, m.trunk_output)
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.out_dim = m.fc.in_features
            
        elif self.name == "resnet18":
            m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            self.stem = nn.Sequential(
                m.conv1, m.bn1, m.relu, m.maxpool,
                m.layer1, m.layer2, m.layer3, m.layer4
            )
            self.out_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.name == "mobilenet_v3_large":
            x = self.features(x)
            x = self.gap(x).squeeze(-1).squeeze(-1)
        elif self.name == "shufflenetv2":
            x = self.features(x)
            x = self.gap(x).squeeze(-1).squeeze(-1)
        elif self.name == "efficientnet_b0":
            x = self.feature_extractor(x)
            x = x.mean(dim=[2, 3])
        elif self.name == "regnety_400mf":
            x = self.features(x)
            x = self.gap(x).squeeze(-1).squeeze(-1)
        elif self.name == "resnet18":
            x = self.stem(x)
            x = x.mean(dim=[2, 3])
        return x
