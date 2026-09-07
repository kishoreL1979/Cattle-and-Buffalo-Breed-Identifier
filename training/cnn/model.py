import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class BreedEfficientNet(nn.Module):
    """
    EfficientNet-B0 transfer learning model for Cattle & Buffalo Breed Classification.
    """
    def __init__(self, num_classes: int, dropout_rate: float = 0.3, pretrained: bool = True):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b0(weights=weights)
        
        # Replace the classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        self.num_classes = num_classes

    def freeze_backbone(self):
        """Freeze all backbone parameters, keeping only the classifier head trainable."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def unfreeze_upper_layers(self, num_blocks: int = 3):
        """Unfreeze the upper MBConv blocks of EfficientNet-B0 for fine-tuning."""
        # Enable all parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Classifier always trainable
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
            
        # Unfreeze top N feature blocks (EfficientNet has 8 feature blocks 0..7)
        total_blocks = len(self.backbone.features)
        start_idx = max(0, total_blocks - num_blocks)
        for i in range(start_idx, total_blocks):
            for param in self.backbone.features[i].parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
