import torch
import torch.nn as nn
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

class BreedEfficientNetV2(nn.Module):
    def __init__(self, num_classes=9, pretrained=True, dropout_rate=0.3):
        super(BreedEfficientNetV2, self).__init__()
        weights = EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_v2_s(weights=weights)
        
        # Replace the final classification head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features=in_features, out_features=num_classes)
        )

    def freeze_backbone(self):
        """
        Freezes feature extraction layers for Stage 1 warm-up training.
        """
        for param in self.backbone.features.parameters():
            param.requires_grad = False
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

    def unfreeze_upper_layers(self, unfreeze_blocks=3):
        """
        Unfreezes upper feature blocks for Stage 2 controlled fine-tuning.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Unfreeze classifier
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True
            
        # Unfreeze last N feature blocks
        feature_blocks = list(self.backbone.features.children())
        for block in feature_blocks[-unfreeze_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)
