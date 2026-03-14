import torch
import torch.nn as nn
from torchvision import models

class StenosisResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(StenosisResNet, self).__init__()
        # Load a pre-trained ResNet18
        # Switch to Wide ResNet-50-2 for significantly more filters/capacity
        self.model = models.resnet50(weights='DEFAULT' if pretrained else None)
        
        # FREEZE BACKBONE: Prevents destroying learned features
        for param in self.model.parameters():
            param.requires_grad = False
            
        # UNFREEZE LAST BLOCK: Allow fine-tuning of high-level features
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace the final fully connected layer
        # ResNet18's last layer is named 'fc' and has 512 input features
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 3)
        ) # 3 Classes: <50%, 50-70%, >70%

    def forward(self, x):
        return self.model(x)