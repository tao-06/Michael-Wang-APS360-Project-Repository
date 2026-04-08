import torch
import torch.nn as nn

from MedViT import MedViT_small 

class StenosisMedViT(nn.Module):
    def __init__(self, checkpoint_path=None):
        super(StenosisMedViT, self).__init__()
        # Initialize the small version (83.7% ImageNet accuracy)
        self.model = MedViT_small(pretrained=True)
        
        # MedViT uses a 'proj_head' attribute for classification
        num_ftrs = self.model.proj_head[0].in_features
        self.model.proj_head[0] = nn.Linear(num_ftrs, 3) 

    def forward(self, x):
        return self.model(x)