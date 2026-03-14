# c:\Users\jcwan\.vscode\projects\aps360 stenosis class\model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class StenosisCNN(nn.Module):
    def __init__(self):
        super(StenosisCNN, self).__init__()
        # Input: 3 x 224 x 224 (Standard ImageNet size)

        # Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)  # Output: 32 x 112 x 112

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Pool -> Output: 64 x 56 x 56

        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        # Pool -> Output: 128 x 28 x 28

        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 3)  # 3 Classes: <50%, 50-70%, >70%

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
