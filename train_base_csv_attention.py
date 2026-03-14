# c:\Users\jcwan\.vscode\projects\aps360 stenosis class\training_utils.py
import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import os
import numpy as np
from PIL import Image


class StenosisDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]

        # 1. Load the full original image
        img_path = os.path.join(self.img_dir, row["filename"])
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Failed to load image at: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 2. Crop using the ROI coordinates from your CSV
        x, y, w, h = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])

        # 1. Expand Context (critical for seeing transitions)
        # Add 20% context so the model sees the "shoulders" of the stenosis
        context_factor = 0.2
        dx = int(w * context_factor)
        dy = int(h * context_factor)
        
        y1 = max(0, y - dy)
        y2 = min(image.shape[0], y + h + dy)
        x1 = max(0, x - dx)
        x2 = min(image.shape[1], x + w + dx)
        
        crop = image[y1:y2, x1:x2]

        # 2. Pad to Square (critical for preserving aspect ratio)
        # Instead of stretching, we paste the crop into the center of a black square
        h_crop, w_crop = crop.shape[:2]
        longest = max(h_crop, w_crop)
        
        square_crop = np.zeros((longest, longest, 3), dtype=np.uint8)
        y_off = (longest - h_crop) // 2
        x_off = (longest - w_crop) // 2
        square_crop[y_off:y_off+h_crop, x_off:x_off+w_crop] = crop

        # 3. Convert to PIL Image for compatibility with torchvision.transforms
        crop = Image.fromarray(square_crop)

        label = int(row["class"])

        if self.transform:
            crop = self.transform(crop)
        else:
            # Fallback if no transform provided: Resize & ToTensor
            import torchvision.transforms.functional as TF
            crop = TF.resize(crop, [224, 224])
            crop = TF.to_tensor(crop)

        return crop, label
