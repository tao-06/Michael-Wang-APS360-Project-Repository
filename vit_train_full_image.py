import modal
import os

# --- 1. App and Volume Setup ---
app = modal.App("stenosis-vit-full-training")
data_volume = modal.Volume.from_name("stenosis-data", create_if_missing=True)
model_volume = modal.Volume.from_name("stenosis-models", create_if_missing=True)
    
# Name of the folder where you cloned the MedViT repository locally
MEDVIT_LOCAL_DIR = "./MedViT_repo" 

# --- 2. Environment Configuration ---
image = (
    modal.Image.debian_slim()
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch", "torchvision", "pandas", "opencv-python", 
        "kagglehub", "numpy", "matplotlib", "tqdm",
        "einops", "timm", "grad-cam"
    )
    .add_local_dir("./Vision_Transformer", remote_path="/root/Vision_Transformer")
    .add_local_dir(MEDVIT_LOCAL_DIR, remote_path="/root/MedViT_repo")
    .add_local_file("training_stenosis_rois.csv", remote_path="/root/training_stenosis_rois.csv")
    .add_local_file("validation_stenosis_rois.csv", remote_path="/root/validation_stenosis_rois.csv")
    .add_local_file("testing_stenosis_rois.csv", remote_path="/root/testing_stenosis_rois.csv")
)

# Configuration for ViT training
CONFIG = {
    "epochs": 30,
    "batch_size": 16,
    "learning_rate": 1e-4, 
    "checkpoint_interval": 5,
    "save_path": "/root/models", 
}

@app.function(
    image=image,
    gpu="A10G",
    volumes={
        "/data": data_volume,    
        "/root/models": model_volume  
    },
    timeout=3600,
    secrets=[modal.Secret.from_name("kaggle-secret")]
)
def train():
    import sys
    import os
    sys.path.append("/root")
    sys.path.append("/root/MedViT_repo")

    import torch
    import kagglehub
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms 
    import matplotlib.pyplot as plt 
    from tqdm import tqdm           
    import pandas as pd
    import cv2
    import numpy as np
    from PIL import Image
    
    from Vision_Transformer.stenosis_VIT import StenosisMedViT
    
    # --- CUSTOM DATASET FOR FULL IMAGES ---
    class FullImageStenosisDataset(Dataset):
        def __init__(self, csv_file, img_dir, transform=None):
            df = pd.read_csv(csv_file)
            # Group by filename and take the maximum severity class for the whole image
            self.annotations = df.groupby("filename")["class"].max().reset_index()
            self.img_dir = img_dir
            self.transform = transform

        def __len__(self):
            return len(self.annotations)

        def __getitem__(self, idx):
            row = self.annotations.iloc[idx]
            img_name = row["filename"]
            label = int(row["class"])

            img_path = os.path.join(self.img_dir, img_name)
            image = cv2.imread(img_path)
            if image is None:
                raise FileNotFoundError(f"Failed to load image at: {img_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

            if self.transform:
                image = self.transform(image)

            return image, label

    # 1. Paths
    train_csv = "/root/training_stenosis_rois.csv"
    val_csv = "/root/validation_stenosis_rois.csv"
    
    persistent_data_path = "/data/arcade_images"
    train_img_dir = os.path.join(persistent_data_path, "arcade", "stenosis", "train", "images")
    val_img_dir = os.path.join(persistent_data_path, "arcade", "stenosis", "val", "images")
    
    print("--- 1. Loading Full Image Datasets ---")
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = FullImageStenosisDataset(train_csv, train_img_dir, transform=train_transform)
    val_dataset = FullImageStenosisDataset(val_csv, val_img_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Data ready: {len(train_dataset)} train, {len(val_dataset)} val images.")

    print("--- 2. Initializing MedViT Model ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StenosisMedViT().to(device)
    
    all_labels = train_dataset.annotations["class"].tolist()
    class_counts = torch.bincount(torch.tensor(all_labels))
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() 
    print(f"Using Class Weights: {class_weights}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)
    scaler = torch.cuda.amp.GradScaler()

    print(f"--- 3. Starting Full Image Training for {CONFIG['epochs']} epochs ---")
    best_val_acc = 0.0
    os.makedirs(CONFIG["save_path"], exist_ok=True)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(CONFIG["epochs"]):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, total=len(train_loader), leave=False)
        train_correct, train_total = 0, 0
        loop.set_description(f"Epoch [{epoch+1}/{CONFIG['epochs']}]")

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100 * train_correct / train_total
        
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch [{epoch+1}/{CONFIG['epochs']}] | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(CONFIG["save_path"], "vit_full_best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  -> New Best Full-Image MedViT Saved! ({val_acc:.2f}%)")
            model_volume.commit()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label='Training Loss')
    plt.plot(history["val_loss"], label='Validation Loss')
    plt.title('Full-Image MedViT Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label='Training Accuracy')
    plt.plot(history["val_acc"], label='Validation Accuracy')
    plt.title('Full-Image MedViT Accuracy')
    plt.legend()

    plot_path = os.path.join(CONFIG["save_path"], "vit_full_training_curves.png")
    plt.savefig(plot_path)
    model_volume.commit()
    print(f"Training finished. Best Accuracy: {best_val_acc:.2f}%")

@app.function(
    image=image,
    gpu="T4",
    volumes={
        "/data": data_volume,    
        "/root/models": model_volume  
    },
    timeout=1200,
)
def generate_gradcam():
    import sys
    import os
    sys.path.append("/root")
    sys.path.append("/root/MedViT_repo")

    import torch
    import torch.nn as nn
    from torchvision import transforms
    from torch.utils.data import DataLoader, Dataset
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import cv2
    from PIL import Image
    
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    
    from Vision_Transformer.stenosis_VIT import StenosisMedViT
    
    # We redefine the dataset here so the Modal function can access it during evaluation
    class FullImageStenosisDataset(Dataset):
        def __init__(self, csv_file, img_dir, transform=None):
            df = pd.read_csv(csv_file)
            self.annotations = df.groupby("filename")["class"].max().reset_index()
            self.img_dir = img_dir
            self.transform = transform
        def __len__(self): return len(self.annotations)
        def __getitem__(self, idx):
            row = self.annotations.iloc[idx]
            image = Image.fromarray(cv2.cvtColor(cv2.imread(os.path.join(self.img_dir, row["filename"])), cv2.COLOR_BGR2RGB))
            return self.transform(image) if self.transform else image, int(row["class"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StenosisMedViT().to(device)
    model_path = "/root/models/vit_full_best_model.pth"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Run training first!")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    last_conv = next((m for m in model.modules() if isinstance(m, nn.Conv2d)), model.model)
    cam = GradCAM(model=model, target_layers=[last_conv])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = FullImageStenosisDataset("/root/validation_stenosis_rois.csv", "/data/arcade_images/arcade/stenosis/val/images", transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    
    fig, axes = plt.subplots(5, 3, figsize=(12, 20))
    class_names = ["<50%", "50-70%", ">70%"]
    mean, std = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(device), torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(device)
    
    for i, (images, labels) in enumerate(val_loader):
        if i >= 5: break
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        grayscale_cam = cam(input_tensor=images, targets=[ClassifierOutputTarget(predicted.item())])[0, :]
        img_display = np.clip((images[0] * std + mean).permute(1, 2, 0).cpu().numpy(), 0, 1)
        visualization = show_cam_on_image(img_display, grayscale_cam, use_rgb=True)
        
        axes[i, 0].imshow(img_display)
        axes[i, 0].set_title(f"True Max Sev: {class_names[labels.item()]}\nPred Max Sev: {class_names[predicted.item()]}")
        axes[i, 1].imshow(grayscale_cam, cmap="jet")
        axes[i, 2].imshow(visualization)
        for ax in axes[i]: ax.axis("off")
        
    save_path = "/root/models/vit_full_gradcam_vis.png"
    plt.tight_layout()
    plt.savefig(save_path)
    model_volume.commit()
    print(f"Visualizations saved to {save_path}")

@app.local_entrypoint()
def main():
    train.remote()