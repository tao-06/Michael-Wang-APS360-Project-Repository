import modal
import os

# --- 1. App and Volume Setup ---
app = modal.App("stenosis-vit-training")
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
        "einops", "timm", "grad-cam" # Added grad-cam for visualization
    )
    .add_local_python_source("train_base_csv_attention") 
    # Mount the Vision_Transformer directory and the cloned MedViT repo
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
    # ViTs usually prefer a lower learning rate than CNNs when fine-tuning
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
    # Add directories to the Python path so the imports inside stenosis_VIT.py resolve properly
    sys.path.append("/root")
    sys.path.append("/root/MedViT_repo")

    import torch
    import kagglehub
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import transforms 
    import matplotlib.pyplot as plt 
    from tqdm import tqdm           
    
    # Import the MedViT model and Dataset
    from Vision_Transformer.stenosis_VIT import StenosisMedViT
    from train_base_csv_attention import StenosisDataset 
    
    # 1. Paths
    train_csv = "/root/training_stenosis_rois.csv"
    val_csv = "/root/validation_stenosis_rois.csv"
    
    persistent_data_path = "/data/arcade_images"
    train_img_dir = os.path.join(persistent_data_path, "arcade", "stenosis", "train", "images")
    val_img_dir = os.path.join(persistent_data_path, "arcade", "stenosis", "val", "images")

    print("--- 1. Checking/Downloading Data ---")
    if not os.path.exists(train_img_dir):
        print("Dataset not found in volume. Downloading from Kaggle...")
        temp_path = kagglehub.dataset_download("nirmalgaud/arcade-dataset")
        
        import shutil
        os.makedirs(persistent_data_path, exist_ok=True)
        shutil.copytree(temp_path, persistent_data_path, dirs_exist_ok=True)
        data_volume.commit()
        print("Dataset saved to persistent volume.")
    else:
        print("Dataset found in persistent volume! Skipping download.")
    
    print("--- 2. Loading Datasets ---")
    # MedViT natively expects 224x224 (or 256x256) image inputs instead of 512x512
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

    train_dataset = StenosisDataset(train_csv, train_img_dir, transform=train_transform)
    val_dataset = StenosisDataset(val_csv, val_img_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Data ready: {len(train_dataset)} train, {len(val_dataset)} val images.")

    print("--- 3. Initializing MedViT Model ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = StenosisMedViT().to(device)
    
    # Calculate Class Weights
    all_labels = train_dataset.annotations["class"].tolist()
    class_counts = torch.bincount(torch.tensor(all_labels))
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() 
    print(f"Using Class Weights: {class_weights}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)
    scaler = torch.cuda.amp.GradScaler()

    print(f"--- 4. Starting MedViT Training for {CONFIG['epochs']} epochs ---")
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
            best_path = os.path.join(CONFIG["save_path"], "vit_best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  -> New Best MedViT Saved! ({val_acc:.2f}%)")
            model_volume.commit()
        
        if (epoch + 1) % CONFIG["checkpoint_interval"] == 0:
            ckpt_path = os.path.join(CONFIG["save_path"], f"vit_checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            model_volume.commit()

    # --- 5. Generate and Save Plots ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label='Training Loss')
    plt.plot(history["val_loss"], label='Validation Loss')
    plt.title('MedViT Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label='Training Accuracy')
    plt.plot(history["val_acc"], label='Validation Accuracy')
    plt.title('MedViT Accuracy vs. Epochs')
    plt.xlabel('Epoch')
    plt.legend()

    plot_path = os.path.join(CONFIG["save_path"], "vit_training_curves.png")
    plt.savefig(plot_path)
    print(f"Plots saved to {plot_path}")
    model_volume.commit()
    print(f"MedViT Training finished. Best Accuracy: {best_val_acc:.2f}%")

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
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import numpy as np
    
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    
    from Vision_Transformer.stenosis_VIT import StenosisMedViT
    from train_base_csv_attention import StenosisDataset
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("--- 1. Loading Trained MedViT Model ---")
    model = StenosisMedViT().to(device)
    model_path = os.path.join(CONFIG["save_path"], "vit_best_model.pth")
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please run training first!")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # MedViT is a hybrid model. We can dynamically find the last Conv2d layer to use for Grad-CAM.
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
            
    if last_conv is None:
        print("Could not find a Conv2d layer! Defaulting to entire model (might fail).")
        target_layers = [model.model]
    else:
        target_layers = [last_conv]
        print(f"Targeting layer for Grad-CAM: {last_conv}")
        
    cam = GradCAM(model=model, target_layers=target_layers)
    
    print("--- 2. Loading Validation Images ---")
    val_csv = "/root/validation_stenosis_rois.csv"
    val_img_dir = "/data/arcade_images/arcade/stenosis/val/images"
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_dataset = StenosisDataset(val_csv, val_img_dir, transform=val_transform)
    # Shuffle to get a random set of images each time
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    
    print("--- 3. Generating Grad-CAM Visualizations ---")
    num_images = 5
    fig, axes = plt.subplots(num_images, 3, figsize=(12, 4 * num_images))
    class_names = ["<50%", "50-70%", ">70%"]
    
    # ImageNet un-normalization constants (to properly display the image)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    
    images_processed = 0
    for images, labels in val_loader:
        if images_processed >= num_images:
            break
            
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        # Ask Grad-CAM to target the class that the model predicted
        targets = [ClassifierOutputTarget(predicted.item())]
        grayscale_cam = cam(input_tensor=images, targets=targets)[0, :]
        
        # Unnormalize and prep image for visualization
        img_display = images[0] * std + mean
        img_display = img_display.permute(1, 2, 0).cpu().numpy()
        img_display = np.clip(img_display, 0, 1)
        
        visualization = show_cam_on_image(img_display, grayscale_cam, use_rgb=True)
        
        ax_orig = axes[images_processed, 0]
        ax_cam = axes[images_processed, 1]
        ax_over = axes[images_processed, 2]
        
        ax_orig.imshow(img_display)
        ax_orig.set_title(f"Original\nTrue: {class_names[labels.item()]} | Pred: {class_names[predicted.item()]}")
        ax_orig.axis("off")
        
        ax_cam.imshow(grayscale_cam, cmap="jet")
        ax_cam.set_title("Grad-CAM Heatmap")
        ax_cam.axis("off")
        
        ax_over.imshow(visualization)
        ax_over.set_title("Overlay")
        ax_over.axis("off")
        
        images_processed += 1
        
    save_path = os.path.join(CONFIG["save_path"], "vit_gradcam_visualizations.png")
    plt.tight_layout()
    plt.savefig(save_path)
    model_volume.commit()
    print(f"--- 4. Success! Plots saved to {save_path} ---")

@app.function(
    image=image,
    gpu="A10G",
    volumes={
        "/data": data_volume,    
        "/root/models": model_volume  
    },
    timeout=1200,
)
def test_vit():
    import sys
    import os
    sys.path.append("/root")
    sys.path.append("/root/MedViT_repo")

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torchvision import transforms 
    
    from Vision_Transformer.stenosis_VIT import StenosisMedViT
    from train_base_csv_attention import StenosisDataset 
    
    test_csv = "/root/testing_stenosis_rois.csv"
    persistent_data_path = "/data/arcade_images"
    test_img_dir = os.path.join(persistent_data_path, "arcade", "stenosis", "test", "images")

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = StenosisDataset(test_csv, test_img_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"Test Data ready: {len(test_dataset)} images.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StenosisMedViT().to(device)
    
    model_path = os.path.join(CONFIG["save_path"], "vit_best_model.pth")
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}! Please run training first.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    correct, total = 0, 0
    print("--- Starting MedViT Testing Evaluation ---")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(images)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_acc = 100 * correct / total
    
    print(f"\n--- MEDVIT TEST SET RESULTS ---")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("-------------------------------\n")

if __name__ == "__main__":
    app.run()