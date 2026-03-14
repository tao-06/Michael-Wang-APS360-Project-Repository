import modal
import os

# --- 1. App and Volume Setup ---
# Using a different app name so you can see it separately in the dashboard
app = modal.App("stenosis-resnet-training")
data_volume = modal.Volume.from_name("stenosis-data", create_if_missing=True)
model_volume = modal.Volume.from_name("stenosis-models", create_if_missing=True)

# --- 2. Environment Configuration ---
image = (
    modal.Image.debian_slim()
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch", "torchvision", "pandas", "opencv-python", 
        "kagglehub", "numpy", "matplotlib", "tqdm"
    )
    # Include resnet_transfer here
    .add_local_python_source("train_base_csv_attention", "resnet_transfer") 
)


# Configuration for training
CONFIG = {
    "epochs": 30,
    "batch_size": 16,
    "learning_rate": 0.001,
    "checkpoint_interval": 5,
    "save_path": "/root/models", 
}

@app.function(
    image=image,
    gpu="T4",
    volumes={
        "/data": data_volume,    
        "/root/models": model_volume  
    },
    timeout=3600,
    secrets=[modal.Secret.from_name("kaggle-secret")]
)
def train():
    import torch
    import kagglehub
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torchvision import transforms 
    import matplotlib.pyplot as plt 
    from tqdm import tqdm           
    import os
    
    # Import the ResNet model
    from resnet_transfer import StenosisResNet 
    from train_base_csv_attention import StenosisDataset 
    
    # 1. Paths
    # Read from the persistent volume path
    train_csv = "/data/training_stenosis_rois.csv"
    val_csv = "/data/validation_stenosis_rois.csv"
    
    # Paths inside the persistent volume
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
    # ResNet typically expects specific normalization
    # ImageNet stats: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        # Resize is now safe because we square-padded the image first
        transforms.Resize((512, 512)),
        # Use Affine instead of ResizedCrop to preserve the "Whole Structure" view
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = StenosisDataset(train_csv, train_img_dir, transform=train_transform)
    val_dataset = StenosisDataset(val_csv, val_img_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    
    print(f"Data ready: {len(train_dataset)} train, {len(val_dataset)} val images.")

    print("--- 3. Initializing ResNet Model ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the Transfer Learning model
    model = StenosisResNet(pretrained=True).to(device)
    
    # Calculate Class Weights
    all_labels = train_dataset.annotations["class"].tolist()
    class_counts = torch.bincount(torch.tensor(all_labels))
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() 
    print(f"Using Class Weights: {class_weights}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=1e-4)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

    print(f"--- 4. Starting ResNet Training for {CONFIG['epochs']} epochs ---")
    best_val_acc = 0.0
    os.makedirs(CONFIG["save_path"], exist_ok=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        }

    for epoch in range(CONFIG["epochs"]):
        model.train()
        running_loss = 0.0
        
        loop = tqdm(train_loader, total=len(train_loader), leave=False)
        train_correct, train_total = 0, 0
        loop.set_description(f"Epoch [{epoch+1}/{CONFIG['epochs']}]")

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Calculate metrics
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

        # Save Best Model (with resnet_ prefix)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(CONFIG["save_path"], "resnet_best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"  -> New Best ResNet Saved! ({val_acc:.2f}%)")
            model_volume.commit()
        
        # Periodic Checkpoints (with resnet_ prefix)
        if (epoch + 1) % CONFIG["checkpoint_interval"] == 0:
            ckpt_path = os.path.join(CONFIG["save_path"], f"resnet_checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            model_volume.commit()

    # --- 5. Generate and Save Plots ---
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label='Training Loss')
    plt.plot(history["val_loss"], label='Validation Loss')
    plt.title('ResNet Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label='Training Accuracy')
    plt.plot(history["val_acc"], label='Validation Accuracy')
    plt.title('ResNet Accuracy vs. Epochs')
    plt.xlabel('Epoch')
    plt.legend()

    plot_path = os.path.join(CONFIG["save_path"], "resnet_training_curves.png")
    plt.savefig(plot_path)
    print(f"Plots saved to {plot_path}")
    
    model_volume.commit()
    print(f"ResNet Training finished. Best Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    app.run()