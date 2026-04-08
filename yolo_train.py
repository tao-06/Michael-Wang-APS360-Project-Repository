import modal
import os

# --- 1. App and Volume Setup ---
app = modal.App("stenosis-yolo-training")
data_volume = modal.Volume.from_name("stenosis-data", create_if_missing=True)
model_volume = modal.Volume.from_name("stenosis-models", create_if_missing=True)

# --- 2. Environment Configuration ---
image = (
    modal.Image.debian_slim()
    .apt_install("libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "ultralytics", "opencv-python", "numpy", "kagglehub", "pyyaml", "matplotlib", "pandas"
    )
)

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
def train_yolo():
    import json
    import cv2
    import numpy as np
    import shutil
    import kagglehub
    import yaml
    from ultralytics import YOLO
    
    # 1. Paths inside the persistent volume
    persistent_data_path = "/data/arcade_images"
    yolo_dataset_path = "/data/yolo_stenosis_expanded"
    
    print("--- 1. Checking Data ---")
    if not os.path.exists(persistent_data_path):
        print("Downloading raw dataset from Kaggle...")
        temp_path = kagglehub.dataset_download("nirmalgaud/arcade-dataset")
        os.makedirs(persistent_data_path, exist_ok=True)
        shutil.copytree(temp_path, persistent_data_path, dirs_exist_ok=True)
        data_volume.commit()
        
    print("--- 2. Preparing YOLO Format Dataset ---")
    # YOLO requires a specific folder structure and .txt label files
    def prep_yolo_split(split_name):
        json_path = os.path.join(persistent_data_path, "arcade", "stenosis", split_name, "annotations", f"{split_name}.json")
        orig_img_dir = os.path.join(persistent_data_path, "arcade", "stenosis", split_name, "images")
        
        yolo_img_dir = os.path.join(yolo_dataset_path, "images", split_name)
        yolo_lbl_dir = os.path.join(yolo_dataset_path, "labels", split_name)
        
        os.makedirs(yolo_img_dir, exist_ok=True)
        os.makedirs(yolo_lbl_dir, exist_ok=True)
        
        with open(json_path) as f:
            data = json.load(f)
            
        img_info = {img['id']: img for img in data['images']}
        processed = 0
        
        # Clear existing labels to prevent duplicates if run multiple times
        for file in os.listdir(yolo_lbl_dir):
            os.remove(os.path.join(yolo_lbl_dir, file))
            
        for ann in data['annotations']:
            img = img_info[ann['image_id']]
            img_w, img_h = img['width'], img['height']
            
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            for seg in ann["segmentation"]:
                poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
                cv2.fillPoly(mask, [poly], 255)
            
            if cv2.countNonZero(mask) > 0:
                x, y, w, h = cv2.boundingRect(mask)
                
                # Expand bounding box by 20%
                context_factor = 0.2
                dx = int(w * context_factor)
                dy = int(h * context_factor)
                
                x1 = max(0, x - dx)
                y1 = max(0, y - dy)
                x2 = min(img_w, x + w + dx)
                y2 = min(img_h, y + h + dy)
                
                new_w, new_h = x2 - x1, y2 - y1
                
                # YOLO format: normalized center_x, center_y, width, height
                cx = (x1 + new_w / 2) / img_w
                cy = (y1 + new_h / 2) / img_h
                nw = new_w / img_w
                nh = new_h / img_h
                
                base_name = os.path.splitext(img['file_name'])[0]
                txt_path = os.path.join(yolo_lbl_dir, f"{base_name}.txt")
                
                with open(txt_path, "a") as lf:
                    lf.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                
                src_img = os.path.join(orig_img_dir, img['file_name'])
                dst_img = os.path.join(yolo_img_dir, img['file_name'])
                if os.path.exists(src_img) and not os.path.exists(dst_img):
                    shutil.copy(src_img, dst_img)
                processed += 1
                
        print(f"Prepared {processed} bounding boxes for {split_name}.")

    prepared_any = False
    for split in ["train", "val", "test"]:
        if not os.path.exists(os.path.join(yolo_dataset_path, "labels", split)):
            prep_yolo_split(split)
            prepared_any = True
            
    if prepared_any:
        data_volume.commit()
    else:
        print("YOLO dataset splits already prepared!")

    # Write the YAML configuration file required by YOLO
    yaml_path = os.path.join(yolo_dataset_path, "dataset.yaml")
    yaml_content = {
        "path": yolo_dataset_path,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "Stenosis"} # Single class for detection
    }
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f)
    data_volume.commit()
        
    print("--- 3. Starting YOLOv8 Training ---")
    model = YOLO("yolov8n.pt") # Load pretrained nano model
    
    results = model.train(
        data=yaml_path,
        epochs=50,           # More epochs to give time to build higher confidence
        imgsz=512,
        batch=16,
        project="/root/models",
        name="yolo_stenosis",
        exist_ok=True,
        patience=15,          # Automatically stop training if mAP doesn't improve for 15 epochs
        # Medical images are sensitive; we use light augmentation and slower learning
        degrees=15.0,         # Randomly rotate images up to 15 degrees
        lr0=0.002             # Lower initial learning rate to prevent early divergence/overfitting
    )
    
    model_volume.commit()
    print("--- 4. Training Complete! Best weights saved to persistent volume. ---")

@app.function(
    image=image,
    gpu="T4",
    volumes={
        "/data": data_volume,    
        "/root/models": model_volume  
    },
    timeout=1200,
)

def evaluate_yolo():
    import os
    import random
    import cv2
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import pandas as pd
    from ultralytics import YOLO

    model_path = "/root/models/yolo_stenosis/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}! Please run training first.")
        return

    print("--- 1. Evaluating YOLO Accuracy on Validation Set ---")
    model = YOLO(model_path)
    # This automatically computes precision, recall, and mAP
    metrics = model.val(data="/data/yolo_stenosis_expanded/dataset.yaml")
    
    print("\n--- PERFORMANCE METRICS ---")
    print(f"mAP@50      : {metrics.box.map50:.4f} (Accuracy at 50% box overlap)")
    print(f"mAP@50-95   : {metrics.box.map:.4f} (Strict accuracy across thresholds)")
    print("---------------------------\n")
    
    print("--- 2. Generating Visualizations ---")
    val_img_dir = "/data/yolo_stenosis_expanded/images/val"
    val_lbl_dir = "/data/yolo_stenosis_expanded/labels/val"
    
    images = [f for f in os.listdir(val_img_dir) if f.endswith(".png") or f.endswith(".jpg")]
    sample_images = random.sample(images, min(5, len(images)))
    
    fig, axes = plt.subplots(len(sample_images), 2, figsize=(10, 5 * len(sample_images)))
    if len(sample_images) == 1:
        axes = [axes]
        
    for i, img_name in enumerate(sample_images):
        img_path = os.path.join(val_img_dir, img_name)
        lbl_path = os.path.join(val_lbl_dir, img_name.rsplit('.', 1)[0] + '.txt')
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        # Lower the confidence threshold for visualization to see more potential ROI guesses
        results = model.predict(img_path, conf=0.15, verbose=False)[0]
        
        # Plot Ground Truth (Green Box)
        ax_gt = axes[i][0]
        ax_gt.imshow(img)
        ax_gt.set_title(f"Ground Truth\n{img_name}")
        ax_gt.axis("off")
        
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, cx, cy, nw, nh = map(float, parts)
                        box_w, box_h = nw * w, nh * h
                        box_x, box_y = (cx * w) - (box_w / 2), (cy * h) - (box_h / 2)
                        rect = patches.Rectangle((box_x, box_y), box_w, box_h, linewidth=2, edgecolor='g', facecolor='none')
                        ax_gt.add_patch(rect)
                        
        # Plot Predictions (Red Box)
        ax_pred = axes[i][1]
        ax_pred.imshow(img)
        ax_pred.set_title("YOLO Prediction")
        ax_pred.axis("off")
        
        for box in results.boxes:
            coords = box.xyxy[0].cpu().numpy() # [xmin, ymin, xmax, ymax]
            conf = box.conf[0].item()
            box_x, box_y = coords[0], coords[1]
            box_w, box_h = coords[2] - coords[0], coords[3] - coords[1]
            
            rect = patches.Rectangle((box_x, box_y), box_w, box_h, linewidth=2, edgecolor='r', facecolor='none')
            ax_pred.add_patch(rect)
            ax_pred.text(box_x, box_y - 5, f"Conf: {conf:.2f}", color='r', fontsize=12, weight='bold')
            
    plt.tight_layout()
    save_path = "/root/models/yolo_predictions_vis.png"
    plt.savefig(save_path)
    print(f"--- 3. Visualizations saved to {save_path} ---")
    
    print("--- 4. Plotting Training Curves ---")
    results_csv = "/root/models/yolo_stenosis/results.csv"
    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
        # YOLO adds leading spaces to column names, so we strip them
        df.columns = df.columns.str.strip() 
        
        plt.figure(figsize=(12, 5))
        
        # Plot Training and Validation Box Loss
        plt.subplot(1, 2, 1)
        plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
        plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
        plt.title('YOLO Box Loss vs. Epochs')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Plot mAP Metrics
        plt.subplot(1, 2, 2)
        plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@50')
        plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@50-95')
        plt.title('YOLO mAP vs. Epochs')
        plt.xlabel('Epoch')
        plt.legend()
        
        curves_path = "/root/models/yolo_training_curves.png"
        plt.savefig(curves_path)
        print(f"--- 5. Training curves saved to {curves_path} ---")
    else:
        print(f"Could not find {results_csv} to plot training curves.")

    model_volume.commit()

@app.function(
    image=image,
    gpu="T4",
    volumes={
        "/data": data_volume,    
        "/root/models": model_volume  
    },
    timeout=1200,
)
def test_yolo():
    import os
    import json
    import cv2
    import numpy as np
    import shutil
    import yaml
    from ultralytics import YOLO

    persistent_data_path = "/data/arcade_images"
    yolo_dataset_path = "/data/yolo_stenosis_expanded"
    test_lbl_dir = os.path.join(yolo_dataset_path, "labels", "test")
    
    # Prepare the test split if it doesn't exist
    if not os.path.exists(test_lbl_dir):
        print("Test split not found in YOLO format. Preparing now...")
        split_name = "test"
        json_path = os.path.join(persistent_data_path, "arcade", "stenosis", split_name, "annotations", f"{split_name}.json")
        orig_img_dir = os.path.join(persistent_data_path, "arcade", "stenosis", split_name, "images")
        yolo_img_dir = os.path.join(yolo_dataset_path, "images", split_name)
        
        os.makedirs(yolo_img_dir, exist_ok=True)
        os.makedirs(test_lbl_dir, exist_ok=True)
        
        with open(json_path) as f:
            data = json.load(f)
            
        img_info = {img['id']: img for img in data['images']}
        processed = 0
        
        for ann in data['annotations']:
            img = img_info[ann['image_id']]
            img_w, img_h = img['width'], img['height']
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            for seg in ann["segmentation"]:
                poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
                cv2.fillPoly(mask, [poly], 255)
            
            if cv2.countNonZero(mask) > 0:
                x, y, w, h = cv2.boundingRect(mask)
                
                context_factor = 0.2
                dx = int(w * context_factor)
                dy = int(h * context_factor)
                
                x1 = max(0, x - dx)
                y1 = max(0, y - dy)
                x2 = min(img_w, x + w + dx)
                y2 = min(img_h, y + h + dy)
                
                new_w, new_h = x2 - x1, y2 - y1
                
                cx = (x1 + new_w / 2) / img_w
                cy = (y1 + new_h / 2) / img_h
                nw = new_w / img_w
                nh = new_h / img_h
                
                base_name = os.path.splitext(img['file_name'])[0]
                txt_path = os.path.join(test_lbl_dir, f"{base_name}.txt")
                with open(txt_path, "a") as lf:
                    lf.write(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                
                src_img = os.path.join(orig_img_dir, img['file_name'])
                dst_img = os.path.join(yolo_img_dir, img['file_name'])
                if os.path.exists(src_img) and not os.path.exists(dst_img):
                    shutil.copy(src_img, dst_img)
                processed += 1
                
        print(f"Prepared {processed} bounding boxes for test set.")
        data_volume.commit()

    # Guarantee dataset.yaml is perfectly up to date and committed to the volume
    yaml_path = os.path.join(yolo_dataset_path, "dataset.yaml")
    yaml_content = {"path": yolo_dataset_path, "train": "images/train", "val": "images/val", "test": "images/test", "names": {0: "Stenosis"}}
    with open(yaml_path, "w") as f: yaml.dump(yaml_content, f)
    data_volume.commit()

    model_path = "/root/models/yolo_stenosis/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}! Please run training first.")
        return

    print("--- 1. Evaluating YOLO Accuracy on Testing Set ---")
    model = YOLO(model_path)
    # Evaluate on the test split defined in the YAML file
    metrics = model.val(data=yaml_path, split="test")
    
    print("\n--- TEST SET PERFORMANCE METRICS ---")
    print(f"mAP@50      : {metrics.box.map50:.4f} (Accuracy at 50% box overlap)")
    print(f"mAP@50-95   : {metrics.box.map:.4f} (Strict accuracy across thresholds)")
    print("---------------------------\n")

@app.local_entrypoint()
def main():
    train_yolo.remote()