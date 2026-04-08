import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import json
import csv


def get_stenosis_severity(mask, visualize=False):
    """
    Calculates stenosis severity by comparing min and max vessel widths.
    """
    # distanceTransform calculates the distance to the closest zero pixel
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # The maximum value in the distance transform is the radius
    # of the largest inscribed circle (the 'normal' part of the vessel)
    radius_ref = np.max(dist_transform)

    if radius_ref == 0:
        return 0

    # To find the narrowest part, we look at the skeleton/midline
    # For a progress report baseline, we can approximate narrowing
    # by looking at the variance in the distance transform.
    # Skeletonize to find the centerline (medial axis)
    # This ensures we measure width along the vessel, not at the edges
    skel = np.zeros(mask.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    temp_img = mask.copy()

    while cv2.countNonZero(temp_img) > 0:
        eroded = cv2.erode(temp_img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(temp_img, temp)
        skel = cv2.bitwise_or(skel, temp)
        temp_img = eroded

    # Get the thickness values along the skeleton
    skeleton_vals = dist_transform[skel > 0]

    if len(skeleton_vals) == 0:
        return 0

    # Use 15th percentile for min width to avoid tapering artifacts at ends
    d_min = np.percentile(skeleton_vals, 15) * 2
    D_ref = np.max(skeleton_vals) * 2

    # Calculate percentage
    severity_pct = ((d_min / D_ref)) * 100
    #severity_pct = (1 - (d_min / D_ref)) * 100
    if visualize:
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(mask, cmap="gray")
        plt.title("Binary Mask")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(dist_transform, cmap="jet")
        plt.title(f"Thickness Map\nMax Width: {D_ref:.1f}px")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        # Convert to RGB to draw red skeleton
        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        vis[skel > 0] = [255, 0, 0]  # Red skeleton
        plt.imshow(vis)
        plt.title(
            f"Skeleton Overlay\nSev: {severity_pct:.1f}% (Min Width: {d_min:.1f})"
        )
        plt.axis("off")
        plt.show()

    # Assign to your three classes
    if severity_pct < 50:
        return 0  # <50%
    elif 50 <= severity_pct <= 70:
        return 1  # 50-70%
    else:
        return 2  # >70%


def process_dataset(json_path, output_csv, dataset_name):
    print(f"Loading {dataset_name} annotations from: {json_path}")
    with open(json_path) as f:
        data = json.load(f)

    # Map image IDs to dimensions to create correct mask sizes
    image_dims = {img["id"]: (img["height"], img["width"]) for img in data["images"]}
    image_filenames = {img["id"]: img["file_name"] for img in data["images"]}

    roi_data = []  # Stores [filename, ann_id, class, x, y, w, h]
    labels = []
    debug_count = 0

    print(f"Processing {dataset_name} annotations...")
    for ann in data["annotations"]:
        # Create empty black mask based on image dimensions
        h, w = image_dims[ann["image_id"]]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Draw the segmentation polygon onto the mask
        for seg in ann["segmentation"]:
            poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [poly], 255)

        # Visualize the first few images to verify logic
        show_debug = debug_count < 3
        if show_debug:
            debug_count += 1

        label = get_stenosis_severity(mask, visualize=show_debug)

        # Get bounding box for the ROI (x, y, w, h)
        if cv2.countNonZero(mask) > 0:
            x, y, w, h = cv2.boundingRect(mask)
            
            # Expand bounding box by 20%
            context_factor = 0.2
            dx = int(w * context_factor)
            dy = int(h * context_factor)
            
            img_h, img_w = image_dims[ann["image_id"]]
            x1 = max(0, x - dx)
            y1 = max(0, y - dy)
            x2 = min(img_w, x + w + dx)
            y2 = min(img_h, y + h + dy)
            
            new_w, new_h = x2 - x1, y2 - y1
            
            # Store individual ROI entry
            roi_data.append(
                [image_filenames[ann["image_id"]], ann["id"], label, x1, y1, new_w, new_h]
            )
            labels.append(label)

    # Save labels to CSV
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "ann_id", "class", "x", "y", "w", "h"])
        writer.writerows(roi_data)

    print(f"Successfully saved {len(roi_data)} ROIs to {output_csv}")
    return labels


def plot_distribution(labels, title):
    class_names = ["<50%", "50-70%", ">70%"]
    counts = [labels.count(0), labels.count(1), labels.count(2)]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(class_names, counts, color=["#A8DADC", "#457B9D", "#1D3557"])
    plt.xlabel("Stenosis Severity Class")
    plt.ylabel("Number of Samples")
    plt.title(title)

    # Add counts on top of bars for professional look
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2, yval + 5, yval, ha="center", va="bottom"
        )

    plt.savefig(f'{title.replace(" ", "_").lower()}.png')
    plt.show()


# Define paths
base_path = r"C:\Users\jcwan\.cache\kagglehub\datasets\nirmalgaud\arcade-dataset\versions\1\arcade\stenosis"

# Process Training
train_json = os.path.join(base_path, "train", "annotations", "train.json")
train_labels = process_dataset(train_json, "training_stenosis_rois.csv", "Training")

# Process Validation
val_json = os.path.join(base_path, "val", "annotations", "val.json")
val_labels = process_dataset(val_json, "validation_stenosis_rois.csv", "Validation")

# Process Testing
test_json = os.path.join(base_path, "test", "annotations", "test.json")
test_labels = process_dataset(test_json, "testing_stenosis_rois.csv", "Testing")

# Plot distributions
plot_distribution(train_labels, "Training Set Distribution")
plot_distribution(val_labels, "Validation Set Distribution")
plot_distribution(test_labels, "Testing Set Distribution")
