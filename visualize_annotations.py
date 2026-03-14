import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# Load JSON and set paths
json_path = r"C:\Users\jcwan\.cache\kagglehub\datasets\nirmalgaud\arcade-dataset\versions\1\arcade\stenosis\train\annotations\train.json"
img_dir = r"C:\Users\jcwan\.cache\kagglehub\datasets\nirmalgaud\arcade-dataset\versions\1\arcade\stenosis\train\images"

def calculate_label(mask):
    """Calculates stenosis severity label based on mask geometry."""
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if np.max(dist_transform) == 0: return "Unknown"

    # Skeletonize
    skel = np.zeros(mask.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    temp_img = mask.copy()
    while cv2.countNonZero(temp_img) > 0:
        eroded = cv2.erode(temp_img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(temp_img, temp)
        skel = cv2.bitwise_or(skel, temp)
        temp_img = eroded

    skeleton_vals = dist_transform[skel > 0]
    if len(skeleton_vals) == 0: return "Unknown"
        
    d_min = np.percentile(skeleton_vals, 15) * 2
    D_ref = np.max(skeleton_vals) * 2
    severity_pct = ((d_min / D_ref)) * 100
    #severity_pct = (1 - (d_min / D_ref)) * 100

    if severity_pct < 50: return f"<50% ({severity_pct:.1f}%)"
    elif 50 <= severity_pct <= 70: return f"50-70% ({severity_pct:.1f}%)"
    else: return f">70% ({severity_pct:.1f}%)"

with open(json_path) as f:
    data = json.load(f)

# Loop through the first 5 images
for i in range(5):
    image_info = data['images'][i]
    img_id = image_info['id']
    img_name = image_info['file_name']

    # Load the actual image
    img = cv2.imread(os.path.join(img_dir, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create mask for calculation
    mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)

    # Find annotations belonging to this image
    image_annotations = [ann for ann in data['annotations'] if ann['image_id'] == img_id]

    for ann in image_annotations:
        for seg in ann['segmentation']:
            poly = np.array(seg).reshape((-1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [poly], 255)
            cv2.polylines(img, [poly], isClosed=True, color=(0, 255, 0), thickness=2)

    # Calculate and display label
    label_text = calculate_label(mask)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"Image: {img_name}\nSeverity: {label_text}")
    plt.axis('off')
    plt.show()