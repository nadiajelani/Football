import torch
import cv2
import numpy as np
import os
import pandas as pd
from torchvision import transforms
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

# Load trained Faster R-CNN model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "/Users/nadiajelani/Desktop/football_project/models/faster_rcnn.pth"  # Update with your trained model path
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confidence threshold
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

predictor = DefaultPredictor(cfg)

# Path to folder containing images
image_folder = "path/to/images"
image_files = sorted(os.listdir(image_folder))  # Sort filenames in order

football_positions = []  # To store football's y-coordinates

# Step 1: Detect Football in Each Image
for i, image_name in enumerate(image_files):
    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path)

    # Run inference
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes if instances.has("pred_boxes") else None

    if boxes is not None and len(boxes) > 0:
        # Assuming there's only one football detected
        box = boxes[0]
        x1, y1, x2, y2 = box.numpy()
        center_y = (y1 + y2) / 2  # Get football's vertical position
        
        football_positions.append((image_name, center_y))

# Step 2: Process Vertical Positions
football_positions = np.array(football_positions, dtype=object)
image_names = football_positions[:, 0]
y_positions = football_positions[:, 1].astype(float)

# Compute velocity (change in y)
velocities = np.diff(y_positions)

# Find ground contact: Velocity near zero before bouncing
ground_contact_index = np.where(np.abs(velocities) < 1)[0]
ground_contact_images = image_names[ground_contact_index]  # Frames where ball is in contact with ground

# Find bounce peak: Velocity goes from positive to negative (highest y-position)
bounce_peak_index = np.where((velocities[:-1] > 0) & (velocities[1:] < 0))[0] + 1
bounce_peak_images = image_names[bounce_peak_index]

# Find leaving ground: Velocity changes from negative to positive (ball starts rising)
leaving_ground_index = np.where((velocities[:-1] < 0) & (velocities[1:] > 0))[0] + 1
leaving_ground_images = image_names[leaving_ground_index]

# Step 3: Save Results to CSV
df = pd.DataFrame({
    "Before Bounce (Ground Contact)": list(ground_contact_images),
    "Maximum Bounce Height": list(bounce_peak_images),
    "After Bounce (Leaving Ground)": list(leaving_ground_images)
})
df.to_csv("bounce_analysis_results.csv", index=False)

print("Results saved to bounce_analysis_results.csv")

# Step 4: Visualization (Optional)
for image_name in image_files:
    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path)

    if image_name in ground_contact_images:
        cv2.putText(image, "Ground Contact", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    if image_name in bounce_peak_images:
        cv2.putText(image, "Max Bounce", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    if image_name in leaving_ground_images:
        cv2.putText(image, "Leaving Ground", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Bounce Analysis", image)
    cv2.waitKey(500)  # Display for 500ms

cv2.destroyAllWindows()
