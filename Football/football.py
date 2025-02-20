import os
import cv2
import torch
import numpy as np
import pandas as pd
from mmdet.apis import init_detector, inference_detector

# 🟢 Step 1: Load the Faster R-CNN Model from MMDetection
config_file = "/Users/nadiajelani/Documents/GitHub/Football/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py"
checkpoint_file = "/Users/nadiajelani/Documents/GitHub/Football/models/faster_rcnn.pth"

# Load model
model = init_detector(config_file, checkpoint_file, device="cuda" if torch.cuda.is_available() else "cpu")

# 🟢 Step 2: Define Paths
image_folder = "/Users/nadiajelani/OneDrive/football/Data and Videos/Ball 1/Drop 1"
output_csv = "/Users/nadiajelani/Desktop/football_project/mmdet_bounce_analysis.csv"

image_files = sorted(os.listdir(image_folder))  # Sort filenames
football_positions = []  # Store detected football positions

# 🟢 Step 3: Detect Football in Each Image
for image_name in image_files:
    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path)

    # Run inference using MMDetection
    results = inference_detector(model, image)

    # Extract bounding boxes for football (COCO Class ID 32)
    for result in results:
        for box in result:
            x1, y1, x2, y2, score = box  # Bounding box & confidence score
            if score > 0.5:  # Confidence threshold
                center_y = (y1 + y2) / 2  # Compute y-center
                football_positions.append((image_name, center_y))

# Convert list to NumPy array
football_positions = np.array(football_positions, dtype=object)
image_names = football_positions[:, 0]
y_positions = football_positions[:, 1].astype(float)

# 🟢 Step 4: Analyze Motion to Find Bounce Events
velocities = np.diff(y_positions)

# Find Ground Contact: Velocity near zero (ball stops falling)
ground_contact_index = np.where(np.abs(velocities) < 1)[0]
ground_contact_images = image_names[ground_contact_index]

# Find Maximum Bounce: Where velocity goes from positive to negative (ball starts falling)
bounce_peak_index = np.where((velocities[:-1] > 0) & (velocities[1:] < 0))[0] + 1
bounce_peak_images = image_names[bounce_peak_index]

# Find Leaving Ground: Where velocity changes from negative to positive (ball starts rising)
leaving_ground_index = np.where((velocities[:-1] < 0) & (velocities[1:] > 0))[0] + 1
leaving_ground_images = image_names[leaving_ground_index]

# 🟢 Step 5: Save Results to CSV
df = pd.DataFrame({
    "Before Bounce (Ground Contact)": list(ground_contact_images),
    "Maximum Bounce Height": list(bounce_peak_images),
    "After Bounce (Leaving Ground)": list(leaving_ground_images)
})
df.to_csv(output_csv, index=False)

print(f"✅ MMDetection Bounce Analysis Completed! Results saved to: {output_csv}")
