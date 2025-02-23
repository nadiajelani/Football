import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import cv2
import os
import numpy as np
import pandas as pd
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# 🟢 Step 1: Load Faster R-CNN Model (TorchVision)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_file = "/Users/nadiajelani/Documents/GitHub/Football/models/faster_rcnn.pth"

# Load the full checkpoint
checkpoint = torch.load(checkpoint_file, map_location="cpu")

# Extract only the state_dict
checkpoint = checkpoint["state_dict"]

# Load the model
model = fasterrcnn_resnet50_fpn(pretrained=False)
model.load_state_dict(checkpoint, strict=False)  # strict=False allows missing keys
model.eval()

print("✅ Model loaded successfully!")


# 🟢 Step 2: Define Input Folder & Output CSV
image_folder = "/Users/nadiajelani/Library/CloudStorage/OneDrive-SheffieldHallamUniversity/football/Data and Videos/Ball 1/Drop 1"
output_csv = "/Users/nadiajelani/Desktop/football_project/bounce_analysis_results.csv"

image_files = sorted(os.listdir(image_folder))  # Sort filenames in order

# Transform function for images
transform = transforms.Compose([
    transforms.ToTensor(),
])

football_positions = []  # To store football's y-coordinates

# 🟢 Step 3: Detect Football in Each Image
for image_name in image_files:
    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path)

    if image is None:
        print(f"⚠️ Warning: Could not read {image_name}. Skipping...")
        continue  # Skip unreadable files

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = transform(image_rgb).to(device).unsqueeze(0)

    # Run inference
    with torch.no_grad():
        outputs = model(image_tensor)

    # Extract bounding boxes and class IDs
    boxes = outputs[0]["boxes"].cpu().numpy()
    scores = outputs[0]["scores"].cpu().numpy()
    labels = outputs[0]["labels"].cpu().numpy()  # Class IDs

    # Keep only football detections (COCO Class ID 32) with confidence > 0.5
    keep = (scores > 0.5) & (labels == 32)
    boxes = boxes[keep]

    if len(boxes) > 0:
        # Select the largest detected football (if multiple)
        largest_box = max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
        x1, y1, x2, y2 = largest_box
        center_y = (y1 + y2) / 2  # Get football's vertical position
        football_positions.append((image_name, center_y))
    else:
        print(f"⚠️ Warning: No football detected in {image_name}")

# Convert to NumPy array for processing
football_positions = np.array(football_positions, dtype=object)
image_names = football_positions[:, 0]
y_positions = football_positions[:, 1].astype(float)

# 🟢 Step 4: Analyze Motion to Find Bounce Events
velocities = np.diff(y_positions)

# Find ground contact: Velocity near zero before bouncing
ground_contact_index = np.where(np.abs(velocities) < 1)[0]
ground_contact_images = image_names[ground_contact_index]

# Find maximum bounce: Where velocity goes from positive to negative (ball starts falling)
bounce_peak_index = np.where((velocities[:-1] > 0) & (velocities[1:] < 0))[0] + 1
bounce_peak_images = image_names[bounce_peak_index]

# Find leaving ground: Where velocity changes from negative to positive (ball starts rising)
leaving_ground_index = np.where((velocities[:-1] < 0) & (velocities[1:] > 0))[0] + 1
leaving_ground_images = image_names[leaving_ground_index]

# 🟢 Step 5: Save Results to CSV
df = pd.DataFrame({
    "Before Bounce (Ground Contact)": list(ground_contact_images),
    "Maximum Bounce Height": list(bounce_peak_images),
    "After Bounce (Leaving Ground)": list(leaving_ground_images)
})
df.to_csv(output_csv, index=False)

print(f"✅ Bounce analysis completed! Results saved to: {output_csv}")
