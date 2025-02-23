import cv2
import numpy as np
import os
import pandas as pd

# Define input and output folders
image_folder = "/Users/nadiajelani/Library/CloudStorage/OneDrive-SheffieldHallamUniversity/football/Data and Videos/Ball 1/Drop 1"
output_csv = "/Users/nadiajelani/Desktop/football_project/football_analysis_results/football_tracking.csv"
pixel_change_csv = "/Users/nadiajelani/Desktop/football_project/football_analysis_results/pixel_changes.csv"

os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# Initialize variables
tracking_records = []
pixel_changes = []
prev_frame = None  # To store the previous frame for pixel difference calculation
ground_contact_frame = None
lowest_y_position = float('inf')
ground_level = None  # Set dynamically based on ball's lowest point
MAX_SHIFT_THRESHOLD = 30
prev_circle = None  # Store previous detected circle for stability

# Process each image in sequence
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.bmp'))])
for image_name in image_files:
    image_path = os.path.join(image_folder, image_name)
    current_frame = cv2.imread(image_path)
    if current_frame is None:
        continue

    gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Use adaptive thresholding for better separation
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 11, 2)

    # Apply edge detection
    edges = cv2.Canny(adaptive_thresh, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    diameter = 0  # Default diameter to avoid NameError
    cx, cy = 0, 0  # Default values to prevent NameError
    bottom_point = (0, 0)  # Store the lowest detected point dynamically

    if contours:
        # Find the largest contour by area (assumed to be the ball)
        largest_contour = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
        cx, cy, radius = int(cx), int(cy), int(radius)
        diameter = 2 * radius  # Compute diameter
        prev_circle = (cx, cy, radius)  # Store for stability

        # Find the lowest point of the ball dynamically
        bottom_point = max(largest_contour, key=lambda p: p[0][1])
        bottom_point = tuple(bottom_point[0])

        # Create a mask for extracting ball pixels
        ball_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(ball_mask, (cx, cy), radius, 255, -1)

        # Extract ball pixels
        ball_pixels = cv2.bitwise_and(gray, gray, mask=ball_mask)

        # Compute pixel change if a previous frame exists
        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, ball_pixels)
            pixel_change = np.sum(diff)  # Sum of pixel intensity differences
            pixel_changes.append([image_name, pixel_change])

        # Update previous frame for next iteration
        prev_frame = ball_pixels.copy()

        # Define eight boundary points dynamically
        points = [
            (cx - radius, cy), (cx + radius, cy),  # Left, Right
            (cx, cy - radius), bottom_point,  # Top, Dynamic bottom point
            (cx - radius // 2, cy - radius // 2), (cx + radius // 2, cy - radius // 2),  # Top-left, Top-right
            (cx - radius // 2, bottom_point[1]), (cx + radius // 2, bottom_point[1])  # Bottom-left, Bottom-right
        ]

        # Draw circle around the ball
        cv2.circle(current_frame, (cx, cy), radius, (255, 0, 0), 2)  # Blue circle

        # Draw green dots at key points
        for point in points:
            cv2.circle(current_frame, point, 5, (0, 255, 0), -1)

        tracking_records.append([image_name, cx, cy, diameter])

    # Detect ground contact
    if ground_level is None or bottom_point[1] > lowest_y_position:
        ground_level = lowest_y_position  # Set ground level dynamically
    lowest_y_position = min(lowest_y_position, bottom_point[1])  # Update lowest detected position

    if ground_contact_frame is None and cx != 0 and cy != 0 and bottom_point[1] >= ground_level:
        cv2.imwrite(os.path.join(os.path.dirname(output_csv), 'ground_contact_frame.jpg'), current_frame)
        ground_contact_frame = image_name

    # Show tracking result
    cv2.imshow("Football Tracking", current_frame)
    cv2.waitKey(30)

cv2.destroyAllWindows()

# Save tracking data to CSV
df_tracking = pd.DataFrame(tracking_records, columns=["Frame", "Center X", "Center Y", "Diameter (pixels)"])
df_tracking.to_csv(output_csv, index=False)

# Save pixel changes to CSV
df_pixel_changes = pd.DataFrame(pixel_changes, columns=["Frame", "Pixel Change"])
df_pixel_changes.to_csv(pixel_change_csv, index=False)

print(f"✅ Football tracking data saved to: {output_csv}")
print(f"✅ Pixel change data saved to: {pixel_change_csv}")

if ground_contact_frame is not None:
    print(f"⚽ Ball touched the ground at frame: {ground_contact_frame}")
