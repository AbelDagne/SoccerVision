# SoccerVision: Automated Player and Ball Tracking for Soccer Analytics
**Project Documentation & Development Plan**

---

## Table of Contents
1. [Overview](#overview)
2. [Project Objectives](#project-objectives)
3. [System Architecture](#system-architecture)
4. [Data and Preprocessing](#data-and-preprocessing)
5. [Core Modules](#core-modules)
   - [1. Field Detection](#1-field-detection)
   - [2. Player Detection](#2-player-detection)
   - [3. Team Classification](#3-team-classification)
   - [4. Ball Detection](#4-ball-detection)
   - [5. 2D Mapping & Output Generation](#5-2d-mapping--output-generation)
6. [Workflow / Pipeline](#workflow--pipeline)
7. [Implementation Notes](#implementation-notes)
8. [Milestones & Tasks](#milestones--tasks)
9. [Example Pseudocode Snippets](#example-pseudocode-snippets)

---

## Overview
SoccerVision is a computer vision tool that processes soccer game images to automatically:
1. Detect **players**.
2. Classify teams using jersey color.
3. Identify **the ball**.
4. Generate a **2D top-down mapping** of the game from the input image.

This documentation provides the key design elements, module breakdown, and recommended approach to implement the SoccerVision pipeline in code.

---

## Project Objectives
1. **Automated Player Detection:** Use existing images to detect soccer players on the field.
2. **Team Identification:** Classify each detected player by jersey color (e.g., Team A vs Team B).
3. **Ball Detection:** Locate the ball in the image.
4. **Field Isolation:** Identify the soccer field lines and boundaries to transform player/ball coordinates into a consistent plane.
5. **2D Bird's-Eye Mapping:** Map all detected players and the ball onto a top-down, 2D field representation.

---

## System Architecture

Raw Input -> [Field & Lines Detection] -> [Player Detection] -> [Team Classification] -> [Ball Detection] -> [2D Projection & Output]


### High-Level Modules
1. **Preprocessing & Field Segmentation**
2. **Player Detection**
3. **Classification (teams)**
4. **Ball Detection**
5. **Coordinate Transformation (to 2D plane)**
6. **Result Generation & Visualization**

---

## Data and Preprocessing
- **Assumption**: We have access to panoramic or high-quality images of soccer matches.
- **Resolution & Format**: Input images standardized (e.g. 1280x720, or user-chosen).
- **ROI (Region of Interest)**: Optionally, crop to the soccer field area if existing camera vantage is consistent.

---

## Core Modules

### 1. Field Detection
- **Goal**: Separate the playing field from surrounding stands or backgrounds.
- **Approach**:
  1. **Color-based thresholding** (e.g., using HSV to find typical green ranges).
  2. **Edge/line detection** (Canny/Hough transform) to detect field boundaries.
- **Output**:
  - A mask that identifies the "field region" in the image.
  - Key line positions for geometric transform (for the final 2D mapping step).

### 2. Player Detection
- **Goal**: Identify all players present on the field.
- **Approach**:
  1. Use **HOG-based** detectors to detect persons on the field.
  2. Alternatively, **Deep neural networks** (e.g., YOLO or Faster R-CNN) can detect players more reliably.
- **Output**:
  - Bounding boxes around each detected player.

### 3. Team Classification
- **Goal**: Distinguish teams by jersey color.
- **Approach**:
  1. Extract the region from each detected player bounding box, compute average color in HSV or a color histogram.
  2. Perform a basic **k-means** on color features or a simpler threshold approach if the teams wear distinctly different colors.
  3. Assign each bounding box to the closest color centroid (Team A or Team B).
- **Output**:
  - Updated bounding boxes with team labels (e.g. "Team A", "Team B").

### 4. Ball Detection
- **Goal**: Identify the ball in the image. The ball is typically much smaller than players.
- **Approach**:
  1. **Hough circle transform** to detect circular objects.
  2. Filter by size, color (typically white), and position (on the field).
  3. Alternatively, use a specialized small-object detection CNN.
- **Output**:
  - Position of the ball in the image.

### 5. 2D Mapping & Output Generation
- **Goal**: Convert each bounding box (players) and ball coordinate into a **top-down** layout (2D plane).
- **Approach**:
  1. **Homography**: Use corners of field lines (4 corner flags or known reference points) and compute the homography H that maps image coordinates -> top-down coordinates.
  2. Apply H to each bounding box's center or footpoint to get the position on a standard 2D field (dimensions, e.g., 105m x 68m).
  3. Visualize or store the positions.

---

## Workflow / Pipeline

1. **Load the image**.
2. **Field detection**:
   - Generate a mask for the field.
   - Identify lines for homography references.
3. **Player detection**:
   - Use the mask to limit searching region to field area.
   - Detect all players within the field area.
4. **Team classification**:
   - For each bounding box, measure jersey color, compare to reference color clusters.
5. **Ball detection**:
   - Identify small white or bright circular region on the field.
6. **Homography & 2D projection**:
   - If references for lines/corners are known, transform each bounding box center & ball coordinate to 2D plane.
7. **Output**: 
   - Provide visualization: a 2D top-down layout showing positions.
   - Possibly store in .JSON or .CSV for further analysis.

---

## Implementation Notes
1. **Libraries**:
   - Recommended: `OpenCV` for image operations, `numpy` for array ops, `scikit-learn` or `scipy` for clustering, etc.
   - For deep detection: `PyTorch` or `TensorFlow`.
2. **Edge Cases**:
   - Partial occlusions of players or ball.
   - Varying lighting conditions. 
   - Teams wearing similarly colored kits.
3. **Data Storage**:
   - Store bounding boxes, team labels, ball location in a structured format (e.g., a dictionary with player IDs, positions, team labels).
4. **Configuration**:
   - Might store thresholds (colors, or confidence thresholds) in a config file. 
   - Possibly read known field dimension to interpret final positions in meters.

---

## Milestones & Tasks

| Milestone                 | Tasks                                                                                         |
|---------------------------|-----------------------------------------------------------------------------------------------|
| **1. Preprocessing**      | - Acquire sample images <br>- Check resolution, standardize data shape                      |
| **2. Field detection**    | - Implement color-based threshold <br>- Detect lines for reference                |
| **3. Player detection**   | - HOG or CNN approach <br>- Validate bounding boxes                                          |
| **4. Team classification**| - Identify color features from bounding boxes <br>- K-means color clusters or thresholding  |
| **5. Ball detection**     | - Implement circle detection or small object detection                                      |
| **6. Homography**         | - Mark reference field corners/lines <br>- Solve for H, transform bounding boxes & ball coords |
| **7. Output**            | - Format data structure <br>- Produce 2D top-down layout visualization                       |

---

## Example Pseudocode Snippets

### Player Detection (HOG-based)
```python
import cv2

# Initialize HOG detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Load image
image = cv2.imread('soccer_match.jpg')

# Field detection mask (optional)
mask = field_detect(image)

# Apply mask or just run on entire image
masked_image = cv2.bitwise_and(image, image, mask=mask)

# Detect players
rects, weights = hog.detectMultiScale(masked_image, winStride=(4,4))

# Draw bounding boxes
for (x, y, w, h) in rects:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display result
cv2.imshow('Player Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Team Classification
```python
import cv2
import numpy as np
from sklearn.cluster import KMeans

# For each detected player
team_labels = []
jersey_colors = []

for (x, y, w, h) in player_rects:
    # Extract player region
    player_roi = image[y:y+h, x:x+w]
    
    # Convert to HSV for better color analysis
    hsv_roi = cv2.cvtColor(player_roi, cv2.COLOR_BGR2HSV)
    
    # Create a mask to focus on jersey (exclude skin, etc.)
    # This is simplified and would need refinement
    mask = cv2.inRange(hsv_roi, lower_color_bound, upper_color_bound)
    
    # Calculate average color in the masked region
    avg_color = cv2.mean(hsv_roi, mask=mask)[:3]
    jersey_colors.append(avg_color)

# Cluster jersey colors into teams (assuming 2 teams)
kmeans = KMeans(n_clusters=2, random_state=0).fit(jersey_colors)
team_labels = kmeans.labels_

# Now you can assign team labels to each player rectangle
for i, (x, y, w, h) in enumerate(player_rects):
    team = "Team A" if team_labels[i] == 0 else "Team B"
    # Draw with team-specific color
    color = (0, 0, 255) if team == "Team A" else (255, 0, 0)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, team, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
```

### Ball Detection
```python
import cv2
import numpy as np

# Load image
image = cv2.imread('soccer_match.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply field mask if available
if 'field_mask' in locals():
    gray = cv2.bitwise_and(gray, gray, mask=field_mask)

# Blur the image to reduce noise
gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# Apply Hough Circle Transform
circles = cv2.HoughCircles(
    gray_blurred,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=50,
    param1=50,
    param2=30,
    minRadius=5,
    maxRadius=15
)

# If circles are found
if circles is not None:
    circles = np.uint16(np.around(circles))
    
    # Filter circles by additional criteria (e.g., color, position)
    for i in circles[0, :]:
        # Extract circle region
        x, y, r = i[0], i[1], i[2]
        
        # Check if circle is on the field
        if field_mask[y, x] == 0:
            continue
            
        # Check if circle is white (simplified)
        roi = image[y-r:y+r, x-r:x+r]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # White has low saturation and high value
        if np.mean(hsv_roi[:,:,1]) < 50 and np.mean(hsv_roi[:,:,2]) > 200:
            # Draw the ball
            cv2.circle(image, (x, y), r, (0, 255, 255), 2)
            ball_position = (x, y)
            break
```

### 2D Mapping
```python
import cv2
import numpy as np

# Define standard soccer field dimensions (in meters)
field_width = 105
field_height = 68

# Create a blank top-down view
scale = 10  # pixels per meter
top_down = np.zeros((int(field_height * scale), int(field_width * scale), 3), dtype=np.uint8)

# Draw field lines
# ... (code to draw field markings)

# Assuming we have reference points for homography
# These would be key points on the field identified in the image
src_points = np.array([
    [x1, y1],  # top-left corner
    [x2, y2],  # top-right corner
    [x3, y3],  # bottom-right corner
    [x4, y4]   # bottom-left corner
], dtype=np.float32)

# Corresponding points in the top-down view
dst_points = np.array([
    [0, 0],
    [field_width * scale, 0],
    [field_width * scale, field_height * scale],
    [0, field_height * scale]
], dtype=np.float32)

# Compute homography matrix
H = cv2.findHomography(src_points, dst_points)[0]

# Transform player positions to top-down view
for i, (x, y, w, h) in enumerate(player_rects):
    # Use bottom-center of bounding box as player position
    player_pos = np.array([x + w/2, y + h], dtype=np.float32).reshape(1, 1, 2)
    
    # Apply homography
    transformed_pos = cv2.perspectiveTransform(player_pos, H)[0][0]
    
    # Draw player on top-down view
    tx, ty = int(transformed_pos[0]), int(transformed_pos[1])
    team_color = (0, 0, 255) if team_labels[i] == 0 else (255, 0, 0)
    cv2.circle(top_down, (tx, ty), 5, team_color, -1)

# Transform ball position
if 'ball_position' in locals():
    ball_pos = np.array([ball_position], dtype=np.float32).reshape(1, 1, 2)
    transformed_ball = cv2.perspectiveTransform(ball_pos, H)[0][0]
    tx, ty = int(transformed_ball[0]), int(transformed_ball[1])
    cv2.circle(top_down, (tx, ty), 3, (0, 255, 255), -1)

# Display result
cv2.imshow('2D Mapping', top_down)
cv2.waitKey(0)
cv2.destroyAllWindows()