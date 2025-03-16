"""
Player detection module for the simplified SoccerVision project.

This module implements player detection using techniques covered in class.
"""

import cv2
import numpy as np

def detect_players(image, threshold=0.3, method="custom"):
    """
    Detect players in an image using color-based segmentation approach.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        threshold (float): Threshold for edge detection.
        method (str): Detection method - can be ignored, always using custom implementation
        
    Returns:
        list: List of player bounding boxes in the format [(x, y, w, h), ...].
    """
    # Implementation using techniques from class
    return detect_players_with_canny_and_contours(image, threshold)

def detect_players_with_canny_and_contours(image, threshold=0.3):
    """
    Detect players using Canny edge detection and contour finding.
    This implements techniques covered in class homework.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        threshold (float): Threshold for edge detection.
        
    Returns:
        list: List of player bounding boxes in the format [(x, y, w, h), ...].
    """
    # 1. Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Apply Gaussian blur to reduce noise (from HW1)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Apply Canny edge detector (from HW1 and HW2)
    # Automatically compute lower and upper thresholds
    median_intensity = np.median(blurred)
    lower_threshold = int(max(0, (1.0 - threshold) * median_intensity))
    upper_threshold = int(min(255, (1.0 + threshold) * median_intensity))
    edges = cv2.Canny(blurred, lower_threshold, upper_threshold)
    
    # 4. Extract field mask using color segmentation (from HW1 color segmentation techniques)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define green color range for soccer field
    lower_green = np.array([35, 30, 30])
    upper_green = np.array([85, 255, 255])
    
    # Create a mask for the field (green areas)
    field_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # 5. Get non-field areas to find potential players
    player_mask = cv2.bitwise_not(field_mask)
    
    # 6. Combine edge information with player mask
    combined_mask = cv2.bitwise_and(edges, player_mask)
    
    # 7. Morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
    combined_mask = cv2.erode(combined_mask, kernel, iterations=1)
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=3)
    
    # 8. Find contours (from HW2 techniques)
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 9. Filter contours by size to identify players
    player_boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by size - adjust for soccer field aerial view
        if 50 < area < 5000:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Ensure minimum size
            if w > 5 and h > 5:
                player_boxes.append((x, y, w, h))
    
    # 10. Apply non-maximum suppression 
    final_boxes = non_maximum_suppression(player_boxes, 0.2)
    
    return final_boxes

def non_maximum_suppression(boxes, overlap_threshold):
    """
    Apply non-maximum suppression to remove overlapping boxes.
    
    Args:
        boxes (list): List of bounding boxes in the format [(x, y, w, h), ...].
        overlap_threshold (float): Maximum allowed overlap between boxes.
        
    Returns:
        list: Filtered list of bounding boxes.
    """
    if len(boxes) == 0:
        return []
    
    # Convert to numpy array for easier computation
    boxes_array = np.array(boxes)
    
    # Extract coordinates
    x = boxes_array[:, 0]
    y = boxes_array[:, 1]
    w = boxes_array[:, 2]
    h = boxes_array[:, 3]
    
    # Compute area of each box
    area = w * h
    
    # Sort by bottom-right y-coordinate
    indices = np.argsort(y + h)
    
    keep = []
    while len(indices) > 0:
        # Pick the last box (highest y + h)
        last = len(indices) - 1
        i = indices[last]
        keep.append(i)
        
        # Find boxes with significant overlap
        xx1 = np.maximum(x[i], x[indices[:last]])
        yy1 = np.maximum(y[i], y[indices[:last]])
        xx2 = np.minimum(x[i] + w[i], x[indices[:last]] + w[indices[:last]])
        yy2 = np.minimum(y[i] + h[i], y[indices[:last]] + h[indices[:last]])
        
        # Compute width and height of overlapping area
        w_overlap = np.maximum(0, xx2 - xx1)
        h_overlap = np.maximum(0, yy2 - yy1)
        
        # Compute overlap ratio
        overlap_area = w_overlap * h_overlap
        overlap_ratio = overlap_area / area[indices[:last]]
        
        # Find indices of boxes to remove
        to_delete = np.concatenate(([last], np.where(overlap_ratio > overlap_threshold)[0]))
        
        # Update indices
        indices = np.delete(indices, to_delete)
    
    # Return kept boxes
    return [boxes[i] for i in keep] 