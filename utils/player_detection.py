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
    
    # 2. Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Apply Canny edge detector
    # Automatically compute lower and upper thresholds
    median_intensity = np.median(blurred)
    lower_threshold = int(max(0, (1.0 - threshold) * median_intensity))
    upper_threshold = int(min(255, (1.0 + threshold) * median_intensity))
    edges = cv2.Canny(blurred, lower_threshold, upper_threshold)
    
    # 4. Extract field mask using color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define green color range for soccer field
    lower_green = np.array([30, 25, 25])
    upper_green = np.array([90, 255, 255])
    
    # Create a mask for the field (green areas)
    field_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Apply morphological operations to clean up the field mask
    field_kernel = np.ones((5, 5), np.uint8)
    field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, field_kernel, iterations=2)
    field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_OPEN, field_kernel, iterations=1)
    
    # Fill holes in the field mask
    field_mask_floodfill = field_mask.copy()
    h, w = field_mask.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(field_mask_floodfill, mask, (0, 0), 255)
    field_mask_inv = cv2.bitwise_not(field_mask_floodfill)
    field_mask = field_mask | field_mask_inv
    
    # 5. Find the largest contour in the field mask, which should be the playing field
    field_contours, _ = cv2.findContours(field_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if field_contours:
        # Sort contours by area and get the largest one
        field_contours = sorted(field_contours, key=cv2.contourArea, reverse=True)
        largest_field_contour = field_contours[0]
        
        # Create a binary mask for the largest field contour
        field_contour_mask = np.zeros_like(field_mask)
        cv2.drawContours(field_contour_mask, [largest_field_contour], 0, 255, -1)
        
        # Update field mask to use only the largest contour
        field_mask = field_contour_mask
    
    # 6. Find potential player candidates on the field
    # Extract non-green areas within the field (potential players)
    player_mask = cv2.bitwise_not(cv2.inRange(hsv, lower_green, upper_green))
    
    # Only consider candidates that are within the field
    player_mask = cv2.bitwise_and(player_mask, field_mask)
    
    # 7. Combine edge information with player mask to get more precise boundaries
    combined_mask = cv2.bitwise_and(edges, player_mask)
    
    # 8. Morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
    combined_mask = cv2.erode(combined_mask, kernel, iterations=1)
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
    
    # 9. Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 10. Filter contours by size and aspect ratio to identify players
    player_boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by size - adjust for soccer field aerial view
        if 100 < area < 3000:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check aspect ratio - players are usually taller than they are wide
            aspect_ratio = float(h) / w if w > 0 else 0
            if 0.5 <= aspect_ratio <= 3.0 and w > 5 and h > 10:
                # Check if the bounding box is mostly inside the field mask
                box_mask = np.zeros_like(field_mask)
                cv2.rectangle(box_mask, (x, y), (x + w, y + h), 255, -1)
                overlap = cv2.bitwise_and(box_mask, field_mask)
                overlap_area = cv2.countNonZero(overlap)
                
                # Only keep the box if at least 60% is inside the field
                if overlap_area > 0.6 * (w * h):
                    player_boxes.append((x, y, w, h))
    
    # 11. Apply non-maximum suppression with stricter overlap threshold
    final_boxes = non_maximum_suppression(player_boxes, 0.3)  # Increased overlap threshold
    
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