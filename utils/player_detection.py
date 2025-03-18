"""
Player detection module for the simplified SoccerVision project.

This module implements player detection using techniques covered in class.
"""

import cv2
import numpy as np
import os

def detect_players(image, threshold=0.3, visualize_steps=False):
    """
    Detect players in a soccer image using Canny edge detection and contour finding.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        threshold (float): Threshold for edge detection.
        visualize_steps (bool): Whether to visualize intermediate steps.
        
    Returns:
        list: List of player bounding boxes in the format (x, y, w, h).
    """
    # Create output directory for visualization if needed
    if visualize_steps:
        vis_dir = "edge_detection_steps"
        os.makedirs(vis_dir, exist_ok=True)
    
    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if visualize_steps:
        cv2.imwrite(f"{vis_dir}/01_grayscale.jpg", gray)
    
    # Step 2: Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 3)
    if visualize_steps:
        cv2.imwrite(f"{vis_dir}/02_gaussian_blur.jpg", blurred)
    
    # Step 3: Calculate gradients with Sobel (for visualization)
    if visualize_steps:
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        sobelx = cv2.convertScaleAbs(sobelx)
        sobely = cv2.convertScaleAbs(sobely)
        gradient_magnitude = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
        cv2.imwrite(f"{vis_dir}/03_gradient_magnitude.jpg", gradient_magnitude)
    
    # Step 4: Apply Canny edge detection
    lower_threshold = int(threshold * 100)
    upper_threshold = lower_threshold * 3
    edges = cv2.Canny(blurred, lower_threshold, upper_threshold)
    if visualize_steps:
        cv2.imwrite(f"{vis_dir}/04_canny_edges.jpg", edges)
    
    # Step 5: Create field mask using HSV thresholding
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Green field mask
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    field_mask = cv2.inRange(hsv_image, lower_green, upper_green)
    # Invert to get non-field mask
    non_field_mask = cv2.bitwise_not(field_mask)
    if visualize_steps:
        cv2.imwrite(f"{vis_dir}/05_field_mask.jpg", field_mask)
        cv2.imwrite(f"{vis_dir}/06_non_field_mask.jpg", non_field_mask)
    
    # Step 6: Combine edge detection with field mask
    player_edges = cv2.bitwise_and(edges, edges, mask=non_field_mask)
    if visualize_steps:
        cv2.imwrite(f"{vis_dir}/07_player_edges.jpg", player_edges)
    
    # Step 7: Find contours
    contours, _ = cv2.findContours(player_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Step 8: Create visualization of contours
    if visualize_steps:
        contour_vis = image.copy()
        cv2.drawContours(contour_vis, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(f"{vis_dir}/08_contours.jpg", contour_vis)
    
    # Step 9: Filter contours by size and create bounding boxes
    player_boxes = []
    min_area = 200  # Minimum contour area to be considered a player
    max_area = 5000  # Maximum contour area to filter out large regions
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter based on aspect ratio to avoid elongated shapes
            aspect_ratio = float(w) / h
            if 0.3 < aspect_ratio < 3.0:
                player_boxes.append((x, y, w, h))
    
    # Step 10: Apply non-maximum suppression
    final_boxes = non_maximum_suppression(player_boxes, 0.3)  # Increased overlap threshold
    
    # Step 11: Create final visualization with bounding boxes
    if visualize_steps:
        final_vis = image.copy()
        for x, y, w, h in final_boxes:
            cv2.rectangle(final_vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imwrite(f"{vis_dir}/09_final_detection.jpg", final_vis)
    
    return final_boxes

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
    # Lower the lower threshold and increase the upper threshold for better edge detection
    lower_threshold = int(max(0, (1.0 - threshold * 1.2) * median_intensity))
    upper_threshold = int(min(255, (1.0 + threshold * 1.2) * median_intensity))
    edges = cv2.Canny(blurred, lower_threshold, upper_threshold)
    
    # 4. Extract field mask using color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define green color range for soccer field - wider range to include more variations
    lower_green = np.array([25, 20, 20])
    upper_green = np.array([95, 255, 255])
    
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
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=5)  # Increased iterations for better connection
    combined_mask = cv2.erode(combined_mask, kernel, iterations=1)
    combined_mask = cv2.dilate(combined_mask, kernel, iterations=2)
    
    # 9. Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 10. Filter contours by size and aspect ratio to identify players
    player_boxes = []
    
    # Find the maximum area of all contours to set a relative size threshold
    max_area = 0
    max_dimension = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
        x, y, w, h = cv2.boundingRect(contour)
        max_dim = max(w, h)
        if max_dim > max_dimension:
            max_dimension = max_dim
    
    # Set a minimum area threshold relative to the maximum area
    # This helps adapt to different image scales
    min_area_threshold = max(150, max_area * 0.05)  # At least 5% of the max area or 150px
    min_dimension_threshold = max(10, max_dimension * 0.15)  # At least 15% of the max dimension or 10px
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filter by size - using more aggressive minimum thresholds
        if area > min_area_threshold and area < 8000:  # Increased minimum area and maximum
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if either width or height is large enough
            larger_dimension = max(w, h)
            if larger_dimension < min_dimension_threshold:
                continue
                
            # Check aspect ratio - prefer more square-like shapes for players seen from above
            aspect_ratio = float(h) / w if w > 0 else 0
            if 0.5 <= aspect_ratio <= 3.0 and w > 5 and h > 5:  # Stricter aspect ratio criteria
                # Check if the bounding box is mostly inside the field mask
                box_mask = np.zeros_like(field_mask)
                cv2.rectangle(box_mask, (x, y), (x + w, y + h), 255, -1)
                overlap = cv2.bitwise_and(box_mask, field_mask)
                overlap_area = cv2.countNonZero(overlap)
                
                # Only keep the box if at least 60% is inside the field (increased threshold)
                if overlap_area > 0.6 * (w * h):
                    player_boxes.append((x, y, w, h))
    
    # 11. Apply non-maximum suppression with a higher overlap threshold to remove redundant boxes
    final_boxes = non_maximum_suppression(player_boxes, 0.3)  # Increased overlap threshold
    
    return final_boxes

def non_maximum_suppression(boxes, overlap_threshold):
    """
    Apply non-maximum suppression to remove overlapping boxes.
    Prioritizes boxes with larger maximum dimension (width or height).
    
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
    
    # Get the larger dimension (width or height) for each box
    max_dimension = np.maximum(w, h)
    
    # Also compute area as a secondary sorting metric
    area = w * h
    
    # Create a combined score that prioritizes larger dimension but also considers area
    # This gives preference to boxes with at least one large dimension
    score = max_dimension * 2 + np.sqrt(area)
    
    # Sort by score (boxes with larger dimension first)
    indices = np.argsort(score)[::-1]
    
    keep = []
    while len(indices) > 0:
        # Pick the box with the largest score
        i = indices[0]
        keep.append(i)
        
        # Find boxes with significant overlap
        xx1 = np.maximum(x[i], x[indices[1:]])
        yy1 = np.maximum(y[i], y[indices[1:]])
        xx2 = np.minimum(x[i] + w[i], x[indices[1:]] + w[indices[1:]])
        yy2 = np.minimum(y[i] + h[i], y[indices[1:]] + h[indices[1:]])
        
        # Compute width and height of overlapping area
        w_overlap = np.maximum(0, xx2 - xx1)
        h_overlap = np.maximum(0, yy2 - yy1)
        
        # Compute overlap ratio
        overlap_area = w_overlap * h_overlap
        overlap_ratio = np.zeros(len(indices) - 1)
        valid_indices = (area[indices[1:]] > 0)
        if np.any(valid_indices):
            overlap_ratio[valid_indices] = overlap_area[valid_indices] / area[indices[1:]][valid_indices]
        
        # Find indices of boxes to remove
        to_delete = np.concatenate(([0], np.where(overlap_ratio > overlap_threshold)[0] + 1))
        
        # Update indices
        indices = np.delete(indices, to_delete)
    
    # Return kept boxes
    return [boxes[i] for i in keep] 