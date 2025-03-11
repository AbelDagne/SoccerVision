"""
Player detection module for the SoccerVision project.

This module contains functions for detecting players in soccer images
using HOG-based and deep learning-based methods.
"""

import cv2
import numpy as np

def detect_players_hog(image, field_mask=None, debug=False):
    """
    Detect players using HOG (Histogram of Oriented Gradients) detector.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        field_mask (numpy.ndarray, optional): Binary mask of the field to limit detection area.
        debug (bool): If True, return intermediate results for debugging.
        
    Returns:
        list: List of player bounding boxes in the format [(x, y, w, h), ...].
        dict (optional): Dictionary containing intermediate results for debugging.
    """
    # Initialize HOG detector
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    # Apply field mask if provided
    if field_mask is not None:
        # Create a copy of the image
        masked_image = image.copy()
        # Set pixels outside the field to black
        masked_image[field_mask == 0] = [0, 0, 0]
    else:
        masked_image = image
    
    # Detect people in the image
    # TODO: Tune these parameters for better detection
    boxes, weights = hog.detectMultiScale(
        masked_image, 
        winStride=(4, 4),
        padding=(8, 8),
        scale=1.05
    )
    
    # Filter out detections with low confidence
    # TODO: Make this threshold configurable
    confidence_threshold = 0.3
    filtered_boxes = []
    
    for i, (x, y, w, h) in enumerate(boxes):
        if weights[i] > confidence_threshold:
            # If field mask is provided, check if the detection is on the field
            if field_mask is not None:
                # Check if the bottom center of the bounding box is on the field
                bottom_center_x = x + w // 2
                bottom_center_y = y + h
                
                # Ensure coordinates are within image bounds
                if (0 <= bottom_center_x < field_mask.shape[1] and 
                    0 <= bottom_center_y < field_mask.shape[0]):
                    if field_mask[bottom_center_y, bottom_center_x] > 0:
                        filtered_boxes.append((x, y, w, h))
            else:
                filtered_boxes.append((x, y, w, h))
    
    if debug:
        # Create a visualization of the detections
        detection_image = image.copy()
        for (x, y, w, h) in filtered_boxes:
            cv2.rectangle(detection_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        debug_info = {
            'masked_image': masked_image,
            'all_boxes': boxes,
            'weights': weights,
            'filtered_boxes': filtered_boxes,
            'detection_image': detection_image
        }
        return filtered_boxes, debug_info
    
    return filtered_boxes

def detect_players_yolo(image, field_mask=None, debug=False):
    """
    Detect players using YOLO (You Only Look Once) object detection.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        field_mask (numpy.ndarray, optional): Binary mask of the field to limit detection area.
        debug (bool): If True, return intermediate results for debugging.
        
    Returns:
        list: List of player bounding boxes in the format [(x, y, w, h), ...].
        dict (optional): Dictionary containing intermediate results for debugging.
    """
    # TODO: Implement YOLO-based player detection
    # This is a placeholder function that returns an empty list
    # In a real implementation, you would:
    # 1. Load a pre-trained YOLO model
    # 2. Run inference on the image
    # 3. Filter detections to keep only person class
    # 4. Apply field mask if provided
    # 5. Return bounding boxes
    
    print("YOLO-based player detection is not implemented yet.")
    print("Using HOG-based detection as a fallback.")
    
    return detect_players_hog(image, field_mask, debug)

def non_max_suppression(boxes, overlap_threshold=0.3):
    """
    Apply non-maximum suppression to eliminate redundant overlapping bounding boxes.
    
    Args:
        boxes (list): List of bounding boxes in the format [(x, y, w, h), ...].
        overlap_threshold (float): Threshold for considering boxes as overlapping.
        
    Returns:
        list: Filtered list of bounding boxes.
    """
    if len(boxes) == 0:
        return []
    
    # Convert (x, y, w, h) to (x1, y1, x2, y2)
    boxes_xyxy = []
    for x, y, w, h in boxes:
        boxes_xyxy.append([x, y, x + w, y + h])
    
    boxes_xyxy = np.array(boxes_xyxy)
    
    # Extract coordinates
    x1 = boxes_xyxy[:, 0]
    y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]
    y2 = boxes_xyxy[:, 3]
    
    # Calculate area of each box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    
    # Sort boxes by bottom-right y-coordinate
    idxs = np.argsort(y2)
    
    pick = []
    while len(idxs) > 0:
        # Pick the last box (highest y2) and add its index to the list
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        
        # Find the intersection with all other boxes
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        
        # Calculate width and height of the intersection
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        
        # Calculate intersection area
        overlap = (w * h) / area[idxs[:last]]
        
        # Delete all indexes from the index list that have overlap greater than the threshold
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))
    
    # Convert back to (x, y, w, h) format
    result = []
    for i in pick:
        x, y, x2, y2 = boxes_xyxy[i]
        w = x2 - x
        h = y2 - y
        result.append((int(x), int(y), int(w), int(h)))
    
    return result

def detect_players(image, method='hog', field_mask=None, debug=False):
    """
    Detect players in a soccer image using the specified method.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        method (str): Detection method to use ('hog' or 'yolo').
        field_mask (numpy.ndarray, optional): Binary mask of the field to limit detection area.
        debug (bool): If True, return intermediate results for debugging.
        
    Returns:
        list: List of player bounding boxes in the format [(x, y, w, h), ...].
        dict (optional): Dictionary containing intermediate results for debugging.
    """
    if method.lower() == 'hog':
        boxes, debug_info = detect_players_hog(image, field_mask, True)
    elif method.lower() == 'yolo':
        boxes, debug_info = detect_players_yolo(image, field_mask, True)
    else:
        raise ValueError(f"Unknown player detection method: {method}")
    
    # Apply non-maximum suppression to eliminate redundant detections
    filtered_boxes = non_max_suppression(boxes)
    
    if debug:
        # Create a visualization of the final detections
        final_image = image.copy()
        for (x, y, w, h) in filtered_boxes:
            cv2.rectangle(final_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        debug_info['filtered_boxes'] = filtered_boxes
        debug_info['final_image'] = final_image
        return filtered_boxes, debug_info
    
    return filtered_boxes 