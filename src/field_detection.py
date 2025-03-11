"""
Field detection module for the SoccerVision project.

This module contains functions for detecting and segmenting the soccer field
from an input image using color-based thresholding and edge detection.
"""

import cv2
import numpy as np

def detect_field_hsv(image, debug=False):
    """
    Detect the soccer field using HSV color thresholding.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        debug (bool): If True, return intermediate results for debugging.
        
    Returns:
        numpy.ndarray: Binary mask of the field (255 for field pixels, 0 for non-field).
        dict (optional): Dictionary containing intermediate results for debugging.
    """
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range of green color in HSV
    # TODO: Make these thresholds configurable or adaptive based on the image
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([90, 255, 255])
    
    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find the largest contour (assuming it's the field)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create a new mask with only the largest contour
        field_mask = np.zeros_like(mask)
        cv2.drawContours(field_mask, [largest_contour], 0, 255, -1)
        
        # Fill holes in the mask
        field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, kernel)
    else:
        # If no contours found, return the original mask
        field_mask = mask
    
    if debug:
        debug_info = {
            'hsv': hsv,
            'initial_mask': mask,
            'field_mask': field_mask,
            'contours': contours
        }
        return field_mask, debug_info
    
    return field_mask

def detect_field_edges(image, debug=False):
    """
    Detect the soccer field using edge detection and Hough transform.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        debug (bool): If True, return intermediate results for debugging.
        
    Returns:
        numpy.ndarray: Binary mask of the field (255 for field pixels, 0 for non-field).
        dict (optional): Dictionary containing intermediate results for debugging.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    # TODO: Make these thresholds adaptive based on the image
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate edges to connect broken lines
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Find lines using Hough transform
    lines = cv2.HoughLinesP(dilated_edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=50)
    
    # Create a mask for the lines
    line_mask = np.zeros_like(gray)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
    
    # Use the HSV method as a fallback/complement
    hsv_mask = detect_field_hsv(image)
    
    # Combine the line mask with the HSV mask
    combined_mask = cv2.bitwise_and(hsv_mask, hsv_mask, mask=line_mask)
    
    # If the combined mask is too small, use the HSV mask
    if np.sum(combined_mask) < np.sum(hsv_mask) * 0.1:
        field_mask = hsv_mask
    else:
        field_mask = combined_mask
    
    if debug:
        debug_info = {
            'gray': gray,
            'blurred': blurred,
            'edges': edges,
            'dilated_edges': dilated_edges,
            'line_mask': line_mask,
            'hsv_mask': hsv_mask,
            'combined_mask': combined_mask,
            'field_mask': field_mask
        }
        return field_mask, debug_info
    
    return field_mask

def detect_field_lines(image, field_mask, debug=False):
    """
    Detect field lines for homography reference points.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        field_mask (numpy.ndarray): Binary mask of the field.
        debug (bool): If True, return intermediate results for debugging.
        
    Returns:
        list: List of detected lines in the format [(x1, y1, x2, y2), ...].
        dict (optional): Dictionary containing intermediate results for debugging.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply the field mask
    masked_gray = cv2.bitwise_and(gray, gray, mask=field_mask)
    
    # Apply adaptive thresholding to highlight white lines
    # TODO: Make these parameters configurable
    thresh = cv2.adaptiveThreshold(
        masked_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Invert the threshold to get white lines as white pixels
    thresh = cv2.bitwise_not(thresh)
    
    # Apply morphological operations to clean up the lines
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Apply Hough transform to detect lines
    lines = cv2.HoughLinesP(
        thresh, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10
    )
    
    detected_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            detected_lines.append((x1, y1, x2, y2))
    
    if debug:
        # Create a visualization of the detected lines
        line_image = image.copy()
        for x1, y1, x2, y2 in detected_lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        debug_info = {
            'masked_gray': masked_gray,
            'thresh': thresh,
            'line_image': line_image
        }
        return detected_lines, debug_info
    
    return detected_lines

def find_field_corners(field_mask, lines, debug=False):
    """
    Find the four corners of the soccer field for homography calculation.
    
    Args:
        field_mask (numpy.ndarray): Binary mask of the field.
        lines (list): List of detected lines in the format [(x1, y1, x2, y2), ...].
        debug (bool): If True, return intermediate results for debugging.
        
    Returns:
        numpy.ndarray: Array of four corner points in the format [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
        dict (optional): Dictionary containing intermediate results for debugging.
    """
    # TODO: Implement a more robust method to find field corners
    # This is a simplified approach that may not work for all images
    
    # Find the contour of the field mask
    contours, _ = cv2.findContours(field_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # If no contours found, return None
        if debug:
            return None, {'error': 'No contours found in field mask'}
        return None
    
    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Approximate the contour with a polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # If we have more than 4 points, find the 4 corners
    if len(approx) > 4:
        # Find the bounding rectangle
        rect = cv2.minAreaRect(approx)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        corners = box
    elif len(approx) == 4:
        # If we already have 4 points, use them
        corners = approx.reshape(4, 2)
    else:
        # If we have less than 4 points, use the convex hull
        hull = cv2.convexHull(largest_contour)
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        if len(approx) >= 4:
            # Find the bounding rectangle
            rect = cv2.minAreaRect(approx)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            corners = box
        else:
            # If still less than 4 points, return None
            if debug:
                return None, {'error': 'Could not find 4 corners in field mask'}
            return None
    
    # Sort corners in order: top-left, top-right, bottom-right, bottom-left
    # First, sort by y-coordinate (top to bottom)
    corners = corners[np.argsort(corners[:, 1])]
    
    # Sort the top two points by x-coordinate (left to right)
    top_points = corners[:2]
    top_points = top_points[np.argsort(top_points[:, 0])]
    
    # Sort the bottom two points by x-coordinate (left to right)
    bottom_points = corners[2:]
    bottom_points = bottom_points[np.argsort(bottom_points[:, 0])]
    
    # Combine the sorted points
    corners = np.vstack((top_points, bottom_points[::-1]))
    
    if debug:
        # Create a visualization of the corners
        corner_image = cv2.cvtColor(field_mask, cv2.COLOR_GRAY2BGR)
        for i, (x, y) in enumerate(corners):
            cv2.circle(corner_image, (x, y), 10, (0, 0, 255), -1)
            cv2.putText(corner_image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        debug_info = {
            'contours': contours,
            'largest_contour': largest_contour,
            'approx': approx,
            'corners': corners,
            'corner_image': corner_image
        }
        return corners, debug_info
    
    return corners 