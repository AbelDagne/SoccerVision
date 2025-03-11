"""
Ball detection module for the SoccerVision project.

This module contains functions for detecting the soccer ball in an image
using circle detection and color filtering.
"""

import cv2
import numpy as np

def detect_ball_hough(image, field_mask=None, debug=False):
    """
    Detect the soccer ball using Hough Circle Transform.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        field_mask (numpy.ndarray, optional): Binary mask of the field to limit detection area.
        debug (bool): If True, return intermediate results for debugging.
        
    Returns:
        tuple: Ball position (x, y) or None if not found.
        dict (optional): Dictionary containing intermediate results for debugging.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply field mask if provided
    if field_mask is not None:
        gray = cv2.bitwise_and(gray, gray, mask=field_mask)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Apply Hough Circle Transform
    # TODO: Make these parameters configurable or adaptive
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=50,
        param2=30,
        minRadius=5,
        maxRadius=30
    )
    
    ball_position = None
    detected_circles = []
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        # Filter circles by additional criteria (e.g., color, position)
        for i in circles[0, :]:
            x, y, r = i[0], i[1], i[2]
            
            # Check if circle is on the field (if field mask is provided)
            if field_mask is not None:
                if y >= field_mask.shape[0] or x >= field_mask.shape[1]:
                    continue
                if field_mask[y, x] == 0:
                    continue
            
            # Extract circle region
            # Ensure coordinates are within image bounds
            y_min = max(0, y - r)
            y_max = min(image.shape[0], y + r)
            x_min = max(0, x - r)
            x_max = min(image.shape[1], x + r)
            
            roi = image[y_min:y_max, x_min:x_max]
            
            if roi.size == 0:
                continue
            
            # Convert to HSV for color analysis
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            # Check if circle is white (simplified)
            # TODO: Make these thresholds configurable
            # White has low saturation and high value
            saturation = np.mean(hsv_roi[:, :, 1])
            value = np.mean(hsv_roi[:, :, 2])
            
            if saturation < 50 and value > 150:
                detected_circles.append((x, y, r))
                
                # If we haven't found a ball yet, use this one
                if ball_position is None:
                    ball_position = (x, y)
    
    if debug:
        # Create a visualization of the ball detection
        detection_image = image.copy()
        
        # Draw all detected circles
        if circles is not None:
            for i in circles[0, :]:
                x, y, r = i[0], i[1], i[2]
                # Draw the outer circle
                cv2.circle(detection_image, (x, y), r, (0, 255, 0), 2)
                # Draw the center of the circle
                cv2.circle(detection_image, (x, y), 2, (0, 0, 255), 3)
        
        # Highlight the selected ball
        if ball_position is not None:
            x, y = ball_position
            # Find the radius of the selected ball
            for x_c, y_c, r_c in detected_circles:
                if x_c == x and y_c == y:
                    # Draw the ball with a different color
                    cv2.circle(detection_image, (x, y), r_c, (0, 255, 255), 2)
                    cv2.circle(detection_image, (x, y), 2, (0, 0, 255), 3)
                    break
        
        debug_info = {
            'gray': gray,
            'blurred': blurred,
            'circles': circles,
            'detected_circles': detected_circles,
            'ball_position': ball_position,
            'detection_image': detection_image
        }
        return ball_position, debug_info
    
    return ball_position

def detect_ball_color(image, field_mask=None, debug=False):
    """
    Detect the soccer ball using color thresholding.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        field_mask (numpy.ndarray, optional): Binary mask of the field to limit detection area.
        debug (bool): If True, return intermediate results for debugging.
        
    Returns:
        tuple: Ball position (x, y) or None if not found.
        dict (optional): Dictionary containing intermediate results for debugging.
    """
    # Convert to HSV for color thresholding
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range for white color in HSV
    # TODO: Make these thresholds configurable
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([180, 50, 255])
    
    # Create a mask for white pixels
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Apply field mask if provided
    if field_mask is not None:
        white_mask = cv2.bitwise_and(white_mask, white_mask, mask=field_mask)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    ball_position = None
    ball_contour = None
    ball_circularity = 0  # Initialize ball_circularity
    
    if contours:
        # Filter contours by area and circularity
        for contour in contours:
            # Calculate contour area
            area = cv2.contourArea(contour)
            
            # Skip very small contours
            if area < 100:
                continue
            
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # A perfect circle has circularity = 1
            if circularity > 0.7:
                # Calculate the center of the contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # If we haven't found a ball yet, use this one
                    if ball_position is None:
                        ball_position = (cx, cy)
                        ball_contour = contour
                        ball_circularity = circularity
                    # Otherwise, use the one with higher circularity
                    elif circularity > ball_circularity:
                        ball_position = (cx, cy)
                        ball_contour = contour
                        ball_circularity = circularity
    
    if debug:
        # Create a visualization of the ball detection
        detection_image = image.copy()
        
        # Draw all contours
        cv2.drawContours(detection_image, contours, -1, (0, 255, 0), 2)
        
        # Highlight the selected ball
        if ball_position is not None:
            x, y = ball_position
            # Draw the ball contour
            cv2.drawContours(detection_image, [ball_contour], 0, (0, 255, 255), 2)
            # Draw the center of the ball
            cv2.circle(detection_image, (x, y), 5, (0, 0, 255), -1)
        
        debug_info = {
            'hsv': hsv,
            'white_mask': white_mask,
            'contours': contours,
            'ball_position': ball_position,
            'ball_contour': ball_contour,
            'detection_image': detection_image
        }
        return ball_position, debug_info
    
    return ball_position

def detect_ball(image, method='hough', field_mask=None, debug=False):
    """
    Detect the soccer ball in an image using the specified method.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        method (str): Detection method to use ('hough' or 'color').
        field_mask (numpy.ndarray, optional): Binary mask of the field to limit detection area.
        debug (bool): If True, return intermediate results for debugging.
        
    Returns:
        tuple: Ball position (x, y) or None if not found.
        dict (optional): Dictionary containing intermediate results for debugging.
    """
    if method.lower() == 'hough':
        return detect_ball_hough(image, field_mask, debug)
    elif method.lower() == 'color':
        return detect_ball_color(image, field_mask, debug)
    else:
        raise ValueError(f"Unknown ball detection method: {method}") 