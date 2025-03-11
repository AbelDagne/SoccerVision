"""
Mapping module for the SoccerVision project.

This module contains functions for transforming player and ball positions
from the image plane to a 2D top-down view using homography.
"""

import cv2
import numpy as np
from utils import draw_field_lines

def compute_homography(src_points, dst_points):
    """
    Compute the homography matrix between source and destination points.
    
    Args:
        src_points (numpy.ndarray): Source points in the format [[x1, y1], [x2, y2], ...].
        dst_points (numpy.ndarray): Destination points in the format [[x1, y1], [x2, y2], ...].
        
    Returns:
        numpy.ndarray: 3x3 homography matrix.
    """
    # Ensure we have at least 4 point correspondences
    if len(src_points) < 4 or len(dst_points) < 4:
        raise ValueError("At least 4 point correspondences are required to compute homography")
    
    # Compute homography matrix
    H, status = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    
    return H

def create_field_template(width=105, height=68, scale=10):
    """
    Create a template of a soccer field for the 2D top-down view.
    
    Args:
        width (int): Width of the field in meters.
        height (int): Height of the field in meters.
        scale (int): Scale factor (pixels per meter).
        
    Returns:
        numpy.ndarray: Image of the field template.
        numpy.ndarray: Array of field corner points in the format [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
    """
    # Create a blank image for the field
    field_template = np.zeros((int(height * scale), int(width * scale), 3), dtype=np.uint8)
    
    # Draw field lines
    field_template = draw_field_lines(field_template, scale)
    
    # Define the corner points of the field
    field_corners = np.array([
        [0, 0],  # Top-left
        [width * scale, 0],  # Top-right
        [width * scale, height * scale],  # Bottom-right
        [0, height * scale]  # Bottom-left
    ], dtype=np.float32)
    
    return field_template, field_corners

def transform_point(point, homography):
    """
    Transform a point using the homography matrix.
    
    Args:
        point (tuple): Point coordinates (x, y).
        homography (numpy.ndarray): 3x3 homography matrix.
        
    Returns:
        tuple: Transformed point coordinates (x, y).
    """
    # Convert point to homogeneous coordinates
    p = np.array([point[0], point[1], 1], dtype=np.float32).reshape(1, 1, 3)
    
    # Apply homography
    p_transformed = cv2.perspectiveTransform(p, homography)[0, 0]
    
    # Convert back to Cartesian coordinates
    return (int(p_transformed[0]), int(p_transformed[1]))

def map_to_2d(image, field_mask, player_boxes, team_labels, ball_position, debug=False):
    """
    Map players and ball positions to a 2D top-down view.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        field_mask (numpy.ndarray): Binary mask of the field.
        player_boxes (list): List of player bounding boxes in the format [(x, y, w, h), ...].
        team_labels (list): List of team labels for each player.
        ball_position (tuple): Position of the ball (x, y) or None if not found.
        debug (bool): If True, return intermediate results for debugging.
        
    Returns:
        numpy.ndarray: 2D top-down view with players and ball positions.
        dict (optional): Dictionary containing intermediate results for debugging.
    """
    # Import field detection functions here to avoid circular imports
    from field_detection import find_field_corners, detect_field_lines
    
    # Detect field lines
    field_lines = detect_field_lines(image, field_mask)
    
    # Find field corners
    image_corners = find_field_corners(field_mask, field_lines)
    
    if image_corners is None:
        # If corners cannot be detected, use the bounding box of the field mask
        contours, _ = cv2.findContours(field_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("Could not find field contours")
        
        # Find the bounding rectangle
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        image_corners = np.array([
            [x, y],  # Top-left
            [x + w, y],  # Top-right
            [x + w, y + h],  # Bottom-right
            [x, y + h]  # Bottom-left
        ], dtype=np.float32)
    
    # Create field template
    field_width = 105  # meters
    field_height = 68  # meters
    scale = 10  # pixels per meter
    
    field_template, template_corners = create_field_template(field_width, field_height, scale)
    
    # Compute homography from image to template
    H = compute_homography(image_corners, template_corners)
    
    # Create a copy of the field template for drawing
    top_down_view = field_template.copy()
    
    # Transform player positions to the top-down view
    player_positions_2d = []
    
    for i, (x, y, w, h) in enumerate(player_boxes):
        # Use the bottom center of the bounding box as the player's position
        player_pos = (x + w // 2, y + h)
        
        # Transform to the top-down view
        player_pos_2d = transform_point(player_pos, H)
        player_positions_2d.append(player_pos_2d)
        
        # Draw the player on the top-down view
        team_color = (0, 0, 255) if team_labels[i] == 0 else (255, 0, 0)  # Red for team A, Blue for team B
        cv2.circle(top_down_view, player_pos_2d, 5, team_color, -1)
        cv2.putText(top_down_view, f"{i}", 
                   (player_pos_2d[0] - 5, player_pos_2d[1] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, team_color, 2)
    
    # Transform ball position to the top-down view
    ball_position_2d = None
    if ball_position is not None:
        ball_position_2d = transform_point(ball_position, H)
        
        # Draw the ball on the top-down view
        cv2.circle(top_down_view, ball_position_2d, 3, (0, 255, 255), -1)
    
    if debug:
        # Create a visualization of the homography
        # Draw the field corners on the original image
        corners_image = image.copy()
        for i, (x, y) in enumerate(image_corners):
            cv2.circle(corners_image, (x, y), 10, (0, 0, 255), -1)
            cv2.putText(corners_image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw the player positions on the original image
        for i, (x, y, w, h) in enumerate(player_boxes):
            player_pos = (x + w // 2, y + h)
            team_color = (0, 0, 255) if team_labels[i] == 0 else (255, 0, 0)
            cv2.circle(corners_image, player_pos, 5, team_color, -1)
            cv2.putText(corners_image, f"{i}", 
                       (player_pos[0] - 5, player_pos[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, team_color, 2)
        
        # Draw the ball position on the original image
        if ball_position is not None:
            cv2.circle(corners_image, ball_position, 5, (0, 255, 255), -1)
        
        debug_info = {
            'image_corners': image_corners,
            'template_corners': template_corners,
            'homography_matrix': H,
            'corners_image': corners_image,
            'field_template': field_template,
            'player_positions_2d': player_positions_2d,
            'ball_position_2d': ball_position_2d,
            'top_down_view': top_down_view
        }
        return top_down_view, debug_info
    
    return top_down_view

def warp_field(image, field_mask, debug=False):
    """
    Warp the field to a top-down view.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        field_mask (numpy.ndarray): Binary mask of the field.
        debug (bool): If True, return intermediate results for debugging.
        
    Returns:
        numpy.ndarray: Warped field image.
        numpy.ndarray: Homography matrix.
        dict (optional): Dictionary containing intermediate results for debugging.
    """
    # Import field detection functions here to avoid circular imports
    from field_detection import find_field_corners, detect_field_lines
    
    # Detect field lines
    field_lines = detect_field_lines(image, field_mask)
    
    # Find field corners
    image_corners = find_field_corners(field_mask, field_lines)
    
    if image_corners is None:
        # If corners cannot be detected, use the bounding box of the field mask
        contours, _ = cv2.findContours(field_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("Could not find field contours")
        
        # Find the bounding rectangle
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        image_corners = np.array([
            [x, y],  # Top-left
            [x + w, y],  # Top-right
            [x + w, y + h],  # Bottom-right
            [x, y + h]  # Bottom-left
        ], dtype=np.float32)
    
    # Create field template
    field_width = 105  # meters
    field_height = 68  # meters
    scale = 10  # pixels per meter
    
    field_template, template_corners = create_field_template(field_width, field_height, scale)
    
    # Compute homography from image to template
    H = compute_homography(image_corners, template_corners)
    
    # Warp the field
    warped_field = cv2.warpPerspective(
        image, H, (field_template.shape[1], field_template.shape[0])
    )
    
    if debug:
        # Create a visualization of the homography
        # Draw the field corners on the original image
        corners_image = image.copy()
        for i, (x, y) in enumerate(image_corners):
            cv2.circle(corners_image, (x, y), 10, (0, 0, 255), -1)
            cv2.putText(corners_image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        debug_info = {
            'image_corners': image_corners,
            'template_corners': template_corners,
            'homography_matrix': H,
            'corners_image': corners_image,
            'field_template': field_template,
            'warped_field': warped_field
        }
        return warped_field, H, debug_info
    
    return warped_field, H 