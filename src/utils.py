"""
Utility functions for the SoccerVision project.
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def load_image(image_path):
    """
    Load an image from the specified path.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        numpy.ndarray: Loaded image in BGR format.
        
    Raises:
        FileNotFoundError: If the image file does not exist.
        Exception: If the image cannot be loaded.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        raise Exception(f"Failed to load image: {image_path}")
    
    return image

def save_image(image, output_path):
    """
    Save an image to the specified path.
    
    Args:
        image (numpy.ndarray): Image to save.
        output_path (str): Path where the image will be saved.
        
    Returns:
        bool: True if the image was saved successfully, False otherwise.
    """
    directory = os.path.dirname(output_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    return cv2.imwrite(output_path, image)

def visualize_results(original_image, field_mask, player_boxes, team_labels, ball_position, top_down_view, output_path=None):
    """
    Visualize the results of the SoccerVision pipeline.
    
    Args:
        original_image (numpy.ndarray): Original input image.
        field_mask (numpy.ndarray): Field detection mask.
        player_boxes (list): List of player bounding boxes (x, y, w, h).
        team_labels (list): List of team labels for each player.
        ball_position (tuple): Position of the ball (x, y).
        top_down_view (numpy.ndarray): 2D top-down mapping.
        output_path (str, optional): Path to save the visualization. If None, the visualization is displayed.
    """
    # Convert BGR to RGB for matplotlib
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Create a copy of the original image for drawing
    result_image = original_rgb.copy()
    
    # Draw player bounding boxes and team labels
    if player_boxes and len(player_boxes) > 0:
        for i, (x, y, w, h) in enumerate(player_boxes):
            if i < len(team_labels):  # Make sure we have a team label for this player
                color = (0, 0, 255) if team_labels[i] == 0 else (255, 0, 0)  # Red for team A, Blue for team B
                cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(result_image, f"Team {'A' if team_labels[i] == 0 else 'B'}", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Draw ball position
    if ball_position is not None:
        x, y = ball_position
        cv2.circle(result_image, (x, y), 10, (0, 255, 255), -1)
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original image with detections
    axs[0, 0].imshow(result_image)
    axs[0, 0].set_title('Player and Ball Detection')
    axs[0, 0].axis('off')
    
    # Field mask
    axs[0, 1].imshow(field_mask, cmap='gray')
    axs[0, 1].set_title('Field Mask')
    axs[0, 1].axis('off')
    
    # Original image with field overlay
    field_overlay = cv2.addWeighted(original_rgb, 0.7, 
                                   cv2.cvtColor(cv2.merge([field_mask, field_mask, field_mask]), 
                                               cv2.COLOR_BGR2RGB), 0.3, 0)
    axs[1, 0].imshow(field_overlay)
    axs[1, 0].set_title('Field Detection Overlay')
    axs[1, 0].axis('off')
    
    # 2D top-down mapping
    axs[1, 1].imshow(top_down_view)
    axs[1, 1].set_title('2D Top-Down Mapping')
    axs[1, 1].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def draw_field_lines(top_down_view, scale=10):
    """
    Draw soccer field lines on a top-down view.
    
    Args:
        top_down_view (numpy.ndarray): Top-down view image.
        scale (int): Scale factor (pixels per meter).
        
    Returns:
        numpy.ndarray: Top-down view with field lines.
    """
    # Standard soccer field dimensions (in meters)
    field_width = 105
    field_height = 68
    
    # Create a copy of the top-down view
    field_lines = top_down_view.copy()
    
    # Field color (green)
    cv2.rectangle(field_lines, (0, 0), 
                 (int(field_width * scale), int(field_height * scale)), 
                 (0, 128, 0), -1)
    
    # Outer boundary (white)
    cv2.rectangle(field_lines, (0, 0), 
                 (int(field_width * scale), int(field_height * scale)), 
                 (255, 255, 255), 2)
    
    # Center line
    cv2.line(field_lines, 
            (int(field_width * scale / 2), 0), 
            (int(field_width * scale / 2), int(field_height * scale)), 
            (255, 255, 255), 2)
    
    # Center circle (radius = 9.15m)
    cv2.circle(field_lines, 
              (int(field_width * scale / 2), int(field_height * scale / 2)), 
              int(9.15 * scale), 
              (255, 255, 255), 2)
    
    # Penalty areas
    # Left penalty area
    cv2.rectangle(field_lines, 
                 (0, int((field_height * scale / 2) - 20.16 * scale / 2)), 
                 (int(16.5 * scale), int((field_height * scale / 2) + 20.16 * scale / 2)), 
                 (255, 255, 255), 2)
    
    # Right penalty area
    cv2.rectangle(field_lines, 
                 (int(field_width * scale - 16.5 * scale), int((field_height * scale / 2) - 20.16 * scale / 2)), 
                 (int(field_width * scale), int((field_height * scale / 2) + 20.16 * scale / 2)), 
                 (255, 255, 255), 2)
    
    # Goal areas
    # Left goal area
    cv2.rectangle(field_lines, 
                 (0, int((field_height * scale / 2) - 7.32 * scale / 2)), 
                 (int(5.5 * scale), int((field_height * scale / 2) + 7.32 * scale / 2)), 
                 (255, 255, 255), 2)
    
    # Right goal area
    cv2.rectangle(field_lines, 
                 (int(field_width * scale - 5.5 * scale), int((field_height * scale / 2) - 7.32 * scale / 2)), 
                 (int(field_width * scale), int((field_height * scale / 2) + 7.32 * scale / 2)), 
                 (255, 255, 255), 2)
    
    # Penalty spots
    cv2.circle(field_lines, 
              (int(11 * scale), int(field_height * scale / 2)), 
              int(0.5 * scale), 
              (255, 255, 255), -1)
    
    cv2.circle(field_lines, 
              (int(field_width * scale - 11 * scale), int(field_height * scale / 2)), 
              int(0.5 * scale), 
              (255, 255, 255), -1)
    
    return field_lines 