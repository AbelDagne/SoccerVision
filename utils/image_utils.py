"""
Utility functions for the simplified SoccerVision project.
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
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
    return cv2.imwrite(output_path, image)

def visualize_results(image, player_boxes, team_labels, output_path=None):
    """
    Visualize player detection and team classification results.
    
    Args:
        image (numpy.ndarray): Original input image.
        player_boxes (list): List of player bounding boxes in format [(x, y, w, h), ...].
        team_labels (list): List of team labels corresponding to each player box (0 or 1).
        output_path (str, optional): Path to save the visualization.
        
    Returns:
        numpy.ndarray: Visualization image with player bounding boxes color-coded by team.
    """
    visualization = image.copy()
    
    # Define team colors (BGR format) - using more distinctive colors
    team_a_color = (0, 0, 255)  # Red (Team A)
    team_b_color = (255, 0, 0)  # Blue (Team B)
    
    # Add a semi-transparent overlay to make the visualization more clear
    overlay = visualization.copy()
    
    # Draw player bounding boxes with team colors
    for i, ((x, y, w, h), team_label) in enumerate(zip(player_boxes, team_labels)):
        color = team_a_color if team_label == 0 else team_b_color
        team_name = "Team A" if team_label == 0 else "Team B"
        
        # Draw a thicker rectangle around player for better visibility
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 3)
        
        # Draw a filled rectangle for the team label background for better readability
        label_bg_color = (0, 0, 200, 0.7) if team_label == 0 else (200, 0, 0, 0.7)
        text_size = cv2.getTextSize(team_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(overlay, 
                     (x, y - 25), 
                     (x + text_size[0] + 10, y), 
                     color, 
                     -1)  # Filled rectangle
        
        # Draw team label with white text for better contrast
        cv2.putText(overlay, team_name, (x + 5, y - 8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw player number
        cv2.putText(overlay, f"#{i+1}", (x + w - 20, y + h - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Blend overlay with original image for semi-transparency
    alpha = 0.7  # Transparency factor
    cv2.addWeighted(overlay, alpha, visualization, 1 - alpha, 0, visualization)
    
    # Draw summary at the bottom of image
    team_a_count = len(team_labels) - sum(team_labels)
    team_b_count = sum(team_labels)
    summary_text = f"Detected: {len(player_boxes)} players (Team A: {team_a_count}, Team B: {team_b_count})"
    
    # Put text at the bottom of the image
    cv2.rectangle(visualization, 
                 (10, visualization.shape[0] - 40), 
                 (10 + len(summary_text)*10, visualization.shape[0] - 10), 
                 (0, 0, 0), 
                 -1)  # Black background
    cv2.putText(visualization, summary_text, (15, visualization.shape[0] - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Save the visualization if output path is provided
    if output_path:
        save_image(visualization, output_path)
    
    return visualization

def enhance_contrast(image):
    """
    Enhance the contrast of an image to improve feature detection.
    
    Args:
        image (numpy.ndarray): Input image.
        
    Returns:
        numpy.ndarray: Contrast-enhanced image.
    """
    # Convert to grayscale if the image is in color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply histogram equalization to enhance contrast
    enhanced = cv2.equalizeHist(gray)
    
    return enhanced 