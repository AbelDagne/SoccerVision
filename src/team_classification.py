"""
Team classification module for the SoccerVision project.

This module contains functions for classifying detected players into teams
based on jersey colors using color clustering techniques.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans

def extract_jersey_color(image, player_box, debug=False):
    """
    Extract the dominant color from a player's jersey region.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        player_box (tuple): Player bounding box in the format (x, y, w, h).
        debug (bool): If True, return intermediate results for debugging.
        
    Returns:
        numpy.ndarray: Dominant color in HSV format.
        dict (optional): Dictionary containing intermediate results for debugging.
    """
    x, y, w, h = player_box
    
    # Extract the player region
    player_roi = image[y:y+h, x:x+w]
    
    # Convert to HSV for better color analysis
    hsv_roi = cv2.cvtColor(player_roi, cv2.COLOR_BGR2HSV)
    
    # Create a mask to focus on jersey (exclude skin, etc.)
    # This is a simplified approach - in a real implementation, you might want to:
    # 1. Use a more sophisticated method to exclude skin tones
    # 2. Focus on the torso region rather than the entire bounding box
    # 3. Apply additional filtering to exclude background pixels
    
    # TODO: Improve jersey region extraction
    # For now, we'll use a simple approach: focus on the middle part of the bounding box
    # Assuming the jersey is in the middle 60% of the bounding box
    jersey_y_start = int(h * 0.2)
    jersey_y_end = int(h * 0.8)
    
    jersey_roi = hsv_roi[jersey_y_start:jersey_y_end, :]
    
    # Reshape the ROI for color clustering
    pixels = jersey_roi.reshape((-1, 3))
    
    # Use K-means to find the dominant colors
    # TODO: Make the number of clusters configurable
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    # Get the dominant color (largest cluster)
    cluster_sizes = np.bincount(kmeans.labels_)
    dominant_cluster = np.argmax(cluster_sizes)
    dominant_color = kmeans.cluster_centers_[dominant_cluster]
    
    if debug:
        # Create a visualization of the jersey extraction
        jersey_mask = np.zeros_like(player_roi)
        jersey_mask[jersey_y_start:jersey_y_end, :] = 255
        
        # Create a color swatch of the dominant color
        color_swatch = np.ones((50, 50, 3), dtype=np.uint8)
        color_swatch[:, :] = dominant_color
        
        # Convert back to BGR for visualization
        color_swatch_bgr = cv2.cvtColor(color_swatch.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        debug_info = {
            'player_roi': player_roi,
            'hsv_roi': hsv_roi,
            'jersey_roi': jersey_roi,
            'jersey_mask': jersey_mask,
            'dominant_color': dominant_color,
            'color_swatch_bgr': color_swatch_bgr
        }
        return dominant_color, debug_info
    
    return dominant_color

def classify_teams(image, player_boxes, debug=False):
    """
    Classify players into teams based on jersey colors.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        player_boxes (list): List of player bounding boxes in the format [(x, y, w, h), ...].
        debug (bool): If True, return intermediate results for debugging.
        
    Returns:
        list: List of team labels (0 or 1) for each player.
        dict (optional): Dictionary containing intermediate results for debugging.
    """
    if not player_boxes:
        return []
    
    # Extract jersey colors for all players
    jersey_colors = []
    debug_info_list = []
    
    for player_box in player_boxes:
        color, debug_info = extract_jersey_color(image, player_box, debug)
        jersey_colors.append(color)
        if debug:
            debug_info_list.append(debug_info)
    
    # Convert to numpy array
    jersey_colors = np.array(jersey_colors)
    
    # Use K-means to cluster players into two teams
    # TODO: Handle cases where there might be more than two teams (e.g., referees)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    team_labels = kmeans.fit_predict(jersey_colors)
    
    # Get the average color for each team
    team_colors = []
    for label in range(2):
        team_mask = team_labels == label
        if np.any(team_mask):
            team_color = np.mean(jersey_colors[team_mask], axis=0)
            team_colors.append(team_color)
        else:
            team_colors.append(np.zeros(3))
    
    if debug:
        # Create color swatches for team colors
        team_swatches = []
        for color in team_colors:
            swatch = np.ones((50, 50, 3), dtype=np.uint8)
            swatch[:, :] = color
            swatch_bgr = cv2.cvtColor(swatch.astype(np.uint8), cv2.COLOR_HSV2BGR)
            team_swatches.append(swatch_bgr)
        
        # Create a visualization of the team classification
        team_image = image.copy()
        for i, (x, y, w, h) in enumerate(player_boxes):
            color = (0, 0, 255) if team_labels[i] == 0 else (255, 0, 0)
            cv2.rectangle(team_image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(team_image, f"Team {team_labels[i]}", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        debug_info = {
            'jersey_colors': jersey_colors,
            'team_labels': team_labels,
            'team_colors': team_colors,
            'team_swatches': team_swatches,
            'team_image': team_image,
            'player_debug_info': debug_info_list
        }
        return team_labels, debug_info
    
    return team_labels

def classify_teams_with_reference(image, player_boxes, reference_team_colors, debug=False):
    """
    Classify players into teams based on reference team colors.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        player_boxes (list): List of player bounding boxes in the format [(x, y, w, h), ...].
        reference_team_colors (list): List of reference team colors in HSV format.
        debug (bool): If True, return intermediate results for debugging.
        
    Returns:
        list: List of team labels (0 or 1) for each player.
        dict (optional): Dictionary containing intermediate results for debugging.
    """
    if not player_boxes:
        return []
    
    # Extract jersey colors for all players
    jersey_colors = []
    debug_info_list = []
    
    for player_box in player_boxes:
        color, debug_info = extract_jersey_color(image, player_box, debug)
        jersey_colors.append(color)
        if debug:
            debug_info_list.append(debug_info)
    
    # Assign players to teams based on color similarity
    team_labels = []
    for color in jersey_colors:
        # Calculate color distance to each reference team color
        distances = [np.linalg.norm(color - ref_color) for ref_color in reference_team_colors]
        # Assign to the team with the closest color
        team_label = np.argmin(distances)
        team_labels.append(team_label)
    
    if debug:
        # Create color swatches for reference team colors
        team_swatches = []
        for color in reference_team_colors:
            swatch = np.ones((50, 50, 3), dtype=np.uint8)
            swatch[:, :] = color
            swatch_bgr = cv2.cvtColor(swatch.astype(np.uint8), cv2.COLOR_HSV2BGR)
            team_swatches.append(swatch_bgr)
        
        # Create a visualization of the team classification
        team_image = image.copy()
        for i, (x, y, w, h) in enumerate(player_boxes):
            color = (0, 0, 255) if team_labels[i] == 0 else (255, 0, 0)
            cv2.rectangle(team_image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(team_image, f"Team {team_labels[i]}", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        debug_info = {
            'jersey_colors': jersey_colors,
            'team_labels': team_labels,
            'reference_team_colors': reference_team_colors,
            'team_swatches': team_swatches,
            'team_image': team_image,
            'player_debug_info': debug_info_list
        }
        return team_labels, debug_info
    
    return team_labels 