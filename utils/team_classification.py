"""
Team classification module for the simplified SoccerVision project.

This module implements color-based team classification approach using techniques covered in class.
"""

import cv2
import numpy as np

def extract_jersey_color(image, player_box):
    """
    Extract the dominant color from a player's jersey region.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        player_box (tuple): Player bounding box in the format (x, y, w, h).
        
    Returns:
        tuple: (average_h, average_s, average_v, dominant_h, dominant_s) in HSV color space
    """
    x, y, w, h = player_box
    
    # Ensure coordinates are within image bounds
    x = max(0, x)
    y = max(0, y)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)
    
    # Extract the player region
    player_roi = image[y:y+h, x:x+w]
    
    if player_roi.size == 0:
        # Return a default color if ROI is empty
        return (0, 0, 0, 0, 0)
    
    # Convert to HSV for better color analysis (technique from HW1/HW4)
    hsv_roi = cv2.cvtColor(player_roi, cv2.COLOR_BGR2HSV)
    
    # Compute average HSV values (this will be more robust)
    avg_h = np.mean(hsv_roi[:,:,0])
    avg_s = np.mean(hsv_roi[:,:,1]) 
    avg_v = np.mean(hsv_roi[:,:,2])
    
    # For aerial view, we use the whole player blob
    # Compute histogram for color analysis (technique from HW4)
    hist_h = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
    hist_s = cv2.calcHist([hsv_roi], [1], None, [256], [0, 256])
    
    # Find dominant hue and saturation
    dominant_h = np.argmax(hist_h)
    dominant_s = np.argmax(hist_s)
    
    # Return a more comprehensive set of color features
    return (avg_h, avg_s, avg_v, dominant_h, dominant_s)

def classify_teams(image, player_boxes):
    """
    Classify players into teams based on jersey colors.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        player_boxes (list): List of player bounding boxes in the format [(x, y, w, h), ...].
        
    Returns:
        list: List of team labels corresponding to each player box (0 or 1).
    """
    if not player_boxes:
        return []
    
    # Extract jersey colors for all players
    jersey_colors = [extract_jersey_color(image, box) for box in player_boxes]
    
    # Create feature vectors using average hue and saturation (more stable)
    # We're combining both average and dominant values for better classification
    feature_vectors = np.array([(color[0], color[1]) for color in jersey_colors])
    
    # We'll use a more robust clustering algorithm
    # that considers distribution of colors
    team_labels = improved_binary_clustering(feature_vectors)
    
    # Debug information (uncomment if needed)
    # print("Feature vectors:", feature_vectors)
    # print("Team labels:", team_labels)
    # print("Team A count:", len(team_labels) - sum(team_labels))
    # print("Team B count:", sum(team_labels)) 
    
    # If classification is too unbalanced, try manual median-based approach
    team_a_count = len(team_labels) - sum(team_labels)
    team_b_count = sum(team_labels)
    min_expected = len(team_labels) * 0.2  # Expect at least 20% of players in each team
    
    if team_a_count < min_expected or team_b_count < min_expected:
        # Fall back to median-based classification if clustering gave unbalanced results
        return classify_teams_manual_threshold(image, player_boxes)
    
    return team_labels

def improved_binary_clustering(features):
    """
    Improved implementation of binary clustering for team classification.
    
    Args:
        features (numpy.ndarray): Feature vectors to cluster with shape (n, 2).
        
    Returns:
        list: Cluster labels (0 or 1) for each feature vector.
    """
    if len(features) <= 1:
        return [0] * len(features)
    
    # Normalize features to 0-1 range for better clustering
    # This helps when dealing with different scales (hue vs saturation)
    min_vals = np.min(features, axis=0)
    max_vals = np.max(features, axis=0)
    
    # Avoid division by zero
    range_vals = np.maximum(max_vals - min_vals, 1e-10)
    
    # Normalize features
    normalized_features = (features - min_vals) / range_vals
    
    # Better initialization strategy:
    # Find the two points that are furthest apart
    # This is a better initialization than just min and max
    distances = np.zeros((len(normalized_features), len(normalized_features)))
    for i in range(len(normalized_features)):
        for j in range(i+1, len(normalized_features)):
            distances[i, j] = np.linalg.norm(normalized_features[i] - normalized_features[j])
            distances[j, i] = distances[i, j]
    
    # Find two points with maximum distance
    i, j = np.unravel_index(np.argmax(distances), distances.shape)
    centroid_1 = normalized_features[i]
    centroid_2 = normalized_features[j]
    
    # Iterative refinement with more iterations for convergence
    max_iterations = 20
    labels = None
    
    for _ in range(max_iterations):
        # Compute distances to centroids
        dist_to_c1 = np.array([np.linalg.norm(f - centroid_1) for f in normalized_features])
        dist_to_c2 = np.array([np.linalg.norm(f - centroid_2) for f in normalized_features])
        
        # Assign each point to the nearest centroid
        new_labels = np.where(dist_to_c1 <= dist_to_c2, 0, 1)
        
        # If labels haven't changed, we've converged
        if labels is not None and np.all(new_labels == labels):
            break
            
        labels = new_labels
        
        # Update centroids
        if np.any(labels == 0):
            centroid_1 = np.mean(normalized_features[labels == 0], axis=0)
        if np.any(labels == 1):
            centroid_2 = np.mean(normalized_features[labels == 1], axis=0)
    
    return labels.tolist()

def classify_teams_manual_threshold(image, player_boxes):
    """
    Alternative approach using manual thresholding for team classification.
    Useful when teams have distinctly different colors.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        player_boxes (list): List of player bounding boxes in the format [(x, y, w, h), ...].
        
    Returns:
        list: List of team labels corresponding to each player box (0 or 1).
    """
    # Extract jersey colors for all players
    jersey_colors = [extract_jersey_color(image, box) for box in player_boxes]
    
    # Use average hue values instead of just dominant ones
    hue_values = [color[0] for color in jersey_colors]
    
    # Compute the median hue
    median_hue = np.median(hue_values)
    
    # Classify based on hue relative to median
    team_labels = [0 if hue < median_hue else 1 for hue in hue_values]
    
    return team_labels 