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
    
    # Convert to HSV for better color analysis
    hsv_roi = cv2.cvtColor(player_roi, cv2.COLOR_BGR2HSV)
    
    # For aerial view, focus more on the center of the player blob
    # This helps avoid including field pixels in the jersey color
    center_y = h // 2
    center_x = w // 2
    center_roi_size = min(w, h) // 2
    
    # Extract center region (more likely to be jersey)
    start_y = max(0, center_y - center_roi_size)
    end_y = min(h, center_y + center_roi_size)
    start_x = max(0, center_x - center_roi_size)
    end_x = min(w, center_x + center_roi_size)
    
    center_roi = hsv_roi[start_y:end_y, start_x:end_x]
    
    if center_roi.size == 0:
        # Fall back to using the whole ROI if center extraction failed
        center_roi = hsv_roi
    
    # Create a mask to filter out green (field) pixels
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(center_roi, lower_green, upper_green)
    non_green_mask = cv2.bitwise_not(green_mask)
    
    # Apply the mask to focus on non-green pixels (likely jersey colors)
    masked_roi = cv2.bitwise_and(center_roi, center_roi, mask=non_green_mask)
    
    # Count non-zero pixels to ensure we have enough data
    non_zero_pixels = cv2.countNonZero(non_green_mask)
    
    if non_zero_pixels > 10:
        # Calculate average HSV on non-green pixels
        non_zero_indices = np.where(non_green_mask > 0)
        avg_h = np.mean(center_roi[non_zero_indices[0], non_zero_indices[1], 0])
        avg_s = np.mean(center_roi[non_zero_indices[0], non_zero_indices[1], 1])
        avg_v = np.mean(center_roi[non_zero_indices[0], non_zero_indices[1], 2])
        
        # Compute histogram for non-green pixels
        hist_h = cv2.calcHist([center_roi], [0], non_green_mask, [180], [0, 180])
        hist_s = cv2.calcHist([center_roi], [1], non_green_mask, [256], [0, 256])
        
        # Find dominant hue and saturation
        dominant_h = np.argmax(hist_h)
        dominant_s = np.argmax(hist_s)
    else:
        # Fall back to whole ROI if not enough non-green pixels
        avg_h = np.mean(hsv_roi[:,:,0])
        avg_s = np.mean(hsv_roi[:,:,1])
        avg_v = np.mean(hsv_roi[:,:,2])
        
        hist_h = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([hsv_roi], [1], None, [256], [0, 256])
        
        dominant_h = np.argmax(hist_h)
        dominant_s = np.argmax(hist_s)
    
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
    
    # Create feature vectors using multiple color attributes for better robustness
    # We'll use both hue and saturation from average and dominant values
    feature_vectors = np.array([
        (color[0], color[1], color[3], color[4]) for color in jersey_colors
    ])
    
    # Weight features to emphasize hue more than saturation
    weights = np.array([1.5, 0.8, 1.0, 0.5])
    weighted_features = feature_vectors * weights
    
    # Use DBSCAN clustering which can automatically determine the number of clusters
    # and is more robust to noise
    try:
        from sklearn.cluster import DBSCAN
        
        # Normalize features for DBSCAN
        if len(weighted_features) > 0:
            # Avoid division by zero by adding a small epsilon
            feature_range = np.max(weighted_features, axis=0) - np.min(weighted_features, axis=0)
            feature_range = np.maximum(feature_range, 1e-10)
            norm_features = (weighted_features - np.min(weighted_features, axis=0)) / feature_range
            
            # Apply DBSCAN clustering
            db = DBSCAN(eps=0.3, min_samples=2).fit(norm_features)
            cluster_labels = db.labels_
            
            # Count number of clusters found (excluding noise points labeled as -1)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            
            # If DBSCAN found exactly 2 clusters, use them
            if n_clusters == 2:
                # Convert DBSCAN labels (-1, 0, 1) to binary labels (0, 1)
                # First, handle noise points by assigning them to the nearest cluster
                noise_indices = np.where(cluster_labels == -1)[0]
                valid_clusters = np.unique(cluster_labels[cluster_labels != -1])
                
                for idx in noise_indices:
                    # Find distances to each valid cluster center
                    distances = []
                    for cluster in valid_clusters:
                        cluster_points = norm_features[cluster_labels == cluster]
                        cluster_center = np.mean(cluster_points, axis=0)
                        distance = np.linalg.norm(norm_features[idx] - cluster_center)
                        distances.append((cluster, distance))
                    
                    # Assign to nearest cluster
                    nearest_cluster = min(distances, key=lambda x: x[1])[0]
                    cluster_labels[idx] = nearest_cluster
                
                # Convert to binary labels (0, 1)
                return [0 if label == valid_clusters[0] else 1 for label in cluster_labels]
            
            # If not exactly 2 clusters, fall back to our binary clustering
            team_labels = improved_binary_clustering(feature_vectors)
            return team_labels
            
    except (ImportError, Exception) as e:
        # Fall back to our binary clustering if DBSCAN unavailable
        pass
    
    # Use improved binary clustering as fallback
    team_labels = improved_binary_clustering(feature_vectors[:, :2])  # Use only the first two features
    
    # Check if the results are balanced
    team_a_count = len(team_labels) - sum(team_labels)
    team_b_count = sum(team_labels)
    min_expected = len(team_labels) * 0.2  # Expect at least 20% of players in each team
    
    if team_a_count < min_expected or team_b_count < min_expected:
        # Fall back to manual thresholding if clustering gave unbalanced results
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
    
    # Use both average and dominant hue values for more robustness
    avg_hue_values = [color[0] for color in jersey_colors]
    dominant_hue_values = [color[3] for color in jersey_colors]
    
    # Use k-means clustering instead of simple median thresholding
    try:
        from sklearn.cluster import KMeans
        
        # Combine average and dominant hue into feature vector
        features = np.column_stack((avg_hue_values, dominant_hue_values))
        
        # Apply k-means clustering with 2 clusters
        kmeans = KMeans(n_clusters=2, random_state=0).fit(features)
        team_labels = kmeans.labels_
        
        # Ensure team A (label 0) has more players than team B (label 1) if possible
        if np.sum(team_labels == 0) < np.sum(team_labels == 1):
            team_labels = 1 - team_labels  # Flip labels
            
        return team_labels.tolist()
        
    except (ImportError, Exception) as e:
        # Fall back to simpler approach if k-means is not available
        pass
    
    # Simple median-based classification as fallback
    median_hue = np.median(avg_hue_values)
    team_labels = [0 if hue < median_hue else 1 for hue in avg_hue_values]
    
    return team_labels 