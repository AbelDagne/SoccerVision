"""
Test script for field detection module.
"""

import cv2
import numpy as np
from src.field_detection import detect_field_edges, detect_players_on_field

def visualize_results(image_path):
    """
    Visualize the field detection results.
    
    Args:
        image_path (str): Path to the input image.
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    # Get field detection results with debug info
    field_mask, debug_info = detect_field_edges(image, debug=True)
    
    # Detect players
    players, player_debug = detect_players_on_field(image, field_mask, debug=True)
    
    # Create visualizations
    # 1. Original image with detected lines, field outline, and players
    overlay = player_debug['debug_image'].copy()
    
    # Draw the rectangular field outline
    if debug_info['corners'] is not None:
        # Convert corners to integer points for drawing
        corners = debug_info['corners']
        if isinstance(corners, dict):  # If corners is in debug info format
            corners = corners['corners']
        corners = corners.astype(np.int32)
        
        # Draw the rectangle
        cv2.polylines(overlay, [corners], True, (0, 255, 0), 2)
        
        # Draw corner points
        for i, corner in enumerate(corners):
            cv2.circle(overlay, tuple(corner), 5, (0, 0, 255), -1)
            cv2.putText(overlay, str(i), tuple(corner), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Draw detected lines
    if debug_info['detected_lines'] is not None:
        for line in debug_info['detected_lines']:
            x1, y1, x2, y2 = line[0]
            cv2.line(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # 2. Create warped perspective (bird's eye view) with players
    if debug_info['perspective_matrix'] is not None:
        # Get standard dimensions from the matrix calculation
        std_height = 800
        field_width = int(std_height * (overlay.shape[1] / overlay.shape[0]))
        
        # Warp the image
        warped = cv2.warpPerspective(
            overlay,  # Use overlay to include player boxes
            debug_info['perspective_matrix'],
            (field_width, std_height)
        )
    else:
        warped = np.zeros_like(image)
    
    # 3. Player detection mask visualization
    player_mask_vis = cv2.cvtColor(player_debug['player_mask'], cv2.COLOR_GRAY2BGR)
    
    # 4. White mask visualization
    white_mask_vis = cv2.cvtColor(debug_info['white_mask'], cv2.COLOR_GRAY2BGR)
    
    # 5. Line segments visualization
    line_segments_vis = cv2.cvtColor(debug_info['line_segments'], cv2.COLOR_GRAY2BGR)
    
    # Display results
    cv2.imshow('Field Detection with Players', overlay)
    cv2.imshow('Bird\'s Eye View with Players', warped)
    cv2.imshow('Player Detection Mask', player_mask_vis)
    cv2.imshow('White Mask', white_mask_vis)
    cv2.imshow('Line Segments', line_segments_vis)
    
    # Save results
    cv2.imwrite('output/field_detection_with_players.jpg', overlay)
    cv2.imwrite('output/birds_eye_view_with_players.jpg', warped)
    cv2.imwrite('output/player_detection_mask.jpg', player_mask_vis)
    cv2.imwrite('output/white_mask.jpg', white_mask_vis)
    cv2.imwrite('output/line_segments.jpg', line_segments_vis)
    
    # Wait for key press
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Create output directory if it doesn't exist
    import os
    if not os.path.exists('output'):
        os.makedirs('output')
    
    # Test with a sample image
    image_path = 'test_field2.png'  # Replace with your image path
    visualize_results(image_path) 