"""
Main module for the SoccerVision project.

This module ties all the components together and provides a command-line interface
for processing soccer images.
"""

import os
import argparse
import cv2
import numpy as np
import time

from src.utils import load_image, save_image, visualize_results
from src.field_detection import detect_field_hsv, detect_field_edges, detect_field_lines
from src.player_detection import detect_players
from src.team_classification import classify_teams
from src.ball_detection import detect_ball
from src.mapping import map_to_2d

def process_image(image_path, output_dir, debug=False, player_detection_method='hog', 
                 field_detection_method='hsv', ball_detection_method='hough'):
    """
    Process a soccer image and generate a 2D top-down mapping.
    
    Args:
        image_path (str): Path to the input image.
        output_dir (str): Directory to save the output.
        debug (bool): If True, save intermediate results for debugging.
        player_detection_method (str): Method to use for player detection ('hog' or 'yolo').
        field_detection_method (str): Method to use for field detection ('hsv' or 'edge').
        ball_detection_method (str): Method to use for ball detection ('hough' or 'color').
        
    Returns:
        dict: Dictionary containing the results of the processing.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the image
    print(f"Loading image: {image_path}")
    image = load_image(image_path)
    
    # Get the base filename without extension
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Step 1: Field detection
    print("Step 1: Detecting the field...")
    start_time = time.time()
    
    if field_detection_method.lower() == 'hsv':
        if debug:
            field_mask, field_debug = detect_field_hsv(image, debug=True)
            # Save field detection debug images
            save_image(field_debug['initial_mask'], os.path.join(output_dir, f"{base_filename}_field_initial_mask.jpg"))
            save_image(field_debug['field_mask'], os.path.join(output_dir, f"{base_filename}_field_mask.jpg"))
        else:
            field_mask = detect_field_hsv(image)
    elif field_detection_method.lower() == 'edge':
        if debug:
            field_mask, field_debug = detect_field_edges(image, debug=True)
            # Save field detection debug images
            save_image(field_debug['edges'], os.path.join(output_dir, f"{base_filename}_field_edges.jpg"))
            save_image(field_debug['line_mask'], os.path.join(output_dir, f"{base_filename}_field_line_mask.jpg"))
            save_image(field_debug['field_mask'], os.path.join(output_dir, f"{base_filename}_field_mask.jpg"))
        else:
            field_mask = detect_field_edges(image)
    else:
        raise ValueError(f"Unknown field detection method: {field_detection_method}")
    
    field_detection_time = time.time() - start_time
    print(f"Field detection completed in {field_detection_time:.2f} seconds")
    
    # Save the field mask
    save_image(field_mask, os.path.join(output_dir, f"{base_filename}_field_mask.jpg"))
    
    # Step 2: Player detection
    print("Step 2: Detecting players...")
    start_time = time.time()
    
    if debug:
        player_boxes, player_debug = detect_players(image, player_detection_method, field_mask, debug=True)
        # Save player detection debug images
        save_image(player_debug['detection_image'], os.path.join(output_dir, f"{base_filename}_player_detection.jpg"))
        save_image(player_debug['final_image'], os.path.join(output_dir, f"{base_filename}_player_detection_final.jpg"))
    else:
        player_boxes = detect_players(image, player_detection_method, field_mask)
    
    player_detection_time = time.time() - start_time
    print(f"Player detection completed in {player_detection_time:.2f} seconds")
    print(f"Detected {len(player_boxes)} players")
    
    # Step 3: Team classification
    print("Step 3: Classifying teams...")
    start_time = time.time()
    
    # Check if any players were detected
    if len(player_boxes) > 0:
        if debug:
            team_labels, team_debug = classify_teams(image, player_boxes, debug=True)
            # Save team classification debug images
            save_image(team_debug['team_image'], os.path.join(output_dir, f"{base_filename}_team_classification.jpg"))
        else:
            team_labels = classify_teams(image, player_boxes)
    else:
        # If no players were detected, create an empty team_labels list
        team_labels = []
        if debug:
            # Create a dummy debug info
            team_debug = {'team_image': image.copy()}
    
    team_classification_time = time.time() - start_time
    print(f"Team classification completed in {team_classification_time:.2f} seconds")
    
    # Step 4: Ball detection
    print("Step 4: Detecting the ball...")
    start_time = time.time()
    
    if debug:
        ball_position, ball_debug = detect_ball(image, ball_detection_method, field_mask, debug=True)
        # Save ball detection debug images
        if ball_position is not None:
            save_image(ball_debug['detection_image'], os.path.join(output_dir, f"{base_filename}_ball_detection.jpg"))
    else:
        ball_position = detect_ball(image, ball_detection_method, field_mask)
    
    ball_detection_time = time.time() - start_time
    print(f"Ball detection completed in {ball_detection_time:.2f} seconds")
    if ball_position is not None:
        print(f"Ball detected at position: {ball_position}")
    else:
        print("Ball not detected")
    
    # Step 5: 2D mapping
    print("Step 5: Generating 2D mapping...")
    start_time = time.time()
    
    try:
        # Only attempt mapping if players were detected
        if len(player_boxes) > 0:
            if debug:
                top_down_view, mapping_debug = map_to_2d(image, field_mask, player_boxes, team_labels, ball_position, debug=True)
                # Save mapping debug images
                save_image(mapping_debug['corners_image'], os.path.join(output_dir, f"{base_filename}_field_corners.jpg"))
                save_image(mapping_debug['field_template'], os.path.join(output_dir, f"{base_filename}_field_template.jpg"))
            else:
                top_down_view = map_to_2d(image, field_mask, player_boxes, team_labels, ball_position)
            
            # Save the 2D mapping
            save_image(top_down_view, os.path.join(output_dir, f"{base_filename}_2d_mapping.jpg"))
        else:
            # If no players were detected, create a dummy top-down view
            from src.utils import draw_field_lines
            field_width = 105  # meters
            field_height = 68  # meters
            scale = 10  # pixels per meter
            top_down_view = np.zeros((int(field_height * scale), int(field_width * scale), 3), dtype=np.uint8)
            top_down_view = draw_field_lines(top_down_view, scale)
            save_image(top_down_view, os.path.join(output_dir, f"{base_filename}_2d_mapping.jpg"))
            print("No players detected, creating empty 2D mapping")
        
        mapping_time = time.time() - start_time
        print(f"2D mapping completed in {mapping_time:.2f} seconds")
        
        # Step 6: Visualize results
        print("Step 6: Visualizing results...")
        visualization_path = os.path.join(output_dir, f"{base_filename}_visualization.jpg")
        visualize_results(image, field_mask, player_boxes, team_labels, ball_position, top_down_view, visualization_path)
        print(f"Results visualization saved to: {visualization_path}")
        
        # Return the results
        results = {
            'field_mask': field_mask,
            'player_boxes': player_boxes,
            'team_labels': team_labels,
            'ball_position': ball_position,
            'top_down_view': top_down_view,
            'timing': {
                'field_detection': field_detection_time,
                'player_detection': player_detection_time,
                'team_classification': team_classification_time,
                'ball_detection': ball_detection_time,
                'mapping': mapping_time
            }
        }
        
        return results
    
    except Exception as e:
        print(f"Error during 2D mapping: {e}")
        # If mapping fails, still visualize the detections
        visualization_path = os.path.join(output_dir, f"{base_filename}_detections.jpg")
        
        # Create a simple visualization without the 2D mapping
        result_image = image.copy()
        
        # Draw player bounding boxes and team labels
        for i, (x, y, w, h) in enumerate(player_boxes):
            if i < len(team_labels):  # Make sure we have a team label for this player
                color = (0, 0, 255) if team_labels[i] == 0 else (255, 0, 0)
                cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
                cv2.putText(result_image, f"Team {'A' if team_labels[i] == 0 else 'B'}", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw ball position
        if ball_position is not None:
            x, y = ball_position
            cv2.circle(result_image, (x, y), 10, (0, 255, 255), -1)
        
        save_image(result_image, visualization_path)
        print(f"Detections visualization saved to: {visualization_path}")
        
        # Return partial results
        results = {
            'field_mask': field_mask,
            'player_boxes': player_boxes,
            'team_labels': team_labels,
            'ball_position': ball_position,
            'error': str(e),
            'timing': {
                'field_detection': field_detection_time,
                'player_detection': player_detection_time,
                'team_classification': team_classification_time,
                'ball_detection': ball_detection_time
            }
        }
        
        return results

def main():
    """
    Main function for the SoccerVision project.
    """
    parser = argparse.ArgumentParser(description='SoccerVision: Automated Player and Ball Tracking for Soccer Analytics')
    parser.add_argument('--image', required=True, help='Path to the input image')
    parser.add_argument('--output', default='output', help='Directory to save the output')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--player_detection', choices=['hog', 'yolo'], default='hog', help='Player detection method')
    parser.add_argument('--field_detection', choices=['hsv', 'edge'], default='hsv', help='Field detection method')
    parser.add_argument('--ball_detection', choices=['hough', 'color'], default='hough', help='Ball detection method')
    
    args = parser.parse_args()
    
    # Process the image
    process_image(
        args.image, 
        args.output, 
        args.debug, 
        args.player_detection, 
        args.field_detection, 
        args.ball_detection
    )

if __name__ == '__main__':
    main() 