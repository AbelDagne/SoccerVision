"""
Test script for the SoccerVision package.

This script verifies that all modules are working correctly by running
a simple test on a sample image.
"""

import os
import sys
import cv2
import numpy as np

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import SoccerVision modules
from src.utils import load_image, save_image
from src.field_detection import detect_field_hsv
from src.player_detection import detect_players
from src.team_classification import classify_teams
from src.ball_detection import detect_ball
from src.mapping import map_to_2d

def create_sample_image():
    """
    Create a simple sample image for testing.
    
    Returns:
        numpy.ndarray: Sample image.
    """
    # Create a green field
    image = np.ones((720, 1280, 3), dtype=np.uint8) * np.array([0, 128, 0], dtype=np.uint8)
    
    # Draw field lines
    # Center line
    cv2.line(image, (640, 0), (640, 720), (255, 255, 255), 2)
    
    # Center circle
    cv2.circle(image, (640, 360), 91, (255, 255, 255), 2)
    
    # Penalty areas
    cv2.rectangle(image, (0, 180), (165, 540), (255, 255, 255), 2)
    cv2.rectangle(image, (1115, 180), (1280, 540), (255, 255, 255), 2)
    
    # Goal areas
    cv2.rectangle(image, (0, 270), (55, 450), (255, 255, 255), 2)
    cv2.rectangle(image, (1225, 270), (1280, 450), (255, 255, 255), 2)
    
    # Add some "players" (red and blue rectangles)
    # Team A (red)
    for pos in [(200, 300), (400, 200), (400, 500), (600, 300), (800, 400)]:
        x, y = pos
        cv2.rectangle(image, (x-20, y-40), (x+20, y+40), (0, 0, 255), -1)
    
    # Team B (blue)
    for pos in [(1000, 300), (800, 200), (800, 500), (600, 400), (400, 300)]:
        x, y = pos
        cv2.rectangle(image, (x-20, y-40), (x+20, y+40), (255, 0, 0), -1)
    
    # Add a "ball" (white circle)
    cv2.circle(image, (640, 360), 10, (255, 255, 255), -1)
    
    return image

def test_modules():
    """
    Test all SoccerVision modules.
    """
    print("Testing SoccerVision modules...")
    
    # Create output directory if it doesn't exist
    output_dir = 'test_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a sample image
    print("Creating sample image...")
    image = create_sample_image()
    save_image(image, os.path.join(output_dir, 'sample_image.jpg'))
    
    # Test field detection
    print("Testing field detection...")
    field_mask = detect_field_hsv(image)
    save_image(field_mask, os.path.join(output_dir, 'field_mask.jpg'))
    
    # Test player detection
    print("Testing player detection...")
    player_boxes = detect_players(image, 'hog', field_mask)
    
    # Draw player bounding boxes
    player_image = image.copy()
    for x, y, w, h in player_boxes:
        cv2.rectangle(player_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    save_image(player_image, os.path.join(output_dir, 'player_detection.jpg'))
    
    # Test team classification
    print("Testing team classification...")
    team_labels = classify_teams(image, player_boxes)
    
    # Draw team labels
    team_image = image.copy()
    for i, (x, y, w, h) in enumerate(player_boxes):
        color = (0, 0, 255) if team_labels[i] == 0 else (255, 0, 0)
        cv2.rectangle(team_image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(team_image, f"Team {'A' if team_labels[i] == 0 else 'B'}", 
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    save_image(team_image, os.path.join(output_dir, 'team_classification.jpg'))
    
    # Test ball detection
    print("Testing ball detection...")
    ball_position = detect_ball(image, 'hough', field_mask)
    
    # Draw ball position
    ball_image = image.copy()
    if ball_position is not None:
        x, y = ball_position
        cv2.circle(ball_image, (x, y), 10, (0, 255, 255), -1)
    save_image(ball_image, os.path.join(output_dir, 'ball_detection.jpg'))
    
    # Test 2D mapping
    print("Testing 2D mapping...")
    try:
        top_down_view = map_to_2d(image, field_mask, player_boxes, team_labels, ball_position)
        save_image(top_down_view, os.path.join(output_dir, '2d_mapping.jpg'))
        print("2D mapping successful!")
    except Exception as e:
        print(f"Error during 2D mapping: {e}")
    
    print("Testing completed! Check the 'test_output' directory for results.")

if __name__ == '__main__':
    test_modules() 