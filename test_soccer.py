#!/usr/bin/env python3
"""
Test script for the improved SoccerVision soccer player detection and team classification.
"""

import os
import cv2
from utils import load_image, detect_players, classify_teams, visualize_results
import argparse
import matplotlib.pyplot as plt

def process_image(image_name):
    """
    Main function to test the improved player detection and team classification.
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Input image path - update to use the correct location
    input_path = os.path.join(script_dir, "data", image_name)
    
    if not os.path.exists(input_path):
        print(f"Error: Input image not found at {input_path}")
        return
    
    # Load the image
    print(f"Loading image from {input_path}...")
    image = load_image(input_path)
    
    # Detect players
    print("Detecting players...")
    player_boxes = detect_players(image)
    print(f"Detected {len(player_boxes)} players")
    
    # Classify teams
    print("Classifying teams...")
    team_labels = classify_teams(image, player_boxes)
    
    # Count players in each team
    team_a_count = len(team_labels) - sum(team_labels)
    team_b_count = sum(team_labels)
    print(f"Team A: {team_a_count} players, Team B: {team_b_count} players")
    
    # Visualize results
    print("Generating visualization...")
    output_path = os.path.join(script_dir, 'results', 'results_' + image_name)
    visualization = visualize_results(image, player_boxes, team_labels, output_path)
    
    # Display the visualization (optional, only works in environments with GUI)
    try:
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB))
        plt.title("Player Detection and Team Classification")
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Could not display image: {e}")
    
    print(f"Visualization saved to {output_path}")

def main():
    """
    Main function to parse arguments and process images.
    """
    parser = argparse.ArgumentParser(description="SoccerVision - Player Detection and Team Classification")
    parser.add_argument("name", help="Name of the input image within the data folder")
    
    args = parser.parse_args()
    
    # Process the image
    process_image(
        args.name
    )

if __name__ == "__main__":
    main() 