"""
Main script for the simplified SoccerVision project.

This script demonstrates how to use the player detection and team classification modules
to process soccer images using techniques covered in class.
"""

import argparse
import cv2
import matplotlib.pyplot as plt

from utils import load_image, visualize_results, detect_players, classify_teams

def process_image(image_path, output_path=None, show_results=True, threshold=0.3, visualize_steps=False):
    """
    Process a soccer image to detect players and classify teams.
    
    Args:
        image_path (str): Path to the input image.
        output_path (str, optional): Path to save the output visualization.
        show_results (bool): Whether to display the results.
        threshold (float): Threshold for edge detection.
        visualize_steps (bool): Whether to visualize intermediate steps of edge detection.
        
    Returns:
        tuple: (player_boxes, team_labels)
    """
    # Load the image
    print(f"Loading image: {image_path}")
    image = load_image(image_path)
    
    # Step 1: Detect players
    print(f"Step 1: Detecting players using Canny edge detection and contour finding...")
    player_boxes = detect_players(image, threshold, visualize_steps=visualize_steps)
    print(f"Detected {len(player_boxes)} players.")
    
    # Step 2: Classify teams
    if player_boxes:
        print("Step 2: Classifying teams using color-based clustering...")
        team_labels = classify_teams(image, player_boxes)
        print(f"Classified players into teams: {sum(team_labels)} players in Team B, {len(team_labels) - sum(team_labels)} players in Team A.")
    else:
        team_labels = []
        print("No players detected, skipping team classification.")
    
    # Step 3: Visualize results
    print("Step 3: Visualizing results...")
    visualization = visualize_results(image, player_boxes, team_labels, output_path)
    
    # Show results if requested
    if show_results:
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB))
        plt.title("Player Detection and Team Classification")
        plt.axis('off')
        plt.show()
    
    return player_boxes, team_labels

def main():
    """
    Main function to parse arguments and process images.
    """
    parser = argparse.ArgumentParser(description="SoccerVision - Player Detection and Team Classification")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--output", "-o", help="Path to save the output visualization")
    parser.add_argument("--threshold", "-t", type=float, default=0.3, help="Threshold for edge detection")
    parser.add_argument("--no-display", action="store_true", help="Do not display the results")
    parser.add_argument("--visualize-steps", action="store_true", help="Visualize edge detection steps")
    
    args = parser.parse_args()
    
    # Process the image
    process_image(
        args.image_path,
        args.output,
        not args.no_display,
        args.threshold,
        args.visualize_steps
    )

if __name__ == "__main__":
    main() 