"""
Test script for the SoccerVision package using a real image.

This script processes the test-soccer.png image using the SoccerVision pipeline.
"""

import os
import sys

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the process_image function from the main module
from src.main import process_image

def main():
    """
    Main function to test the SoccerVision system with a real image.
    """
    # Path to the test image
    image_path = 'data/test-soccer.png'
    
    # Output directory
    output_dir = 'output'
    
    # Process the image with debug mode enabled
    print(f"Processing image: {image_path}")
    results = process_image(
        image_path=image_path,
        output_dir=output_dir,
        debug=True,
        player_detection_method='hog',
        field_detection_method='hsv',
        ball_detection_method='hough'
    )
    
    print("\nProcessing completed!")
    print(f"Results saved to: {output_dir}")
    
    # Print some statistics
    print("\nStatistics:")
    print(f"Number of players detected: {len(results['player_boxes'])}")
    print(f"Ball detected: {'Yes' if results['ball_position'] is not None else 'No'}")
    
    if 'timing' in results:
        print("\nTiming information:")
        for step, time in results['timing'].items():
            print(f"  {step}: {time:.2f} seconds")

if __name__ == '__main__':
    main() 