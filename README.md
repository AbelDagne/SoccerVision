# SoccerVision: Player Detection and Team Classification

An implementation of computer vision algorithms for soccer player detection and team classification.

## Overview

## Technical Approach

### 1. Player Detection
We use a custom implementation of Canny edge detection and contour finding:

- **Grayscale Conversion & Blurring**: Convert to grayscale and apply Gaussian blur to reduce noise
- **Edge Detection**: Apply Canny edge detector to find boundaries
- **Field Masking**: Segment the green field using HSV color thresholding
- **Contour Finding**: Extract contours from non-field areas that likely represent players
- **Size Filtering**: Filter contours by area to identify player regions
- **Non-Maximum Suppression**: Remove overlapping detections

### 2. Team Classification
We use color-based segmentation with a custom clustering implementation:

- **Color Extraction**: Extract dominant colors from player regions in HSV color space
- **Histogram Analysis**: Create histograms of hue values to identify jersey colors
- **Custom Clustering**: Implement a custom binary clustering algorithm to separate teams
- **Team Assignment**: Assign each player to one of two teams based on jersey color

## Implementation Details

### Key Components
1. **Player Detection Module**: Implements Canny edge detection and contour-based player detection
2. **Team Classification Module**: Implements custom color-based team classification
3. **Utilities**: Helper functions for loading, processing, and visualizing images

### Core Algorithms
1. **Player Detection**:
   - Canny edge detection to find boundaries
   - Color-based field segmentation
   - Contour extraction for player identification
   - Non-maximum suppression to clean up overlapping detections

2. **Color-based Team Classification**:
   - HSV color space analysis of player regions
   - Histogram-based dominant color extraction
   - Custom binary clustering algorithm (similar to k-means but without external libraries)
