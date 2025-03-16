"""
Field detection module for the SoccerVision project.

This module contains functions for detecting and segmenting the soccer field
from an input image using color-based thresholding and edge detection.
"""

import cv2
import numpy as np

def detect_field_hsv(image, debug=False):
    """
    Detect the soccer field using HSV color thresholding.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        debug (bool): If True, return intermediate results for debugging.
        
    Returns:
        numpy.ndarray: Binary mask of the field (255 for field pixels, 0 for non-field).
        dict (optional): Dictionary containing intermediate results for debugging.
    """
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define range of green color in HSV
    # TODO: Make these thresholds configurable or adaptive based on the image
    lower_green = np.array([35, 50, 50])
    upper_green = np.array([90, 255, 255])
    
    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find the largest contour (assuming it's the field)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Create a new mask with only the largest contour
        field_mask = np.zeros_like(mask)
        cv2.drawContours(field_mask, [largest_contour], 0, 255, -1)
        
        # Fill holes in the mask
        field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, kernel)
    else:
        # If no contours found, return the original mask
        field_mask = mask
    
    if debug:
        debug_info = {
            'hsv': hsv,
            'initial_mask': mask,
            'field_mask': field_mask,
            'contours': contours
        }
        return field_mask, debug_info
    
    return field_mask

def gaussian_kernel(size, sigma):
    """Implementation of Gaussian Kernel."""
    kernel = np.zeros((size, size))
    k = size // 2
    for i in range(size):
        for j in range(size):
            kernel[i,j] = np.exp(-((i-k)**2 + (j-k)**2)/(2*sigma**2)) / (2*np.pi*sigma**2)
    return kernel

def conv(image, kernel):
    """Implementation of convolution filter."""
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')
    kernel = np.flip(kernel, axis=0)
    kernel = np.flip(kernel, axis=1)
    for i in range(Hi):
        for j in range(Wi):
            out[i, j] = np.sum(kernel * padded[i:i+Hk,j:j+Wk])
    return out

def partial_x(img):
    """Computes partial x-derivative of input img."""
    dx = np.array([[1, 0, -1]])/2
    return conv(img, dx)

def partial_y(img):
    """Computes partial y-derivative of input img."""
    dy = np.array([[1, 0, -1]]).T/2
    return conv(img, dy)

def gradient(img):
    """Returns gradient magnitude and direction of input img."""
    Gx = partial_x(img)
    Gy = partial_y(img)
    G = np.sqrt(Gx*Gx+Gy*Gy)
    theta = np.arctan2(Gy, Gx) / (2*np.pi) * 360
    theta = theta % 360
    return G, theta

def non_maximum_suppression(G, theta):
    """Performs non-maximum suppression."""
    H, W = G.shape
    out = np.zeros((H, W))
    theta = np.floor((theta + 22.5) / 45) * 45
    theta = (theta % 360.0).astype(np.int32)
    
    for i in range(H):
        for j in range(W):
            if theta[i,j] == 0 or theta[i,j] == 180:
                before = G[i, max(j-1,0)]
                after = G[i, min(j+1,W-1)]
            elif theta[i,j] == 45 or theta[i,j] == (45+180):
                if i-1 < 0 or j-1 < 0:
                    before = 0
                else:
                    before = G[i-1, j-1]
                if i+1 > H-1 or j+1 > W-1:
                    after = 0
                else:
                    after = G[i+1, j+1]
            elif theta[i,j] == 90 or theta[i,j] == (90+180):
                before = G[max(i-1,0), j]
                after = G[min(i+1,H-1), j]
            else:
                if i-1 < 0 or j+1 > W-1:
                    before = 0
                else:
                    before = G[i-1, j+1]
                if i+1 > H-1 or j-1 < 0:
                    after = 0
                else:
                    after = G[i+1, j-1]
            out[i,j] = G[i,j] if max(G[i,j], before, after) == G[i,j] else 0
    return out

def double_thresholding(img, high, low):
    """Performs double thresholding."""
    strong_edges = img > high
    weak_edges = (img > low) * (img <= high)
    return strong_edges, weak_edges

def get_neighbors(y, x, H, W):
    """Return indices of valid neighbors of (y, x)."""
    neighbors = []
    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))
    return neighbors

def link_edges(strong_edges, weak_edges):
    """Find weak edges connected to strong edges and link them."""
    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.copy(strong_edges)
    weak_edges = np.copy(weak_edges)
    
    for r, c in indices:
        Q = []
        Q.append((r, c))
        while Q:
            y, x = Q.pop(0)
            for i, j in get_neighbors(y, x, H, W):
                if weak_edges[i, j] == True:
                    weak_edges[i, j] = False
                    edges[i, j] = True
                    Q.append((i, j))
    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """Implement canny edge detector."""
    kernel = gaussian_kernel(kernel_size, sigma)
    smoothed = conv(img, kernel)
    G, theta = gradient(smoothed)
    nms = non_maximum_suppression(G, theta)
    strong_edges, weak_edges = double_thresholding(nms, high, low)
    edge = link_edges(strong_edges, weak_edges)
    return edge

def hough_transform(img):
    """Transform points in the input image into Hough space."""
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2 + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)
    
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)
    
    for x, y in zip(xs, ys):
        for theta_idx in range(num_thetas):
            rho = int(x * cos_t[theta_idx] + y * sin_t[theta_idx])
            rho_idx = np.round(rho + diag_len).astype(int)
            accumulator[rho_idx, theta_idx] += 1
    
    return accumulator, rhos, thetas

def create_rectangular_field(image, lines, debug=False):
    """
    Create a field mask by finding the best-fitting soccer field shape from the HSV field mask.
    Uses four lines to create a trapezoid that optimally covers the field area.
    
    Args:
        image (numpy.ndarray): Input image
        lines (list): Detected lines from HoughLinesP
        debug (bool): Whether to return debug information
        
    Returns:
        tuple: (field_mask, perspective_matrix, corners)
    """
    h, w = image.shape[:2]
    
    # Get the field mask from HSV
    hsv_mask = detect_field_hsv(image)
    
    # Find contours in the field mask
    contours, _ = cv2.findContours(hsv_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None
    
    # Get the largest contour (the field)
    field_contour = max(contours, key=cv2.contourArea)
    
    # Get initial bounding box
    rect = cv2.minAreaRect(field_contour)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    
    # Sort points to get consistent order: top-left, top-right, bottom-right, bottom-left
    box = box[np.argsort(box[:, 1])]
    top_points = box[:2]
    bottom_points = box[2:]
    top_points = top_points[np.argsort(top_points[:, 0])]
    bottom_points = bottom_points[np.argsort(bottom_points[:, 0])]
    corners = np.vstack((top_points, bottom_points[::-1]))
    
    # Extract initial points
    top_left, top_right, bottom_right, bottom_left = corners
    
    # Function to calculate coverage ratio for a given trapezoid
    def calculate_coverage(corners):
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [corners.astype(np.int32)], 255)
        intersection = cv2.bitwise_and(hsv_mask, mask)
        coverage_area = np.sum(intersection > 0)
        trapezoid_area = np.sum(mask > 0)
        if trapezoid_area == 0:
            return 0
        return coverage_area / trapezoid_area
    
    # Function to create trapezoid points with variable top inset
    def create_trapezoid(top_inset):
        top_vector = top_right - top_left
        top_unit = top_vector / np.linalg.norm(top_vector)
        top_left_new = top_left + (top_unit * top_inset)
        top_right_new = top_right - (top_unit * top_inset)
        return np.vstack(([top_left_new, top_right_new], [bottom_right, bottom_left]))
    
    # Find optimal top inset by maximizing coverage ratio
    bottom_width = np.linalg.norm(bottom_right - bottom_left)
    best_coverage = 0
    best_corners = corners
    
    # Try different inset values from 0% to 30% of bottom width
    for inset_factor in np.linspace(0, 0.3, 30):
        inset = bottom_width * inset_factor
        test_corners = create_trapezoid(inset)
        coverage = calculate_coverage(test_corners)
        
        # Update best if this coverage is better
        if coverage > best_coverage:
            best_coverage = coverage
            best_corners = test_corners
    
    # Create source points for perspective transform
    src_points = best_corners.astype(np.float32)
    
    # Calculate aspect ratio based on standard soccer field dimensions
    std_ratio = 1.5
    std_height = 800
    std_width = int(std_height * std_ratio)
    
    # Create destination points for a perfect rectangle
    dst_points = np.float32([
        [0, 0],                  # Top left
        [std_width, 0],          # Top right
        [std_width, std_height], # Bottom right
        [0, std_height]          # Bottom left
    ])
    
    # Calculate perspective transform matrix
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Create field mask
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [src_points.astype(np.int32)], 255)
    
    if debug:
        debug_info = {
            'corners': src_points,
            'dst_corners': dst_points,
            'mask': mask,
            'matrix': perspective_matrix,
            'field_contour': field_contour,
            'hsv_mask': hsv_mask,
            'coverage_ratio': best_coverage
        }
        return mask, perspective_matrix, debug_info
    
    return mask, perspective_matrix, src_points

def detect_field_edges(image, debug=False):
    """
    Detect the soccer field by first isolating white lines and then finding field boundaries.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Get field mask to limit our search area
    hsv_mask = detect_field_hsv(image)
    
    # Step 1: Isolate white pixels
    # Apply adaptive thresholding to handle varying lighting
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 25, 15
    )
    
    # Also apply regular thresholding to catch bright lines
    _, bright_thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    # Combine both thresholds
    white_mask = cv2.bitwise_or(adaptive_thresh, bright_thresh)
    
    # Only look within the field area
    white_mask = cv2.bitwise_and(white_mask, hsv_mask)
    
    # Step 2: Clean up the white mask
    # Remove small noise
    kernel_small = np.ones((3,3), np.uint8)
    kernel_medium = np.ones((5,5), np.uint8)
    
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel_medium)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel_medium)
    
    # Step 3: Find connected components and filter by size and shape
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        white_mask, connectivity=8
    )
    
    # Create a mask for valid line segments
    line_segments_mask = np.zeros_like(white_mask)
    min_area = 50  # Minimum area to consider
    min_ratio = 3  # Minimum length/width ratio to be considered a line
    
    for i in range(1, num_labels):  # Skip background label 0
        width = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        if area < min_area:
            continue
            
        # Check if the component is line-like
        ratio = max(width, height) / min(width, height)
        if ratio > min_ratio:
            # Add this component to our line segments mask
            line_segments_mask[labels == i] = 255
    
    # Step 4: Use Hough transform to find lines from the segments
    lines = cv2.HoughLinesP(
        line_segments_mask, 
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=50,
        maxLineGap=20
    )
    
    # Step 5: Group and filter lines
    if lines is not None:
        # Convert lines to a more manageable format
        line_list = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            line_list.append((x1, y1, x2, y2, angle, length))
        
        # Separate vertical and horizontal lines
        vertical_lines = []
        horizontal_lines = []
        angle_threshold = 15  # Degrees from vertical/horizontal
        
        for line in line_list:
            x1, y1, x2, y2, angle, length = line
            # Normalize angle to 0-90 range
            norm_angle = angle if angle <= 90 else 180 - angle
            
            if norm_angle < angle_threshold:  # Horizontal
                horizontal_lines.append(line)
            elif norm_angle > 90 - angle_threshold:  # Vertical
                vertical_lines.append(line)
        
        # Sort lines by length
        vertical_lines.sort(key=lambda x: x[5], reverse=True)
        horizontal_lines.sort(key=lambda x: x[5], reverse=True)
        
        # Function to check if a line is inward slanting
        def is_inward_slanting(x1, y1, x2, y2):
            if x1 > image.shape[1]/2:  # Right side
                return x2 < x1  # Should slant left
            else:  # Left side
                return x2 > x1  # Should slant right
        
        # Find the best vertical lines (one for each side)
        best_verticals = []
        if vertical_lines:
            # Separate into left and right sides
            left_lines = []
            right_lines = []
            mid_x = image.shape[1] / 2
            
            for line in vertical_lines:
                x1, y1, x2, y2 = line[:4]
                mid_point_x = (x1 + x2) / 2
                
                # Check if the line slants inward
                if is_inward_slanting(x1, y1, x2, y2):
                    if mid_point_x < mid_x:
                        left_lines.append(line)
                    else:
                        right_lines.append(line)
            
            # Take the longest line from each side
            if left_lines:
                best_verticals.append(left_lines[0])
            if right_lines:
                best_verticals.append(right_lines[0])
        
        # Find the best horizontal line (should be near the middle)
        best_horizontal = None
        if horizontal_lines:
            mid_y = image.shape[0] / 2
            # Look for horizontal lines near the middle
            for line in horizontal_lines:
                x1, y1, x2, y2 = line[:4]
                mid_point_y = (y1 + y2) / 2
                # Check if line is in middle third of image
                if image.shape[0]/3 < mid_point_y < 2*image.shape[0]/3:
                    best_horizontal = line
                    break
        
        # Create the final line mask
        line_mask = np.zeros_like(gray)
        
        # Draw the best lines
        if best_horizontal:
            x1, y1, x2, y2 = best_horizontal[:4]
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 9)
            
        for line in best_verticals:
            x1, y1, x2, y2 = line[:4]
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 9)
    else:
        line_mask = np.zeros_like(gray)
    
    # Create final mask
    field_mask = cv2.bitwise_and(hsv_mask, hsv_mask, mask=line_mask)
    
    # After line detection, create rectangular field
    rect_mask, perspective_matrix, corners = create_rectangular_field(image, lines, debug)
    
    if debug:
        debug_info = {
            'gray': gray,
            'white_mask': white_mask,
            'line_segments': line_segments_mask,
            'line_mask': line_mask,
            'hsv_mask': hsv_mask,
            'field_mask': field_mask,
            'detected_lines': lines if lines is not None else [],
            'rect_mask': rect_mask,
            'perspective_matrix': perspective_matrix,
            'corners': corners
        }
        return field_mask, debug_info
    
    return field_mask

def detect_field_lines(image, field_mask, debug=False):
    """
    Detect field lines for homography reference points.
    
    Args:
        image (numpy.ndarray): Input image in BGR format.
        field_mask (numpy.ndarray): Binary mask of the field.
        debug (bool): If True, return intermediate results for debugging.
        
    Returns:
        list: List of detected lines in the format [(x1, y1, x2, y2), ...].
        dict (optional): Dictionary containing intermediate results for debugging.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply the field mask
    masked_gray = cv2.bitwise_and(gray, gray, mask=field_mask)
    
    # Apply adaptive thresholding to highlight white lines
    # TODO: Make these parameters configurable
    thresh = cv2.adaptiveThreshold(
        masked_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Invert the threshold to get white lines as white pixels
    thresh = cv2.bitwise_not(thresh)
    
    # Apply morphological operations to clean up the lines
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Apply Hough transform to detect lines
    lines = cv2.HoughLinesP(
        thresh, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10
    )
    
    detected_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            detected_lines.append((x1, y1, x2, y2))
    
    if debug:
        # Create a visualization of the detected lines
        line_image = image.copy()
        for x1, y1, x2, y2 in detected_lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        debug_info = {
            'masked_gray': masked_gray,
            'thresh': thresh,
            'line_image': line_image
        }
        return detected_lines, debug_info
    
    return detected_lines

def find_field_corners(field_mask, lines, debug=False):
    """
    Find the four corners of the soccer field for homography calculation.
    
    Args:
        field_mask (numpy.ndarray): Binary mask of the field.
        lines (list): List of detected lines in the format [(x1, y1, x2, y2), ...].
        debug (bool): If True, return intermediate results for debugging.
        
    Returns:
        numpy.ndarray: Array of four corner points in the format [[x1, y1], [x2, y2], [x3, y3], [x4, y4]].
        dict (optional): Dictionary containing intermediate results for debugging.
    """
    # TODO: Implement a more robust method to find field corners
    # This is a simplified approach that may not work for all images
    
    # Find the contour of the field mask
    contours, _ = cv2.findContours(field_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # If no contours found, return None
        if debug:
            return None, {'error': 'No contours found in field mask'}
        return None
    
    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Approximate the contour with a polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # If we have more than 4 points, find the 4 corners
    if len(approx) > 4:
        # Find the bounding rectangle
        rect = cv2.minAreaRect(approx)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        corners = box
    elif len(approx) == 4:
        # If we already have 4 points, use them
        corners = approx.reshape(4, 2)
    else:
        # If we have less than 4 points, use the convex hull
        hull = cv2.convexHull(largest_contour)
        epsilon = 0.02 * cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, epsilon, True)
        
        if len(approx) >= 4:
            # Find the bounding rectangle
            rect = cv2.minAreaRect(approx)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            corners = box
        else:
            # If still less than 4 points, return None
            if debug:
                return None, {'error': 'Could not find 4 corners in field mask'}
            return None
    
    # Sort corners in order: top-left, top-right, bottom-right, bottom-left
    # First, sort by y-coordinate (top to bottom)
    corners = corners[np.argsort(corners[:, 1])]
    
    # Sort the top two points by x-coordinate (left to right)
    top_points = corners[:2]
    top_points = top_points[np.argsort(top_points[:, 0])]
    
    # Sort the bottom two points by x-coordinate (left to right)
    bottom_points = corners[2:]
    bottom_points = bottom_points[np.argsort(bottom_points[:, 0])]
    
    # Combine the sorted points
    corners = np.vstack((top_points, bottom_points[::-1]))
    
    # Adjust corners to better match a soccer field shape
    # The field should be wider at the bottom (near) than at the top (far)
    top_left, top_right, bottom_right, bottom_left = corners
    
    # Calculate the width at bottom
    bottom_width = np.linalg.norm(bottom_right - bottom_left)
    
    # Calculate how much to bring in the top corners (40% of bottom width)
    inward_factor = 0.4
    top_inward_dist = bottom_width * inward_factor
    
    # Calculate unit vector along top edge
    top_vector = top_right - top_left
    top_unit = top_vector / np.linalg.norm(top_vector)
    
    # Move top corners inward
    top_left_new = top_left + (top_unit * top_inward_dist)
    top_right_new = top_right - (top_unit * top_inward_dist)
    
    # Create new corners array with adjusted top points
    corners = np.vstack((
        [top_left_new, top_right_new],
        [bottom_right, bottom_left]
    ))
    
    if debug:
        # Create a visualization of the corners
        corner_image = cv2.cvtColor(field_mask, cv2.COLOR_GRAY2BGR)
        for i, (x, y) in enumerate(corners):
            cv2.circle(corner_image, (x, y), 10, (0, 0, 255), -1)
            cv2.putText(corner_image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        debug_info = {
            'contours': contours,
            'largest_contour': largest_contour,
            'approx': approx,
            'corners': corners,
            'corner_image': corner_image
        }
        return corners, debug_info
    
    return corners

def detect_players(image, field_mask, debug=False):
    """
    Detect players on the soccer field using color detection and contour analysis.
    
    Args:
        image (numpy.ndarray): Input image in BGR format
        field_mask (numpy.ndarray): Binary mask of the field
        debug (bool): Whether to return debug information
        
    Returns:
        list: List of player detections, each containing:
              - bounding box (x, y, w, h)
              - team classification (0 for team1, 1 for team2, 2 for referee/other)
              - confidence score
        dict (optional): Debug information including visualizations
    """
    h, w = image.shape[:2]
    
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create a mask for the field area we're interested in
    roi_mask = field_mask.copy()
    
    # Erode the field mask slightly to remove players touching the edges
    kernel = np.ones((5,5), np.uint8)
    roi_mask = cv2.erode(roi_mask, kernel, iterations=2)
    
    # Apply ROI mask to the image
    hsv_roi = cv2.bitwise_and(hsv, hsv, mask=roi_mask)
    
    # Function to create a color mask within HSV ranges
    def create_color_mask(hsv_img, lower_bound, upper_bound):
        mask = cv2.inRange(hsv_img, np.array(lower_bound), np.array(upper_bound))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask
    
    # Detect players based on common jersey colors
    # These ranges should be adjusted based on the specific teams' colors
    color_ranges = {
        'white': ([0, 0, 180], [180, 30, 255]),
        'black': ([0, 0, 0], [180, 255, 50]),
        'red': ([0, 100, 100], [10, 255, 255]),
        'blue': ([100, 50, 50], [130, 255, 255]),
        'green': ([40, 50, 50], [80, 255, 255]),
        'yellow': ([20, 100, 100], [35, 255, 255])
    }
    
    # Create masks for each color
    color_masks = {}
    for color, (lower, upper) in color_ranges.items():
        color_masks[color] = create_color_mask(hsv_roi, lower, upper)
    
    # Combine all color masks
    player_mask = np.zeros_like(field_mask)
    for mask in color_masks.values():
        player_mask = cv2.bitwise_or(player_mask, mask)
    
    # Find contours of potential players
    contours, _ = cv2.findContours(player_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on size and shape
    players = []
    min_player_area = 100  # Minimum area for a player
    max_player_area = 5000  # Maximum area for a player
    min_aspect_ratio = 1.2  # Minimum height/width ratio for a player
    max_aspect_ratio = 4.0  # Maximum height/width ratio for a player
    
    debug_image = image.copy() if debug else None
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_player_area or area > max_player_area:
            continue
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Check aspect ratio
        aspect_ratio = h / w
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            continue
        
        # Determine team based on dominant color in the bounding box
        roi = hsv[y:y+h, x:x+w]
        max_color_count = 0
        team_classification = 2  # Default to "other"
        
        for i, (color, mask) in enumerate(color_masks.items()):
            color_count = np.sum(mask[y:y+h, x:x+w] > 0)
            if color_count > max_color_count:
                max_color_count = color_count
                # Assign team based on color (can be customized based on known team colors)
                team_classification = 0 if color in ['red', 'blue'] else 1 if color in ['white', 'yellow'] else 2
        
        # Calculate confidence based on area and color match
        confidence = min((area - min_player_area) / (max_player_area - min_player_area), 
                        max_color_count / area)
        
        players.append({
            'bbox': (x, y, w, h),
            'team': team_classification,
            'confidence': float(confidence)
        })
        
        if debug:
            # Draw bounding box and team classification
            color = [(0,0,255), (255,0,0), (0,255,0)][team_classification]
            cv2.rectangle(debug_image, (x,y), (x+w,y+h), color, 2)
            cv2.putText(debug_image, f'Team {team_classification}', 
                       (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    if debug:
        debug_info = {
            'player_mask': player_mask,
            'color_masks': color_masks,
            'debug_image': debug_image,
            'hsv_roi': hsv_roi
        }
        return players, debug_info
    
    return players

def detect_players_on_field(image, field_mask, debug=False):
    """
    Detect players on the soccer field using color detection and contour analysis.
    
    Args:
        image (numpy.ndarray): Input image in BGR format
        field_mask (numpy.ndarray): Binary mask of the field
        debug (bool): Whether to return debug information
        
    Returns:
        list: List of player detections, each containing:
              - bounding box (x, y, w, h)
              - team classification (0 for team1, 1 for team2, 2 for referee/other)
              - confidence score
        dict (optional): Debug information including visualizations
    """
    h, w = image.shape[:2]
    
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create a mask for the field area we're interested in
    roi_mask = field_mask.copy()
    
    # Erode the field mask slightly to remove players touching the edges
    kernel = np.ones((5,5), np.uint8)
    roi_mask = cv2.erode(roi_mask, kernel, iterations=2)
    
    # Apply ROI mask to the image
    hsv_roi = cv2.bitwise_and(hsv, hsv, mask=roi_mask)
    
    # Function to create a color mask within HSV ranges
    def create_color_mask(hsv_img, lower_bound, upper_bound):
        mask = cv2.inRange(hsv_img, np.array(lower_bound), np.array(upper_bound))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask
    
    # Detect players based on common jersey colors
    # These ranges should be adjusted based on the specific teams' colors
    color_ranges = {
        'white': ([0, 0, 180], [180, 30, 255]),
        'black': ([0, 0, 0], [180, 255, 50]),
        'red': ([0, 100, 100], [10, 255, 255]),
        'blue': ([100, 50, 50], [130, 255, 255]),
        'green': ([40, 50, 50], [80, 255, 255]),
        'yellow': ([20, 100, 100], [35, 255, 255])
    }
    
    # Create masks for each color
    color_masks = {}
    for color, (lower, upper) in color_ranges.items():
        color_masks[color] = create_color_mask(hsv_roi, lower, upper)
    
    # Combine all color masks
    player_mask = np.zeros_like(field_mask)
    for mask in color_masks.values():
        player_mask = cv2.bitwise_or(player_mask, mask)
    
    # Find contours of potential players
    contours, _ = cv2.findContours(player_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours based on size and shape
    players = []
    min_player_area = 100  # Minimum area for a player
    max_player_area = 5000  # Maximum area for a player
    min_aspect_ratio = 1.2  # Minimum height/width ratio for a player
    max_aspect_ratio = 4.0  # Maximum height/width ratio for a player
    
    debug_image = image.copy() if debug else None
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_player_area or area > max_player_area:
            continue
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Check aspect ratio
        aspect_ratio = h / w
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            continue
        
        # Determine team based on dominant color in the bounding box
        roi = hsv[y:y+h, x:x+w]
        max_color_count = 0
        team_classification = 2  # Default to "other"
        
        for i, (color, mask) in enumerate(color_masks.items()):
            color_count = np.sum(mask[y:y+h, x:x+w] > 0)
            if color_count > max_color_count:
                max_color_count = color_count
                # Assign team based on color (can be customized based on known team colors)
                team_classification = 0 if color in ['red', 'blue'] else 1 if color in ['white', 'yellow'] else 2
        
        # Calculate confidence based on area and color match
        confidence = min((area - min_player_area) / (max_player_area - min_player_area), 
                        max_color_count / area)
        
        players.append({
            'bbox': (x, y, w, h),
            'team': team_classification,
            'confidence': float(confidence)
        })
        
        if debug:
            # Draw bounding box and team classification
            color = [(0,0,255), (255,0,0), (0,255,0)][team_classification]
            cv2.rectangle(debug_image, (x,y), (x+w,y+h), color, 2)
            cv2.putText(debug_image, f'Team {team_classification}', 
                       (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    if debug:
        debug_info = {
            'player_mask': player_mask,
            'color_masks': color_masks,
            'debug_image': debug_image,
            'hsv_roi': hsv_roi
        }
        return players, debug_info
    
    return players 