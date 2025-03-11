# SoccerVision

Automated Player and Ball Detection for Soccer Analytics

## Overview

SoccerVision is a computer vision tool that processes soccer game images to automatically:
1. Detect players on the field
2. Classify teams using jersey color
3. Identify the ball
4. Generate a 2D top-down mapping of the game from the input image

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

```python
# Basic usage
python src/main.py --image path/to/soccer_image.jpg --output path/to/output_directory
```

### Optional Arguments

- `--debug`: Enable debug mode to visualize intermediate steps
- `--model`: Choose player detection model (hog or yolo, default: hog)
- `--field_method`: Choose field detection method (hsv or edge, default: hsv)

## Project Structure

- `src/`: Source code
  - `main.py`: Main entry point
  - `field_detection.py`: Field detection and segmentation
  - `player_detection.py`: Player detection
  - `team_classification.py`: Team classification
  - `ball_detection.py`: Ball detection
  - `mapping.py`: 2D mapping and visualization
  - `utils.py`: Utility functions
- `data/`: Input data directory
- `output/`: Output directory for results

## Example

Input image:
![Input Image](docs/example_input.jpg)

Output 2D mapping:
![Output Mapping](docs/example_output.jpg)

## License

MIT 