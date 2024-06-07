# My Project

## Overview

This project uses a Raspberry Pi with a camera to detect AprilTags and determine the position of a box with AprilTags on its faces. It includes camera calibration using a chessboard pattern and supports saving images and displaying a live video feed.

## Project Structure

- `camera_calibration.py`: Script to calibrate the camera using a chessboard pattern.
- `camera_thread.py`: Handles the camera feed in a separate thread.
- `apriltag_detector.py`: Detects AprilTags in the camera feed.
- `box_position.py`: Determines which face(s) the camera is currently pointing at based on detected AprilTags.
- `main.py`: Main script to run the entire detection system.
- `requirements.txt`: List of dependencies.
- `config.json`: Configuration file for various settings.
- `documents/`: Folder containing the chessboard and AprilTags PDFs.

## Documents

The `documents` folder contains:
- `Checkerboard-A3-40mm-9x6.pdf`: A 9x6 chessboard pattern with 40mm squares for camera calibration.
- `tag36h11_100mm_id000.pdf`: AprilTag 36h11 ID 0 used for right face.
- `tag36h11_100mm_id001.pdf`: AprilTag 36h11 ID 1 used for bottom face.
- `tag36h11_100mm_id024.pdf`: AprilTag 36h11 ID 0 used for left face.
- `tag36h11_100mm_id025.pdf`: AprilTag 36h11 ID 0 used for top face(not in use yet).

## Setup Instructions

1. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2. **Calibrate the Camera**:
    ```bash
    python camera_calibration.py
    ```

3. **Run the Detection System**:
    ```bash
    python main.py
    ```

## Configuration

Edit `config.json` to customize the settings:
```json
{
    "live": false,
    "save": false,
    "calibration": "camera_calibration_data.npz",
    "print_delay": 2,
    "num_images": 20,
    "chessboard_size": [9, 6],
    "square_size": 40
}
