import cv2
import numpy as np
import json
import argparse
import os
from apriltag_detector import AprilTagDetector

def load_config(config_path='config.json'):
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = json.load(file)
        print(f"Loaded configuration from {config_path}.")
    else:
        config = {}
        print(f"Configuration file {config_path} not found. Using default settings.")
    return config

def save_initial_positions(initial_positions, file_path='initial_camera_position.json'):
    with open(file_path, 'w') as file:
        json.dump(initial_positions, file)
    print(f"Initial camera position saved to {file_path}")

def main():
    parser = argparse.ArgumentParser(description="Initial Camera Position Calibration.")
    parser.add_argument('-cal', '--calibration', type=str, default='camera_calibration_data.npz', help='Path to the camera calibration data.')
    args = parser.parse_args()

    config = load_config()
    calibration_file = args.calibration or config.get('calibration', 'camera_calibration_data.npz')
    calibration_data = np.load(calibration_file) if os.path.exists(calibration_file) else None

    if calibration_data is not None:
        camera_matrix = calibration_data['camera_matrix']
        dist_coeffs = calibration_data['dist_coeffs']
    else:
        camera_matrix = None
        dist_coeffs = None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    tag_detector = AprilTagDetector(camera_matrix, dist_coeffs)
    seen_tags = {}
    rotation_count = 0
    rotation_order = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        detections, gray_frame = tag_detector.detect(frame)
        frame = tag_detector.draw_detections(frame, detections)

        if len(detections) > 0:
            positions_orientations = tag_detector.get_position_and_orientation(detections)
            for tag_id, position, tvec, orientation in positions_orientations:
                if tag_id not in seen_tags:
                    seen_tags[tag_id] = {
                        'position': position.tolist() if position is not None else None,
                        'distance': np.linalg.norm(tvec) if tvec is not None else None,
                        'orientation': orientation
                    }

                if tag_id not in rotation_order:
                    rotation_order.append(tag_id)

                if len(rotation_order) == 6:  # Assuming we have 6 tags to detect in one view
                    rotation_count += 1
                    rotation_order = []

            print("Seen tags:", seen_tags)
            print("Rotation count:", rotation_count)
            break

        cv2.imshow('Initial Position Calibration', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    save_initial_positions(seen_tags)

if __name__ == "__main__":
    main()
