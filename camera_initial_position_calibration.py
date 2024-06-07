import cv2
import numpy as np
import json
import argparse
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

def main():
    parser = argparse.ArgumentParser(description="Initial Camera Position Calibration.")
    parser.add_argument('--calibration', type=str, default='camera_calibration_data.npz', help='Path to the camera calibration data.')
    args = parser.parse_args()

    config = load_config()
    calibration_file = args.calibration or config.get('calibration', 'camera_calibration_data.npz')
    calibration_data = np.load(calibration_file) if os.path.exists(calibration_file) else None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    tag_detector = AprilTagDetector(calibration_data)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        detections, frame = tag_detector.detect(frame)
        frame = tag_detector.draw_detections(frame, detections)

        if len(detections) > 0:
            initial_positions = []
            for detection in detections:
                distance, orientation = tag_detector.get_distance_and_orientation(detection)
                initial_positions.append({
                    'tag_id': detection.tag_id,
                    'distance': distance,
                    'orientation': orientation,
                    'center': detection.center
                })
            print("Initial positions:", initial_positions)
            break

        cv2.imshow('Initial Position Calibration', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    with open('initial_camera_position.json', 'w') as file:
        json.dump(initial_positions, file)
    print("Initial camera position saved to initial_camera_position.json")

if __name__ == "__main__":
    main()
