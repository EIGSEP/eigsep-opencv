import os
import time
import cv2
import shutil
import argparse
import numpy as np
import logging
import json
import queue
from datetime import datetime
from camera_thread import CameraThread, DisplayThread, DetectionThread
from apriltag_detector import AprilTagDetector
from box_position import BoxPosition

# Base directory to save images
BASE_SAVE_DIR = "saved_images"
os.makedirs(BASE_SAVE_DIR, exist_ok=True)
BASE_DATA_DIR = "saved_data"
os.makedirs(BASE_DATA_DIR, exist_ok=True)

def get_next_run_number(base_dir):
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    run_numbers = [int(d.split('_')[0][3:]) for d in subdirs if d.startswith('run')]
    if run_numbers:
        return max(run_numbers) + 1
    else:
        return 1

def parse_args():
    parser = argparse.ArgumentParser(description="AprilTag Box Position Detection")
    parser.add_argument("-l", "--live", action="store_true", help="Show live video feed")
    parser.add_argument("-s", "--save", action="store_true", help="Save images with detections")
    parser.add_argument("-d", "--data", action="store_true", help="Save output data from the run")
    parser.add_argument("-cal", "--calibration", type=str, default="camera_calibration_data.npz", help="Path to camera calibration data")
    parser.add_argument("-con", "--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("-ip", "--initial_position", type=str, default="initial_camera_position.json", help="Path to initial camera position data")
    return parser.parse_args()

def load_config(config_path):
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = json.load(file)
        logging.info(f"Loaded configuration from {config_path}.")
    else:
        config = {}
        logging.warning(f"Configuration file {config_path} not found. Using default settings.")
    return config

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_run_data(run_data, run_dir):
    data_path = os.path.join(run_dir, 'run_data.json')
    with open(data_path, 'w') as file:
        json.dump(run_data, file, indent=4)
    logging.info(f"Saved run data: {data_path}")
    return data_path

def main():
    setup_logging()
    args = parse_args()

    config = load_config(args.config)
    live = args.live if args.live is not None else config.get("live", False)
    save = args.save or config.get("save", False)
    save_data = args.data or config.get("save_data", False)
    calibration_path = args.calibration or config.get("calibration", "camera_calibration_data.npz")
    initial_position_path = args.initial_position or config.get("initial_position", "initial_camera_position.json")
    tag_size = config.get("tag_size", 0.1)  # Default tag size to 0.1 meters if not in config
    print_delay = config.get("print_delay", 2)

    # Load camera calibration data
    if os.path.exists(calibration_path):
        calibration_data = np.load(calibration_path)
        camera_matrix = calibration_data['camera_matrix']
        dist_coeffs = calibration_data['dist_coeffs']
        logging.info("Loaded camera calibration data.")
    else:
        calibration_data = None
        camera_matrix = None
        dist_coeffs = None
        logging.warning("Camera calibration data not found. Proceeding without calibration.")

    # Load initial camera position data
    if os.path.exists(initial_position_path):
        with open(initial_position_path, 'r') as file:
            initial_positions = json.load(file)
        logging.info("Loaded initial camera position data.")
    else:
        initial_positions = {}
        logging.warning("Initial camera position data not found.")

    logging.info("Initializing camera...")
    camera_thread = CameraThread()
    camera_thread.start()

    # Wait until the first frame is captured
    camera_thread.frame_ready.wait()

    logging.info("Creating AprilTag detector...")
    detector = AprilTagDetector(camera_matrix, dist_coeffs, tag_size)
    box_position = BoxPosition(initial_positions)  # Assuming BoxPosition takes initial_positions as an argument

    # Create subdirectories for this run
    run_number = get_next_run_number(BASE_SAVE_DIR)
    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_image_dir = os.path.join(BASE_SAVE_DIR, f'run{run_number}_{run_timestamp}')
    os.makedirs(run_image_dir)

    run_data_dir = os.path.join(BASE_DATA_DIR, f'run{run_number}_{run_timestamp}')
    os.makedirs(run_data_dir)

    display_queue = queue.Queue()

    logging.info("Starting video capture...")
    
    display_thread = DisplayThread(camera_thread, live, display_queue)
    display_thread.start()

    detection_thread = DetectionThread(camera_thread, detector, display_queue, print_delay, save, run_image_dir, box_position)
    detection_thread.start()

    run_data = []

    try:
        while True:
            time.sleep(0.1)
            if save_data:
                detections, _ = detector.detect(camera_thread.frame)
                positions_orientations = detector.get_position_and_orientation(detections)
                current_position, current_orientation = box_position.calculate_position(positions_orientations)
                run_data_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'detected_tags': [
                        {
                            'tag_id': tag_id,
                            'position': position.tolist() if position is not None else None,
                            'distance': np.linalg.norm(tvec) if tvec is not None else None,
                            'orientation': orientation
                        } for tag_id, position, tvec, orientation in positions_orientations
                    ],
                    'box_position': current_position.tolist() if current_position is not None else None,
                    'box_orientation': current_orientation,
                    'rotation_count': box_position.rotation_count
                }
                run_data.append(run_data_entry)

    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        logging.info("Releasing resources...")
        display_thread.stop()
        detection_thread.stop()
        camera_thread.stop()
        display_thread.join()
        detection_thread.join()
        camera_thread.join()
        cv2.destroyAllWindows()

        if save:
            user_input_images = input("Do you want to keep the saved images? (y/n): ").strip().lower()
            if user_input_images == 'n':
                logging.info("Deleting saved images...")
                shutil.rmtree(run_image_dir)
                logging.info("Images deleted.")
            else:
                logging.info(f"Images kept in {run_image_dir}")

        if save_data:
            data_path = save_run_data(run_data, run_data_dir)
            user_input_data = input("Do you want to keep the saved run data? (y/n): ").strip().lower()
            if user_input_data == 'n':
                logging.info("Deleting saved run data...")
                shutil.rmtree(run_data_dir)
                logging.info("Run data deleted.")
            else:
                logging.info(f"Run data kept in {run_data_dir}")

if __name__ == "__main__":
    main()
