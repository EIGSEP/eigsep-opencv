import os
import time
import cv2
import shutil
import argparse
import numpy as np
import logging
import json
import threading
import queue
from camera_thread import CameraThread
from apriltag_detector import AprilTagDetector
from box_position import BoxPosition

# Base directory to save images
BASE_SAVE_DIR = "saved_images"
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

def get_next_run_number():
    subdirs = [d for d in os.listdir(BASE_SAVE_DIR) if os.path.isdir(os.path.join(BASE_SAVE_DIR, d))]
    run_numbers = [int(d.split('_')[0][3:]) for d in subdirs if d.startswith('run')]
    if run_numbers:
        return max(run_numbers) + 1
    else:
        return 1

def parse_args():
    parser = argparse.ArgumentParser(description="AprilTag Box Position Detection")
    parser.add_argument("-l", "--live", action="store_true", help="Show live video feed")
    parser.add_argument("-s", "--save", action="store_true", help="Save images with detections")
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

class DisplayThread(threading.Thread):
    def __init__(self, camera_thread, live, display_queue):
        threading.Thread.__init__(self)
        self.camera_thread = camera_thread
        self.live = live
        self.running = True
        self.display_queue = display_queue

    def run(self):
        while self.running:
            if self.camera_thread.frame_ready.wait(1):
                frame = self.camera_thread.frame
                if self.live:
                    cv2.imshow('AprilTag Detection', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        break

                if not self.display_queue.empty():
                    frame_with_detections = self.display_queue.get()
                    cv2.imshow('AprilTag Detection', frame_with_detections)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        break

    def stop(self):
        self.running = False

class DetectionThread(threading.Thread):
    def __init__(self, camera_thread, detector, display_queue, print_delay, save, run_dir):
        threading.Thread.__init__(self)
        self.camera_thread = camera_thread
        self.detector = detector
        self.display_queue = display_queue
        self.print_delay = print_delay
        self.save = save
        self.run_dir = run_dir
        self.running = True
        self.image_count = 0

    def run(self):
        last_print_time = time.time()
        while self.running:
            if self.camera_thread.frame_ready.wait(1):
                frame = self.camera_thread.frame
                detections, undistorted_frame = self.detector.detect(frame)
                positions_orientations = self.detector.get_position_and_orientation(detections)

                current_time = time.time()
                if current_time - last_print_time >= self.print_delay:
                    for tag_id, position, tvec, orientation in positions_orientations:
                        pos_str = f"Position: {position}" if position is not None else "Position: N/A"
                        if tvec is not None:
                            distance = np.linalg.norm(tvec)
                            dist_str = f"Distance: {distance:.2f} meters"
                            logging.info(f"Tag ID: {tag_id}, Position: {position}, Distance: {distance:.2f} meters, Orientation: {orientation}")
                        else:
                            dist_str = "Distance: N/A"
                            logging.info(f"Tag ID: {tag_id}, {pos_str}, {dist_str}")
                    
                    frame_with_detections = self.detector.draw_detections(undistorted_frame, detections)
                    self.display_queue.put(frame_with_detections)

                    if self.save:
                        image_path = os.path.join(self.run_dir, f'apriltag_detection_{self.image_count}.png')
                        cv2.imwrite(image_path, frame_with_detections)
                        logging.info(f"Saved image: {image_path}")
                        self.image_count += 1

                    last_print_time = current_time

    def stop(self):
        self.running = False

def main():
    setup_logging()
    args = parse_args()

    config = load_config(args.config)
    live = args.live or config.get("live", False)
    save = args.save or config.get("save", False)
    calibration_path = args.calibration or config.get("calibration", "camera_calibration_data.npz")
    initial_position_path = args.initial_position or config.get("initial_position", "initial_camera_position.json")
    tag_size = config.get("tag_size", 0.0275)  # Default tag size to 0.0275 meters if not in config
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
        initial_positions = None
        logging.warning("Initial camera position data not found.")

    logging.info("Initializing camera...")
    camera_thread = CameraThread()
    camera_thread.start()

    # Wait until the first frame is captured
    camera_thread.frame_ready.wait()

    logging.info("Creating AprilTag detector...")
    detector = AprilTagDetector(camera_matrix, dist_coeffs, tag_size)
    box_position = BoxPosition()

    # Create a subdirectory for this run
    run_number = get_next_run_number()
    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(BASE_SAVE_DIR, f'run{run_number}_{run_timestamp}')
    os.makedirs(run_dir)

    display_queue = queue.Queue()

    logging.info("Starting video capture...")
    
    display_thread = DisplayThread(camera_thread, live, display_queue)
    display_thread.start()

    detection_thread = DetectionThread(camera_thread, detector, display_queue, print_delay, save, run_dir)
    detection_thread.start()

    try:
        while True:
            time.sleep(0.1)

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
            user_input = input("Do you want to keep the saved images? (y/n): ").strip().lower()
            if user_input == 'n':
                logging.info("Deleting saved images...")
                shutil.rmtree(run_dir)
                logging.info("Images deleted.")
            else:
                logging.info(f"Images kept in {run_dir}")

if __name__ == "__main__":
    main()
