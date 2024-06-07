import os
import time
import cv2
import json
import argparse
import logging
import numpy as np
from camera_thread import CameraThread
from apriltag_detector import AprilTagDetector
from box_position import BoxPosition

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
    parser = argparse.ArgumentParser(description="Box Position Tracker.")
    parser.add_argument('--live', action='store_true', help='Display live video feed.')
    parser.add_argument('--save', action='store_true', help='Save images during the run.')
    parser.add_argument('--calibration', type=str, default='camera_calibration_data.npz', help='Path to the camera calibration data.')
    parser.add_argument('--initial_position', type=str, default='initial_camera_position.json', help='Path to the initial camera position data.')
    args = parser.parse_args()

    config = load_config()
    live = args.live or config.get('live', False)
    save = args.save or config.get('save', False)
    calibration_file = args.calibration or config.get('calibration', 'camera_calibration_data.npz')
    initial_position_file = args.initial_position or config.get('initial_position', 'initial_camera_position.json')
    print_delay = config.get('print_delay', 2)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    calibration_data = np.load(calibration_file) if os.path.exists(calibration_file) else None

    with open(initial_position_file, 'r') as file:
        initial_positions = json.load(file)
    logger.info(f"Loaded initial camera position from {initial_position_file}.")

    logger.info("Starting video capture...")
    camera_thread = CameraThread()
    camera_thread.start()

    tag_detector = AprilTagDetector(calibration_data)
    box_position = BoxPosition()

    run_number = get_next_run_number()
    run_dir = os.path.join("saved_images", f"run{run_number}_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(run_dir, exist_ok=True)

    try:
        while True:
            if camera_thread.frame_ready.wait(1):
                frame = camera_thread.frame
                detections, frame = tag_detector.detect(frame)
                if live:
                    frame = tag_detector.draw_detections(frame, detections)
                    cv2.imshow('Box Position Tracker', frame)

                detected_faces = box_position.determine_position(detections)
                positions_orientations = tag_detector.get_position_and_orientation(detections)

                for tag_id, position, distance, orientation in positions_orientations:
                    logger.info(f"Tag ID: {tag_id}, Position: {position}, Distance: {distance}, Orientation: {orientation}")

                logger.info(f"Detected Faces: {detected_faces}")

                if save:
                    image_path = os.path.join(run_dir, f'image_{int(time.time())}.png')
                    cv2.imwrite(image_path, frame)

                if cv2.waitKey(print_delay) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        logger.info("Keyboard Interrupt detected. Stopping...")
    finally:
        logger.info("Releasing resources...")
        camera_thread.stop()
        cv2.destroyAllWindows()

        if save:
            keep_photos = input("Do you want to keep the saved photos? (y/n): ").strip().lower()
            if keep_photos != 'y':
                shutil.rmtree(run_dir)
                logger.info(f"Deleted saved images in {run_dir}.")
            else:
                logger.info(f"Saved images are kept in {run_dir}.")

if __name__ == "__main__":
    main()
