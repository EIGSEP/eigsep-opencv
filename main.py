import os
import time
import cv2
import shutil
import argparse
import numpy as np
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
    parser.add_argument("-c", "--calibration", type=str, default="camera_calibration_data.npz", help="Path to camera calibration data")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load camera calibration data
    if os.path.exists(args.calibration):
        calibration_data = np.load(args.calibration)
        print("Loaded camera calibration data.")
    else:
        calibration_data = None
        print("Camera calibration data not found. Proceeding without calibration.")

    print("Initializing camera...")
    camera_thread = CameraThread()
    camera_thread.start()

    # Wait until the first frame is captured
    camera_thread.frame_ready.wait()

    print("Creating AprilTag detector...")
    detector = AprilTagDetector(calibration_data)
    box_position = BoxPosition()

    # Create a subdirectory for this run
    run_number = get_next_run_number()
    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(BASE_SAVE_DIR, f'run{run_number}_{run_timestamp}')
    os.makedirs(run_dir)

    print("Starting video capture...")
    last_print_time = time.time()
    print_delay = 2  # Delay in seconds between prints
    image_count = 0
    saved_images = []

    try:
        while True:
            if not camera_thread.ret:
                print("Error: Could not read frame.")
                break

            frame = camera_thread.frame
            detections, undistorted_frame = detector.detect(frame)
            detected_faces = box_position.determine_position(detections)
            positions_orientations = detector.get_position_and_orientation(detections)

            current_time = time.time()
            if current_time - last_print_time >= print_delay:
                print(f"Detected faces: {detected_faces}")
                for tag_id, position, orientation in positions_orientations:
                    print(f"Tag ID: {tag_id}, Position: {position}, Orientation: {orientation:.2f} degrees")
                last_print_time = current_time

                frame_with_detections = detector.draw_detections(undistorted_frame, detections)

                if args.save:
                    # Save the frame with detections to a file
                    image_path = os.path.join(run_dir, f'apriltag_detection_{image_count}.png')
                    cv2.imwrite(image_path, frame_with_detections)
                    saved_images.append(image_path)
                    print(f"Saved image: {image_path}")
                    image_count += 1

                if args.live:
                    cv2.imshow('AprilTag Detection', frame_with_detections)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            time.sleep(1)  # Small delay to control the loop frequency

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        print("Releasing resources...")
        camera_thread.stop()
        cv2.destroyAllWindows()

        if saved_images:
            user_input = input("Do you want to keep the saved images? (y/n): ").strip().lower()
            if user_input == 'n':
                print("Deleting saved images...")
                shutil.rmtree(run_dir)
                print("Images deleted.")
            else:
                print(f"Images kept in {run_dir}")

if __name__ == "__main__":
    main()
