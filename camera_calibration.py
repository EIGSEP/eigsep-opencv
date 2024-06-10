import cv2
import numpy as np
import os
import glob
import json
import argparse
from camera_thread import CameraThread

def load_config(config_path='config.json'):
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = json.load(file)
        print(f"Loaded configuration from {config_path}.")
    else:
        config = {}
        print(f"Configuration file {config_path} not found. Using default settings.")
    return config

def capture_images_and_calibrate(save_dir, num_images=30, chessboard_size=(3, 3), square_size=40, live=False):
    camera_thread = CameraThread()
    camera_thread.start()

    os.makedirs(save_dir, exist_ok=True)
    image_count = 0
    obj_points = []
    img_points = []

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    try:
        while image_count < num_images:
            if camera_thread.frame_ready.wait(1):
                frame = camera_thread.frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
                if ret:
                    print(f"Chessboard detected: {image_count + 1}/{num_images}")
                    obj_points.append(objp)
                    img_points.append(corners)
                    cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)
                    if live:
                        cv2.imshow('Chessboard', frame)
                        cv2.waitKey(1)  # Short delay to keep the display responsive

                    image_path = os.path.join(save_dir, f'chessboard_{image_count}.png')
                    cv2.imwrite(image_path, frame)
                    print(f"Saved image {image_count + 1}/{num_images}: {image_path}")
                    image_count += 1
                else:
                    print("Chessboard not detected")
                    if live:
                        cv2.imshow('Chessboard', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        camera_thread.stop()
        cv2.destroyAllWindows()

    if len(obj_points) > 0 and len(img_points) > 0:
        print("Starting camera calibration...")
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
        if ret:
            print("Calibration succeeded")
            print(f"Camera matrix: \n{camera_matrix}")
            print(f"Distortion coefficients: \n{dist_coeffs}")
            return camera_matrix, dist_coeffs
        else:
            print("Calibration failed")
            return None, None
    else:
        print("Calibration failed: No valid chessboard corners were found.")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Camera calibration with chessboard patterns.")
    parser.add_argument("-l", "--live", action="store_true", help="Show live video feed during calibration")
    args = parser.parse_args()

    config = load_config()
    save_dir = 'calibration_images'
    chessboard_size = tuple(config.get("chessboard_size", [3, 3]))
    num_images = config.get("num_images", 30)
    square_size = config.get("square_size", 40)  # Default to 40mm
    print(chessboard_size, num_images, square_size)

    print("Capturing images and calibrating camera...")
    camera_matrix, dist_coeffs = capture_images_and_calibrate(save_dir, num_images, chessboard_size, square_size, live=args.live)

    if camera_matrix is not None and dist_coeffs is not None:
        print("Camera calibration complete.")
        print("Camera matrix:")
        print(camera_matrix)
        print("Distortion coefficients:")
        print(dist_coeffs)

        # Save the camera matrix and distortion coefficients to a file
        np.savez('camera_calibration_data.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    else:
        print("Camera calibration failed.")

if __name__ == "__main__":
    main()
