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

def capture_images(save_dir, num_images=30, chessboard_size=(9, 6), square_size=40, live=False):
    camera_thread = CameraThread()
    camera_thread.start()

    os.makedirs(save_dir, exist_ok=True)
    image_count = 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    try:
        while image_count < num_images:
            if camera_thread.frame_ready.wait(1):
                frame = camera_thread.frame
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

                if ret:
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    cv2.drawChessboardCorners(frame, chessboard_size, corners2, ret)

                    if live:
                        cv2.imshow('Chessboard', frame)
                        cv2.waitKey(1)

                    image_path = os.path.join(save_dir, f'chessboard_{image_count}.png')
                    cv2.imwrite(image_path, frame)
                    print(f"Saved image {image_count + 1}/{num_images}: {image_path}")
                    image_count += 1
                else:
                    if live:
                        cv2.imshow('Chessboard', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        camera_thread.stop()
        cv2.destroyAllWindows()

def calibrate_camera(image_dir, chessboard_size=(9, 6), square_size=40):
    obj_points = []
    img_points = []

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    images = glob.glob(os.path.join(image_dir, '*.png'))
    print(f"Found {len(images)} images for calibration.")

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            obj_points.append(objp)
            img_points.append(corners2)

            cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv2.imshow('Chessboard', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    if len(obj_points) > 0 and len(img_points) > 0:
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
        print(f"Calibration reprojection error: {ret}")

        return camera_matrix, dist_coeffs, rvecs, tvecs, ret
    else:
        print("Calibration failed: No valid chessboard corners were found.")
        return None, None, None, None, None

def main():
    parser = argparse.ArgumentParser(description="Camera calibration with chessboard patterns.")
    parser.add_argument("-l", "--live", action="store_true", help="Show live video feed during calibration")
    args = parser.parse_args()

    config = load_config()
    save_dir = 'calibration_images'
    chessboard_size = tuple(config.get("chessboard_size", [9, 6]))
    num_images = config.get("num_images", 30)
    square_size = config.get("square_size", 40)  # Default to 40mm

    try:
        capture_images(save_dir, num_images, chessboard_size, square_size, live=args.live)

        print("Calibrating camera...")
        camera_matrix, dist_coeffs, rvecs, tvecs, error = calibrate_camera(save_dir, chessboard_size, square_size)

        if camera_matrix is not None and dist_coeffs is not None:
            print("Camera calibration complete.")
            print("Camera matrix:")
            print(camera_matrix)
            print("Distortion coefficients:")
            print(dist_coeffs)
            print("Reprojection error:")
            print(error)

            np.savez('camera_calibration_data.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs, rvecs=rvecs, tvecs=tvecs, error=error)
        else:
            print("Camera calibration failed.")
    except KeyboardInterrupt:
        print("Camera calibration failed.")

if __name__ == "__main__":
    main()