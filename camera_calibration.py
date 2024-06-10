import numpy as np
import cv2 as cv
import glob
import json
import os
from camera_thread import CameraThread

def load_config(config_path='config.json'):
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = json.load(file)
        print(f"Loaded configuration from {config_path}.")
    else:
        config = {
            "chessboard_size": [9, 6],
            "square_size": 40,
            "num_images": 30
        }
        print(f"Configuration file {config_path} not found. Using default settings.")
    return config

def capture_images(save_dir, num_images=30, chessboard_size=(9, 6), live=False):
    camera_thread = CameraThread()
    camera_thread.start()

    os.makedirs(save_dir, exist_ok=True)
    image_count = 0
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    try:
        while image_count < num_images:
            if camera_thread.frame_ready.wait(1):
                frame = camera_thread.frame
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)

                if ret:
                    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                    cv.drawChessboardCorners(frame, chessboard_size, corners2, ret)

                    if live:
                        cv.imshow('Chessboard', frame)
                        cv.waitKey(1)

                    image_path = os.path.join(save_dir, f'chessboard_{image_count}.png')
                    cv.imwrite(image_path, frame)
                    print(f"Saved image {image_count + 1}/{num_images}: {image_path}")
                    image_count += 1
                else:
                    print(f"Chessboard not detected in image {image_count + 1}")
                    if live:
                        cv.imshow('Chessboard', frame)
                        if cv.waitKey(1) & 0xFF == ord('q'):
                            break

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        camera_thread.stop()
        cv.destroyAllWindows()

def calibrate_camera(image_dir, chessboard_size=(9, 6), square_size=40):
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob(os.path.join(image_dir, '*.png'))
    print(f"Found {len(images)} images for calibration.")

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (chessboard_size[0], chessboard_size[1]), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv.drawChessboardCorners(img, (chessboard_size[0], chessboard_size[1]), corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(500)
        else:
            print(f"Failed to detect chessboard in {fname}")

    cv.destroyAllWindows()

    if len(objpoints) > 0 and len(imgpoints) > 0:
        print(f"Number of valid images: {len(objpoints)}")
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print(f"Calibration reprojection error: {ret}")

        return camera_matrix, dist_coeffs, rvecs, tvecs, ret
    else:
        print("Calibration failed: No valid chessboard corners were found.")
        return None, None, None, None, None

def main():
    config = load_config()
    save_dir = 'calibration_images'
    chessboard_size = tuple(config.get("chessboard_size", [9, 6]))
    num_images = config.get("num_images", 30)
    square_size = config.get("square_size", 40)  # Default to 40mm

    live = config.get("live", False)

    capture_images(save_dir, num_images, chessboard_size, live=live)

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

if __name__ == "__main__":
    main()
