import cv2
import numpy as np
import os
import glob
import json

def load_config(config_path='config.json'):
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config = json.load(file)
        print(f"Loaded configuration from {config_path}.")
    else:
        config = {}
        print(f"Configuration file {config_path} not found. Using default settings.")
    return config

def capture_images(save_dir, num_images=20, chessboard_size=(9, 6), square_size=40):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    os.makedirs(save_dir, exist_ok=True)
    image_count = 0

    while image_count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)
            cv2.imshow('Chessboard', frame)
            cv2.waitKey(500)  # Show the frame for 500 ms

            image_path = os.path.join(save_dir, f'chessboard_{image_count}.png')
            cv2.imwrite(image_path, frame)
            print(f"Saved image {image_count + 1}/{num_images}: {image_path}")
            image_count += 1

        cv2.imshow('Chessboard', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def calibrate_camera(image_dir, chessboard_size=(9, 6), square_size=40):
    obj_points = []
    img_points = []

    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    images = glob.glob(os.path.join(image_dir, '*.png'))

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            obj_points.append(objp)
            img_points.append(corners)

            cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
            cv2.imshow('Chessboard', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    return camera_matrix, dist_coeffs

def main():
    config = load_config()
    save_dir = 'calibration_images'
    chessboard_size = tuple(config.get("chessboard_size", [9, 6]))
    num_images = config.get("num_images", 20)
    square_size = config.get("square_size", 40)  # Default to 40mm

    capture_images(save_dir, num_images, chessboard_size, square_size)

    print("Calibrating camera...")
    camera_matrix, dist_coeffs = calibrate_camera(save_dir, chessboard_size, square_size)

    print("Camera calibration complete.")
    print("Camera matrix:")
    print(camera_matrix)
    print("Distortion coefficients:")
    print(dist_coeffs)

    # Save the camera matrix and distortion coefficients to a file
    np.savez('camera_calibration_data.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

if __name__ == "__main__":
    main()
