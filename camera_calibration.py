import cv2
import numpy as np
import os
import glob
import json
import threading
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

class CaptureThread(threading.Thread):
    def __init__(self, camera_thread, save_dir, num_images=20, chessboard_size=(9, 6), square_size=40):
        threading.Thread.__init__(self)
        self.camera_thread = camera_thread
        self.save_dir = save_dir
        self.num_images = num_images
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.image_count = 0

    def run(self):
        os.makedirs(self.save_dir, exist_ok=True)

        try:
            while self.image_count < self.num_images:
                if self.camera_thread.frame_ready.wait(1):
                    frame = self.camera_thread.frame
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

                    if ret:
                        cv2.drawChessboardCorners(frame, self.chessboard_size, corners, ret)
                        cv2.imshow('Chessboard', frame)
                        cv2.waitKey(1)  # Short delay to keep the display responsive

                        image_path = os.path.join(self.save_dir, f'chessboard_{self.image_count}.png')
                        cv2.imwrite(image_path, frame)
                        print(f"Saved image {self.image_count + 1}/{self.num_images}: {image_path}")
                        self.image_count += 1

                    else:
                        cv2.imshow('Chessboard', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break

        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            self.camera_thread.stop()
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
            cv2.waitKey(1)

    cv2.destroyAllWindows()

    if len(obj_points) > 0 and len(img_points) > 0:
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
        return camera_matrix, dist_coeffs
    else:
        print("Calibration failed: No valid chessboard corners were found.")
        return None, None

def main():
    config = load_config()
    save_dir = 'calibration_images'
    chessboard_size = tuple(config.get("chessboard_size", [9, 6]))
    num_images = config.get("num_images", 20)
    square_size = config.get("square_size", 40)  # Default to 40mm

    camera_thread = CameraThread()
    camera_thread.start()

    capture_thread = CaptureThread(camera_thread, save_dir, num_images, chessboard_size, square_size)
    capture_thread.start()
    capture_thread.join()

    print("Calibrating camera...")
    camera_matrix, dist_coeffs = calibrate_camera(save_dir, chessboard_size, square_size)

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
