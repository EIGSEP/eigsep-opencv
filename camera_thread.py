import threading
import time
import cv2
import queue
import os
import logging
import json
import numpy as np

class CameraThread(threading.Thread):
    def __init__(self, camera_index=0):
        threading.Thread.__init__(self)
        self.cap = cv2.VideoCapture(camera_index)
        self.ret = False
        self.frame = None
        self.frame_ready = threading.Event()
        self.running = True

    def run(self):
        while self.running:
            self.ret, self.frame = self.cap.read()
            if self.ret:
                self.frame_ready.set()
            time.sleep(0.01)

    def stop(self):
        self.running = False
        self.cap.release()

class DisplayThread(threading.Thread):
    def __init__(self, camera_thread, detector, live, display_queue):
        threading.Thread.__init__(self)
        self.camera_thread = camera_thread
        self.detector = detector
        self.live = live
        self.running = True
        self.display_queue = display_queue

    def run(self):
        while self.running:
            if self.camera_thread.frame_ready.wait(1):
                frame = self.camera_thread.frame
                if self.live:
                    zoomed_frame = self.detector.apply_digital_zoom(frame)  # Apply zoom to the frame
                    cv2.imshow('AprilTag Detection', zoomed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False
                        break

                if not self.display_queue.empty():
                    frame_with_detections = self.display_queue.get()
                    if self.live:
                        cv2.imshow('AprilTag Detection', frame_with_detections)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            self.running = False
                            break

    def stop(self):
        self.running = False

class DetectionThread(threading.Thread):
    def __init__(self, camera_thread, detector, display_queue, print_delay, save, run_dir, save_data, data_dir, box_position):
        threading.Thread.__init__(self)
        self.camera_thread = camera_thread
        self.detector = detector
        self.display_queue = display_queue
        self.print_delay = print_delay
        self.save = save
        self.run_dir = run_dir
        self.save_data = save_data
        self.data_dir = data_dir
        self.box_position = box_position
        self.running = True
        self.image_count = 0
        self.run_data = []
        self.last_save_time = time.time()

    def run(self):
        last_print_time = time.time()
        while self.running:
            if self.camera_thread.frame_ready.wait(1):
                frame = self.camera_thread.frame
                detections, zoomed_frame = self.detector.detect(frame)
                positions_orientations = self.detector.get_position_and_orientation(detections)
                current_position, current_orientation, relative_orientation = self.box_position.calculate_orientation(positions_orientations)

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
                    if current_position is not None:
                        logging.info(f"Current box position: {current_position}, Old Orientation: {old_orientation} degrees, Relative Orientation: {relative_orientation} degrees")
                    else:
                        logging.info(f"Current box position: N/A, Old Orientation: N/A, Relative Orientation: N/A")
                    logging.info(f"Rotation count: {self.box_position.rotation_count}")
                    frame_with_detections = self.detector.draw_detections(zoomed_frame, detections)
                    self.display_queue.put(frame_with_detections)

                    if self.save:
                        image_path = os.path.join(self.run_dir, f'apriltag_detection_{self.image_count}.png')
                        cv2.imwrite(image_path, frame_with_detections)
                        logging.info(f"Saved image: {image_path}")
                        self.image_count += 1

                    if self.save_data:
                        self.run_data.append({
                            'position': current_position.tolist() if current_position is not None else None,
                            'orientation': current_orientation,
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                        })

                        if current_time - self.last_save_time >= 60:  # Save data every minute
                            self.save_run_data()
                            self.last_save_time = current_time
                            self.run_data = []  # Clear the data to start fresh
                        last_print_time = current_time

    def stop(self):
        self.running = False
        self.save_run_data()

    def save_run_data(self):
        data_path = os.path.join(self.data_dir, 'run_data.json')
        with open(data_path, 'w') as file:
            json.dump(self.run_data, file)
        logging.info(f"Saved run data: {data_path}")
        self.run_data = []  # Clear the data to start fresh
