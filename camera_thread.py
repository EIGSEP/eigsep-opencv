import threading
import time
import cv2
import queue
import os
import logging
import numpy as np

class CameraThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.cap = cv2.VideoCapture(0)
        self.ret = False
        self.frame = None
        self.running = True
        self.frame_ready = threading.Event()

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame
                self.ret = ret
                self.frame_ready.set()
            else:
                self.ret = False
                self.frame_ready.clear()

    def stop(self):
        self.running = False
        self.cap.release()

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
                    if self.live:
                        cv2.imshow('AprilTag Detection', frame_with_detections)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            self.running = False
                            break

    def stop(self):
        self.running = False
        
class DetectionThread(threading.Thread):
    def __init__(self, camera_thread, detector, display_queue, print_delay, save, run_dir, box_position):
        threading.Thread.__init__(self)
        self.camera_thread = camera_thread
        self.detector = detector
        self.display_queue = display_queue
        self.print_delay = print_delay
        self.save = save
        self.run_dir = run_dir
        self.box_position = box_position
        self.running = True
        self.image_count = 0

    def run(self):
        last_print_time = time.time()
        try:
            while self.running:
                if self.camera_thread.frame_ready.wait(1):
                    frame = self.camera_thread.frame
                    if frame is None:
                        continue
                    detections, undistorted_frame = self.detector.detect(frame)
                    positions_orientations = self.detector.get_position_and_orientation(detections)
                    current_position, current_orientation = self.box_position.calculate_orientation(positions_orientations)

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

                        if current_orientation is not None:
                            logging.info(f"Current box position: {current_position}, Orientation: {current_orientation:.2f} degrees")
                        else:
                            logging.info(f"Current box position: {current_position}, Orientation: N/A")

                        logging.info(f"Rotation count: {self.box_position.rotation_count}")

                        frame_with_detections = self.detector.draw_detections(undistorted_frame, detections)
                        self.display_queue.put(frame_with_detections)

                        if self.save:
                            image_path = os.path.join(self.run_dir, f'apriltag_detection_{self.image_count}.png')
                            cv2.imwrite(image_path, frame_with_detections)
                            logging.info(f"Saved image: {image_path}")
                            self.image_count += 1

                        last_print_time = current_time
        except Exception as e:
            logging.error(f"Error in DetectionThread: {e}")
        finally:
            self.stop()

    def stop(self):
        self.running = False
