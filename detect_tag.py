import cv2
import apriltag
import numpy as np
import time
import threading
import os
import shutil

# Directory to save images
SAVE_DIR = "saved_images"
os.makedirs(SAVE_DIR, exist_ok=True)

class CameraThread(threading.Thread):
    def __init__(self, camera_index=0):
        threading.Thread.__init__(self)
        self.cap = cv2.VideoCapture(camera_index)
        self.ret = False
        self.frame = None
        self.running = True
        self.frame_ready = threading.Event()

    def run(self):
        while self.running:
            self.ret, self.frame = self.cap.read()
            if self.ret:
                self.frame_ready.set()
            time.sleep(0.01)  # Small delay to reduce CPU usage

    def stop(self):
        self.running = False
        self.cap.release()

def main():
    print("Initializing camera...")
    camera_thread = CameraThread()
    camera_thread.start()

    # Wait until the first frame is captured
    camera_thread.frame_ready.wait()

    print("Creating AprilTag detector...")
    options = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(options)

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
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = detector.detect(gray)

            current_time = time.time()
            if current_time - last_print_time >= print_delay:
                print(f"Number of tags detected: {len(detections)}")
                last_print_time = current_time

                for detection in detections:
                    try:
                        # Draw the bounding box
                        for i in range(4):
                            pt1 = (int(detection.corners[i][0]), int(detection.corners[i][1]))
                            pt2 = (int(detection.corners[(i + 1) % 4][0]), int(detection.corners[(i + 1) % 4][1]))
                            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

                        # Draw the center
                        center = (int(detection.center[0]), int(detection.center[1]))
                        cv2.circle(frame, center, 5, (0, 0, 255), -1)

                        # Display the tag ID
                        tag_id = detection.tag_id  # This should be the unique ID of the detected tag
                        cv2.putText(frame, str(tag_id), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                        # Print position and orientation
                        position = (detection.center[0], detection.center[1])  # Center of the tag in the image
                        orientation = np.degrees(np.arctan2(detection.corners[1][1] - detection.corners[0][1],
                                                            detection.corners[1][0] - detection.corners[0][0]))  # Orientation angle
                        print(f"Tag ID: {tag_id}, Position: {position}, Orientation: {orientation:.2f} degrees")
                    except Exception as e:
                        print(f"Error processing detection: {e}")

                # Save the frame with detections to a file
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                image_path = os.path.join(SAVE_DIR, f'apriltag_detection_{timestamp}_{image_count}.png')
                cv2.imwrite(image_path, frame)
                saved_images.append(image_path)
                print(f"Saved image: {image_path}")
                image_count += 1

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
                for image_path in saved_images:
                    os.remove(image_path)
                print("Images deleted.")
            else:
                print("Images kept.")

if __name__ == "__main__":
    main()
