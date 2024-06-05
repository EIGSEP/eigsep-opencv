import cv2
import apriltag
import numpy as np
import time
import threading

class CameraThread(threading.Thread):
    def __init__(self, camera_index=0):
        threading.Thread.__init__(self)
        self.cap = cv2.VideoCapture(camera_index)
        self.ret = False
        self.frame = None
        self.running = True

    def run(self):
        while self.running:
            self.ret, self.frame = self.cap.read()
            time.sleep(0.01)  # Small delay to reduce CPU usage

    def stop(self):
        self.running = False
        self.cap.release()

def main():
    print("Initializing camera...")
    camera_thread = CameraThread()
    camera_thread.start()

    print("Creating AprilTag detector...")
    options = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(options)

    print("Starting video capture...")
    last_print_time = time.time()
    print_delay = 0.5  # Delay in seconds between prints

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
                    # Draw the bounding box
                    for i in range(4):
                        pt1 = (int(detection.corners[i][0]), int(detection.corners[i][1]))
                        pt2 = (int(detection.corners[(i + 1) % 4][0]), int(detection.corners[(i + 1) % 4][1]))
                        cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

                    # Draw the center
                    center = (int(detection.center[0]), int(detection.center[1]))
                    cv2.circle(frame, center, 5, (0, 0, 255), -1)

                    # Display the tag ID
                    tag_id = detection.tag_id
                    cv2.putText(frame, str(tag_id), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # Print position and orientation
                    position = (detection.center[0], detection.center[1])
                    orientation = np.degrees(np.arctan2(detection.corners[1][1] - detection.corners[0][1],
                                                        detection.corners[1][0] - detection.corners[0][0]))
                    print(f"Tag ID: {tag_id}, Position: {position}, Orientation: {orientation:.2f} degrees")

            cv2.imshow('AprilTag Detection', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        print("Releasing resources...")
        camera_thread.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
