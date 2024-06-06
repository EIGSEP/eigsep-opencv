import cv2
import threading
import time

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
