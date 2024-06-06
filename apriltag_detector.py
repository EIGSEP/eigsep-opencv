import cv2
import apriltag
import numpy as np

class AprilTagDetector:
    def __init__(self):
        self.options = apriltag.DetectorOptions(families="tag36h11")
        self.detector = apriltag.Detector(self.options)

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray)
        return detections

    def draw_detections(self, frame, detections):
        for detection in detections:
            for i in range(4):
                pt1 = (int(detection.corners[i][0]), int(detection.corners[i][1]))
                pt2 = (int(detection.corners[(i + 1) % 4][0]), int(detection.corners[(i + 1) % 4][1]))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            center = (int(detection.center[0]), int(detection.center[1]))
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            tag_id = detection.tag_id
            cv2.putText(frame, str(tag_id), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return frame

    def get_position_and_orientation(self, detections):
        positions_orientations = []
        for detection in detections:
            position = (detection.center[0], detection.center[1])
            orientation = np.degrees(np.arctan2(detection.corners[1][1] - detection.corners[0][1],
                                                detection.corners[1][0] - detection.corners[0][0]))
            positions_orientations.append((detection.tag_id, position, orientation))
        return positions_orientations
