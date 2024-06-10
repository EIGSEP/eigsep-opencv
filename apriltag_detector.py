import cv2
import apriltag
import numpy as np

class AprilTagDetector:
    def __init__(self, camera_matrix=None, dist_coeffs=None, tag_size=0.0275):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.tag_size = tag_size
        self.detector = apriltag.Detector()

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray)
        return detections, gray

    def get_position_and_orientation(self, detections):
        positions_orientations = []
        for detection in detections:
            corners = detection.corners
            tag_id = detection.tag_id

            if self.camera_matrix is not None and self.dist_coeffs is not None:
                object_points = np.array([
                    [-0.5, -0.5, 0],
                    [0.5, -0.5, 0],
                    [0.5, 0.5, 0],
                    [-0.5, 0.5, 0]
                ]) * self.tag_size

                image_points = np.array(corners, dtype=np.float32)

                success, rvec, tvec = cv2.solvePnP(object_points, image_points, self.camera_matrix, self.dist_coeffs)
                if success:
                    position = tvec.flatten()
                    rotation_matrix, _ = cv2.Rodrigues(rvec)
                    orientation = cv2.RQDecomp3x3(rotation_matrix)[0]
                    positions_orientations.append((tag_id, position, tvec, orientation))
                else:
                    positions_orientations.append((tag_id, None, None, None))
            else:
                positions_orientations.append((tag_id, None, None, None))

        return positions_orientations

    def draw_detections(self, frame, detections):
        for detection in detections:
            corners = detection.corners
            corners = np.int32(corners)
            cv2.polylines(frame, [corners], isClosed=True, color=(0, 255, 0), thickness=2)

            center = tuple(np.int32(detection.center))
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            tag_id = str(detection.tag_id)
            cv2.putText(frame, tag_id, center, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        return frame
