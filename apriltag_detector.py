import cv2
import apriltag
import numpy as np

class AprilTagDetector:
    def __init__(self, calibration_data=None):
        self.options = apriltag.DetectorOptions(families="tag36h11")
        self.detector = apriltag.Detector(self.options)
        self.calibration_data = calibration_data
        if calibration_data:
            self.camera_matrix = calibration_data['camera_matrix']
            self.dist_coeffs = calibration_data['dist_coeffs']
        else:
            self.camera_matrix = None
            self.dist_coeffs = None

    def undistort(self, frame):
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs, None, self.camera_matrix)
        return frame

    def detect(self, frame):
        undistorted_frame = self.undistort(frame)
        gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray)
        return detections, undistorted_frame

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

    def get_distance_and_orientation(self, detection):
        if self.camera_matrix is None:
            raise ValueError("Camera matrix is not available for distance and orientation calculation.")

        # Assuming the real size of the tag is known (in meters)
        tag_size = 0.04  # example size, 40mm

        # Calculate distance
        focal_length = self.camera_matrix[0, 0]
        perceived_width = np.linalg.norm(detection.corners[0] - detection.corners[1])
        distance = (tag_size * focal_length) / perceived_width

        # Calculate orientation
        orientation = np.degrees(np.arctan2(detection.corners[1][1] - detection.corners[0][1],
                                            detection.corners[1][0] - detection.corners[0][0]))

        return distance, orientation

    def get_position_and_orientation(self, detections):
        positions_orientations = []
        for detection in detections:
            distance, orientation = self.get_distance_and_orientation(detection)
            position = (detection.center[0], detection.center[1])
            positions_orientations.append((detection.tag_id, position, distance, orientation))
        return positions_orientations
