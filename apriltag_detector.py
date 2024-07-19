import cv2
import apriltag
import numpy as np

class AprilTagDetector:
    def __init__(self, camera_matrix=None, dist_coeffs=None, tag_size=0.080, zoom=1.0):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.tag_size = tag_size
        self.zoom = zoom
        self.detector = apriltag.Detector()
        
        if self.camera_matrix is not None:
            # Adjust camera matrix for zoom
            self.camera_matrix = self.camera_matrix.copy()
            self.camera_matrix[0, 0] *= zoom
            self.camera_matrix[1, 1] *= zoom

    def apply_digital_zoom(self, frame):
        if self.zoom != 1.0:
            height, width = frame.shape[:2]
            new_width = int(width / self.zoom)
            new_height = int(height / self.zoom)
            
            # Calculate cropping box
            x1 = (width - new_width) // 2
            y1 = (height - new_height) // 2
            x2 = x1 + new_width
            y2 = y1 + new_height
            
            # Crop and resize the frame
            cropped_frame = frame[y1:y2, x1:x2]
            zoomed_frame = cv2.resize(cropped_frame, (width, height), interpolation=cv2.INTER_LINEAR)
            return zoomed_frame
        return frame

    def detect(self, frame):
        frame = self.apply_digital_zoom(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray)
        return detections, frame

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
