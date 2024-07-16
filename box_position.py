import numpy as np

class BoxPosition:
    def __init__(self, initial_positions=None):
        self.face_tags = {
            'right': {'top': 0, 'bottom': 1},
            'bottom': {'top': 24, 'bottom': 25},
            'left': {'top': 2, 'bottom': 3},
            'top': {'top': 4, 'bottom': 5}  # Currently not used
        }
        self.initial_positions = initial_positions if initial_positions else {}
        self.rotation_count = 0
        self.previous_orientation = None

    def determine_orientation(self, positions_orientations):
        if len(positions_orientations) < 2:
            return None, None

        tag_positions = np.array([pos for _, pos, _, _ in positions_orientations if pos is not None])
        if tag_positions.shape[0] < 2:
            return None, None

        avg_position = np.mean(tag_positions, axis=0)
        orientation = np.arctan2(avg_position[1], avg_position[2])  # Considering Z-axis for up/down orientation
        return avg_position, orientation

    def calculate_orientation(self, positions_orientations):
        avg_position, orientation = self.determine_orientation(positions_orientations)

        if orientation is not None:
            orientation_degrees = np.degrees(orientation)
            if self.previous_orientation is not None:
                delta_orientation = orientation_degrees - self.previous_orientation
                if delta_orientation > 180:
                    delta_orientation -= 360
                elif delta_orientation < -180:
                    delta_orientation += 360

                if abs(delta_orientation) > 10:  # Threshold to consider a significant rotation
                    self.rotation_count += np.sign(delta_orientation)

            self.previous_orientation = orientation_degrees

        return avg_position, orientation_degrees

    def get_orientation_from_tags(self, detections):
        orientations = [d.orientation for d in detections if d.orientation is not None]
        if orientations:
            avg_orientation = np.mean(orientations)
            return avg_orientation
        else:
            return None
