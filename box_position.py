import numpy as np

class BoxPosition:
    def __init__(self, initial_positions=None):
        self.face_tags = {
            'right': {'top': 0, 'bottom': 1},
            'bottom': {'top': 24, 'bottom': 25},
            'left': {'top': 2, 'bottom': 3},
            'top': {'top': 4, 'bottom': 5}  # Currently not used
        }
        self.tag_relationships = {
            1: 24,  # Bottom of the right face borders the top of the bottom face
            25: 2,  # Bottom of the bottom face borders the top of the left face
            3: 4,   # Bottom of the left face borders the top of the top face (currently not used)
            5: 0,   # Bottom of the top face borders the top of the right face (currently not used)
        }
        self.initial_positions = initial_positions if initial_positions else {}
        self.rotation_order = []
        self.rotation_count = 0

    def determine_orientation(self, positions_orientations):
        # Calculate the orientation of the box using the positions of the detected tags
        if len(positions_orientations) < 2:
            return None, None

        tag_positions = np.array([pos for _, pos, _, _ in positions_orientations if pos is not None])
        if tag_positions.shape[0] < 2:
            return None, None

        avg_position = np.mean(tag_positions, axis=0)
        orientation = np.arctan2(avg_position[1], avg_position[0])
        return avg_position, orientation

    def calculate_orientation(self, positions_orientations):
        if not self.initial_positions:
            return None, None

        avg_position, orientation = self.determine_orientation(positions_orientations)
        return avg_position, orientation

    def get_orientation_from_tags(self, detections):
        orientations = [d.orientation for d in detections if d.orientation is not None]
        if orientations:
            avg_orientation = np.mean(orientations)
            return avg_orientation
        else:
            return None
