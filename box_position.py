import numpy as np

class BoxPosition:
    def __init__(self, initial_positions=None):
        self.face_tags = {
            'right': {'top_left': 0, 'top_right': 1, 'bottom_left': 2, 'bottom_right': 3},
            'bottom': {'top_left': 25, 'top_right': 24, 'bottom_left': 23, 'bottom_right': 22},
            'left': {'top_left': 8, 'top_right': 9, 'bottom_left': 10, 'bottom_right': 11},
        }
        self.tag_relationships = {
            2: 25,  # Bottom right of the right face borders the top right of the bottom face
            3: 24,  # Bottom left of the right face borders the top left of the bottom face
            23: 4,  # Bottom right of the bottom face borders the top right of the left face
            22: 5,  # Bottom left of the bottom face borders the top left of the left face
        }
        self.initial_positions = initial_positions if initial_positions else {}
        self.rotation_order = []
        self.rotation_count = 0

    def determine_orientation(self, positions_orientations):
        if len(positions_orientations) == 0:
            return None, None

        tag_positions = []
        for _, pos, _, _ in positions_orientations:
            if pos is not None:
                tag_positions.append(pos)

        tag_positions = np.array(tag_positions)

        if tag_positions.size == 0:  # Ensure there are valid positions
            return None, None

        avg_position = np.mean(tag_positions, axis=0)
        orientation = np.arctan2(avg_position[1], avg_position[0])
        return avg_position, orientation

    def calculate_orientation(self, positions_orientations):
        avg_position, orientation = self.determine_orientation(positions_orientations)
        orientation_degrees = None
        relative_orientation = None

        if orientation is not None:
            orientation_degrees = np.degrees(orientation)
            if orientation_degrees < 0:
                orientation_degrees += 360
            if self.initial_positions:
                initial_orientation = self.initial_positions.get('orientation', 180)
                relative_orientation = (orientation_degrees - initial_orientation) % 360
            else:
                relative_orientation = orientation_degrees

        return avg_position, orientation_degrees, relative_orientation

    def get_orientation_from_tags(self, detections):
        orientations = [d.orientation for d in detections if d.orientation is not None]
        if orientations:
            avg_orientation = np.mean(orientations)
            return avg_orientation
        else:
            return None
