import numpy as np

class BoxPosition:
    def __init__(self, initial_positions=None):
        self.face_tags = {
            'right': {'top_left': 0, 'top_right': 1, 'bottom_left': 2, 'bottom_right': 3},
            'bottom': {'top_left': 25, 'top_right': 24, 'bottom_left': 23, 'bottom_right': 22},
            'left': {'top_left': 4, 'top_right': 5, 'bottom_left': 6, 'bottom_right': 7},
        }
        self.tag_relationships = {
            2: 25,  # Bottom left of the right face borders the top left of the bottom face
            3: 24,  # Bottom right of the right face borders the top right of the bottom face
            23: 4,  # Bottom left of the bottom face borders the top left of the left face
            22: 5,  # Bottom right of the bottom face borders the top right of the left face
            0: 6,   # Top left of the right face is directly across from tag 6 on the left face
            1: 7,   # Top right of the right face is directly across from tag 7 on the left face
            2: 4,   # Bottom right of the right face is directly across from tag 4 on the left face
            3: 5    # Bottom left of the right face is directly across from tag 5 on the left face
        }
        self.initial_positions = initial_positions if initial_positions else {}
        self.rotation_order = []
        self.rotation_count = 0

    def determine_orientation(self, positions_orientations):
        if len(positions_orientations) < 2:
            return None, None

        tag_positions = []
        for _, pos, _, _ in positions_orientations:
            if pos is not None:
                tag_positions.append(pos)

        tag_positions = np.array(tag_positions)

        if tag_positions.size == 0:
            return None, None

        avg_position = np.mean(tag_positions, axis=0)
        orientation = np.arctan2(avg_position[1], avg_position[0])
        return avg_position, orientation

    def calculate_orientation(self, positions_orientations):
        if not self.initial_positions:
            return None, None, None

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

    def calculate_relative_orientation(self, detections):
        if not self.initial_positions:
            return None

        current_tags = {detection.tag_id for detection in detections}
        initial_tags = set(self.initial_positions.keys())

        # If no tags are detected, guess that we are facing the top face
        if not current_tags:
            return 0  # Assuming 0 degrees for top face

        # Determine relative orientation based on tag relationships
        for current_tag in current_tags:
            for initial_tag in initial_tags:
                if current_tag in self.tag_relationships and initial_tag == self.tag_relationships[current_tag]:
                    return (self.initial_positions[initial_tag]['orientation'] + 180) % 360

        return None

    def get_orientation_from_tags(self, detections):
        orientations = [d.orientation for d in detections if d.orientation is not None]
        if orientations:
            avg_orientation = np.mean(orientations)
            return avg_orientation
        else:
            return None
