import numpy as np

class BoxPosition:
    def __init__(self, initial_positions=None):
        self.face_tags = {
            'right': {'top_left': 0, 'top_right': 1, 'bottom_left': 2, 'bottom_right': 3},
            'bottom': {'top_left': 25, 'top_right': 24, 'bottom_left': 23, 'bottom_right': 22},
            'left': {'top_left': 4, 'top_right': 5, 'bottom_left': 6, 'bottom_right': 7},
        }
        self.tag_relationships = {
            2: (25, 90),  # Bottom left of the right face borders the top left of the bottom face
            25: (2, 90),  # Top left of the bottom face borders the bottom left of the right face
            3: (24, 90),  # Bottom right of the right face borders the top right of the bottom face
            24: (3, 90),  # Top right of the bottom face borders the bottom right of the right face
            23: (4, 90),  # Bottom left of the bottom face borders the top left of the left face
            4: (23, 90),  # Top left of the left face borders the bottom left of the bottom face
            22: (5, 90),  # Bottom right of the bottom face borders the top right of the left face
            5: (22, 90),  # Top right of the left face borders the bottom right of the bottom face
            0: (6, 180),  # Top left of the right face is directly across from tag 6 on the left face
            6: (0, 180),  # Top left of the left face is directly across from tag 0 on the right face
            1: (7, 180),  # Top right of the right face is directly across from tag 7 on the left face
            7: (1, 180),  # Top right of the left face is directly across from tag 1 on the right face
            2: (4, 180),  # Bottom left of the right face is directly across from tag 4 on the left face
            4: (2, 180),  # Top left of the left face is directly across from tag 2 on the right face
            3: (5, 180),  # Bottom right of the right face is directly across from tag 5 on the left face
            5: (3, 180),   # Top right of the left face is directly across from tag 3 on the right face
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
        # Calculate relative orientation based on detected tags
        relative_orientation = self.calculate_relative_orientation(positions_orientations)

        return avg_position, orientation_degrees, relative_orientation  

    def calculate_relative_orientation(self, positions_orientations):
        if not self.initial_positions:
            return None

        initial_tags = set(self.initial_positions.keys())

        # If no tags are detected, guess that we are facing the top face
        if not positions_orientations:
            return 0  # Assuming 0 degrees for top face

        relative_orientation = None

        # Determine relative orientation based on tag relationships
        for tag_id, current_position, current_tvec, current_orientation in positions_orientations:
            if current_position is None:
                continue

            for initial_tag, (related_tag, angle) in self.tag_relationships.items():
                if tag_id == initial_tag and related_tag in initial_tags:
                    initial_position = np.array(self.initial_positions[related_tag]['position'])
                    current_position = np.array(current_position)

                    # Calculate the angle between the initial and current position
                    vector_initial = initial_position - np.array([0, 0, 0])
                    vector_current = current_position - np.array([0, 0, 0])
                    dot_product = np.dot(vector_initial, vector_current)
                    magnitude_initial = np.linalg.norm(vector_initial)
                    magnitude_current = np.linalg.norm(vector_current)
                    angle_between = np.arccos(dot_product / (magnitude_initial * magnitude_current))
                    angle_between_degrees = np.degrees(angle_between)

                    # Calculate the relative orientation
                    relative_orientation = (self.initial_positions[related_tag]['orientation'] + angle + angle_between_degrees) % 360
                    break

        return relative_orientation

    def get_orientation_from_tags(self, detections):
        orientations = [d.orientation for d in detections if d.orientation is not None]
        if orientations:
            avg_orientation = np.mean(orientations)
            return avg_orientation
        else:
            return None
