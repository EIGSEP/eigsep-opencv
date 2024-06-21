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
        self.initial_positions = initial_positions if initial_positions else []

    def determine_position(self, detections):
        detected_faces = set()
        for detection in detections:
            tag_id = detection.tag_id
            for face, tags in self.face_tags.items():
                if tag_id in tags.values():
                    detected_faces.add(face)
        return detected_faces

    def calculate_position(self, positions_orientations):
        if not self.initial_positions:
            return None, None

        positions = []
        orientations = []
        for tag_id, position, tvec, orientation in positions_orientations:
            if position is not None:
                positions.append(position)
                orientations.append(orientation)

        if positions:
            avg_position = np.mean(positions, axis=0)
            avg_orientation = np.mean(orientations) if orientations else None
            return avg_position, avg_orientation
        else:
            return None, None
