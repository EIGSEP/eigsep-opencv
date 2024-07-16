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
        else:
            orientation_degrees = None

        return avg_position, orientation_degrees

    def determine_position(self, detections):
        detected_faces = set()
        for detection in detections:
            tag_id = detection.tag_id
            for face, tags in self.face_tags.items():
                if tag_id in tags.values():
                    detected_faces.add(face)
        return detected_faces

    def calculate_position(self, detections):
        if not self.initial_positions:
            return None, None

        positions = []
        for detection in detections:
            tag_id = detection.tag_id
            for face, tags in self.face_tags.items():
                if tag_id in tags.values():
                    distance = detection.distance
                    orientation = detection.orientation
                    pos = self.get_position_from_tag(face, tag_id, distance, orientation)
                    positions.append(pos)

        if positions:
            avg_position = np.mean(positions, axis=0)
            return avg_position, self.get_orientation_from_tags(detections)
        else:
            return None, None

    def get_position_from_tag(self, face, tag_id, distance, orientation):
        # Simplified example, would need real calculations based on tag and face layout
        x, y, z = 0, 0, 0
        if face == 'right':
            if tag_id == self.face_tags[face]['top']:
                y = distance
            elif tag_id == self.face_tags[face]['bottom']:
                y = -distance
        elif face == 'bottom':
            if tag_id == self.face_tags[face]['top']:
                x = distance
            elif tag_id == self.face_tags[face]['bottom']:
                x = -distance
        elif face == 'left':
            if tag_id == self.face_tags[face]['top']:
                y = distance
            elif tag_id == self.face_tags[face]['bottom']:
                y = -distance
        return np.array([x, y, z])

    def get_orientation_from_tags(self, detections):
        orientations = [d.orientation for d in detections if d.orientation is not None]
        if orientations:
            avg_orientation = np.mean(orientations)
            return avg_orientation
        else:
            return None
