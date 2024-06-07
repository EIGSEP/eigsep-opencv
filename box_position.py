class BoxPosition:
    def __init__(self):
        self.face_tags = {
            'right': 0,
            'bottom': 1,
            'left': 24,
            'top': 25
        }

    def determine_position(self, detections):
        detected_faces = set()
        for detection in detections:
            tag_id = detection.tag_id
            for face, id in self.face_tags.items():
                if tag_id == id:
                    detected_faces.add(face)
        return detected_faces
