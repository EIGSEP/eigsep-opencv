import cv2
import apriltag
import numpy as np

def main():
    # Initialize the camera
    cap = cv2.VideoCapture(0)

    # Create the AprilTag detector
    options = apriltag.DetectorOptions(families="tag36h11")
    detector = apriltag.Detector(options)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale as required by the AprilTag detector
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect AprilTags in the image
        detections = detector.detect(gray)

        # Draw detections on the frame
        for detection in detections:
            # Draw the bounding box
            for i in range(4):
                pt1 = (int(detection.corners[i][0]), int(detection.corners[i][1]))
                pt2 = (int(detection.corners[(i + 1) % 4][0]), int(detection.corners[(i + 1) % 4][1]))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

            # Draw the center
            center = (int(detection.center[0]), int(detection.center[1]))
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            # Display the tag ID
            tag_id = detection.tag_id
            cv2.putText(frame, str(tag_id), center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('AprilTag Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
