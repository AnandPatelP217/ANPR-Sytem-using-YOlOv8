import cv2
import os
from ultralytics import YOLO

def detect_and_count_faces(video_source):
    cap = cv2.VideoCapture(video_source)
    model = YOLO("best (2).pt")
    people_count = 0

    # Create save directory if it doesn't exist
    save_dir = "detected_faces"
    os.makedirs(save_dir, exist_ok=True)

    while True:
        success, img = cap.read()

        if not success:
            break

        # Perform object detection
        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls = int(box.cls[0])

                if cls == 0:  # Assuming class 0 for faces
                    people_count = 2

                    # Save detected face
                    face = img[y1:y2, x1:x2]
                    cv2.imwrite(f"{save_dir}/face_{people_count}.jpg", face)

                # Draw bounding box around the detected face
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

        cv2.putText(img, f"Detected Faces: {people_count}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_source = 0  # Set to the index of the camera or the path to the video file
detect_and_count_faces(video_source)
