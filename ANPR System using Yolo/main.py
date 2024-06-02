import time
import cv2
import math
import pytesseract
from openpyxl import Workbook
from ultralytics import YOLO
import json


# Function to extract OCR from detected images and save to XML
def extract_ocr_to_xml(directory):
    pass  # Placeholder for your implementation


def predict_on_live_cctv(CCTV_URL, platform):
    global currentClass
    cap = cv2.VideoCapture(0)  # Video input
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Set Tesseract executable path
    wb = Workbook()
    ws = wb.active

    ws.append(['License Plate'])

    ClassNames = ['License_Plate']
    myColor = (0, 255, 0)

    alert_sent = False
    alert_threshold = 10
    people_count = 0

    # Interval time
    image_capture_interval = 6

    # Last saved time gap
    last_save_time = time.time()

    # Where you want to save detected images
    save_directory = "detected_images"

    # Load YOLO model
    model = YOLO("best (1).pt")

    # Counter for image names
    image_count = 1

    # Define a dictionary to store the detected position and direction of each license plate
    detected_position = {}

    # Function to calculate movement direction
    def calculate_direction(previous_position, current_position):
        # Check if previous_position is None
        if previous_position is None:
            return None

        # Compare the x-coordinate of previous and current positions
        if current_position[0] > previous_position[0]:
            return "Right"
        elif current_position[0] < previous_position[0]:
            return "Left"
        else:
            return "Straight"

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
                w, h = x2 - x1, y2 - y1
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = ClassNames[cls]

                if currentClass == 'License_Plate':
                    myColor = (0, 255, 0)
                    people_count += 1  # Increment the people count for each detection
                else:
                    myColor = (0, 255, 0)

                plate_img = img[y1:y2, x1:x2]

                plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
                plate_gray = cv2.GaussianBlur(plate_gray, (5, 5), 0)

                plate_text = pytesseract.image_to_string(plate_gray, config='--psm 8')
                print("License Plate OCR:", plate_text)

                # Store the previous and current positions of the license plate
                previous_position = detected_position.get(plate_text)
                current_position = (x1, y1)

                # Calculate direction if previous position exists
                if previous_position:
                    direction = calculate_direction(previous_position["position"], current_position)
                    print("License Plate Movement Direction:", direction)
                    # Store the direction in the dictionary
                    detected_position[plate_text]["direction"] = direction

                # Update the position of the license plate
                detected_position[plate_text] = {"position": current_position}

                # normal ocr in the xmls file
                ws.append([plate_text])

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 1)

                # Display class name
                cv2.putText(img, "", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, myColor, 1)

                # Save only the detected region
                roi_img = img[y1:y2, x1:x2]

                roi_file_name = f"detected{image_count}.jpg"
                cv2.imwrite(f"{save_directory}/{roi_file_name}", roi_img)
                image_count += 1

        current_time = time.time()
        time_difference = current_time - last_save_time

        if time_difference >= image_capture_interval:
            alert_sent = False

        cv2.putText(img, f"Detected License Plates Number is : {people_count}", (30, 330), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, myColor, 1)

        if people_count > alert_threshold and not alert_sent:
            print(f"Alert: More than {alert_threshold} people detected!")
            frame_name = f"{currentClass}_{current_time}.jpg"
            cv2.imwrite(f"{save_directory}/{frame_name}", img)
            print(f"Saved image: {frame_name}")
            last_save_time = current_time
            alert_sent = True

        people_count = 0
        # Exit code 0
        if cv2.waitKey(1) & 0xFF == ord('a'):
            break

        cv2.imshow("Image", img)
        cv2.waitKey(1)

    # Save detected positions and directions in JSON format
    with open('detected_position.json', 'w') as json_file:
        json.dump(detected_position, json_file)

    # Save XMLs format data
    wb.save("license_plates.xlsx")

    # Extract OCR from detected images and save to XML
    extract_ocr_to_xml(save_directory)


# Function calling for live CCTV
predict_on_live_cctv("CCTV_URL", "platform")
