import os
import cv2
import pytesseract
import xml.etree.ElementTree as ET
from image_utils import are_similar  # Import the function for image similarity comparison
from datetime import datetime
import pytz
import json

def image_to_bytes(image):
    # Convert the image to bytes
    _, img_bytes = cv2.imencode('.jpg', image)
    return bytes(img_bytes)

def convert_to_india_time(timestamp):
    # Convert the timestamp to India timezone
    tz_india = pytz.timezone('Asia/Kolkata')
    dt = datetime.fromtimestamp(timestamp)
    dt_india = tz_india.localize(dt)
    return dt_india

def extract_ocr_to_xml(directory):
    # Create XML file
    root = ET.Element("OCR_Detections")
    
    # Create a list to store OCR data for conversion to JSON
    ocr_data_list = []

    # Keep track of previously processed OCR text
    previous_ocr_texts = set()

    # Loop through the detected images in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            image_path = os.path.join(directory, filename)

            # Read the image
            img = cv2.imread(image_path)

            # Perform OCR on the image
            ocr_text = pytesseract.image_to_string(img)

            # Check if OCR text already exists
            if ocr_text in previous_ocr_texts:
                continue  # Skip processing if OCR text is duplicate
            else:
                previous_ocr_texts.add(ocr_text)

            # Get the timestamp of the image and convert to India timezone
            timestamp = os.path.getmtime(image_path)
            timestamp_india = convert_to_india_time(timestamp)

            # Create XML elements for each OCR result
            ocr_element = ET.SubElement(root, "OCR_Result")
            filename_element = ET.SubElement(ocr_element, "Filename")
            filename_element.text = filename
            ocr_text_element = ET.SubElement(ocr_element, "OCR_Text")
            ocr_text_element.text = ocr_text
            timestamp_element = ET.SubElement(ocr_element, "Timestamp")
            timestamp_element.text = str(timestamp_india)

            # Store the OCR data for conversion to JSON
            ocr_data_list.append({
                "Filename": filename,
                "OCR_Text": ocr_text,
                "Timestamp": str(timestamp_india)
            })

    # Write XML tree to file
    tree = ET.ElementTree(root)
    tree.write("ocr_results.xml")

    print("OCR results saved to ocr_results.xml")

    # Convert OCR data to JSON and save to file
    with open("ocr_results.json", "w") as json_file:
        json.dump(ocr_data_list, json_file, indent=4)
    
    print("OCR results saved to ocr_results.json")

# Call the function to extract OCR and save to XML and JSON
extract_ocr_to_xml("detected_images")
