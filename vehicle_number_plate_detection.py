import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Using OpenCV's pre-trained Haar Cascade for plate detection
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

def detect_number_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, 1.1, 10)

    for (x, y, w, h) in plates:
        # Draw rectangle around the plate
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        plate = img[y:y + h, x:x + w]
        # Show extracted number plate (Optional)
        cv2.imshow("Detected Number Plate", plate)
    
    return img

# List of test image paths
image_paths = [
    'car_image1.jpg',
    'car_image2.jpg',
    'car_image3.jpg'  # Add more images as needed
]

# Loop through each image and detect number plates
for image_path in image_paths:
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image {image_path}")
        continue

    # Detect number plate
    detected_img = detect_number_plate(img)

    # Display the result
    cv2.imshow(f"Vehicle Number Plate Detection - {image_path}", detected_img)
    cv2.waitKey(0)  # Wait for a key press to move to the next image
    cv2.destroyAllWindows()
