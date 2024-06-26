import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('tomato_model.h5')

# Known width of a ripe tomato in centimeters
KNOWN_WIDTH = 6.0  # Adjust according to your ripe tomato's actual width

# Focal length of the camera in pixels (this should be calibrated for your specific setup)
FOCAL_LENGTH = 800  # Example value; adjust as per calibration

def detect_ripe_tomatoes(frame, model):
    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the red color (ripe tomatoes)
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # Create a mask for the red color
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through the contours
    for contour in contours:
        # Calculate the area of each contour
        area = cv2.contourArea(contour)

        # Set a threshold for the area to filter out small contours
        if area > 100:
            # Draw a bounding box around the detected tomato
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Calculate the distance to the tomato
            distance = calculate_distance(FOCAL_LENGTH, KNOWN_WIDTH, w)

            # Display the distance on the frame
            cv2.putText(frame, f"Distance: {distance:.2f} cm", (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Extract the tomato from the frame and resize it to the size the model expects
            tomato = frame[y:y + h, x:x + w]
            tomato = cv2.resize(tomato, (224, 224))

            # Normalize the image to a range of 0-1
            tomato = tomato / 255.0

            # Add an extra dimension for the batch size
            tomato = np.expand_dims(tomato, axis=0)

            # Use the model to predict whether the tomato is ripe or unripe
            prediction = model.predict(tomato)[0][0]

            # Convert the prediction to a binary label
            label = "Ripe" if prediction > 0.5 else "Unripe"

            # Draw the label on the frame
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the result
    cv2.imshow("Ripe Tomato Detection", frame)

# Function to calculate the distance to the object
def calculate_distance(focal_length, known_width, pixel_width):
    return (known_width * focal_length) / pixel_width

# Open a connection to the webcam (camera index 0 by default)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the frame was captured successfully
    if not ret:
        print("Error: Couldn't read frame")
        break

    # Perform ripe tomato detection on the frame
    detect_ripe_tomatoes(frame, model)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
