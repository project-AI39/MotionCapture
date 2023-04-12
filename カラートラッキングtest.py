# Import necessary libraries
import cv2

# Open the webcam
cap = cv2.VideoCapture(0)

# Define the color range for blue color
blue_lower = (100, 50, 50)
blue_upper = (130, 255, 255)

while True:
    # Capture frame from webcam
    ret, frame = cap.read()

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the frame to extract blue color
    mask = cv2.inRange(hsv, blue_lower, blue_upper)

    # Perform bitwise AND operation to get only blue color pixels
    blue_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Display the original frame and the blue color extracted frame
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Blue Color Tracking', blue_frame)
    cv2.imshow('Mask', mask)

    # Exit loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
