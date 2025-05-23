import numpy as np
import cv2

# Load the Haar Cascade for face detection
face_classifier = cv2.CascadeClassifier(r"C:\Users\pawar\data_science\Deep Learning\projects\openCV\Haarcascades\haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier(r"C:\Users\pawar\data_science\Deep Learning\projects\openCV\Haarcascades\haarcascade_eye.xml")

# Load the image
image = cv2.imread(r"C:\Users\pawar\OneDrive\Pictures\old_screenshot\Screenshot 2024-09-12 125652.png")

#image = cv2.imread(r'C:\Users\A3MAX SOFTWARE TECH\Desktop\WORK\2. DATASCIENCE PROJECT\10. Computer vision\Computer-Vision-Tutorial-master\Computer-Vision-Tutorial-master\image_examples\5.jpg')
# Check if the image is loaded correctly
if image is None:
    print("Error: Image not found or cannot be loaded!")
    exit()  # Exit if image is not loaded

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_classifier.detectMultiScale(gray, 1.3, 5)

# Check if faces are detected
if len(faces) == 0:
    print("No faces found!")
else:
    # Draw rectangles around the faces
    for (x, y, w, h) in faces:  # (x, y) is the top-left corner, and (w, h) is the width and height of the face
        cv2.rectangle(image, (x, y), (x + w, y + h), (127, 0, 255), 2)

        # Region of interest for eyes
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]

        # Detect eyes within the face region
        eyes = eye_classifier.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Display the output image
    cv2.imshow('Face Detection', image)
    cv2.waitKey(0)  # Wait for a key press to close the window

# Close all OpenCV windows
cv2.destroyAllWindows()