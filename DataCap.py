import os
import cv2 as cv

# Initialize webcam
cap = cv.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

# Load Haar cascade for face detection
facedetect = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

count = 0

# Get user input for the ID
ID = input("Enter your Name: ").lower()
path = 'data/' + ID
Exists = os.path.exists(path)

if Exists:
    print("ID Present")
else:
    os.makedirs(path)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture image")
        break

    # Convert frame to grayscale for face detection
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        count += 1
        name = f'data/{ID}/{count}.jpg'
        print(f'Building Dataset... {name}')
        cv.imwrite(name, frame[y:y + h, x:x + w])
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow('Video', frame)
    k = cv.waitKey(1)

    if k == ord('q') or count > 500:  # Break if 'q' is pressed or if count exceeds 500
        break

# Release resources
cap.release()
cv.destroyAllWindows()
