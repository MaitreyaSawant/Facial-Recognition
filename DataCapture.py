import os
import cv2 as cv


cap = cv.VideoCapture(0)


if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()


facedetect = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

count = 0


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

    if k == ord('q') or count > 500:
        break

# Release resources
cap.release()
cv.destroyAllWindows()
