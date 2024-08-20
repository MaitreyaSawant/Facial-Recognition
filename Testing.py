import numpy as np
import cv2 as cv
import cv2.data
from tensorflow.keras.models import load_model
import joblib


model=load_model('FacialRecognition.h5')
label_encoder=joblib.load('label_encoder.pkl')

cap= cv.VideoCapture(0)
facedet=cv.CascadeClassifier(cv.data.haarcascades+'haarcascade_frontalface_default.xml')

while True:
    ret,frame=cap.read()
    if not ret:
        break

    gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces=facedet.detectMultiScale(frame,1.3,5)

    for(x,y,w,h) in faces:
        face=frame[y:y+h,x:x+w]
        face_resized=cv.resize(face,(128,128))
        face_norm=face_resized/255.0
        face_reshape=np.reshape(face_norm,(1,128,128,3))

        pred=model.predict(face_reshape)
        class_id=np.argmax(pred,axis=1)[0]
        class_name=label_encoder.inverse_transform([class_id])[0]

        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv.putText(frame,class_name,(x,y-10),cv.FONT_HERSHEY_SIMPLEX,0.9,(36,255,12),2)

    cv.imshow('Video',frame)

    if cv.waitKey(1)& 0xFF==ord('q'):
        break

cap.release()
cv.destroyAllWindows()