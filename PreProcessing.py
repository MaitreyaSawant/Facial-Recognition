import os
import cv2 as cv
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout
import joblib


def load_images(dir):
    images=[]
    labels=[]
    for person in os.listdir(dir):
        per_fol=os.path.join(dir,person)
        if os.path.isdir(per_fol):
            for filename in os.listdir(per_fol):
                img_path=os.path.join(per_fol,filename)
                img=cv.imread(img_path)
                if img is not None:
                    img=cv.resize(img,(128,128))
                    images.append(img)
                    labels.append(person)
    return np.array(images),np.array(labels)

images,labels=load_images('C:/Users/Admin/PycharmProjects/FacialRecognition/data')
images=images/255.0

label_encoder=LabelEncoder()
encoded=label_encoder.fit_transform(labels)
categorical=to_categorical(encoded)

x_train,x_test,y_train,y_test=train_test_split(images,categorical,test_size=0.2,random_state=42,stratify=categorical)

model=Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)),
    MaxPooling2D((2,2)),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(128,(3,3),activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128,activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_),activation='softmax')
])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
print(model.summary())

model.fit(x_train,y_train,epochs=20,validation_data=(x_test,y_test))
model.save('FacialRecognition.h5')
joblib.dump(label_encoder,'label_encoder.pkl')