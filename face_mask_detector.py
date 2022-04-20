from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import datetime

#Firstly we detect the face and then we will send ROI to Face mask detector model.

#importing our model
proto_txt_path = 'deploy.prototxt'
model_path = 'res10_300x300_ssd_iter_140000.caffemodel'
face_detector = cv2.dnn.readNetFromCaffe(proto_txt_path, model_path)

mask_detector = load_model('mask_detector.model')
#you can use it for both video and live camera or cctv
#for live camera or cctv
cap = cv2.VideoCapture(0)
#for video
#cap = cv2.VideoCapture('mask.mp4')
#you can use it for both video and live camera or cctv
#for live camera or cctv
while True:
    ret, frame = cap.read() #reading the frames
    frame = imutils.resize(frame, width=400) #resizing the frame according to our model
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123)) #creating a blob to send it to our face mask detector model

    face_detector.setInput(blob) #sending frames to our model
    detections = face_detector.forward()

    faces = [] #list of detected faces
    bbox = [] #list of bounding boxes
    results = [] #list of result of face mask detector

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2] #detecting the faces

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h]) 
            (startX, startY, endX, endY) = box.astype("int") #extracted the coordinates of the face

            face = frame[startY:endY, startX:endX] #extracted face ROI
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) #processing Face ROI
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            faces.append(face)
            bbox.append((startX, startY, endX, endY))
    #passing faces to our mask detector model
    if len(faces) > 0: //whether face is detected or not
        results = mask_detector.predict(faces)

    for (face_box, result) in zip(bbox, results):
        (startX, startY, endX, endY) = face_box
        (mask, withoutMask) = result #here mask and without mask is a floating point number, which ever is greater will be true.

        label = ""
        if mask > withoutMask:
            label = "Mask"
            color = (0, 255, 0) # bounding box in green color
        else:
            label = "No Mask"
            color = (0, 0, 255) # bounding box in red color

        cv2.putText(frame, label, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
