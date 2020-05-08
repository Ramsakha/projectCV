# -*- coding: utf-8 -*-
"""
Created on Mon May  4 20:53:51 2020

@author: Ram
"""

import cv2
import os,sys
import numpy as np

face_casade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = cv2.face.EigenFaceRecognizer_create()
model.read('eigen_feature_data.xml')
face_dir = 'images'
def recog():
    names = {}
    key = 0
    for (subdirs, dirs, files) in os.walk(face_dir):
        for subdir in dirs:
            names[key] = subdir
            key += 1
    cap = cv2.VideoCapture(0)
    ret,frame = cap.read()
    while True:
        ret,frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_casade.detectMultiScale(gray,1.3,5)
        if len(faces)>0:
            for x,y,w,h in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
                gray_f = gray[y:y+h,x:x+w]
                gray_f = cv2.resize(gray_f,(112,92))
                confi = model.predict(gray_f)
                if confi[1]<3500:
                    person = names[confi[0]]
                    cv2.putText(frame, '%s - %.0f' % (person, confi[1]), (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
                else:
                    person = 'Unknown'
                    cv2.putText(frame, person, (x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
        cv2.imshow("recognized faces",frame)
        if cv2.waitKey(1)==13:
            cap.release()
            cv2.destroyAllWindows()
            break
    return


print("press enter to quit")
recog()

                