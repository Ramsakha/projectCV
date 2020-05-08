# -*- coding: utf-8 -*-
"""
Created on Fri May  1 23:38:58 2020

@author: Ram
"""

import cv2
import os,sys
import numpy as np

face_casade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
root = 'images'
model = cv2.face.EigenFaceRecognizer_create()
def create_folder():
    name = sys.argv[1]
    #name = input()
    path = os.path.join("images",name)
    ret = 0
    try:
        os.mkdir(path)
        ret = 1
    except:
        print("folder already exists")
    return path,ret,name

def save_show(path,name):
    vid = cv2.VideoCapture(0)
    ret,frame = vid.read()
    count = 0
    while True:
        ret,frame = vid.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_casade.detectMultiScale(gray,1.3,5)
        
        if len(faces)>0:
            areas = []
            for x,y,w,h in faces:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
                areas.append(w*h)
                #print(w*h)
                #print("\n")
        #max_area, idx = max([(val,idx) for idx,val in enumerate(areas)])
            idx = areas.index(max(areas))
            face_sv = faces[idx]
            x = face_sv[0]
            y = face_sv[1]
            w = face_sv[2]
            h = face_sv[3]
            face = gray[y:y+h,x:x+w]
            face_res = cv2.resize(face,(112,92))
            
            #print(count)
            #print("\n")
            if count<100:
                cv2.imwrite('%s/%d.png' % (path, count), face_res)
                print(count)
                print("\n")
                cv2.waitKey(500)
                count=count+1
            else:
                print("done******************press enter to exit")
            
        #print(w*h)
        #print("\n")
        cv2.imshow("hii "+name+'!',frame)
        if cv2.waitKey(1)==13:
            vid.release()
            cv2.destroyAllWindows()
            break
    return

def train_images(root):
    imgs = []
    tags = []
    index = 0
    for (subdirs, dirs, files) in os.walk(root):
        for subdir in dirs:
            img_path = os.path.join(root, subdir)
            for fn in os.listdir(img_path):
                path = img_path + '/' + fn
                tag = index
                imgs.append(cv2.imread(path, 0))
                tags.append(int(tag))
            index += 1
    (imgs, tags) = [np.array(item) for item in [imgs, tags]]
    return (imgs,tags)



path,ret,name = create_folder()
if ret!=0:
    save_show(path,name)
    imags,tags = train_images(root)
    model.train(imags,tags)
    model.save('eigen_feature_data.xml')
    print('training completed successfully')
    

    

    
    
    
    
    
    
    
    
    