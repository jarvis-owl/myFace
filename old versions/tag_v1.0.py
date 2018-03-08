# @Date:   22-09-2017
# @Last modified time: 02-10-2017
# @License: GNU Public License v3

import numpy as np
import cv2
import os

ROTATE = False

face_cascade = cv2.CascadeClassifier('haar-cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haar-cascades/haarcascade_eye.xml')
#eye_cascade = cv2.CascadeClassifier('../haar-cascades/haarcascade_eye_tree_eyeglasses.xml')


# === walk the dirs ===
for root, dirs, files in os.walk('.'):
    for name in files:
        #r.append(os.path.join(root,name))

        if name.endswith('.jpg') or name.endswith('.JPG'):
            print(name)
            img=cv2.imread(os.path.join(root,name))


            if ROTATE :
                rows,cols,channel = img.shape
                #print(img.shape)
                M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
                img = cv2.warpAffine(img,M,(cols,rows))

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # cv2.imshow('gray',gray)
            # print('press q to quit')
            # k = cv2.waitKey(delay=10000)
            # cv2.destroyAllWindows()

            #find faces
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]

            #find eyes in faces
                eyes = eye_cascade.detectMultiScale(roi_gray)
                print(eyes)
                i = 0
                for (ex,ey,ew,eh) in eyes:
                        if i < 2:
                            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                            i+=1


    #show
            res = cv2.resize(img,None,fx=0.3, fy=0.3, interpolation = cv2.INTER_AREA)
            cv2.imshow(name,res)
            if cv2.waitKey() & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

            cv2.destroyAllWindows()
