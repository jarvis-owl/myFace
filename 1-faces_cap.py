# @Date:   22-09-2017
# @Last modified time: 29-09-2017
# @License: GNU Public License v3




import numpy as np
import cv2
ROTATE = True

face_cascade = cv2.CascadeClassifier('haar-cascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haar-cascades/haarcascade_eye.xml')

cap = cv2.VideoCapture('video/ruben-Z2-frontcam.mp4')
#cap = cv2.VideoCapture('video/ruben-alpha6000.mp4')
while 1:
    ret, img = cap.read()

    #print(img.shape)
    if ROTATE :
        rows,cols,channel = img.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
        img = cv2.warpAffine(img,M,(cols,rows))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
    cv2.imshow('img',img)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

cap.release()
cv2.destroyAllWindows()
