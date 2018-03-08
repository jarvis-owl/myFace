# @Date:   02-10-2017
# @Last modified time: 02-10-2017
# @License: GNU Public License v3



'''
    test script to test
    lbpcascade_profileface.xml <- img also must be flipped!
    and
    haarcascade_profileface.xml
'''

import cv2
import sys


imagePath = '2013-07-02 19.06.19.jpg'

face_cascade = cv2.CascadeClassifier('../haar-cascades/lbpcascade_profileface.xml')
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# faces = faceCascade.detectMultiScale(
#     gray,
#     scaleFactor=1.1,
#     minNeighbors=5,
#     minSize=(30, 30),
#     flags = cv2.CV_HAAR_SCALE_IMAGE
# )
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow('completed',image)
cv2.waitKey()
destroyAllWindows()
