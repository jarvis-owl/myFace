# @Date:   22-09-2017
# @Last modified time: 02-10-2017
# @License: GNU Public License v3

'''
    receives an image via commandline argv and draws rectangles around found faces
    uses multiple classifiers
'''

import numpy as np
import cv2
import os
import sys

ROTATE = False
NR = 5



classifier = [
#    'haarcascade_frontalcatface_extended.xml',
#    'haarcascade_frontalface_alt.xml',
#    'haarcascade_frontalface_alt_tree.xml',
#    'haarcascade_frontalface_alt2.xml',
    'haarcascade_frontalface_default.xml',
    'haarcascade_profileface.xml',
#    'haarcascade_eye_tree_eyeglasses.xml'
]

#get args
if len(sys.argv) is 2:
    img=cv2.imread(sys.argv[1])
else:
    print('no image given')
    sys.exit()


if ROTATE :
    rows,cols,channel = img.shape
    print(img.shape)
    M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
    img = cv2.warpAffine(img,M,(cols,rows))

face_dict = {}
dst = cv2.bitwise_not(np.zeros((img.shape),np.uint8)) #np.255

for i,j in enumerate(classifier):
    face_cascade = cv2.CascadeClassifier(os.path.join('haar-cascades/',j))
    print(j)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #find faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(dst,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(dst,str(i),(x+w+i*3,y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2, cv2.LINE_AA)
       #cv2.putText(dst,'i,(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)


        #roi_gray = gray[y:y+h, x:x+w]
        #roi_color = img[y:y+h, x:x+w]
    print(len(faces))

    face_dict.update({j:faces})

    #shw('histogram with original',np.concatenate((data,res),axis=1)) #axis is horizontal/vertival argument 0/1
    #show
    merge = cv2.bitwise_and(dst,img)
    factor = 1200/merge.shape[1] #always 1200px wide
    res = cv2.resize(merge,None,fx=factor, fy=factor, interpolation = cv2.INTER_AREA)
    cv2.imshow(classifier[i],res)
    if cv2.waitKey(delay=5000) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break


    cv2.destroyAllWindows()
