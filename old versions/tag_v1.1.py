# @Date:   29-09-2017
# @Last modified time: 02-10-2017
# @License: GNU Public License v3
# v1
#
#checked 12GB Images:
#1347 faces in 1531 images found
#in 5206s = 103min
#classifier quality: [32 685 77 769 1488 213]
#
#checked 700MB Images:
#185 faces in 218 images found
#in 744s = 12.4m
#classifier quality: [3 96 7 84 178 33]
#
#classifier[0] and [2] work too badly
#it's need to be determined, if classifier[1] and [4] detect different faces
#classifier[5] is for profilefaces


'''
    crawls through dirs and checks jpgs on faces
    detected faces are asked to be tagged and saved
'''

import numpy as np
import cv2
import os
import sys
import hashlib
from math import sqrt
import time
#from msvcrt import getch

ROTATE = False
NR = 5
SHOW = False




classifier = [
    'haarcascade_frontalcatface_extended.xml',
    'haarcascade_frontalface_alt.xml',
    'haarcascade_frontalface_alt_tree.xml',
    'haarcascade_frontalface_alt2.xml',
    'haarcascade_frontalface_default.xml',
    'haarcascade_profileface.xml',
]

found_summed = np.array([0,0,0,0,0,0])
#found_summed = np.zeros([1,6])
hash_list = [] #runtime 'local'
uniq_faces = 0

t1 = time.time()

if len(sys.argv) is 2:
    path=sys.argv[1]
else:
    path='images'

# === walk the dirs ===
for root, dirs, files in os.walk(path):
    #print(files)
    for name in files:

        # if name is 'choir.png': #never becomes true :/
        #      print('found the choir')
        #      sys.exit()
        if name.endswith('.jpg') or name.endswith('.JPG') or name.endswith('.png') or name.endswith('.PNG'):

# ====== IMAGE LOCAL =======
            print('================={}'.format(name))
            img=cv2.imread(os.path.join(path,name))
            print(img.shape)
            #check if image was already seen
            my_hash = hashlib.md5(img).hexdigest()
            if not my_hash in hash_list:
                # if my_hash in hash_list:
                #     print('found similar image {}'.format(name))
                #     break
                hash_list.append(my_hash)

                draw = cv2.bitwise_not(np.zeros((img.shape),np.uint8)) #np.255 #receives the rectangles around faces
                face_list = np.array([], dtype=np.int64).reshape(0,4) #np.empty([1,4],dtype=int) # contains faces from all classifiers of one image
                all_dist=[] #image local

# ====== CLASSIFIER LOCAL =======
                for counter,j in enumerate(classifier):
                    face_cascade = cv2.CascadeClassifier(os.path.join('haar-cascades/',j))
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    #find faces
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# ====== FACES =======
                    for (x,y,w,h) in faces:
                        #print('faces: ')
                        #print(faces)
                        print('\n')

                        #if distance is lower than 0.2*sqrt(w**2+h**2) <-- tweak!
                        threshold = sqrt(w**2+h**2) * 0.2
                        #print('threshold: {}'.format(threshold))

                        #calc distance to every x,y in faces
                        for xf,yf,wf,hf in face_list:
                            #print('read face_list: {} {} {} {}'.format(xf,yf,wf,hf))
                            all_dist.append(sqrt((x-xf)**2+(y-yf)**2))
                            #print('len(all_dist): {} \n all_dist:'.format(len(all_dist)))
                            #print(all_dist)

                        #draw rectangle only, when:
                        #   all_dist = 0 or
                        #   dist > threshold
                    #    if not all_dist: #when all_dist is empty (first image load and first found face)
                    #        print('all_dist is empty')
                    #        cv2.rectangle(draw,(x,y),(x+w,y+h),(0,255,0),2)
                            #roi_gray = gray[y:y+h, x:x+w]
                            #roi_color = img[y:y+h, x:x+w]

                    #    elif min(all_dist) > threshold: #when detected face is outer formerly know face
                    #        cv2.rectangle(draw,(x,y),(x+w,y+h),(0,255,0),2)

                        #it's important to check not all_dist first, might be empty
                        if not all_dist or  min(all_dist) > threshold :
                            if SHOW:
                                cv2.rectangle(draw,(x,y),(x+w,y+h),(0,255,0),2)
                            uniq_faces+=1
                            roi_color = img[y:y+h, x:x+w]
                        elif min(all_dist) <= threshold: #mutiple detected face
                            print('found equal face')
                            if SHOW:
                                cv2.circle(draw,(x,y),int(threshold),(0,0,255),2)

                        else:
                            print('problem with similar face detection. line ~110+')
                            sys.exit()

                        found_summed[counter]+=1

                        #print('face at: {} {} {} {}'.format(x,y,w,h))
                        face_list = np.vstack([face_list,[x,y,w,h]])

                        #shw('face with original',np.concatenate((img,roi_color),axis=1)) #axis is horizontal/vertival argument 0/1
                        #show
                        if SHOW:
                            merge = cv2.bitwise_and(draw,img)
                            if merge.shape[0]  < merge.shape[1]:
                                factor = 1200/merge.shape[1] #always 1200px wide
                            else:
                                factor = 800/merge.shape[0]
                            res = cv2.resize(merge,None,fx=factor, fy=factor, interpolation = cv2.INTER_AREA)
                            cv2.imshow(j,res)
                            if cv2.waitKey(delay=10000) & 0xFF == ord('q'):
                                cv2.destroyAllWindows()
                                sys.exit()
                            cv2.destroyAllWindows()

                    print('face_list.shape: ')
                    print(face_list.shape)


                    print('{} performed'.format(j))
                    print('\n')
                print(face_list)
            else:
                print('found similar image {}'.format(name))


print('\n')
print('classifier quality:')
print(found_summed)
print('\n')
print('found {} unique faces in '.format(uniq_faces))
print('{} unique images scanned'.format(len(hash_list)))
print('\n')
print('duration: {}'.format(time.time()-t1))
print('\n')
