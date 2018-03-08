# @Date:   02-10-2017
# @Last modified time: 07-10-2017
# @License: GNU Public License v3
# v3


'''
    crawls through dirs and checks jpgs on faces
    detected faces are asked to be tagged and saved
        -tag function added
        -dataset storing implemented
        ! +enclose face finder in function to use multithreading (semaphore for output) - possible? can it really be parallized?
            yes - every classifier needs to run in his unique thread, easy. but the similar face find algorithm needs to be adapted.    !!! nyd
        +reduced amount of classifiers
        +hot to encode classes? - what are unclassified faces? neg-examples?

        imshow could receive a resized image if x < 100: upscale elif x > 800: downscale
##################################################################
Notes:
    try:
        1/x
        print('[+] works: x != 0')
    except Exeption, e:
        print('[-] Error= '.format(str(e)))
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

file_training = 'data/training_data.npy'
file_hash = 'data/hash_list.npy'


classes= [
    'ruben', #0
    'katja', #1
    'michi', #2
    'simon', #3
    'fink',  #4
    'hannes',#5
    'hopf',  #6
    'steini' #7
    #'joe',#8
    #'schelske'#9
]

#load hash file of former scanned images
if os.path.isfile(file_hash):
    print('hash_list found - load ')
    hash_list = list(np.load(file_hash))
else:
    print('no hash_list found - starting fresh')
    hash_list = [] #runtime 'local'

#load training_data
if os.path.isfile(file_training):
    print('[+] {} found - load '.format(file_training))
    training_data = list(np.load(file_training))
else:
    print('[-] {} NOT found - exit'.format(file_training))
    training_data = [] #runtime 'local'


def tag_and_save(roi_gray):
    #ask user for pictured person
    empty=np.zeros(len(classes))
    cv2.imshow('roi_gray',roi_gray)
    print('enter 0-{}: '.format(len(classes)-1))

    nr = cv2.waitKey(delay=10000)
    cv2.destroyAllWindows()
    if nr & 0xFF == ord('q'):
        print('exit')
        sys.exit()
    elif 47 < nr <= 47+len(classes):
        print(nr-48)
        tag = classes[int(nr-48)]
        #choose w < h and scale the smaller one to 96
        #then tanke 96 by 96 off that picture
        #luckily the rois are squared !
        res = cv2.resize(roi_gray,(96,96), interpolation = cv2.INTER_AREA)
        cv2.imwrite('faces/{}.jpg'.format(tag),res)
        #one_hot encoding tag
        empty[classes.index(tag)]=1
        return [res,empty]
    else:
        print('no tag entered - discard face')
        return None


classifier = [
#    'haarcascade_frontalcatface_extended.xml',
    'haarcascade_frontalface_alt.xml',
#    'haarcascade_frontalface_alt_tree.xml',
#    'haarcascade_frontalface_alt2.xml',
    'haarcascade_frontalface_default.xml',
    'haarcascade_profileface.xml',
]

if len(sys.argv) is 2:
    path=sys.argv[1]
else:
    path='images/'

found_summed = np.array([0,0,0,0,0,0])
#found_summed = np.zeros([1,6])

uniq_faces = 0
img_sofar = 0
# === walk the dirs ===
for root, dirs, files in os.walk(path):
    #print(files)
    for name in files:

        if name.endswith('.jpg') or name.endswith('.JPG') or name.endswith('.png') or name.endswith('.PNG'):
        #if name.endswith('mp4'): read Frame from Video - tbd
# ====== IMAGE LOCAL =======
            print('====================\t{}'.format(name))
            try:
                img=cv2.imread(os.path.join(path,name))
            except Exception as e:
                print('[-] Error loading Image:'+str(e))
                sys.exit()
            #check if image was already seen
            try:
                my_hash = hashlib.md5(img).hexdigest()
            except Exception as e:
                print('[-] Error hashing: '+str(e))
                sys.exit()

            if not my_hash in hash_list:

                draw = cv2.bitwise_not(np.zeros((img.shape),np.uint8)) #np.255 #receives the rectangles around faces
                face_list = np.array([], dtype=np.int64).reshape(0,4) #np.empty([1,4],dtype=int) # contains faces from all classifiers of one image
                all_dist=[] #image local

# ====== CLASSIFIER LOCAL =======
                for counter,j in enumerate(classifier):
                    #find faces
                    face_cascade = cv2.CascadeClassifier(os.path.join('haar-cascades/',j))
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# ====== FACES =======
                    for (x,y,w,h) in faces:
                        print('\n')

                        #if distance is lower than  <-- tweak!
                        threshold = sqrt(w**2+h**2) * 0.2

                        #calc distance to every x,y in faces
                        for xf,yf,wf,hf in face_list:

                            all_dist.append(sqrt((x-xf)**2+(y-yf)**2))
                        '''new face found'''
                        #it's important to check not all_dist first, might be empty
                        if not all_dist or  min(all_dist) > threshold :
                            if SHOW:
                                cv2.rectangle(draw,(x,y),(x+w,y+h),(0,255,0),2)
                            uniq_faces+=1
                            roi_gray = gray[y:y+h, x:x+w]

                            output = tag_and_save(roi_gray)
                            if output:
                                training_data.append(output)

                        elif min(all_dist) <= threshold: #mutiple detected face
                            #print('{} found equal face'.format(j))
                            if SHOW:
                                cv2.circle(draw,(x,y),int(threshold),(0,0,255),2)

                        else:
                            print('problem with similar face detection. line ~150+')
                            sys.exit()

                        found_summed[counter]+=1

                        #add actual face to list of all faces (also doubles)
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

                #save training data and img-hash after every image
                np.save(file_training,training_data)
                hash_list.append(my_hash)
                np.save(file_hash,hash_list)

                img_sofar+=1#this could be useful to decrease np.save activity if img_sofar%50 = 0: np.save()

            else:
                print('found similar image {}'.format(name))
            print('images so far: {}'.format(img_sofar))


print('\n')
print('classifier quality:')
print(found_summed)
print('\n')
print('found {} unique faces in '.format(uniq_faces))
print('{} unique images scanned'.format(len(hash_list)))
print('\n')
