# @Date:   09-10-2017
# @Last modified time: 12-10-2017
# @License: GNU Public License v3

'''
	improved tagging/labeling function
	walks dirs and checks images for faces
	faces can be added manually
	multithreading automated finding

	returnvalues for label_fct might be added -> evaluate faces found and faces added
'''

import numpy as np
import cv2
import os
import sys
import hashlib
from math import sqrt
import time
from threading import *
import tag_2_fcts

''' ========================================== init ============================================= '''

file_training = 'data/training_data.npy'
file_hash = 'data/hash_list.npy'


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

#i still use three classifiers, because i did not yet tested, if they find different faces
classifier = [
    'haarcascade_frontalface_alt.xml',
    'haarcascade_frontalface_default.xml',
    'haarcascade_profileface.xml',
]

#if other path is required
if len(sys.argv) is 2:
    path=sys.argv[1]
else:
    path='images/'


''' ========================================== main ============================================= '''

#walk dirs
for root, dirs, files in os.walk(path):
    for name in files:
		if name.endswith('.jpg') or name.endswith('.JPG') or name.endswith('.png') or name.endswith('.PNG'):

			#load img
			print('\t\t\t{}'.format(name))
            try:
                img=cv2.imread(os.path.join(path,name))
            except Exception as e:
                print('[-] Error loading Image:'+str(e))
				#break
                sys.exit()

            #check if image was already seen
            try:
                actual_hash = hashlib.md5(img).hexdigest() #md5 is unsave, but sufficient for this purpose
            except Exception as e:
                print('[-] Error hashing: '+str(e))
                sys.exit()
				#break

			if not actual_hash in hash_list:

				#open threads (img,classifier) - return value = 3x faces_list
				for i in classifier:
					t = Thread(target=classify,args=(classifier[i],img)) 								#!!!! return values ?
					t.start()

				#discard redundant detections - min(point_dists[])

				#new final_faces (list)

				for line in final_faces:
					try:
						face, label = label_fct(line) #might return None
					if face and label:
						dataset.append([face,label])
						#draw rect(line) into img

				#show image with detected faces

				#mark further faces manually

				#label_fct(further_face) and append to dataset

			np.save(training_dataset.npy,dataset)
			hash_list.append(actual_hash)
			np.save(hashfile.npy,hash_list)
