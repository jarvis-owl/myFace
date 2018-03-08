# @Date:   30-09-2017
# @Last modified time: 30-09-2017
# @License: GNU Public License v3

import hashlib
import cv2
import os
#
# h = hashlib.new('md5')
# h.update( b"password")
# print(h.hexdigest())

# hash_dict = []
# hash_dict.update({})

# hash_list = []
# for i in range(0,10):
#     h = hashlib.new('md5')
#     h.update(str(i))
#     hash_list.append(h.hexdigest())
# print(hash_list)
hash_list = []
for root, dirs, files in os.walk('.'):
    for name in files:
        if name.endswith('.jpg') or name.endswith('.JPG') or name.endswith('.png') or name.endswith('.PNG'):


            img = cv2.imread(name)
            print(name)
#            print(hashlib.md5(img).hexdigest())
            hash_list.append(hashlib.md5(img).hexdigest())


print('249ac5f7650e6da54f9522bf64cfc9fd' in hash_list)
