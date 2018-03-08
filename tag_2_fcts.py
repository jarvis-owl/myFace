# @Date:   11-10-2017
# @Last modified time: 11-10-2017
# @License: GNU Public License v3

'''
    contains functions for tag 2.0.py
'''

classes= [
    'ruben', #0
    'katja', #1
    'michi', #2
    'simon', #3
    'fink',  #4
    'hannes',#5
    'hopf',  #6
    'steini', #7
    'joe',#8
    'schelske'#9
]

def classify(classifier,img):
    """DOCSTRING"""

    
def label_fct(line):
    """ask user for pictured person"""
    y=np.zeros(len(classes))

    if line[1] < 100:
        factor = 100/line.shape[1]
    elif line[1] > 800:
        factor = 800/line.shape[1]
    shw = cv2.resize(line,None,fx=factor, fy=factor, interpolation = cv2.INTER_AREA)

    cv2.imshow(line,shw)
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
        #then take 96 by 96 off that picture
        #luckily the rois are squared already !
        res = cv2.resize(line,(96,96), interpolation = cv2.INTER_AREA)
        #cv2.imwrite('faces/{}.jpg'.format(tag),res)

        #one_hot encoding tag
        y[classes.index(tag)]=1
        return [res,y]
    else:
        print('no tag entered - discard face')
        return None
