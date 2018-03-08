# @Date:   07-10-2017
# @Last modified time: 08-10-2017
# @License: GNU Public License v3
'''
    load dataset
    split into test [x,y] and train [x,y]
    normalize classes occurence
'''


import os
import numpy as np
from random import shuffle
import pandas as pd
from collections import Counter


file_training = 'data/training_data.npy'


def extract_data():
    #grab training_data and return 96x96px grayscale img [SHAPE?] as x
    #and label as one_hot encoded y

    #specify TEST img and labels                                                            !!!

    #load training_data
    if os.path.isfile(file_training):
        print('[+] {} found - load '.format(file_training))
        training_data = list(np.load(file_training))
    else:
        print('[-] {} NOT found - exit'.format(file_training))
        sys.exit()

    x_train = []
    y_train = []

    x_test = []
    y_test = []

    classes_count = np.zeros(8) #len(classes)
    cls_dict = {
                0:[np.zeros(9216),np.zeros(10)],#ruben
                1:[np.zeros(9216),np.zeros(10)],#katja
                2:[np.zeros(9216),np.zeros(10)],#michi
                3:[np.zeros(9216),np.zeros(10)],#simon
                4:[np.zeros(9216),np.zeros(10)],#fink
                5:[np.zeros(9216),np.zeros(10)],#hannes
                6:[np.zeros(9216),np.zeros(10)],#hopf
                7:[np.zeros(9216),np.zeros(10)],#steini
                }

    #tie up image and choice
    for data in training_data: #[epoch*batchsize:epoch*batchsize+batchsize]:
                               #get only a window from training_data, regarding memory size - wasted... np.load(complete_dataset)
        choice = data[1]
        img = data[0]
        print(choice)

        if i > len(training_data)*0.90:
            x_test.append(img)
            y_test.append(choice)
        else:
            x_train.append(img)
            y_train.append(choice)
        i+=1
        classes_count[np.argmax(data[1])]+=1
        try:
            cls_dict[np.argmax(choice)]=[img,choice]
        except Exception as e:
            print('[-] no matches'+str(e))

    #trim to min class len - balancing
    for i in range(len(cls_dict)):
        cls_dict[i]=cls_dicti[i][:len(np.argmin(classes_count))]                                    #!!!

    print(classes_count)
    print(np.argmin(classes_count))
    #print(len(training_data[np.argmin(classes_count)]) )
    #balance data

    return x_train,y_train,x_test,y_test

X_Train, Y_Train, X_Test, Y_Test = extract_data()
print('\n')


print('len training_data {}'.format(len(X_Train)))
print('len test_data {}'.format(len(X_Test)))

'''
input('say something:')
training_data = np.load(file_training)
df = pd.DataFrame(training_data)
print(df.head(7))
print('\n')
'''
#print(Counter(df[1].apply(str)))

#shuffle(training_data)


#for i < len(training_data(np.argmin(classes)))
