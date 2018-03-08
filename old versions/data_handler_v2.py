# @Date:   07-10-2017
# @Last modified time: 09-10-2017
# @License: GNU Public License v3
'''
    load dataset
    normaliz/balance classes occurence
    split into test [x,y] and train [x,y]
'''


import os
import numpy as np
from random import shuffle
import pandas as pd
from collections import Counter
import tensorflow as tf


file_training = 'data/training_data.npy'


def extract_data(test_size=0.1):
    #grab training_data and return 96x96px grayscale img [SHAPE?] as x
    #and label as one_hot encoded y

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

    classes_count = np.zeros(len(training_data[0][1]))

    #determine least occurence of a class
    for data in training_data:
        classes_count[np.argmax(data[1])]+=1
    #print(classes_count)
    #print(classes_count[int(np.argmin(classes_count))])

    #balancing

    #this is shit!: I guess pandas should be considered
    ruben=[]
    katja=[]
    michi=[]
    simon=[]
    fink=[]
    hannes=[]
    hopf=[]
    steini=[]

    #could not figure out, how to use a dict or something more variable

    for data in training_data:

        img=data[0]
        img = img.reshape(96*96)    #this was hard to find
        choice=data[1]
        if np.argmax(choice) == 0:
            ruben.append([img,choice])
        elif np.argmax(choice) == 1:
            katja.append([img,choice])
        elif np.argmax(choice) == 2:
            michi.append([img,choice])
        elif np.argmax(choice) == 3:
            simon.append([img,choice])
        elif np.argmax(choice) == 4:
            fink.append([img,choice])
        elif np.argmax(choice) == 5:
            hannes.append([img,choice])
        elif np.argmax(choice) == 6:
            hopf.append([img,choice])
        elif np.argmax(choice) == 7:
            steini.append([img,choice])


    #trim

    ruben=ruben[:int(classes_count[int(np.argmin(classes_count))])]
    katja=katja[:int(classes_count[int(np.argmin(classes_count))])]
    michi=michi[:int(classes_count[int(np.argmin(classes_count))])]
    simon=simon[:int(classes_count[int(np.argmin(classes_count))])]
    fink=fink[:int(classes_count[int(np.argmin(classes_count))])]
    hannes=hannes[:int(classes_count[int(np.argmin(classes_count))])]
    hopf=hopf[:int(classes_count[int(np.argmin(classes_count))])]
    steini=steini[:int(classes_count[int(np.argmin(classes_count))])]


    output_data = ruben+katja+michi+simon+fink+hannes+hopf+steini
    shuffle(output_data)

    output_data  = np.array(output_data)
    testing_size = int(test_size*len(output_data))
    x_train = list(output_data[:,0][:-testing_size])
    y_train = list(output_data[:,1][:-testing_size])
    x_test = list(output_data[:,0][-testing_size:])
    y_test = list(output_data[:,1][-testing_size:])


    return x_train,y_train,x_test,y_test

if __name__ == '__main__' :
    X_Train, Y_Train, X_Test, Y_Test = extract_data()
