
# @Date:   04-10-2017
# @Last modified time: 10-10-2017
# @License: GNU Public License v3
#
# v1
#code from : https://pythonprogramming.net/cnn-tensorflow-convolutional-nerual-network-machine-learning-tutorial/
'''
    the mnist example shall be adapted to my own dataset provided by tag_vX.py
    it  | accuracy [%]
    10  | 94.9
    20  | 95.9
    100 | 97.4
    initialized weights with truncated_normal and got +1%

    this version is to be adapted to 96x96 face images
    96*96=9216
    9216/4 = 2304
'''

import time
import tensorflow as tf
import numpy as np
import sys
from data_handler_v3 import extract_data
import matplotlib.pyplot as plt
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

t1 = time.time()

x_train,y_train,x_test,y_test = extract_data(test_size=0.1)

print('len training_data: {}'.format(len(x_train)))
print('len testing_data: {}'.format(len(x_test)))

print('image size: {}'.format(len(x_train[0])))



n_nodes_hl1 = int(len(x_train[0])/2)
n_nodes_hl2 = int(len(x_train[0])/4)
n_nodes_hl3 = int(len(x_train[0])/8)

n_nodes_hl1 = 2304#2304
n_nodes_hl2 = 1500
n_nodes_hl3 = 750

n_classes = len(y_train[0])

hm_epochs = 100 #100 and batch_size = 100 isn't that bad - even time efficient
batch_size = 50 #consider 10/50/21 ? due to small dataset

losses = [sys.maxsize]

x = tf.placeholder('float', [None, len(x_train[0])])
#x = tf.placeholder('float', [96, 96])
y = tf.placeholder('float')

def neural_network_model(data):
    #hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
    hidden_1_layer = {'weights':tf.Variable(tf.truncated_normal([int(len(x_train[0])),n_nodes_hl1], stddev=0.05)),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl1, n_nodes_hl2], stddev=0.05)),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl2, n_nodes_hl3], stddev=0.05)),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}


    #l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output



def train_neural_network(x):
    prediction = neural_network_model(x)

    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )
    #learning rate = 0.001
    optimizer = tf.train.AdamOptimizer(  learning_rate=0.001,
                                         beta1=0.9,
                                         beta2=0.999,
                                         epsilon=1e-08,
                                         use_locking=False,
                                         name='Adam').minimize(cost)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0

            i = 0
            while i < len(x_train):
                start = i
                end = i+batch_size


                batch_x = np.array(x_train[start:end])
                batch_y = np.array(y_train[start:end])


                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i+=batch_size

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

            #check last 2 losses
            #might be too specific
            if losses[-1]*0.99 < epoch_loss < losses[-1]*1.01 and losses[-2]*0.98 < losses[-1] < losses[-2]*1.05:
                print('no further loss benefits')
                losses.append(epoch_loss)
                break
            elif losses[-1]*1.75 < epoch_loss:
                #print(epoch_loss)
                print('loss increased dramatically')
                losses.append(epoch_loss)
                #break
            elif epoch_loss < 15 and losses[-1] < 20:
                print('losses[-2:] < 20 and < 15')
                losses.append(epoch_loss)
                break
            else:
                losses.append(epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:x_test, y:y_test}))

train_neural_network(x)

t2 = (time.time()-t1)/60
minutes=int(t2)
sec=int((t2%1)*60)
print('{}:{} min'.format(minutes,sec))

print('min loss: {} '.format(np.min(losses)))
print('end loss: {}'.format(losses[-1]))

plt.subplot(121)
#plt.semilogy(range(0,int(len(losses)*0.4)),losses[int(-len(losses)*0.4):])
#plt.loglog(range(0,int(len(losses)*0.4)),losses[int(-len(losses)*0.4):],basex=10)
plt.loglog(range(0,int(len(losses)-1)),np.flip(losses[1:],axis=0),basex=10) #skip first value sys.intsize
#reversed(losses)
plt.title('loss log - flipped')
plt.grid()

plt.subplot(122)
plt.plot(range(0,int(len(losses)*0.4)),losses[int(-len(losses)*0.4):])
plt.title('loss last 40%')
plt.grid()


plt.draw()
plt.savefig('data/last_run.png',transparent=True,bbox_inches='tight',dpi=600)

plt.show()
