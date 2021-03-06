# @Date:   04-10-2017
# @Last modified time: 07-10-2017
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
from data_handler import extract_data
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

file_training = 'data/training_data.npy'
file_hash = 'data/hash_list.npy'

t1 = time.time()

n_nodes_hl1 = 2304
n_nodes_hl2 = 100
n_nodes_hl3 = 500

n_classes = 8
batch_size = 100

x = tf.placeholder('float', [None, 9216])
y = tf.placeholder('float')

def neural_network_model(data):
    #hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
    hidden_1_layer = {'weights':tf.Variable(tf.truncated_normal([9216,n_nodes_hl1], stddev=0.05)),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl1, n_nodes_hl2], stddev=0.05)),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.truncated_normal([n_nodes_hl2, n_nodes_hl3], stddev=0.05)),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}


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

    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            #for _ in range(int(mnist.train.num_examples/batch_size)):
            for _ in range(int(13/batch_size)):
                #epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                epoch_x, epoch_y = extract_data(batch_size,epoch)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        #print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))               #!!!
        print('Accuracy:',accuracy.eval({x:x_test, y:y_test}))

train_neural_network(x)
print('{}min'.format((time.time()-t1)/60))
