#Source: https://www.tensorflow.org/tutorials/layers
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm

import os
import sys
import timeit
import curses


import tensorflow as tf
from matplotlib import pyplot as plt
import cv2
from pathlib import Path
import numpy as np
import tensorflow as tf



class simple_FCNN:



    def __init__(self):

        # parameters, weights and biases
        self.params = {}

        # where to save model
        self.path = "res/model/FCNN_demo"


    def design_model(self):
        # input
        self.X = tf.placeholder(tf.float32, shape = [None, 784])
        # lable
        self.Y_ = tf.placeholder(tf.float32, shape = [None, 10])


        # parameters
        self.weights = {
            'w1' : tf.Variable(tf.truncated_normal([784, 128])),
            'w2' : tf.Variable(tf.truncated_normal([128, 32])),
            'w3' : tf.Variable(tf.truncated_normal([32, 10]))
        }

        # parameters
        self.biases = {
            'b1' : tf.Variable(tf.zeros([128])),
            'b2' : tf.Variable(tf.zeros([32])),
            'b3' : tf.Variable(tf.zeros([10]))
        }

        # hidden layers
        self.h1 = tf.nn.leaky_relu(tf.matmul(self.X, self.weights['w1'])
                                    + self.biases['b1'])
        self.h2 = tf.nn.leaky_relu(tf.matmul(self.h1, self.weights['w2'])
                                    + self.biases['b2'])

        #output layer
        self.Y = tf.matmul(self.h2, self.weights['w3']) + self.biases['b3']

        # graph model
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.Y, labels=self.Y_))
        self.train_step = tf.train.GradientDescentOptimizer(0.02).minimize(self.loss_op)
        self.sess = tf.Session()

        return

    def train(self):

        # initilaze variables
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        # data preprocessing
        mnist = input_data.read_data_sets("res/datasets/MNIST/", one_hot=True)

        # train model
        self.epoches = 1000
        for _ in range(self.epoches):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            self.sess.run(self.train_step, feed_dict={self.X: batch_xs, self.Y_: batch_ys})
        correct_prediction = tf.equal(tf.argmax(self.Y, 1), tf.argmax(self.Y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # tensorBoard
        writer = tf.summary.FileWriter('/Users/Sadegh/Documents/UIO/unik4690/4690-p2018/src/res/my_graph', self.sess.graph)

        # Accuracity print
        print("Accuracity: "+str(self.sess.run(accuracy,
                                            feed_dict={self.X: mnist.test.images, self.Y_: mnist.test.labels})))

        return

    def forward(self, data):
        return self.sess.run(self.Y, feed_dict={self.X: data})


def file_rendering(path):
    img = cv2.imread(path);
    img_small = cv2.resize(img, (28,28))
    img_grey = cv2.cvtColor(255-img_small, cv2.COLOR_RGB2GRAY)

    data = np.reshape(img_grey, (1, -1))
    data = (1/255)*data

    return data

if __name__ == "__main__":
    np.set_printoptions(linewidth=9999999)

    nn = simple_FCNN()

    nn.design_model()
    nn.train()


    print("\n\n\n")
    dirname = 'res/images/'
    for filename in os.listdir(dirname):
        if len(filename) is not 5:
            continue

        path = dirname+filename
        print(path)

        data = file_rendering(path)
        r = nn.forward(data)

        print('r: ', r)
        index = np.argmax(r)
        print("%d : %.2f%%" % (index, np.amax(r)))
