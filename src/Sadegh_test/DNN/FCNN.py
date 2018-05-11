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


class FCNN:
    # input: layers = list with number of nodes for each layer
    def __init__(self, layers):
        # how many nodes for each layer, inc input, hidden and output layer
        self.layer_dimensions = layers
        self.tot_layers = len(layers)

        # weights(w) and biases(b), strings lowercase + nr layer(starting at 0)
        # eg. 'w0', 'b0'
        self.params = {}

        # all layers(nodes) including input, hidden and output layer
        self.layers = {}

        # model path
        self.path = "res/model/FCNN_demo"


    # initilazes all parameters (w and b to zero) in a dict 'params'
    def init_params(self):
        for idx, a in enumerate(self.layer_dimensions):
            if idx != self.tot_layers - 1:
                self.params['w{0}'.format(idx + 1)] = tf.Variable(tf.random_normal([
                                                a,
                                                self.layer_dimensions[idx + 1]]))

                self.params['b{0}'.format(idx + 1)] = tf.Variable(tf.random_normal([
                                                self.layer_dimensions[idx + 1]]))
        return

    # initilazes all layers (input, hidden, output) in a dict 'layers'
    def init_layers(self):
        for idx, a in enumerate(self.layer_dimensions):
            if idx == 0: # input layer, x
                self.layers['x'] = tf.placeholder(
                                            tf.float32,
                                            [None, a])
            elif idx != self.tot_layers - 1: # hidden layers
                if idx == 1:
                    layer_before = self.layers['x']
                else:
                    layer_before = self.layers['l{0}'.format(idx - 1)]
                w_before = self.params['w{0}'.format(idx)]
                b_before = self.params['b{0}'.format(idx)]

                self.layers['l{0}'.format(idx)] = tf.nn.relu(
                                            tf.matmul(layer_before,
                                            w_before)
                                            + b_before)

            else: # output layer
                layer_before = self.layers['l{0}'.format(idx - 1)]
                w_before = self.params['w{0}'.format(idx)]
                b_before = self.params['b{0}'.format(idx)]

                self.layers['y'] = tf.nn.softmax(
                                            tf.matmul(layer_before,
                                            w_before)
                                            + b_before)
        return

    # initilazes the graph with the connections and variables
    def init_graph(self):
        self.init_params()
        self.init_layers()

        print('params')
        for key, value in self.params.items():
            print(key, ': ', value)

        print('layers')
        for key, value in self.layers.items():
            print(key, ': ', value)

        # predicted output
        self.y_ = tf.placeholder(tf.float32, shape = [None, 10])

        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.layers['y']), reduction_indices=[1]))
        self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(self.cross_entropy)
        self.sess = tf.Session()

        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)
        return


    def train(self):

        # data preprocessing
        mnist = input_data.read_data_sets("res/datasets/MNIST/", one_hot=True)

        for _ in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            self.sess.run(self.train_step, feed_dict={self.layers['x']: batch_xs, self.y_: batch_ys})
        correct_prediction = tf.equal(tf.argmax(self.layers['y'],1), tf.argmax(self.y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        writer = tf.summary.FileWriter('/Users/Sadegh/Documents/UIO/unik4690/4690-p2018/src/res/my_graph', self.sess.graph)


        print("Accuracity: "+str(self.sess.run(accuracy,
                                            feed_dict={self.layers['x']: mnist.test.images, self.y_: mnist.test.labels})))

        #self.writer = tf.summary.FileWriter('./res/model/my_graph', self.sess.graph)
        #self.writer.close
        #save_path = saver.save(self.sess, self.path)
        #print("Model saved in path: %s" % save_path)


        '''
        #path = "res/model/mnist_demo"
        saver = tf.train.Saver()
        if Path(self.path+".index").is_file():
            saver.restore(self.sess, self.path)
            print("Model restored")
        else:
            print("Not trained")

            mnist = input_data.read_data_sets("res/datasets/MNIST/", one_hot=True)
            self.init = tf.global_variables_initializer()
            self.sess.run(self.init)
            #np.set_printoptions(linewidth=9999999)
            for _ in range(1000):
                batch_xs, batch_ys = mnist.train.next_batch(100)
                self.sess.run(self.train_step, feed_dict={self.layers['l0']: batch_xs, self.y_: batch_ys})
            correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


            print("Accuracity: "+str(self.sess.run(accuracy,
                                                feed_dict={self.layers['l0']: mnist.test.images, self.y_: mnist.test.labels})))

            save_path = saver.save(self.sess, self.path)
            print("Model saved in path: %s" % save_path)
            '''
        return

    def forward(self, data):
        network = self.layers['y']
        return self.sess.run(network, feed_dict={self.layers['x']: data})


def file_rendering(path):
    img = cv2.imread(path);
    img_small = cv2.resize(img, (28,28))
    img_grey = cv2.cvtColor(255-img_small, cv2.COLOR_RGB2GRAY)

    data = np.reshape(img_grey, (1, -1))
    data = (1/255)*data

    return data

if __name__ == "__main__":
    np.set_printoptions(linewidth=9999999)

    layers = [784, 64, 32, 10]
    nn = FCNN(layers)

    nn.init_graph()
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

        index = np.argmax(r)
        print("%d : %.2f%%" % (index, np.amax(r)*100))
