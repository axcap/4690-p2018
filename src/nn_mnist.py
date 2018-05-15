#Source: https://www.tensorflow.org/tutorials/layers
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm

import os
import sys
import cv2
import timeit
import curses
import numpy as np
import tensorflow as tf

class NN_MNIST:
    sess = None
    model_path = None

    # Parameters
    learning_rate = 0.001
    training_epochs = 1
    batch_size = 10
    display_step = 1

    # Network Parameters
    n_input    = 784    # MNIST data input (img shape: 28*28)
    n_hidden_1 = 728 # 1st layer number of neurons
    n_hidden_2 = 128 # 1st layer number of neurons
    n_classes  = 10   # MNIST total classes (0-9 digits)

    def __init__(self, model_path="res/model/nn_mnist/mnist_demo"):
        # tf Graph input
        self.X = tf.placeholder("float", [None, self.n_input])
        self.Y = tf.placeholder("float", [None, self.n_classes])

        # Store layers weight & bias
        self.weights = {
            'h1':  tf.Variable(tf.random_normal([self.n_input,    self.n_hidden_1])),
            'h2':  tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_classes]))
        }
        self.biases = {
            'b1':  tf.Variable(tf.random_normal([self.n_hidden_1])),
            'b2':  tf.Variable(tf.random_normal([self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

        # Hidden fully connected layer with 256 neurons
        self.layer_1 = tf.nn.leaky_relu(tf.matmul(self.X, self.weights['h1'])       + self.biases['b1'])
        self.layer_2 = tf.nn.leaky_relu(tf.matmul(self.layer_1, self.weights['h2']) + self.biases['b2'])

        # Output fully connected layer with a neuron for each class
        self.logits = tf.matmul(self.layer_2, self.weights['out']) + self.biases['out']
        self.network = tf.nn.softmax(self.logits)

        # Define loss and optimizer
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        #self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss_op)

        # Initializing the variables
        self.init = tf.global_variables_initializer()

        self.sess = tf.Session()
        self.sess.run(self.init)

        self.model_path = model_path

    def train(self, dataset, force_retrain = False, save_model = True):
        saver = tf.train.Saver()
        if force_retrain or not Path(self.model_path+".index").is_file():
            # Training cycle
            for epoch in range(self.training_epochs):
                if curses.wrapper(self.getch) == ord('q'):
                    break

                avg_cost = 0.
                total_batch = int(dataset.train.num_examples/self.batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    batch_x, batch_y = dataset.train.next_batch(self.batch_size)
                    _, c = self.sess.run([self.train_op, self.loss_op], feed_dict={self.X: batch_x,
                                                                                   self.Y: batch_y})
                    # Compute average loss
                    avg_cost += c / total_batch

                # Display logs per epoch step
                if epoch % self.display_step == 0:
                    print("Epoch %d/%d, cost: %.4f" % (epoch+1, self.training_epochs, avg_cost))

            print("Optimization Finished!")

            if save_model:
                save_path = saver.save(self.sess, self.model_path)
                print("Model saved in path: %s" % save_path)
        else:
            saver.restore(self.sess, self.model_path)
            print("Model restored")


    def forward(self, data):
        return self.sess.run(self.network, feed_dict={self.X: data})

    def confusion_matrix(self, dataset):
        """This function takes dataset and tensorflow network and calculates the confusion matrix.
        Args:
            dataset (tensorflow DataSet): Dataset with train_data and labels.
            nn      (Tensor): Tensorflow function.
        Returns:
            numpy array: Confusion matrix with shape n_classes x n_classes.
        """

        y =  self.sess.run(tf.argmax(self.network, 1), {self.X: dataset.images})
        y_ = self.sess.run(tf.argmax(dataset.labels, 1))

        confusion = tf.confusion_matrix(labels=y_, predictions=y)
        return self.sess.run(confusion)


    def set_params(self, learning_rate, training_epochs, batch_size, display_step = 1):
        self.learning_rate   = learning_rate
        self.training_epochs = training_epochs
        self.batch_size      = batch_size
        self.display_step    = display_step

    def getch(self, stdscr):
        """checking for keypress"""
        stdscr.nodelay(True)  # do not wait for input when calling getch
        return stdscr.getch()


if __name__ == "__main__":
    np.set_printoptions(linewidth=9999999)

    #Load training data
    dataset = input_data.read_data_sets("res/datasets/FNIST/", one_hot=True, validation_size=10)

    nn = NN_MNIST()
    nn.set_params(0.001, 100000, 100)
    print("Time used: ", timeit.timeit('nn.train(dataset, force_retrain=False, save_model=False)', number=1, globals=globals()))
    #nn.train(dataset, force_retrain=True)

    #conf = nn.confusion_matrix(dataset.test)
    conf = nn.confusion_matrix(dataset.train)
    print("Accuracy: %.2f %%" % (np.sum(conf.diagonal())/np.sum(conf) * 100))
    print("Error: %.2f %%"    % ((1-np.sum(conf.diagonal())/np.sum(conf)) * 100))
    #print("Confusion:\n", conf)
