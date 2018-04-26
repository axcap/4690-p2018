#Source: https://www.tensorflow.org/tutorials/layers
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
    n_hidden_1 = 728 # 1st layer number of neurons
    n_input = 784    # MNIST data input (img shape: 28*28)
    n_classes = 10   # MNIST total classes (0-9 digits)
    
    def __init__(self, model_path="res/model/nn_mnist/mnist_demo"):
        # tf Graph input
        self.X = tf.placeholder("float", [None, self.n_input])
        self.Y = tf.placeholder("float", [None, self.n_classes])

        # Store layers weight & bias
        self.weights = {
            'h1':  tf.Variable(tf.random_normal([self.n_input,    self.n_hidden_1])),
            'out': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_classes]))
        }
        self.biases = {
            'b1':  tf.Variable(tf.random_normal([self.n_hidden_1])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

        # Hidden fully connected layer with 256 neurons
        self.layer_1 = tf.nn.leaky_relu(tf.add(tf.matmul(self.X, self.weights['h1']), self.biases['b1']))
        # Output fully connected layer with a neuron for each class
        self.logits = tf.matmul(self.layer_1, self.weights['out']) + self.biases['out']

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
        network = tf.nn.softmax(self.logits)
        return self.sess.run(network, feed_dict={self.X: data})

    def confusion_matrix(self, dataset):
        """This function takes dataset and tensorflow network and calculates the confusion matrix.
        Args:
            dataset (tensorflow DataSet): Dataset with train_data and labels.
            nn      (Tensor): Tensorflow function.
        Returns:
            numpy array: Confusion matrix with shape n_classes x n_classes.
        """
        network = tf.nn.softmax(self.logits)
        
        y =  self.sess.run(tf.argmax(network, 1), {self.X: dataset.images})
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
    mnist = input_data.read_data_sets("res/datasets/MNIST/", one_hot=True)
    
    nn = NN_MNIST()
    nn.set_params(1e-4, 500, 200)
    print("Time used: ", timeit.timeit('nn.train(mnist, force_retrain=False, save_model=True)', number=1, globals=globals()))
    #nn.train(mnist, force_retrain=True)

    conf = nn.confusion_matrix(mnist.test)
    print("Accuracy: %.2f %%" % (np.sum(conf.diagonal())/np.sum(conf) * 100))
    print("Error: %.2f %%"    % ((1-np.sum(conf.diagonal())/np.sum(conf)) * 100))
    #print("Confusion:\n", conf)


    print("\n\n\n")
    dirname = 'res/images/'
    for filename in os.listdir(dirname):
        if len(filename) is not 5:
            continue
        
        print(filename)
        path = dirname+filename
        img = cv2.imread(path);
        img_small = cv2.resize(img, (28,28))
        img_grey = cv2.cvtColor(255-img_small, cv2.COLOR_RGB2GRAY)

        data = np.reshape(img_grey, (1, -1))
        data = (1/255)*data
        
        r = nn.forward(data)            
        index = np.argmax(r)
        print("%d : %.2f%%" % (index, np.amax(r)*100))
