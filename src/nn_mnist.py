#Source: https://www.tensorflow.org/tutorials/layers
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
from matplotlib import pyplot as plt
import cv2
from pathlib import Path
import numpy as np
import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data
#tf.logging.set_verbosity(tf.logging.INFO)

class NN_MNIST:
    sess = None
    model_path = None
    
    def __init__(self, model_path="res/model/nn_mnist/mnist_demo"):
        self.x = tf.placeholder(tf.float32, [None, 784])
        self.W = tf.Variable(tf.zeros([784, 10]))
        self.b = tf.Variable(tf.zeros([10]))
        self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)
        
        self.y_ = tf.placeholder(tf.float32, shape = [None, 10])
        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))
        self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(self.cross_entropy)
        self.sess = tf.InteractiveSession()
    
        self.model_path = model_path

    def train(self, dataset, force_retrain = False):
        saver = tf.train.Saver()
        if force_retrain or not Path(self.model_path+".index").is_file():
            print("New training started")
            tf.global_variables_initializer().run()
            for _ in range(1000):
                batch_xs, batch_ys = dataset.train.next_batch(100)
                self.sess.run(self.train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})
                
            correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print("Accuracity: "+str(self.sess.run(accuracy,
                                              feed_dict={self.x: dataset.test.images,
                                                         self.y_: dataset.test.labels})))
            #tf.logging.set_verbosity(old_v)            
            save_path = saver.save(self.sess, self.model_path)
            print("Model saved in path: %s" % save_path)
        else:
            saver.restore(self.sess, self.model_path)
            print("Model restored")       
        

    def forward(self, data):
        return self.sess.run(self.y, feed_dict={self.x: data})

if __name__ == "__main__":
    np.set_printoptions(linewidth=9999999)
    
    #Load training data 
    mnist = input_data.read_data_sets("res/datasets/MNIST/", one_hot=True)
    nn = NN_MNIST()
    
    nn.train(mnist, force_retrain=True)
    
    img = cv2.imread('res/images/1.png');
    img_small = cv2.resize(img, (28,28))
    img_grey = cv2.cvtColor(255-img_small, cv2.COLOR_BGR2GRAY)
    print(img_grey)
    
    data = (1/256)*img_grey
    data = np.reshape(data, (1, 28*28))
    r = nn.forward(data)
    print(np.argmax(r))
