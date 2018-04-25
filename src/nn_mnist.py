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


# Our application logic will be added here
if __name__ == "__main__":
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    y_ = tf.placeholder(tf.float32, shape = [None, 10])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    sess = tf.InteractiveSession()

    path = "res/model/nn_mnist/mnist_demo"
    saver = tf.train.Saver()
    if Path(path+".index").is_file():
        saver.restore(sess, path)
        print("Model restored")
    else:
        print("Not trained")
        
        mnist = input_data.read_data_sets("res/datasets/MNIST/", one_hot=True)    
        tf.global_variables_initializer().run()
        np.set_printoptions(linewidth=9999999)
        for _ in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            '''
            a = np.reshape(batch_xs[0], (28,28))
            print(a, "\n\n"),       
            plt.imshow(a)
            plt.show()
            '''
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracity: "+str(sess.run(accuracy,
                                        feed_dict={x: mnist.test.images, y_: mnist.test.labels})))
        tf.logging.set_verbosity(old_v)

        save_path = saver.save(sess, path)
        print("Model saved in path: %s" % save_path)



    img = cv2.imread('res/images/1.png');
    img_small = cv2.resize(img, (28,28))
    img_grey = cv2.cvtColor(255-img_small, cv2.COLOR_BGR2GRAY)
    demo = np.reshape(img_grey, (1, -1))
    demo = ((1/256)*demo);
    demo = (1/256)*demo

    demo = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.7490196, 1., 0., 0., 0., 0., 0., 0., 0., 0.,],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.2509804, 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0.5019608, 0., 0., 0., 0., 0., 0., 0., 0.,],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.2509804, 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.2509804, 1., 1., 0.5019608, 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.5019608, 1., 1., 0.2509804, 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.2509804, 1., 1., 0.7490196, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.5019608, 1., 1., 0.7490196, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.2509804, 1., 1., 0.7490196, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.5019608, 1., 1., 0.5019608, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.7490196, 1., 1., 0.2509804, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0.7490196, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.5019608, 1., 1., 0.2509804, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.7490196, 1., 0.5019608, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0.5019608, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.7490196, 1., 0.7490196, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.2509804, 1., 0.7490196, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,],
                     [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,]])

    demo = np.reshape(demo, (1, 28*28))
    print(type(demo))
    r = sess.run(y, feed_dict={x: demo})
    print(r)
    print(np.argmax(r))

    plt.imshow(np.reshape(demo, (28,28)))
    plt.show()
