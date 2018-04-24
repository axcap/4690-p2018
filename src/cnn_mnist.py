#Source: https://www.tensorflow.org/tutorials/layers
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
from matplotlib import pyplot as plt
import cv2
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data
#tf.logging.set_verbosity(tf.logging.INFO)


def confusion_matrix(dataset, nn):
    """This function takes dataset and tensorflow network and calculates the confusion matrix.

    Args:
        dataset (tensorflow DataSet): Dataset with train_data and labels.
        nn      (Tensor): Tensorflow function.

    Returns:
        numpy array: Confusion matrix with shape n_classes x n_classes.

    """
    sess = tf.Session()
    y = tf.argmax(nn, 1).eval({X: dataset.images})
    y_ = tf.argmax(dataset.labels, 1).eval()
    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    confusion = tf.confusion_matrix(labels=y_, predictions=y, num_classes=10).eval()
    return confusion


# Our application logic will be added here
if __name__ == "__main__":

    # Parameters
    learning_rate = 0.001
    training_epochs = 1
    batch_size = 10
    display_step = 1

    # Network Parameters
    n_hidden_1 = 128 # 1st layer number of neurons
    n_hidden_2 = 64 # 2nd layer number of neurons
    n_input = 784 # MNIST data input (img shape: 28*28)
    n_classes = 10 # MNIST total classes (0-9 digits)

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    # tf Graph input
    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_classes])


    # Store layers weight & bias
    weights = {
        'h1':  tf.Variable(tf.random_normal([n_input,    n_hidden_1])),
        'h2':  tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases = {
        'b1':  tf.Variable(tf.random_normal([n_hidden_1])),
        'b2':  tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.nn.leaky_relu(tf.add(tf.matmul(X, weights['h1']), biases['b1']))
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.nn.leaky_relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    # Output fully connected layer with a neuron for each class
    logits = tf.matmul(layer_2, weights['out']) + biases['out']

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=Y))
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss_op)
    # Initializing the variables
    init = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                                Y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
                # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
                print("Optimization Finished!")

        save_path = saver.save(sess, "../res/model/mnist")
        print("Model saved in path: %s" % save_path)

        # Test model
        network = tf.nn.softmax(logits)  # Apply softmax to logits
        correct_prediction = tf.equal(tf.argmax(network, 1), tf.argmax(Y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))



        conf = confusion_matrix(mnist.test, network)
        print("Acc: ", np.sum(conf.diagonal())/np.sum(conf))
        print("Confusion:", conf)
        plt.imshow(conf)
        plt.show()

        print("\n\n\n")
        dirname = '../res/images/input/'
        for filename in os.listdir(dirname):
            path = dirname+filename
            img = cv2.imread(path);
            img_small = cv2.resize(img, (28,28))
            img_grey = cv2.cvtColor(255-img_small, cv2.COLOR_BGR2GRAY)
            demo = np.reshape(img_grey, (1, -1))
            demo = (1/256)*demo

            output = network.eval({X: demo})
            index = np.argmax(output)
            #print(output)
            print("%d : %.2f%%" % (index, np.amax(output)*100))
            plt.imshow(np.reshape(demo, (28,28)))
            plt.show()
