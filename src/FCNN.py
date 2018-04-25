
import tensorflow as tf
from matplotlib import pyplot as plt
import cv2
from pathlib import Path
import numpy as np
import tensorflow as tf


'''
I assume we get a CONF dictionary in as input with variable declerations
conf['layer_dimensions']  : A list of length L+1 with the number of nodes in
                            each layer, including the input layer, all hidden
                            layers, and the output layer.


input -- init:
'layer_dimensions'
'

'''

class FCNN:

    # how many nodes for each layer, inc input, hidden and output layer
    self.layer_dimensions = []
    self.tot_layers = None

    # weights(w) and biases(b), strings lowercase + nr layer(starting at 0)
    # eg. 'w0', 'b0'
    self.params = {}

    # all layers(nodes) including input, hidden and output layer
    self.layers = {}

    # model path
    self.path = None

    # input: layers = list with number of nodes for each layer
    def __init__(self, layers):
        self.layer_dimensions = layers
        self.tot_layers = len(layers)
        self.path = "res/model/mnist_demo"

    # initilazes all parameters (w and b to zero) in a dict 'params'
    def init_params(self):
        for idx, a in enumerate(self.layer_dimensions):
            if idx != self.tot_layers:
                self.params['w{0}'.format(idx + 1)] = tf.Variable(tf.zeros([
                                                a,
                                                self.layer_dimensions[idx + 1]]))

                self.params['b{0}'.format(idx + 1)] = tf.Variable(tf.zeros([
                                                self.layer_dimensions[idx + 1]]))
        return


    # initilazes all layers (input, hidden, output) in a dict 'layers'
    def init_layers(self):
        for idx, a in enumerate(self.layer_dimensions):
            if idx == 0: # input layer, x
                self.layers['l{0}'.format(idx)] = tf.placeholder(
                                            tf.float32,
                                            [None, a])
            elif idx == self.tot_layers - 1: # hidden layers
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

                self.layers['l{0}'.format(idx)] = tf.nn.softmax(
                                            tf.matmul(layer_before,
                                            w_before)
                                            + b_before)
        return


    def init_graph(self):
        #x = tf.placeholder(tf.float32, [None, 784])     # init_layers
        #W = tf.Variable(tf.zeros([784, 10]))            # init_param
        #b = tf.Variable(tf.zeros([10]))                 # init_param
        #y = tf.nn.softmax(tf.matmul(x, W) + b)          # init_layers

        init_param(self)
        init_layers(self)

        # last layer
        y = layers['l{0}'.format(tot_layers - 1)]

        # predicted output
        y_ = tf.placeholder(tf.float32, shape = [None, 10])

        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        sess = tf.InteractiveSession()
        return


    def train(self):

        #path = "res/model/mnist_demo"
        saver = tf.train.Saver()
        if Path(path+".index").is_file():
            saver.restore(sess, self.path)
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
        return

    def test(self):
        print("Accuracity: "+str(sess.run(accuracy,
                                            feed_dict={x: mnist.test.images, y_: mnist.test.labels})))
        tf.logging.set_verbosity(old_v)

        save_path = saver.save(sess, self.path)
        print("Model saved in path: %s" % save_path)
        return

    def run(self):
        return
