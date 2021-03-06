#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np
import functools
import timeit
import os

from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data

tf.logging.set_verbosity(tf.logging.DEBUG)

np.set_printoptions(linewidth=9999999)
np.set_printoptions(edgeitems=9999999)

def imshow(text, img):
    plt.title(text)
    plt.xticks([]), plt.yticks([])
    plt.imshow(img, cmap="gray")
    plt.draw()
    plt.pause(0.1)
    return input("<Hit Enter To Continue>")


dataset_path = "res/datasets/"

class COMBINED_CNN:
    def __init__(self, model_dir=None):
        self.model_dir = model_dir if model_dir != None else"res/model/cnn_combined"
        #self.mapping = np.loadtxt(dataset_path+"bymerge-mapping.txt", dtype=np.uint8)
        self.mapping = np.array([
            [0, 48],
            [1, 49],
            [2, 50],
            [3, 51],
            [4, 52],
            [5, 53],
            [6, 54],
            [7, 55],
            [8, 56],
            [9, 57],
            [10, 65],
            [11, 66],
            [12, 67],
            [13, 68],
            [14, 69],
            [15, 70],
            [16, 71],
            [17, 72],
            [18, 73],
            [19, 74],
            [20, 75],
            [21, 76],
            [22, 77],
            [23, 78],
            [24, 79],
            [25, 80],
            [26, 81],
            [27, 82],
            [28, 83],
            [29, 84],
            [30, 85],
            [31, 86],
            [32, 87],
            [33, 88],
            [34, 89],
            [35, 90],
            [36, 97],
            [37, 98],
            [38, 100],
            [39, 101],
            [40, 102],
            [41, 103],
            [42, 104],
            [43, 110],
            [44, 113],
            [45, 114],
            [46, 116]])

        # Create the Estimator
        self.classifier = tf.estimator.Estimator(
            model_fn=self.cnn_model_fn,
            model_dir=self.model_dir)


    def cnn_model_fn(self, features, labels, mode):
        """Model function for CNN."""
        # Input Layer
        # Reshape X to 4-D tensor: [batch_size, width, height, channels]
        # MNIST images are 28x28 pixels, and have one color channel
        input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

        # Convolutional Layer #1
        # Computes 32 features using a 5x5 filter with ReLU activation.
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, 28, 28, 1]
        # Output Tensor Shape: [batch_size, 28, 28, 32]
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            name = "conv1",
            padding="same",
        activation=tf.nn.relu)

        # Pooling Layer #1
        # First max pooling layer with a 2x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, 28, 28, 32]
        # Output Tensor Shape: [batch_size, 14, 14, 32]
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # Convolutional Layer #2
        # Computes 64 features using a 5x5 filter.
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, 14, 14, 32]
        # Output Tensor Shape: [batch_size, 14, 14, 64]
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            name = "conv2",
            padding="same",
            activation=tf.nn.relu)

        # Pooling Layer #2
        # Second max pooling layer with a 2x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, 14, 14, 64]
        # Output Tensor Shape: [batch_size, 7, 7, 64]
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # Convolutional Layer #3
        # Computes 64 features using a 5x5 filter.
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, 7, 7, 64]
        # Output Tensor Shape: [batch_size, 7, 7, 64]
        conv3 = tf.layers.conv2d(
            inputs=pool2,
            filters=64,
            kernel_size=[5, 5],
            name = "conv3",
            padding="same",
            activation=tf.nn.relu)

        # Flatten tensor into a batch of vectors
        # Input Tensor Shape: [batch_size, 7, 7, 64]
        # Output Tensor Shape: [batch_size, 7 * 7 * 64]
        conv3_flat = tf.reshape(conv3, [-1, 7 * 7 * 64])

        # Dense Layer
        # Densely connected layer with 1024 neurons
        # Input Tensor Shape: [batch_size, 7 * 7 * 64]
        # Output Tensor Shape: [batch_size, 1024]
        dense = tf.layers.dense(inputs=conv3_flat, units=1024, activation=tf.nn.relu)

        # Add dropout operation; 0.6 probability that element will be kept
        dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

        # Logits layer
        # Input Tensor Shape: [batch_size, 1024]
        # Output Tensor Shape: [batch_size, 48]

        logits = tf.layers.dense(inputs=dropout, units=48, name="output")

        predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        # Calculate Loss (for both TRAIN and EVAL modes)
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        # Compute evaluation metrics.
        accuracy = tf.metrics.accuracy(labels=labels,
                                       predictions=predictions['classes'],
                                       name='acc_op')
        metrics = {'accuracy': accuracy}
        tf.summary.scalar('accuracy', accuracy[1])
        tf.summary.scalar("loss", loss)

        # Configure the Training Op (for TRAIN mode)
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # Add evaluation metrics (for EVAL mode)
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


    def forward(self, data):
        # Forbid fowward to generate unnecessary output
        old_v = tf.logging.get_verbosity()
        tf.logging.set_verbosity(tf.logging.ERROR)

        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": data},
            num_epochs=1,
            shuffle=False)

        predict_results = self.classifier.predict(input_fn=predict_input_fn)
        predicted_class = next(predict_results)['classes']

        tf.logging.set_verbosity(old_v)
        return predicted_class


    def class2char(self, class_n):
        index = np.argwhere(self.mapping[:,0] == class_n)[0][0]
        char = self.mapping[index][1]
        return chr(char)


    def load_data(self, dataset):
        self.train_data   = dataset.train.images  # Returns np.array
        self.train_labels = np.asarray(dataset.train.labels, dtype=np.int32)
        self.eval_data    = dataset.test.images  # Returns np.array
        self.eval_labels  = np.asarray(dataset.test.labels, dtype=np.int32)

    def train(self, steps=2000, batch_size=100, num_epochs=None, shuffle=True):
        # Enable debug output while training
        old_v = tf.logging.get_verbosity()
        tf.logging.set_verbosity(tf.logging.DEBUG)

        # Train the model
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": self.train_data},
            y=self.train_labels,
            batch_size=batch_size,
            num_epochs=num_epochs,
            shuffle=shuffle)

        self.classifier.train(
            input_fn=train_input_fn,
            steps=steps)

        for x in [1, 2, 3]:
            weights = self.classifier.get_variable_value("conv"+str(x)+"/kernel")
            np.savetxt("conv"+str(x)+"_"+str(np.shape(weights))+".txt", weights.flatten())



    def evaluate(self, num_epochs=1, shuffle=False):
        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": self.eval_data},
            y=self.eval_labels,
            num_epochs=num_epochs,
            shuffle=shuffle)
        eval_results = self.classifier.evaluate(input_fn=eval_input_fn)
        print(eval_results)
        return eval_results



if __name__ == "__main__":

    nn = COMBINED_CNN()

    for sett in [('SANS', 100, 50000, 0), ('FNIST', 100, 50000, 5000), ('EMNIST_ByMerge/', 100, 30000, 5000)]:
        dataset = input_data.read_data_sets(dataset_path+sett[0], validation_size=sett[3])
        nn.load_data(dataset = dataset)
        nn.train(sett[2], batch_size=sett[1])
        nn.evaluate()
