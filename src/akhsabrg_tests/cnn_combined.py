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


dataset_path = "res/datasets/EMNIST_ByMerge/"
model_dir="res/model/cnn_combined"

mapping = np.loadtxt(dataset_path+"emnist-bymerge-mapping.txt", dtype=np.uint8)
print(mapping)

def cnn_model_fn(features, labels, mode):
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
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 14, 14, 64]
  # Output Tensor Shape: [batch_size, 7, 7, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

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


def forward(network, data):
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": data},
        num_epochs=1,
        shuffle=False)
    predict_results = network.predict(input_fn=predict_input_fn)
    predicted_class = next(predict_results)['classes']
    print(predicted_class)
    return predicted_class

def class2char(class_n):
    index = np.argwhere(mapping[:,0] == class_n)[0][0]
    char = mapping[index][1]
    return chr(char)

def main(unused_argv):
  # Load training and eval data
  emnist = input_data.read_data_sets(dataset_path)

  train_data   = emnist.train.images  # Returns np.array
  train_labels = np.asarray(emnist.train.labels, dtype=np.int32)
  eval_data    = emnist.test.images  # Returns np.array
  eval_labels  = np.asarray(emnist.test.labels, dtype=np.int32)

  # Create the Estimator
  emnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir=model_dir)

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=10,
      num_epochs=None,
      shuffle=True)

  t = timeit.Timer(functools.partial(emnist_classifier.train,
                                     input_fn=train_input_fn,
                                     steps=30000))
  print("Time: ", t.timeit(1)/60)

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = emnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)

if __name__ == "__main__":
  #tf.app.run()
  # Load training and eval data
  emnist = input_data.read_data_sets(dataset_path)

  train_data   = emnist.train.images  # Returns np.array
  train_labels = np.asarray(emnist.train.labels, dtype=np.int32)
  eval_data    = emnist.test.images  # Returns np.array
  eval_labels  = np.asarray(emnist.test.labels, dtype=np.int32)

  # Create the Estimator
  emnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir=model_dir)

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)

  t = timeit.Timer(functools.partial(emnist_classifier.train,
                                     input_fn=train_input_fn,
                                     steps=1))
  print("Time: ", t.timeit(1)/60)

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  #eval_results = emnist_classifier.evaluate(input_fn=eval_input_fn)
  #print(eval_results)

  for idx in range(eval_data.size):
      img = eval_data[idx]
      label = eval_labels[idx]
      print(label)
      demo = np.reshape(img, (1, -1))
      print("forward")
      guessed = forward(emnist_classifier, demo)
      print(guessed)
      print("%s : %s" % (class2char(label) ,
                         class2char(guessed)))
      imshow("Test: ", np.transpose(np.reshape(demo, (28,28))))
