from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
old_v = tf.logging.get_verbosity()
#tf.logging.set_verbosity(tf.logging.ERROR)
tf.logging.set_verbosity(tf.logging.INFO)

from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm

import utils as utils
import numpy as np
import sys
import cv2

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=25,
      kernel_size=12,
      padding="same",
      activation=tf.nn.relu)

  conv1_flat = tf.reshape(conv1, [-1, 28 * 28 * 1])

  dense = tf.layers.dense(inputs=conv1_flat, units=1, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  logits = tf.layers.dense(inputs=dropout, units=1)

  #cost = tf.reduce_mean(tf.square(logits - labels))
  loss = tf.losses.mean_squared_error(labels, logits)

  predictions = {
      "classes": tf.argmax(input=logits, axis=1),
      "probabilities": tf.losses.mean_squared_error(labels, logits)
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
  print("111111111111")
  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    print("111111111111")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    print("111111111111")
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    print("111111111111")
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Load training and eval data
  mnist = input_data.read_data_sets("res/datasets/MNIST/")

  train_data = mnist.train.images
  train_labels = np.zeros((len(mnist.train.labels), 1), dtype=np.int32)
  eval_data = mnist.test.images
  eval_labels = np.zeros((len(mnist.test.labels),1), dtype=np.int32)

  print(len(train_data), len(train_labels))
  print(len(eval_data), len(eval_labels))

  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

  # Train the model
  print("Training started")
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=3,
    shuffle=True)

  mnist_classifier.train(
    input_fn=train_input_fn,
    steps=200)

  print("Evaluation started")
  # Load training and eval data
  fnist = input_data.read_data_sets("res/datasets/FNIST/")

  eval_data = fnist.train.images
  eval_labels = np.zeros((len(fnist.train.images), 1), dtype=np.int32) #np.asarray(fnist.train.labels, dtype=np.int32)

  print(np.shape(eval_labels), "\n\n\n")

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)

  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print("Results: ", eval_results)
  exit()


if __name__ == "__main__":
  tf.app.run()
