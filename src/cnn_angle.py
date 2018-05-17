from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
#tf.logging.set_verbosity(tf.logging.DEBUG)

from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm

import utils as utils
import numpy as np
import sys
import cv2

np.set_printoptions(linewidth=9999999)
np.set_printoptions(edgeitems=9999999)


dataset_path = "res/datasets/ROTFNIST/"

def cnn_model_fn2(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=25,
      kernel_size=5,
      padding="same",
      activation=tf.nn.relu)

  conv1 = tf.reshape(conv1, [-1, 28 * 28 * 1])

  logits = tf.layers.dense(inputs=conv1, units=1, activation=None)

  # Add dropout operation; 0.6 probability that element will be kept
  #dropout = tf.layers.dropout(
  #    inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  #cost = tf.reduce_mean(tf.square(logits - labels))
  loss = tf.losses.mean_squared_error(labels, logits)

  predictions = {
      "classes": tf.argmax(input=logits),
      "probabilities": tf.losses.mean_squared_error(labels, logits)
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    print("PREDICT")  
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    print("TRAINT")
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss = loss)
    print("111111111111")
    with tf.Session() as sess:
      print(sess.run(tf.shape(loss)))
      #print(sess.run(tf.shape(conv1)))
      #print(sess.run(loss))
      pass
    
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel
  input_layer = tf.reshape(
    features["x"],
    [-1, 28, 28, 1])

  conv1 = tf.layers.conv2d(
    inputs=input_layer,
    filters=25,
    kernel_size=5,
    padding="same",
    activation=tf.nn.relu)

  conv1 = tf.reshape(
    conv1,
    [-1, 28*28*25])

  logits = tf.layers.dense(
    inputs=conv1,
    units=1)

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

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss=loss,
                                  global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
  # Load training and eval data
  dataset = input_data.read_data_sets(dataset_path, validation_size=10)

  train_data = dataset.train.images
  train_labels = np.zeros((len(dataset.train.labels), 1), dtype=np.int32)
  eval_data = dataset.train.images
  eval_labels = np.zeros((len(dataset.train.labels), 1), dtype=np.int32)

  print(len(train_data), len(train_labels))
  print(len(eval_data), len(eval_labels))

  # Create the Estimator
  dataset_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/rotfnist")

  # Train the model
  print("Training started")
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=100,
    num_epochs=1000,
    shuffle=True)
  
  dataset_classifier.train(input_fn = train_input_fn, steps=1)    
  
  print("Evaluation started")
  eval_data = dataset.train.images
  eval_labels = np.zeros((len(dataset.train.images), 1), dtype=np.int32) #np.asarray(fnist.train.labels, dtype=np.int32)

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)

  eval_results = dataset_classifier.evaluate(input_fn=eval_input_fn)
  print("Results: ", eval_results)

  for x in range(10):
    img = np.reshape(train_data[x], (28,28))

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": img.reshape(1, -1)},
      num_epochs=1,
      shuffle=False)
    
    predict_results = dataset_classifier.predict(input_fn = predict_input_fn)
    print(img)
    for result in predict_results:
      print(result)
    

  
if __name__ == "__main__":
  tf.app.run()
