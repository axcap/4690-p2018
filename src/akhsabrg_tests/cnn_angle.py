from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.DEBUG)

from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
from pathlib import Path
from tqdm import tqdm
import utils as utils
import shutil

import utils as utils
import numpy as np
import sys
import cv2

np.set_printoptions(linewidth=9999999)
np.set_printoptions(edgeitems=9999999)


dataset_path = "res/datasets/ROTMNIST/"

class RNIST:
  def cnn_model_fn(self, features, labels, mode):
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

    logits = tf.layers.dense(inputs=conv1, units=4)

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
    #loss = tf.losses.mean_squared_error(labels, logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
      print("TRAIN")
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
      train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
        labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


  def __init__(self, model_path="res/model/rotmnist"):
    self.model_path = model_path

    # Create the Estimator
    self.dataset_classifier = tf.estimator.Estimator(
      model_fn=self.cnn_model_fn, model_dir=self.model_path)

  def set_params(self, learning_rate, num_epochs, batch_size, steps = 1):
    self.learning_rate   = learning_rate
    self.num_epochs      = num_epochs
    self.batch_size      = batch_size
    self.steps           = steps

  def train(self, dataset, force_retrain = False):
    old_v = tf.logging.get_verbosity()
    tf.logging.set_verbosity(tf.logging.INFO)

    train_data = dataset.train.images
    train_labels = np.asarray(dataset.train.labels, dtype=np.int32)
    train_labels = np.asarray(dataset.train.labels, dtype=np.int32)
    eval_data = dataset.test.images
    eval_labels = np.asarray(dataset.test.labels, dtype=np.int32)

    if (force_retrain):
      shutil.rmtree(self.model_path)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=self.batch_size,
      num_epochs=self.num_epochs,
      shuffle=True)

    self.dataset_classifier.train(input_fn = train_input_fn, steps=self.steps)

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
    eval_results = self.dataset_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

    tf.logging.set_verbosity(old_v)


  def predict(self, data):
    # Evaluate the model and print results
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": data},
      num_epochs=1,
      shuffle=False)

    return self.dataset_classifier.predict(input_fn=predict_input_fn)


  def main(unused_argv):
    # Load training and eval data
    dataset = input_data.read_data_sets(dataset_path)

    train_data = dataset.train.images
    train_labels = np.asarray(dataset.train.labels, dtype=np.int32)
    eval_data = dataset.test.images
    eval_labels = np.asarray(dataset.test.labels, dtype=np.int32)

    # Create the Estimator
    dataset_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="res/model/rotfnist")

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

    dataset_classifier.train(input_fn = train_input_fn, steps=20000)

    idx = np.random.randint(low=0, high=dataset.train.num_examples, size=10)
    for x in range(10):
      img = np.reshape(train_data[idx[x]], (28,28))
      predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": img.reshape(1, -1)},
        num_epochs=1,
        shuffle=False)

      predict_results = dataset_classifier.predict(input_fn = predict_input_fn)
      print(img*255)
      for result in predict_results:
        print(result)
      print("")



if __name__ == "__main__":
  #Load training data
  dataset = input_data.read_data_sets(dataset_path)

  nn = RNIST()
  nn.set_params(0.001, None, 100, 50000)

  nn.train(dataset, force_retrain = False)
