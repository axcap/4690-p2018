from nn_mnist import NN_MNIST
import utils as utils

import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
from nn_mnist import NN_MNIST
from cnn_angle import RNIST


import cv2
import sys
import time
import numpy as np
from pathlib import Path
import utils as utils

np.set_printoptions(linewidth=9999999)

if __name__ == "__main__":

    image_path = sys.argv[1]

    # load the image from disk
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.bitwise_not(gray)
    y,x = binary.shape

    nn = NN_MNIST(model_path="res/model/nn_fnist/fnist_demo")
    dataset = input_data.read_data_sets("res/datasets/FNIST/", one_hot=True, validation_size=10)
    images = input_data.read_data_sets("res/datasets/ROTFNIST/", validation_size=10)

    nn.train(dataset, force_retrain=False)

    rn = RNIST()

    for img in images.train.images:
      img = img.reshape((28,28))
      print(img*255)
      angle = next(rn.predict(img.astype(np.float32)))['classes'] * 90
      print("Angle: %d" % (angle))

      corrected_img = utils.rotate2angle(img, -angle)
      r = nn.forward(np.reshape(corrected_img, (1, 28*28)))
      index = np.argmax(r)
      print("Class: %d - %.2f%%" % (index, r[0, index]*100))
      out = np.hstack((img, np.zeros((28, 5)), corrected_img))
      utils.imshow("img", out)
