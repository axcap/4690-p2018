from nn_mnist import NN_MNIST
import utils as utils

import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
from nn_mnist import NN_MNIST
from cnn_angle import RNIST
from pathlib import Path


import cv2
import sys
import time
import numpy as np
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


    linesHist = utils.find_lines(binary)
    lines = utils.segment_lines(binary, linesHist)
    for i, l in enumerate(lines):
        single_line = binary[l[0]:l[1], 0:x]
        symbolHist = utils.find_symbol(single_line)
        symbols    = utils.segment_symbols(binary, symbolHist)
        for s in symbols:
            single_symbol = binary[l[0]:l[1], s[0]:s[1]]
            single_symbol = cv2.resize(single_symbol, (28,28))
            single_symbol[single_symbol > 30] = 255
            single_symbol[single_symbol <= 30] = 0
            for angle in [0, 90, 180, 270]:
                symbol = utils.rotate2angle(single_symbol, angle)

                angle = next(rn.predict(symbol.astype(np.float32)))['classes'] * 90
                print("Angle: %d" % (angle))

                utils.imshow("Orig", symbol)

                corrected_img = utils.rotate2angle(symbol, -angle)
                r = nn.forward(np.reshape(corrected_img, (1, 28*28)))
                index = np.argmax(r)
                print("Class: %d - %.2f%%" % (index, r[0, index]*100))
                out = np.hstack((symbol, np.zeros((28, 5)), corrected_img))
                utils.imshow("img", out)
