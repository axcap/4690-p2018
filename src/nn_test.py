from nn_mnist import NN_MNIST
import utils as utils

import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
from nn_mnist import NN_MNIST

import cv2
import sys
import time
import numpy as np
from pathlib import Path

np.set_printoptions(linewidth=9999999)

def imshow(text, img):
    plt.title(text)
    plt.xticks([]), plt.yticks([])
    plt.imshow(img, cmap="gray")
    plt.draw()
    plt.pause(0.1)
    input()

if __name__ == "__main__":

    image_path = sys.argv[1]

    # load the image from disk
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.bitwise_not(gray)
    y,x = binary.shape

    nn = NN_MNIST(model_path="res/model/nn_mnist/mnist_demo")
    dataset = input_data.read_data_sets("res/datasets/FNIST/", one_hot=True, validation_size=10)
    nn.train(dataset, force_retrain=False)

    linesHist = utils.find_lines(binary)
    lines = utils.segment_lines(binary, linesHist)
    print(len(lines))

    output = ""
    start = time.time()
    for i, l in enumerate(lines):
        single_line = binary[l[0]:l[1], 0:x]
        #imshow("line", single_line)
        symbolHist = utils.find_symbol(single_line)
        symbols    = utils.segment_symbols(binary, symbolHist)
        print("%d/%d lines processed" % (i, len(lines)))
        line_out = ""
        for s in symbols:
            single_symbol = binary[l[0]:l[1], s[0]:s[1]]
            single_symbol[single_symbol > 30] = 255
            single_symbol[single_symbol <= 30] = 0
            #print(single_symbol)
            data = cv2.resize(single_symbol, (28,28))
            r = nn.forward(np.reshape(data, (1, 28*28)))
            index = np.argmax(r)
            #print(single_symbol)
            #print("Class: %d - %.2f%%" % (index, r[0, index]*100))
            line_out += " " + str(index)
            #imshow("Symbol", data)

        output += line_out[1:] + "\n"

    with open('res/images/results/tall_result.txt','w+') as fd:
        fd.write(output)

    contents = "".join(Path('res/images/tall.txt').read_text().replace("\n", " ").split())
    output   = "".join(output.replace("\n", " ").split())
    print(output)

    count = 0
    for x, y in zip(output, contents):
        if x != y: count += 1

    print(count, len(output))
    print("Accuracity?: ", 100 - count/(len(output)/100))
    finish = time.time()
    #print(output)
    print("Time: ", finish-start)
