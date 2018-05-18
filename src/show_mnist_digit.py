from tensorflow.examples.tutorials.mnist import input_data
from matplotlib.widgets import Button
from matplotlib import pyplot as plt

import sys
import numpy as np
np.set_printoptions(linewidth=9999999)

if(len(sys.argv) < 2):
    print("\n\nUsage: python %s N" % sys.argv[0])
    exit(1)

digit = int(sys.argv[1])
database = input_data.read_data_sets("res/datasets/ROTMNIST/", one_hot=True, validation_size=10)

for i in range(len(database.train.images)):
    img = np.reshape((database.train.images[i]*255).astype(int), (28, 28))
    print(img)
    print(database.train.labels[i])
    input()


'''
labels = np.argmax(database.train.labels, axis=1)
arr = np.where(labels == digit)[0]
print("Len: ", len(arr))

for i in range(len(arr)):
    img = np.reshape((database.train.images[arr[i]]*255).astype(np.int), (28, 28))
    print(img)
    input()
'''
