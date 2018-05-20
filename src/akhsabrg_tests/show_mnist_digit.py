import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.examples.tutorials.mnist import input_data
from matplotlib.widgets import Button
from matplotlib import pyplot as plt

import sys
import numpy as np
np.set_printoptions(linewidth=9999999)

if(len(sys.argv) < 2):
    print("\n\nUsage: python %s N" % sys.argv[0])
    exit(1)

path  = sys.argv[1]
digit = int(sys.argv[2])
dataset = input_data.read_data_sets(path)


collection = np.where(dataset.train.labels == digit)

num = 8
idx = 0
while True:
  fig = plt.figure()
  for y in range(num):
    for x in range(num):
      img = dataset.train.images[collection[0][idx]]
      img = (np.reshape(img, (28,28))*255).astype(int)
      ax = fig.add_subplot(num,num, y*num+x +1)
      print(img, "\n")
      #ax.imshow(img, cmap='gray')
      ax.imshow(img)
      idx += 1

  plt.show()
