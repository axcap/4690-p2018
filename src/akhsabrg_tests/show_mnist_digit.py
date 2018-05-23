import sys
sys.path.append('src')
import utils as utils

import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
import numpy as np

np.set_printoptions(linewidth=9999999)

if(len(sys.argv) < 2):
    print("\n\nUsage: python %s N" % sys.argv[0])
    exit(1)

path  = sys.argv[1]
digit = utils.char2class(sys.argv[2])
dataset = input_data.read_data_sets(path, validation_size=0)
# Print some public variables
print(dataset.train.num_examples)
print(dataset.test.num_examples)
print(np.reshape(dataset.train.images[-1], (28,28)))
print(dataset.train.labels[-1]*255)

img = np.reshape(dataset.train.images[0], (28,28))

collection = np.where(dataset.train.labels == digit)[0]
num = 4
idx = 0

print("Showing class: ", digit)
while True:
  fig = plt.figure()
  for y in range(num):
    for x in range(num):
      if idx >= len(collection): break
      img = dataset.train.images[collection[idx]]
      img = np.reshape(img, (28,28))
      ax = fig.add_subplot(num,num, y*num+x +1)
      #ax.imshow(img, cmap='gray')
      ax.imshow(img)
      idx += 1

  plt.show()
