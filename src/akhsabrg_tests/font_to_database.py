'''
Converts font file to MNIST-like dataset
to possibly train NN to better recognize
computer 'written' digits/letters
'''

import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.examples.tutorials.mnist import input_data
from pathlib import Path

import sys
sys.path.append('src')

import matplotlib.pyplot as plt
import pygame.freetype
import utils as utils
import numpy as np
import struct
import pygame
import time
import gzip
import cv2
import os

np.set_printoptions(linewidth=9999999)
np.set_printoptions(edgeitems=9999999)

def write_data_to_set(fd_train, fd_test, n, images, labels):
  fd_train.write(struct.pack(str(n*28*28)+'B', *images))
  fd_test.write(struct.pack(str(n)+'B',        *labels))

#Path where dataset files will be saves
#same path used by MNIST's input_data.read_data_sets(..)
dataset_path = "res/datasets/FNIST/"

#fontpath = "res/fonts/"
fontpath = "/run/media/akhsarbg/47E8-126A/homo/all/"
#fontpath = None # use system default

# Which digits/lettes to export from each font
alphabeth = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# Number of entries in dataset
alphabeth_lenth = len(alphabeth)

# Scaling font chars to match MNIST image resolution
rows = 28
cols = 28

# Black background
bgcolor = None
# White background
fgcolor  = (255, 255,255)

# Using pyGame library to render font chars
pygame.freetype.init()
#font = pygame.font.Font(fontpath, 40)
dst  = pygame.Surface((cols, rows), 0, 8)

# Number of fonts in fontpath (expect only font files in fontpath)
n_fonts = 1 if fontpath is None else len(next(os.walk(fontpath))[2])
# Will contain total number of chars read from font
n_images   = 0
tot_images = n_fonts*alphabeth_lenth

# First loop to count total number of chars, find how much of it will be test etc.
for fontname in os.listdir(fontpath):
    font = pygame.freetype.Font(fontpath+fontname, 40)
    font.antialiased = False

    # Traverse through alphabeth and save each char data to dataset files
    for index in range(alphabeth_lenth):
        text = font.render(alphabeth[index], fgcolor, bgcolor)[0]
        img = pygame.surfarray.pixels2d(text)
        if img.size == 0:
          continue
        else:
          n_images += 1

if n_images != tot_images:
  print("Found incomplete font(s), continue? y/n: ", end="")
  yn = input()
  if(yn == "n" or yn == "no"):
    exit()

tot_images = n_images
n_images   = 0
div        = 6 # every 6th image goes to test collection
n_test     = tot_images // div
n_train    = tot_images - n_test

# Create output files needed for database load by MNIST interface
train_img = gzip.open(dataset_path + "train-images-idx3-ubyte.gz", 'wb')
train_lbl = gzip.open(dataset_path + "train-labels-idx1-ubyte.gz", 'wb')
test_img  = gzip.open(dataset_path + "t10k-images-idx3-ubyte.gz", 'wb')
test_lbl  = gzip.open(dataset_path + "t10k-labels-idx1-ubyte.gz", 'wb')

# Input file headers
# For train images/labels
train_img.write((2051).to_bytes(4, byteorder='big'))            # magic number
train_img.write((n_train).to_bytes(4, byteorder='big'))         # number of images
train_img.write((rows).to_bytes(4, byteorder='big'))            # number of rows
train_img.write((cols).to_bytes(4, byteorder='big'))            # number of columns

train_lbl.write((2049).to_bytes(4, byteorder='big'))            # magic number
train_lbl.write((n_train).to_bytes(4, byteorder='big'))         # number of images

# For test images/labels
test_img.write((2051).to_bytes(4, byteorder='big'))             # magic number
test_img.write((n_test).to_bytes(4, byteorder='big'))           # number of images
test_img.write((rows).to_bytes(4, byteorder='big'))             # number of rows
test_img.write((cols).to_bytes(4, byteorder='big'))             # number of columns

test_lbl.write((2049).to_bytes(4, byteorder='big'))             # magic number
test_lbl.write((n_test).to_bytes(4, byteorder='big'))           # number of images

# Hold ut to chars_in_buff chars in ram before writing to disk
chars_in_buff = 50000

# Create temp buffers of same shape as out data
train_images_buffer =  np.zeros(chars_in_buff * 28*28, dtype = int)
train_labels_buffer =  np.zeros(chars_in_buff, dtype = int)

test_images_buffer =  np.zeros(chars_in_buff *  28*28, dtype = int)
test_labels_buffer =  np.zeros(chars_in_buff, dtype = int)

train_idx = 0
test_idx  = 0
t_train = 0

kernel = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]], dtype=np.uint8)

# For each font in fontpath
for i, fontname in enumerate(os.listdir(fontpath)):
    print("%d/%d - %s" % (i, n_fonts, fontname))
    font = pygame.freetype.Font(fontpath+fontname, 60)
    font.antialiased = False

    # Traverse through alphabeth and save each char data to dataset files
    for index in range(alphabeth_lenth):
        char = font.render(alphabeth[index], fgcolor, bgcolor)[0]
        img = pygame.surfarray.pixels2d(char)
        # pygames scale segfault on empty surface
        #pygame.transform.scale(char, (cols, rows), dst)
        if img.size == 0:
          continue

        img = np.transpose(img)*255

        img = utils.img2data(img)
        label = utils.char2class(alphabeth[index])

        if ((i*alphabeth_lenth + index) % div) == 0:
          test_images_buffer[test_idx*28*28:(test_idx+1)*28*28] = np.ravel(img)
          test_labels_buffer[test_idx] = label
          test_idx += 1
          if test_idx == chars_in_buff:
            write_data_to_set(test_img, test_lbl, test_idx,
                              test_images_buffer, test_labels_buffer)
            test_images_buffer[:]=0
            test_labels_buffer[:]=0
            test_idx = 0
        else:
          train_images_buffer[(train_idx*28*28):(train_idx+1)*28*28] = img.reshape(-1)
          train_labels_buffer[train_idx] = label
          train_idx += 1

          if train_idx == chars_in_buff:
            write_data_to_set(train_img, train_lbl, train_idx,
                              train_images_buffer, train_labels_buffer)
            train_images_buffer[:]=0
            train_labels_buffer[:]=0
            train_idx = 0

        n_images += 1

write_data_to_set(test_img, test_lbl, test_idx,
                  test_images_buffer[:test_idx*28*28], test_labels_buffer[:test_idx])
write_data_to_set(train_img, train_lbl, train_idx,
                  train_images_buffer[:train_idx*28*28], train_labels_buffer[:train_idx])

# Free ram
test_images_buffer[:]=0
test_labels_buffer[:]=0
train_images_buffer[:]=0
train_labels_buffer[:]=0

print("Collected: %d/%d images" % (n_images, tot_images))

# Close train files to force file saving (flushing)
train_img.close()
train_lbl.close()
test_img.close()
test_lbl.close()

# Test in MNIST interface can load our dataset
dataset = input_data.read_data_sets("res/datasets/FNIST/")

# Print some public variables
print(dataset.train.num_examples)
print(dataset.test.num_examples)
print(np.reshape(dataset.train.images[-1], (28,28)))
print(dataset.train.labels[-1]*255)

while True:
    fig = plt.figure()
    for y in range(4):
        for x in range(4):
            rnd = np.random.randint(low=0, high=dataset.train.num_examples)
            img = np.reshape((dataset.train.images[rnd]).astype(int), (28,28))
            lbl = dataset.train.labels[rnd]
            ax = fig.add_subplot(4,4, y*4+x  +1)
            ax.set_ylabel(str(lbl))
            ax.imshow(img, cmap='gray')

    plt.show()
