import tensorflow as tf

import sys
sys.path.append('src')

from tensorflow.examples.tutorials.mnist import input_data
from pathlib import Path

import utils as utils
import matplotlib.pyplot as plt
import pygame.freetype
import numpy as np
import struct
import pygame
import time
import gzip
import sys
import cv2
import os

np.set_printoptions(linewidth=9999999)
np.set_printoptions(edgeitems=9999999)

#Path where dataset files will be saves
#same path used by MNIST's input_data.read_data_sets(..)
dataset_path = sys.argv[1] #"res/datasets/ROTFNIST/"

fontpath = "res/fonts/"
#fontpath = "/uio/hume/student-u11/akhsarbg/Downloads/homo/all/"
#fontpath = None # use system default

# Which digits/lettes to export from each font
alphabeth = "0123456789"

# Number of entries in dataset
alphabeth_lenth = len(alphabeth)

# Rotate each char with this angles and save to dataset
angles = [0, 90, 180, 270]

# Scaling font chars to match MNIST image resolution
rows = 28
cols = 28

# Black background
bgcolor = None
# White background
fgcolor  = (255, 255,255)

def write_data_to_set(fd_train, fd_test, n, images, labels):
  fd_train.write(struct.pack(str(n*28*28)+'B', *images))
  fd_test.write(struct.pack(str(n)+'B',       *labels))


# Using pyGame library to render font chars
pygame.freetype.init()
#font = pygame.font.Font(fontpath, 40)
dst = pygame.Surface((cols, rows), 0, 8)

# Number of fonts in fontpath (expect only font files in fontpath)
n_fonts = 1 if fontpath is None else len(next(os.walk(fontpath))[2])
# Will contain total number of chars read collected
tot_expected = n_fonts * len(alphabeth) * len(angles)
tot_images   = 0
# Each div'th char goes to test set
div = 6

chars_in_buff = 50000

n_test = int(np.ceil(tot_expected/div))
n_train = int(tot_expected - n_test)

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Create output files needed for database load by MNIST interface
train_img = gzip.open(dataset_path + "train-images-idx3-ubyte.gz", 'wb')
train_lbl = gzip.open(dataset_path + "train-labels-idx1-ubyte.gz", 'wb')
test_img  = gzip.open(dataset_path + "t10k-images-idx3-ubyte.gz",  'wb')
test_lbl  = gzip.open(dataset_path + "t10k-labels-idx1-ubyte.gz",  'wb')

# Input file headers

# For train images/labels
train_img.write((2051).to_bytes(4, byteorder='big'))            # magic number
train_img.write((n_train).to_bytes(4, byteorder='big'))        # number of images
train_img.write((rows).to_bytes(4, byteorder='big'))            # number of rows
train_img.write((cols).to_bytes(4, byteorder='big'))            # number of columns

train_lbl.write((2049).to_bytes(4, byteorder='big'))            # magic number
train_lbl.write((n_train).to_bytes(4, byteorder='big'))        # number of images

# For test images/labels (Currently none, close to empy files)
test_img.write((2051).to_bytes(4, byteorder='big'))             # magic number
test_img.write((n_test).to_bytes(4, byteorder='big'))  # number of images
test_img.write((rows).to_bytes(4, byteorder='big'))             # number of rows
test_img.write((cols).to_bytes(4, byteorder='big'))             # number of columns

test_lbl.write((2049).to_bytes(4, byteorder='big'))             # magic number
test_lbl.write((n_test).to_bytes(4, byteorder='big'))                # number of images

# Create temp buffers of same shape as out data
train_images_buffer =  np.zeros(chars_in_buff * 28*28, dtype = int)
train_labels_buffer =  np.zeros(chars_in_buff, dtype = int)

test_images_buffer =  np.zeros(chars_in_buff *  28*28, dtype = int)
test_labels_buffer =  np.zeros(chars_in_buff, dtype = int)

train_idx = 0
test_idx  = 0
t_train = 0

# For each font in fontpath
for i, fontname in enumerate(os.listdir(fontpath)):
    print("%d/%d - %s" % (i, n_fonts, fontname))
    font = pygame.freetype.Font(fontpath+fontname, 40)
    font.antialiased = False

    # Traverse through alphabeth and save each char data to dataset files
    for index in range(alphabeth_lenth):
        text = font.render(alphabeth[index], fgcolor, bgcolor)[0]
        for angle in angles:
            rotated = pygame.transform.rotate(text, angle)
            img = pygame.surfarray.pixels2d(rotated)
            if img.size == 0: continue
            img *= 255
            img = np.transpose(img)
            img = utils.img2data(img)
            print(img)

            if n_test > 0 and ((i + index + angle//90) % div) == 0:
              test_images_buffer[test_idx*28*28:(test_idx+1)*28*28] = np.ravel(img)
              test_labels_buffer[test_idx] = angle//90
              test_idx += 1
              if test_idx == chars_in_buff:
                write_data_to_set(test_img, test_lbl, test_idx, test_images_buffer, test_labels_buffer)
                test_images_buffer[:]=0
                test_labels_buffer[:]=0
                test_idx = 0
            else:
              train_images_buffer[(train_idx*28*28):(train_idx+1)*28*28] = img.reshape(-1)
              train_labels_buffer[train_idx] = angle//90
              train_idx += 1
              t_train += 1

              if train_idx == chars_in_buff:
                write_data_to_set(train_img, train_lbl, train_idx, train_images_buffer, train_labels_buffer)
                train_images_buffer[:]=0
                train_labels_buffer[:]=0
                train_idx = 0

            tot_images += 1

print("Got all: ", tot_expected == tot_images)
write_data_to_set(test_img, test_lbl, test_idx, test_images_buffer[:test_idx*28*28], test_labels_buffer[:test_idx])
write_data_to_set(train_img, train_lbl, train_idx, train_images_buffer[:train_idx*28*28], train_labels_buffer[:train_idx])

print("Collected: %d/%d images" % (t_train, n_train))

# Close files to force file saving (flushing)
train_img.close()
train_lbl.close()
test_img.close()
test_lbl.close()

# Test in MNIST interface can load our dataset
dataset = input_data.read_data_sets(dataset_path, validation_size=0)
# Print some public variables
#print("Lables: \n", dataset.train.labels)
print(dataset.train.num_examples)
print(dataset.test.num_examples)
print(np.reshape(dataset.train.images[5], (28,28)))
print(dataset.train.labels[5])

while True:
    fig = plt.figure()
    for y in range(4):
        for x in range(4):
            rnd = np.random.randint(low=0, high=n_train)
            print("N_train: ", n_train)
            print("Rand: ", rnd)
            img = np.reshape((dataset.train.images[rnd]).astype(int), (28,28))
            lbl = dataset.train.labels[rnd]
            print(img)
            #fig.suptitle(, fontsize=16)
            ax = fig.add_subplot(4,4, y*4+x  +1)
            ax.set_ylabel(str(lbl*90))
            ax.imshow(img, cmap='gray')

    plt.show()
