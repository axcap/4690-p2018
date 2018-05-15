'''
Converts font file to MNIST-like dataset
to possibly train NN to better recognize
computer 'written' digits/letters


 TODO
 * Code refactoring
'''

import tensorflow as tf
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

from tensorflow.examples.tutorials.mnist import input_data
from pathlib import Path

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
dataset_path = "res/datasets/FNIST/"

#fontpath = "res/fonts/"
fontpath = "/uio/hume/student-u11/akhsarbg/Downloads/homo/all/"
#fontpath = None # use system default

# Which digits/lettes to export from each font
alphabeth = "0123456789"

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

# Create output files needed for database load by MNIST interface
train_img = gzip.open(dataset_path + "train-images-idx3-ubyte.gz", 'wb')
train_lbl = gzip.open(dataset_path + "train-labels-idx1-ubyte.gz", 'wb')
test_img  = gzip.open(dataset_path + "t10k-images-idx3-ubyte.gz", 'wb')
test_lbl  = gzip.open(dataset_path + "t10k-labels-idx1-ubyte.gz", 'wb')

# Input file headers

# For train images/labels
train_img.write((2051).to_bytes(4, byteorder='big'))            # magic number
train_img.write((tot_images).to_bytes(4, byteorder='big'))        # number of images
train_img.write((rows).to_bytes(4, byteorder='big'))            # number of rows
train_img.write((cols).to_bytes(4, byteorder='big'))            # number of columns

train_lbl.write((2049).to_bytes(4, byteorder='big'))            # magic number
train_lbl.write((tot_images).to_bytes(4, byteorder='big'))        # number of images

# For test images/labels (Currently none, close to empy files)
test_img.write((2051).to_bytes(4, byteorder='big'))             # magic number
test_img.write((0).to_bytes(4, byteorder='big'))  # number of images
test_img.write((rows).to_bytes(4, byteorder='big'))             # number of rows
test_img.write((cols).to_bytes(4, byteorder='big'))             # number of columns

test_lbl.write((2049).to_bytes(4, byteorder='big'))             # magic number
test_lbl.write((0).to_bytes(4, byteorder='big'))                # number of images

# As we dont have any test data, close right after filling the file header
test_img.close()
test_lbl.close()


# For each font in fontpath
for i, fontname in enumerate(os.listdir(fontpath)):
    print("%d/%d - %s" % (i, n_fonts, fontname))
    font = pygame.freetype.Font(fontpath+fontname, 40)
    font.antialiased = False

    # Create temp buffers of same shape as out data
    images_buffer =  np.zeros(alphabeth_lenth*28*28, dtype = int)
    labels_buffer =  np.zeros(alphabeth_lenth, dtype = int)
    curr_images = 0
    # Traverse through alphabeth and save each char data to dataset files
    for index in range(alphabeth_lenth):
        text = font.render(alphabeth[index], fgcolor, bgcolor)[0]
        pygame.transform.scale(text, (cols, rows), dst)

        #font.render_raw_to(img, alphabeth[2], dest=None, rotation=0, size=0, invert=False)
        img = pygame.surfarray.pixels2d(dst)
        #print(index, '\n', data, "\n\n")
        #img = cv2.resize(data, dsize=(rows, cols))
        img[img == 1] = 255
        img = np.transpose(img)

        images_buffer[(index*28*28):(index+1)*28*28] = img.reshape(-1)
        labels_buffer[index] = index
        curr_images += 1

    train_img.write(struct.pack(str(curr_images*28*28)+'B', *images_buffer))
    train_lbl.write(struct.pack(str(curr_images)+'B', *labels_buffer))
    n_images += curr_images


print("Collected: %d/%d images" % (n_images, tot_images))

# Close train files to force file saving (flushing)
train_img.close()
train_lbl.close()

# Test in MNIST interface can load our dataset
dataset = input_data.read_data_sets("res/datasets/FNIST/", validation_size=0)
# Print some public variables
print(dataset.train.labels)
print(dataset.train.num_examples)
print(dataset.test.num_examples)
print(dataset.train.next_batch(3))

print("\n\n\n")
print(np.reshape((dataset.train.images[2]*255).astype(int), (28,28)))
print(dataset.train.labels[2])
