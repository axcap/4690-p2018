'''
Converts font file to MNIST-like dataset
to possibly train NN to better recognize
computer 'written' digits/letters  


 TODO
 * Traverse multiple font files from directory
 * Code refactoring
'''

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pygame
import gzip

np.set_printoptions(linewidth=9999999)

#Path where dataset files will be saves
#same path used by MNIST's input_data.read_data_sets(..)
dataset_path = "res/datasets/FNIST/"

# Which digits/lettes to export from each font
alphabeth = "0123456789"

# Number of entries in dataset 
alphabeth_lenth = len(alphabeth)

# Scaling font chars to match MNIST image resolution
rows = 28
cols = 28

#fontpath = "Im Wunderland.otf"
fontpath = None # use system default
# Black background
background = (0, 0, 0)
# White background
forground  = (255, 255,255)

# Using pyGame library to render font chars
pygame.font.init()
font = pygame.font.Font(fontpath, 40)
dst  = pygame.Surface((cols, rows), 0, 8)


# Create output files needed for database load by MNIST interface 
train_img = gzip.open(dataset_path + "train-images-idx3-ubyte.gz", 'wb')
train_lbl = gzip.open(dataset_path + "train-labels-idx1-ubyte.gz", 'wb')

test_img = gzip.open(dataset_path + "t10k-images-idx3-ubyte.gz", 'wb')
test_lbl = gzip.open(dataset_path + "t10k-labels-idx1-ubyte.gz", 'wb')

# Input file headers

# For train images/labels
train_img.write((2051).to_bytes(4, byteorder='big'))             # magic number 
train_img.write((alphabeth_lenth).to_bytes(4, byteorder='big'))  # number of images 
train_img.write((rows).to_bytes(4, byteorder='big'))             # number of rows 
train_img.write((cols).to_bytes(4, byteorder='big'))             # number of columns

train_lbl.write((2049).to_bytes(4, byteorder='big'))             # magic number 
train_lbl.write((alphabeth_lenth).to_bytes(4, byteorder='big'))  # number of images 

# For test images/labels (Currently none, close to empy files)
test_img.write((2051).to_bytes(4, byteorder='big'))             # magic number 
test_img.write((0).to_bytes(4, byteorder='big'))  # number of images 
test_img.write((rows).to_bytes(4, byteorder='big'))             # number of rows 
test_img.write((cols).to_bytes(4, byteorder='big'))             # number of columns

test_lbl.write((2049).to_bytes(4, byteorder='big'))             # magic number 
test_lbl.write((0).to_bytes(4, byteorder='big'))  # number of images 


# As we dont have any test data, close right after filling the file header
test_img.close()
test_lbl.close()

# Traverse through alphabeth and save each char data to dataset files
for index in range(alphabeth_lenth):
    text = font.render(alphabeth[index], True, forground, background)
    pygame.transform.scale(text, (cols, rows), dst)

    img = pygame.surfarray.pixels2d(dst)
    img = np.transpose(img)
    train_img.write(img)
    train_lbl.write(bytes([index]))
    #print(img)
       

# Close train files to force file saving (flushing)
train_img.close()
train_lbl.close()

# Test in MNIST interface can load our dataset
dataset = input_data.read_data_sets("res/datasets/FNIST/",
                                    validation_size=0)
# Print some public variables 
print(dataset.train.labels)
print(dataset.train.num_examples)
print(dataset.test.num_examples)
print(dataset.train.next_batch(3))
