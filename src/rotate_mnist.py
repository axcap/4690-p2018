from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt
import utils as utils
import numpy as np
import struct
import gzip
import cv2
import os

np.set_printoptions(linewidth=9999999)
np.set_printoptions(edgeitems=9999999)

def write_data_to_set(fd_train, fd_lbl, n, images, labels):
  nt = n*28*28
  for x in range(0, n, n//10):
    print("%d/%d" % (x//(n//10), 10))
    print(x, n, n//10, x*28*28, (x+n//10)*28*28)
    fd_train.write(struct.pack(str(n//10*28*28)+'B', *(images[x*28*28:(x+n//10)*28*28])))
    fd_lbl.write(struct.pack(str(n//10)+'B',         *(labels[x:x+n//10])))


#Path where dataset files will be saves
#same path used by MNIST's input_data.read_data_sets(..)
mnist_path   = "res/datasets/MNIST/"
dataset_path = "res/datasets/ROTMNIST/"

# Rotate each char with this angles and save to dataset
angles = [0, 90, 180, 270]

# Scaling font chars to match MNIST image resolution
rows = 28
cols = 28

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

mnist  = input_data.read_data_sets(mnist_path, validation_size=0)
num_train = mnist.train.num_examples
num_test  = mnist.test.num_examples
print(num_train, num_test)

# Create output files needed for database load by MNIST interface
train_img = gzip.open(dataset_path + "train-images-idx3-ubyte.gz", 'wb')
train_lbl = gzip.open(dataset_path + "train-labels-idx1-ubyte.gz", 'wb')
test_img  = gzip.open(dataset_path + "t10k-images-idx3-ubyte.gz",  'wb')
test_lbl  = gzip.open(dataset_path + "t10k-labels-idx1-ubyte.gz",  'wb')

print(4*num_train*28*28, 4*num_test*28*28)

# For train images/labels
train_img.write((2051).to_bytes(4, byteorder='big'))           # magic number
train_img.write((4*num_train).to_bytes(4, byteorder='big'))    # number of images
train_img.write((rows).to_bytes(4, byteorder='big'))           # number of rows
train_img.write((cols).to_bytes(4, byteorder='big'))           # number of columns

train_lbl.write((2049).to_bytes(4, byteorder='big'))           # magic number
train_lbl.write((4*num_train).to_bytes(4, byteorder='big'))    # number of images

# For test images/labels
test_img.write((2051).to_bytes(4, byteorder='big'))             # magic number
test_img.write((4*num_test).to_bytes(4, byteorder='big'))       # number of images
test_img.write((rows).to_bytes(4, byteorder='big'))             # number of rows
test_img.write((cols).to_bytes(4, byteorder='big'))             # number of columns

test_lbl.write((2049).to_bytes(4, byteorder='big'))             # magic number
test_lbl.write((4*num_test).to_bytes(4, byteorder='big'))       # number of images

# Create temp buffers of same shape as out data
train_images_buffer =  np.zeros(4 * num_train * 28*28, dtype = int)
train_labels_buffer =  np.zeros(4 * num_train, dtype = int)

test_images_buffer =  np.zeros(4 * num_test *  28*28, dtype = int)
test_labels_buffer =  np.zeros(4 * num_test, dtype = int)


# For each image in train images
train_imgs = np.ceil(mnist.train.images*255).astype(int)
train_imgs[train_imgs > 0] = 255
total = 0
for i, img in enumerate(train_imgs):
    print("%d/%d processed" % (i*4, 4*num_train))
    img = img.reshape((cols, rows))
    for angle in angles:
      rotated = utils.rotate2angle(img, angle)
      train_images_buffer[((i*4)+angle//90)*28*28:(((i*4+1)+angle//90)*28*28)] = np.ravel(rotated)
      train_labels_buffer[(i*4 + angle//90)] = angle//90
      total += 1

print("Total: ", total)
write_data_to_set(train_img, train_lbl, 4*num_train, train_images_buffer, train_labels_buffer)

train_img.close()
train_lbl.close()

# For each image in test images
test_imgs = np.ceil(mnist.test.images*255).astype(int)
test_imgs[test_imgs > 0] = 255
for i, img in enumerate(test_imgs):
    print("%d/%d processed" % (i, num_test))
    img = img.reshape((cols, rows))
    for angle in angles:
      rotated = utils.rotate2angle(img, angle)
      test_images_buffer[((i*4)+angle//90)*28*28:(((i*4+1)+angle//90)*28*28)] = np.ravel(rotated)
      test_labels_buffer[(i*4 + angle//90)] = angle//90

write_data_to_set(test_img,  test_lbl,  4*num_test,  test_images_buffer,  test_labels_buffer)

# Close files to force file saving (flushing)
test_img.close()
test_lbl.close()

print("DONE")
# Create temp buffers of same shape as out data
del train_images_buffer
del train_labels_buffer
del test_images_buffer
del test_labels_buffer


# Test in MNIST interface can load our dataset
dataset = input_data.read_data_sets(dataset_path)
# Print some public variables
print("Testing")
print(dataset.train.num_examples)
print(dataset.test.num_examples)
print(np.reshape(dataset.train.images[5], (28,28)))
print(dataset.train.labels[5])

while True:
    fig = plt.figure()
    for y in range(4):
        for x in range(4):
            rnd = np.random.randint(low=0, high=num_train)
            img = np.reshape((dataset.train.images[rnd]).astype(int), (28,28))
            lbl = dataset.train.labels[rnd]
            ax = fig.add_subplot(4,4, y*4+x  +1)
            ax.set_ylabel(str(lbl*90))
            ax.imshow(img, cmap='gray')

    plt.show()
