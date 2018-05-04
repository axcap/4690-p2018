from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pygame
import gzip

np.set_printoptions(linewidth=9999999)

dataset_path = "res/datasets/FNIST/"

alphabeth = "0123456789"
#alphabeth = "7"
alphabeth_lenth = len(alphabeth)

rows = 28
cols = 28

#fontpath = "Im Wunderland.otf"
fontpath = None # use system default
background = (0, 0, 0)
forground  = (255, 255,255)

pygame.font.init()
font = pygame.font.Font(fontpath, 40)
dst  = pygame.Surface((cols, rows), 0, 8)


train_img = gzip.open(dataset_path + "train-images-idx3-ubyte.gz", 'wb')
train_lbl = gzip.open(dataset_path + "train-labels-idx1-ubyte.gz", 'wb')

test_img = gzip.open(dataset_path + "t10k-images-idx3-ubyte.gz", 'wb')
test_lbl = gzip.open(dataset_path + "t10k-labels-idx1-ubyte.gz", 'wb')


train_img.write((2051).to_bytes(4, byteorder='big'))             # magic number 
train_img.write((alphabeth_lenth).to_bytes(4, byteorder='big'))  # number of images 
train_img.write((rows).to_bytes(4, byteorder='big'))             # number of rows 
train_img.write((cols).to_bytes(4, byteorder='big'))             # number of columns

train_lbl.write((2049).to_bytes(4, byteorder='big'))             # magic number 
train_lbl.write((alphabeth_lenth).to_bytes(4, byteorder='big'))  # number of images 


test_img.write((2051).to_bytes(4, byteorder='big'))             # magic number 
test_img.write((0).to_bytes(4, byteorder='big'))  # number of images 
test_img.write((rows).to_bytes(4, byteorder='big'))             # number of rows 
test_img.write((cols).to_bytes(4, byteorder='big'))             # number of columns

test_lbl.write((2049).to_bytes(4, byteorder='big'))             # magic number 
test_lbl.write((0).to_bytes(4, byteorder='big'))  # number of images 

test_img.close()
test_lbl.close()

for index in range(alphabeth_lenth):
    text = font.render(alphabeth[index], True, forground, background)
    pygame.transform.scale(text, (cols, rows), dst)


    #pygame way
    #img = pygame.PixelArray(dst)

    #numpy way (pretty print)
    img = pygame.surfarray.pixels2d(dst)
    img = np.transpose(img)
    #print(img)
    #print(type(img))
    train_img.write(img)
    train_lbl.write(bytes([index]))

train_img.close()
train_lbl.close()

dataset = input_data.read_data_sets("res/datasets/FNIST/",
                                    validation_size=0)

print(dataset.train.labels)
print(dataset.train.num_examples)
print(dataset.test.num_examples)
print(dataset.train.next_batch(3))
