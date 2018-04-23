#Draw while holding LMB
#Enter to run through neural network
#RMB to clear one pixel under the cursor
#Middle mouse click to clear screen (auto clear after Enter)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
from matplotlib import pyplot as plt
import cv2
from pathlib import Path
import numpy as np
import tensorflow as tf
import sys, pygame

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data


def drawPixel(x, y):
    pygame.draw.rect(screen, (0,0,0), pygame.Rect((x*10, y*10), (10, 10)), 0)    

def clearPixel(x, y):
    pygame.draw.rect(screen, bgColor, pygame.Rect((x*10, y*10), (10, 10)), 0)    

def initGraph():
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    y_ = tf.placeholder(tf.float32, shape = [None, 10])
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    sess = tf.InteractiveSession()
    return y, x, sess
    
if __name__ == "__main__":
    
    pygame.init()
    
    bgColor = (255, 255, 255)
    lines = False
    size = width, height = 280, 280
    screen = pygame.display.set_mode(size)
    screen.fill(bgColor)
    pygame.display.update()

    if lines == True:
        for i in range(84):
            pygame.draw.line(screen, (0,0,0), [i*10, 0], [i*10, 480], 1)
        for i in range(48):
            pygame.draw.line(screen, (0,0,0), [0, i*10], [840, i*10], 1)


    y_, x_ , sess = initGraph()
    sess = tf.InteractiveSession()

    path = "../res/model/mnist_demo"
    saver = tf.train.Saver()
    if Path(path+".index").is_file():
        saver.restore(sess, path)
        print("Model restored")
    else:
        print("Model not found")
        exit(1)

    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    img = pygame.surfarray.array3d(screen)
                    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) 
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)                    
                    img = cv2.resize(img, (28,28))
                    img = ((1/256)* (255-img))
                    img = np.reshape(img, (1, 28*28))
                    r = sess.run(y_, feed_dict={x_: img})
                    print(r)
                    print(np.argmax(r))
                    
            if event.type == pygame.MOUSEMOTION or event.type == pygame.MOUSEBUTTONDOWN:
                status = pygame.mouse.get_pressed()
                if status[0] == 1:
                    #draw pixel under mouse
                    pos = pygame.mouse.get_pos()
                    x,y = pos[0]//10, pos[1]//10
                    drawPixel(x, y)
                elif status[1] == 1:
                    #clear the screen
                    screen.fill(bgColor)
                elif status[2] == 1:
                    #clear pixel under mouse
                    pos = pygame.mouse.get_pos()
                    x,y = pos[0]//10, pos[1]//10
                    clearPixel(x, y)
                pygame.display.update()

