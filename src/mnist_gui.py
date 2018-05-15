#Draw while holding LMB
#Enter to run through neural network
#RMB to clear one pixel under the cursor
#Middle mouse click to clear screen (auto clear after Enter)

# Imports
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
from nn_mnist import NN_MNIST
from pathlib import Path

import cv2
import numpy as np
import sys, pygame
np.set_printoptions(linewidth=9999999)

def drawPixel(x, y):
    pygame.draw.circle(screen, (0, 0, 0), (x*10, y*10), 10, 0)

def clearPixel(x, y):
    pygame.draw.rect(screen, bgColor, pygame.Rect((x*10, y*10), (10, 10)), 0)

def image2data(img):
    #convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #Fit neural network input size
    img = cv2.resize(img, (28,28))
    #Invert colors and rotate from graph to display coordinates
    img = 255-img.transpose()
    #Create kernel for convolution
    kernel = np.ones((3,3),np.float32)/25
    #Aply 3x3 kernel to smoothen up our image
    img = cv2.filter2D(img,-1,kernel)
    #Convert from 0-255 to 0-1 range
    #img = ((1/256)*img)
    #Normalize image values after smoothening
    img = img * (255/img.max())
    print(img.astype(np.int))
    #Flatten out to feed into network
    img = np.reshape(img, (1, 28*28))
    return img


if __name__ == "__main__":
    pygame.init()
    bgColor = (255, 255, 255)
    lines = False
    size = width, height = 280, 280
    screen = pygame.display.set_mode(size)
    screen.fill(bgColor)


    if lines == True:
        for i in range(84):
            pygame.draw.line(screen, (0,0,0), [i*10, 0], [i*10, 480], 1)
        for i in range(48):
            pygame.draw.line(screen, (0,0,0), [0, i*10], [840, i*10], 1)


    nn = NN_MNIST()
    mnist = input_data.read_data_sets("res/datasets/MNIST/", one_hot=True)
    nn.train(mnist, force_retrain=False)

    pygame.display.update()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    img = pygame.surfarray.array3d(screen)
                    data = image2data(img)
                    r = nn.forward(data)
                    index = np.argmax(r)
                    print("Class: %d - %.2f%%" % (index, r[0, index]*100))

            if event.type == pygame.MOUSEMOTION or event.type == pygame.MOUSEBUTTONDOWN:
                status = pygame.mouse.get_pressed()
                if status[0] == 1:
                    #draw pixel under mouse
                    pos = pygame.mouse.get_pos()
                    x,y = pos[0]//10, pos[1]//10
                    drawPixel(x, y)
                elif status[2] == 1:
                    #clear the screen
                    screen.fill(bgColor)
                pygame.display.update()
