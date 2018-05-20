import matplotlib.pyplot as plt
import pygame.freetype
import utils as utils
import numpy as np
import pygame
import cv2
import os

np.set_printoptions(linewidth=9999999)
np.set_printoptions(edgeitems=9999999)

fontpath = "res/fonts/"
#fontpath = "/uio/hume/student-u11/akhsarbg/Downloads/homo/all/"
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
kernel = np.array([[0, 1, 0],
                   [1, 1, 1],
                   [0, 1, 0]], dtype=np.uint8)
print(kernel)
# For each font in fontpath
for i, fontname in enumerate(os.listdir(fontpath)):
    font = pygame.freetype.Font(fontpath+fontname, 30)
    font.antialiased = False

    # Traverse through alphabeth and save each char data to dataset files
    for index in range(alphabeth_lenth):
        char = font.render(alphabeth[index], fgcolor, bgcolor)[0]
        img = pygame.surfarray.pixels2d(char)
        img = np.transpose(img)*255

        center = img.shape[1]
        pad = (28-center)//2
        img = np.pad(img, ((4, 4), (pad, pad)), 'constant', constant_values=0)
        img = cv2.filter2D(img, -1, kernel)
        eroded = cv2.erode(img, kernel, iterations = 1)
        img = img - (img-eroded)//2
        img = cv2.resize(img, (28, 28))
