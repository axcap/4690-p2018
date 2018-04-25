#Draw while holding LMB
#Enter to run through neural network
#RMB to clear one pixel under the cursor
#Middle mouse click to clear screen (auto clear after Enter)

# Imports
from matplotlib import pyplot as plt
import cv2
from pathlib import Path
import numpy as np
import sys, pygame
from nn_mnist import NN_MNIST

def drawPixel(x, y):
    pygame.draw.circle(screen, (0, 0, 0), (x*10, y*10), 10, 0)

def clearPixel(x, y):
    pygame.draw.rect(screen, bgColor, pygame.Rect((x*10, y*10), (10, 10)), 0)    

    
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


    nn = NN_MNIST()
    nn.train(None, force_retrain=False)
    
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
                    r = nn.forward(img)
                    print(np.argmax(r))
                    
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
                
