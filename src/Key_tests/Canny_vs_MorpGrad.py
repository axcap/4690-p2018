# import utils as utils
import numpy as np
import cv2

def segmentText(img):
    #img = cv2.pyrDown(img)
    #small = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    grad = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    Canny = cv2.Canny(img,100,200)

    cv2.imwrite('Canny.png', Canny)
    cv2.imwrite('MorpGrad.png', grad)

def main():
    IMAGE_PATH = '../res/images/'
    image_filename = 'ReceiptSwiss.jpg'
    # image_filename = 'lorem.png'

    image = cv2.imread(IMAGE_PATH+image_filename,0)
    segmentText(image)

if __name__ == '__main__':
    main()