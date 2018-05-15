import cv2
import numpy as np
import math



def detect_paper(image):
    edges = cv2.Canny(image,50,150,apertureSize = 3)

    minLineLength = 500
    maxLineGap = 50
    lines = cv2.HoughLinesP(edges,1,np.pi/180,15,minLineLength,maxLineGap)
    for x in range(0, len(lines)):
        for x1,y1,x2,y2 in lines[x]:
            cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.imwrite('houghlines3.jpg', image)


def main():

    IMAGE_PATH = '../../res/images/'
    image_filename = 'ReceiptSwiss.jpg'
    image = cv2.imread(IMAGE_PATH+image_filename,0)
    detect_paper(image)

if __name__ == '__main__':
    main()