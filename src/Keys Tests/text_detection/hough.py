import cv2 
import numpy as np 
from skimage.filters.rank import median
from skimage.morphology import disk


def main():
    IMAGE_PATH = '../../../res/images/'
    image_filename = 'ReceiptSwiss.jpg'
    image = cv2.imread(IMAGE_PATH+image_filename,0)

    edges = cv2.Canny(image,50,150,apertureSize = 3)
    img = image.copy()
    lines = cv2.HoughLinesP(edges,1,np.pi/180,200,1000,20)
    print(lines)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.imshow("text", img) 
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()