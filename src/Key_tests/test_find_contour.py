import cv2
import numpy as np 
 

def findContours(image): 
    # read and scale down image
    # this is not right imprementation, need redo!! 

    # Threshold image, this need to be can be done by finding image
    # ret, threshed_img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    img = image.copy()
    
    # find contours, this is also need more experimenting, green theorem are used to find contour. 
    edge_img = cv2.Canny(img,100,200)
    kernel = np.ones((3,3),np.uint8)
    morph_img = cv2.morphologyEx(edge_img, cv2.MORPH_CLOSE, kernel)

    img, contours, hier = cv2.findContours(morph_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_found = []

    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)

        # the treshold need to be evaluated, with histogram
        # if w >= 15 and (h >= 30 and h <= 40):
        #     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
        #     digitFound.append(c)
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 1)
        contour_found.append(image[y:y+h, x:x+w])
  
    
    cv2.imshow("Image segment", image) 
    cv2.waitKey()
    cv2.destroyAllWindows()

    return contour_found

def main():
    img = cv2.imread('../../res/images/text_example1.png', 0)

if __name__ == '__main__':
    main()