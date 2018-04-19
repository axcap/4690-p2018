import cv2
import numpy as np 
 
# read and scale down image
img = cv2.imread('../../res/images/numbers.png', 0)

# Threshold image, this need to be can be done by finding image
ret, threshed_img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)


# find contours, this is also need more experimenting, green theorem are used to find contour. 
image, contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

digitFound = []
for c in contours:
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(c)

    # the treshold need to be evaluated, with histogram
    if w >= 15 and (h >= 30 and h <= 40):
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
        digitFound.append(c)

print(len(digitFound))
 
cv2.imshow("Image segment", img) 
cv2.waitKey()
cv2.destroyAllWindows()
