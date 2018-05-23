# import utils as utils
import numpy as np
import cv2

def segmentText(img):
  #img = cv2.pyrDown(img)
  #small = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

  
  _, bw = cv2.threshold(img, 0.0, 255.0, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
  im_floodfill = bw.copy()

  h, w = bw.shape[:2]
  mask = np.zeros((h+2, w+2), np.uint8)
  cv2.floodFill(im_floodfill, mask, (0,0), 255)

  # Floodfill from point (0, 0)
  cv2.floodFill(im_floodfill, mask, (0,0), 255);\
  
  # Invert floodfilled image
  im_floodfill_inv = cv2.bitwise_not(im_floodfill)
  
  # Combine the two images to get the foreground.
  im_out = bw | im_floodfill_inv

  im2, contours, hierarchy = cv2.findContours(im_out.copy(),
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)

  cv2.imshow("Image segment", im_out) 
  cv2.waitKey()
  cv2.destroyAllWindows()
  
  # mask = np.zeros(bw.shape, dtype=np.uint8)
  boundRect = []
  for idx in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[idx])
    # mask[y:y+h, x:x+w] = 0
    # cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
    # r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

    if w > 2 and h > 6:
      boundRect.append( (x, y, w, h) )

  return boundRect

def highlightSegments(img, segments):
  # Copy input array as cv2 drawing function work inplace
  temp = np.array(img)
  for (x,y,w,h) in segments:
    cv2.rectangle(temp, (x, y),(x+w, y+h), (0,255,0), 1, 8, 0)

  return temp

def main():
    IMAGE_PATH = '../res/images/'
    # image_filename = 'ReceiptSwiss.jpg'
    # image_filename = 'lorem.png'
    # image_filename = 'doc.jpg'
    image_filename = 'simpleR.png'

    image = cv2.imread(IMAGE_PATH+image_filename,0)

    rect = segmentText(image)
    show_img = highlightSegments(image,rect)

    cv2.imshow("Image segment", show_img) 
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
  main()