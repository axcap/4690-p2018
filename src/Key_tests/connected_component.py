# import utils as utils
import numpy as np
import cv2

def find_combonent_median(img):
  #img = cv2.pyrDown(img)
  #small = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

  
  _, bw = cv2.threshold(img, 0.0, 255.0, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
  # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
  # bw = cv2.morphologyEx(bw,cv2.MORPH_CLOSE,kernel)

  output = cv2.connectedComponentsWithStats(bw, 4, cv2.CV_32S)
  stats = output[2]
  areas = stats[1:,cv2.CC_STAT_AREA]
  median = np.median(areas)

  cv2.imshow("Image segment", bw) 
  cv2.waitKey()
  cv2.destroyAllWindows()

  print(median)
  return median

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
    image_filename = 'Android_image.jpg'

    image = cv2.imread(IMAGE_PATH+image_filename,0)

    find_combonent_median(image)
    # show_img = highlightSegments(image,rect)

    # cv2.imshow("Image segment", show_img) 
    # cv2.waitKey()
    # cv2.destroyAllWindows()

if __name__ == '__main__':
  main()