import utils as utils
import numpy as np
import cv2

def findBoundingRect(img, contours, min_w=8, min_h=8, min_r = 0.45):
  boundRect = []
  mask = np.zeros(img.shape, dtype=np.uint8)

  for idx in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[idx])

    mask[y:y+h, x:x+w] = 0
    cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
    r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

    if r > min_r and w > min_w and h > min_h:
      boundRect.append( (x, y, w, h) )

  return boundRect

def segmentText(img, point = (21,21)):
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
  grad = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
  _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, point)
  connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
  # using RETR_EXTERNAL instead of RETR_CCOMP
  im2, contours, hierarchy = cv2.findContours(connected.copy(),
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)
  return findBoundingRect(img, contours)

def segmentLetters(image):
  [M,N] = image.shape

  kernel = np.ones((5,1), np.uint8)
  img = cv2.dilate(image, kernel)

  #img = image
  # invert since we are working black on white
  _, tresh_img = cv2.threshold(img, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  # tresh_img[0] = 0
  # tresh_img[M-1] = 0
  #tresh_img = img
  
  im_floodfill = tresh_img.copy()

  h, w = img.shape[:2]
  mask = np.zeros((h+2, w+2), np.uint8)
  cv2.floodFill(im_floodfill, mask, (0,0), 255)

  
  # Invert floodfilled image
  im_floodfill_inv = cv2.bitwise_not(im_floodfill)
  
  # Combine the two images to get the foreground.
  im_out = tresh_img | im_floodfill_inv  
  
  _, contours, __ = cv2.findContours(im_out.copy(),
                                     cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_NONE)

  return findBoundingRect(img, contours, min_h=10, min_w=0, min_r=0)


def highlightSegments(text, img, segments):
  # Copy input array as cv2 drawing function work inplace
  temp = img.copy()
  for (x,y,w,h) in segments:
    cv2.rectangle(temp, (x, y),(x+w, y+h), (255,255,255), 1, 8, 0)
  utils.imshow(text, temp)

def main():
  SAVE_IMAGE_PATH = '../doc/res/'
  IMAGE_PATH = '../res/images/'
  image_filename = 'ReceiptSwiss.jpg'
  # image_filename = 'lorem.png'
  # image_filename = 'doc.jpg'

  image = cv2.imread(IMAGE_PATH+image_filename,0)

  text_regions = segmentText(image)
  text_regions = highlightSegments(image,text_regions)

  cv2.imwrite(SAVE_IMAGE_PATH + "segment_text1.png", text_regions)

  cv2.imshow("text_regions", text_regions)
  cv2.waitKey()
  cv2.destroyAllWindows()

  image_filename = 'lorem.png'
  image = cv2.imread(IMAGE_PATH+image_filename,0)

  rect = segmentsLetters(image)
  show_img = highlightSegments(image,rect)

  cv2.imshow("Image segment", show_img)
  cv2.waitKey()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()
