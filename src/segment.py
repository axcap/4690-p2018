import utils as utils
import numpy as np
import cv2

def segmentText(img):
  #img = cv2.pyrDown(img)
  #small = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
  grad = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
  _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 50))
  connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
  # using RETR_EXTERNAL instead of RETR_CCOMP
  im2, contours, hierarchy = cv2.findContours(connected.copy(),
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)

  mask = np.zeros(bw.shape, dtype=np.uint8)

  boundRect = []
  for idx in range(len(contours)):
    x, y, w, h = cv2.boundingRect(contours[idx])
    mask[y:y+h, x:x+w] = 0
    cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
    r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

    if r > 0.45 and w > 8 and h > 8:
      boundRect.append( (x, y, w, h) )

  return boundRect

def highlightSegments(img, segments):
  # Copy input array as cv2 drawing function work inplace
  temp = np.array(img)
  for (x,y,w,h) in segments:
    cv2.rectangle(temp, (x, y),(x+w, y+h), (0,255,0), 1, 8, 0)

  return temp
