import utils as utils
import numpy as np
import cv2

<<<<<<< HEAD
<<<<<<< HEAD
def findBoundingRect(img, contours, min_w=8, min_h=8, min_r = 0.35):
=======
def findBoundingRect(img, contours, min_w=8, min_h=8, min_r = 0.25):
>>>>>>> 968d29c916de6aa01f8830a1c873a695405f7094
=======
def findBoundingRect(img, contours, min_w=8, min_h=8, min_r = 0.25):
>>>>>>> 968d29c916de6aa01f8830a1c873a695405f7094
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

def segmentText(img, point = (21, 21)):
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

<<<<<<< HEAD
def segmentLetters(image):
  [M,N] = image.shape

  kernel = np.ones((M//2,1), np.uint8)
  img = cv2.dilate(image, kernel)
  #utils.imshow("dilate", img)

  #img = image
  # invert since we are working black on white
  _, tresh_img = cv2.threshold(img, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  tresh_img[0] = 0
  tresh_img[M-1] = 0
  #tresh_img = img
=======
def segmentLetters(img):
  [M,N] = img.shape

  # invert since we are working black on white
  _, tresh_img = cv2.threshold(img, 0.0, 255.0, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

  kernel = np.ones((M//8,1), np.uint8)
  tresh_img = cv2.dilate(tresh_img, kernel)

  tresh_img[0] = 0
  tresh_img[M-1] = 0
<<<<<<< HEAD
>>>>>>> 968d29c916de6aa01f8830a1c873a695405f7094
=======
>>>>>>> 968d29c916de6aa01f8830a1c873a695405f7094

  im_floodfill = tresh_img.copy()
  #print(im_floodfill)
  #utils.imshow("food", im_floodfill)

  h, w = img.shape[:2]
  mask = np.zeros((h+2, w+2), np.uint8)
  cv2.floodFill(im_floodfill, mask, (0,0), 255)


  # Invert floodfilled image
  im_floodfill_inv = cv2.bitwise_not(im_floodfill)

  # Combine the two images to get the foreground.
  im_out = tresh_img | im_floodfill_inv

  _, contours, __ = cv2.findContours(im_out.copy(),
<<<<<<< HEAD
                                     cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_NONE)

  return findBoundingRect(img, contours, min_h=10, min_w=0, min_r=0)

<<<<<<< HEAD
=======
=======
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)

  return findBoundingRect(img, contours, min_h=10, min_w=2, min_r=0)

>>>>>>> 968d29c916de6aa01f8830a1c873a695405f7094
def find_largest_component_rect(b_img):
  """ 
  input: binaery image
  return: coordinate of largest compnent
  """

  # add boundary
  [M,N] = b_img.shape
  b_img[0] = 0
  b_img[M-1] = 0

  b_img = b_img.astype(np.uint8)
  output = cv2.connectedComponentsWithStats(b_img, 4, cv2.CV_32S)
  stats = output[2]
  areas = stats[1:,cv2.CC_STAT_AREA]
  heights = stats[1:,cv2.CC_STAT_HEIGHT]
  widhts = stats[1:,cv2.CC_STAT_WIDTH]

  i = np.argmax(areas)
  x = stats[i+1, cv2.CC_STAT_LEFT]
  y = stats[i+1, cv2.CC_STAT_TOP]
  w = stats[i+1, cv2.CC_STAT_WIDTH]
  h = stats[i+1, cv2.CC_STAT_HEIGHT]

  return [x,y,w,h]
  # return orig[y:y+h, x:x+w] 

def segmentPaper(img):
  """ 
  we assume the paper is white and s the larges component in the image
  """
  
  canny = cv2.Canny(img, 100, 200, apertureSize=3)
  lines = cv2.HoughLines(canny,1,np.pi/180,200)
  
  bw = np.ones(img.shape)

  for line in lines:
    for rho,theta in line:
      a = np.cos(theta)
      b = np.sin(theta)
      x0 = a*rho
      y0 = b*rho
      x1 = int(x0 + 2000*(-b))
      y1 = int(y0 + 2000*(a))
      x2 = int(x0 - 2000*(-b))
      y2 = int(y0 - 2000*(a))

      cv2.line(bw,(x1,y1),(x2,y2),0,2)

  [x,y,w,h] = find_largest_component_rect(bw)
  return img[y:y+h, x:x+w]


<<<<<<< HEAD
>>>>>>> 968d29c916de6aa01f8830a1c873a695405f7094
=======
>>>>>>> 968d29c916de6aa01f8830a1c873a695405f7094

def highlightSegments(text, img, segments):
  # Copy input array as cv2 drawing function work inplace
  temp = img.copy()
  for (x,y,w,h) in segments:
<<<<<<< HEAD
<<<<<<< HEAD
    cv2.rectangle(temp, (x, y),(x+w, y+h), (255,255,255), 3, 8, 0)
  utils.imshow(text, temp)
=======
=======
>>>>>>> 968d29c916de6aa01f8830a1c873a695405f7094
    cv2.rectangle(temp, (x, y),(x+w, y+h), (0,0,255), 1, 8, 0)

  utils.imshow(text,temp)
  return temp
>>>>>>> 968d29c916de6aa01f8830a1c873a695405f7094


def _DEMO_segment_text():
  SAVE_IMAGE_PATH = 'doc/res/'
  IMAGE_PATH = 'res/images/'
  image_filename1 = 'text_skew.png'
  image_filename2 = 'Android_image.jpg'
  image_filename3 = 'bad.jpg'

  image1 = cv2.imread(IMAGE_PATH+image_filename1,0)
  image2 = cv2.imread(IMAGE_PATH+image_filename2,0)
  image3 = cv2.imread(IMAGE_PATH+image_filename3,0)

  # segment paper not finish
  paper_region1 = image1 #segmentPaper(image1)
  paper_region2 = image2 #segmentPaper(image2)
  paper_region3 = image3 #segmentPaper(image3)

  text_regions1 = segmentText(paper_region1)
  text_regions2 = segmentText(paper_region2)
  text_regions3 = segmentText(paper_region3)

  demo_image1 = highlightSegments("demo_image1",paper_region1,text_regions1)
  demo_image2 = highlightSegments("demo_image2",paper_region2,text_regions2)
  demo_image3 = highlightSegments("demo_image3",paper_region3,text_regions3)

  cv2.imwrite(SAVE_IMAGE_PATH + "segment_text1.png", demo_image1)
  cv2.imwrite(SAVE_IMAGE_PATH + "segment_text2.png", demo_image2)
  cv2.imwrite(SAVE_IMAGE_PATH + "segment_text3.png", demo_image3)


def _DEMO_segment_letter():
  SAVE_IMAGE_PATH = 'doc/res/'
  IMAGE_PATH = 'res/images/'
  image_filename = 'simpleR2.png'
  image = cv2.imread(IMAGE_PATH+image_filename,0)
  letter_rect = segmentLetters(image)
  demo_image = highlightSegments("demo_letter_image", image, letter_rect)


def main():
  _DEMO_segment_text()
  _DEMO_segment_letter()

if __name__ == '__main__':
  main()
