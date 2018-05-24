# import utils as utils
import numpy as np
import cv2

def find_combonent_median(img):
  _, bw = cv2.threshold(img, 0.0, 255.0, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

  output = cv2.connectedComponentsWithStats(bw, 4, cv2.CV_32S)
  stats = output[2]
  areas = stats[1:,cv2.CC_STAT_AREA]
  median = np.median(areas)

  cv2.imshow("Image segment", bw) 
  cv2.waitKey()
  cv2.destroyAllWindows()
  print(median)
  return median

def findBoundingRect(img, contours, min_w=7, min_h=7, min_r = 0.35):
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

def segmentText(img):

  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
  grad = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
  _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21,21))
  connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

  cv2.imshow("connect", bw)
  cv2.waitKey()
  cv2.destroyAllWindows()

  im2, contours, hierarchy = cv2.findContours(connected.copy(),
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)
                                              
  return findBoundingRect(img, contours)

def segmentsLetters(img):
  [M,N] = img.shape
  # invert since we are working black on white
  _, tresh_img = cv2.threshold(img, 0.0, 255.0, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

  tresh_img[0] = 0
  tresh_img[M-1] = 0

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

  cv2.imshow("text_regions", im_out)
  cv2.waitKey()
  cv2.destroyAllWindows()
  
  return findBoundingRect(img, contours, min_r=0)

def segmentsLettersByComp(img):
  _, bw = cv2.threshold(img, 0.0, 255.0, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

  output = cv2.connectedComponentsWithStats(bw, 4, cv2.CV_32S)
  stats = output[2]
  areas = stats[1:,cv2.CC_STAT_AREA]
  heights = stats[1:,cv2.CC_STAT_HEIGHT]
  widhts = stats[1:,cv2.CC_STAT_WIDTH]

  m_areal = np.median(areas)
  m_height = np.median(heights)
  m_width = np.median(widhts)
  boundingrect = []
  for i in range(len(stats)):
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i,cv2.CC_STAT_AREA]
    
    # if w > m_width/2 and h > m_height/2:
    boundingrect.append((x, y, w, h))
  return boundingrect  

                  
def highlightSegments(img, segments):
  # Copy input array as cv2 drawing function work inplace
  temp = np.array(img)
  for (x,y,w,h) in segments:
    cv2.rectangle(temp, (x, y),(x+w, y+h), (0,255,0), 1, 8, 0)

  return temp

def segmentPaper(img):
  _, bw = cv2.threshold(img, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
  bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

  cv2.imshow("Paper",bw) 
  cv2.waitKey()
  cv2.destroyAllWindows()


  output = cv2.connectedComponentsWithStats(bw, 4, cv2.CV_32S)
  stats = output[2]
  areas = stats[1:,cv2.CC_STAT_AREA]
  heights = stats[1:,cv2.CC_STAT_HEIGHT]
  widhts = stats[1:,cv2.CC_STAT_WIDTH]

  i = np.argmax(areas)
  x = stats[i+1, cv2.CC_STAT_LEFT]
  y = stats[i+1, cv2.CC_STAT_TOP]
  w = stats[i+1, cv2.CC_STAT_WIDTH]
  h = stats[i+1, cv2.CC_STAT_HEIGHT]
  print(i)
  return img[y:y+h, x:x+w] 


def rotate(img):
  thresh = cv2.threshold(img, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

  coords = np.column_stack(np.where(thresh > 0))
  angle = cv2.minAreaRect(coords)[-1]

  if angle < -45:
  	angle = -(90 + angle)
  else:
    angle = -angle

  # Affine Transformation used to do rotation
  (h, w) = img.shape[:2]
  center = (w // 2, h // 2)
  M = cv2.getRotationMatrix2D(center, angle, 1.0)
  rotated = cv2.warpAffine(img, M, (w, h),
    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
  
  # show the output image
  print("[INFO] angle: {:.3f}".format(angle))
  cv2.imshow("Input", img)
  cv2.imshow("Rotated", rotated)
  cv2.waitKey(0)

  return rotated

def main():
  SAVE_IMAGE_PATH = '../doc/res/'
  IMAGE_PATH = '../res/images/'
  image_filename = 'Android_image.jpg'
  # image_filename = 'Android2.jpg'
  # image_filename = 'lorem.png'
  # image_filename = 'ReceiptSwiss.jpg'
  # image_filename = 'doc.jpg'

  image = cv2.imread(IMAGE_PATH+image_filename,0)


  paper_region = segmentPaper(image)
  cv2.imshow("Paper",paper_region) 
  cv2.waitKey()
  cv2.destroyAllWindows()


  text_regions = segmentText(paper_region)
  text_reg = highlightSegments(paper_region,text_regions)

  # cv2.imwrite(SAVE_IMAGE_PATH + "segment_text1.png", text_regions) 
  cv2.imshow("Image segment",text_reg) 
  cv2.waitKey()
  cv2.destroyAllWindows()

  (x,y,w,h) = text_regions[1]
  text_region_image = image[y:y+h, x:x+w]
  cv2.imshow("Image segment",text_region_image) 
  cv2.waitKey()
  cv2.destroyAllWindows()

  rotated_image = rotate(text_region_image)

  rect = segmentsLettersByComp(rotated_image)
  show_img = highlightSegments(rotated_image,rect)

  cv2.imshow("Image segment", show_img) 
  cv2.waitKey()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()
