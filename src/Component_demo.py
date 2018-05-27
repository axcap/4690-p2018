import cv2
import segment as seg
import numpy as np
import utils as utils

def rotate(img):
  """ 
  Note, not the same method used in main project, but same approach. This is only used for demo purpose.
  """
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

  return rotated



def _DEMO_rotate_text():
  SAVE_IMAGE_PATH = 'doc/res/'
  IMAGE_PATH = 'res/images/'
  image_filename = 'text_skew_region.png'
  image = cv2.imread(IMAGE_PATH+image_filename,0)
  rotated = rotate(image)

  cv2.imwrite("Rotated.png",rotated)
  utils.imshow("Non-rotat Input", image)
  utils.imshow("Rotated", rotated)

  return

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

  text_regions1 = seg.segmentText(paper_region1)
  text_regions2 = seg.segmentText(paper_region2)
  text_regions3 = seg.segmentText(paper_region3)

  demo_image1 = seg.highlightSegments("demo_image1",paper_region1,text_regions1)
  demo_image2 = seg.highlightSegments("demo_image2",paper_region2,text_regions2)
  demo_image3 = seg.highlightSegments("demo_image3",paper_region3,text_regions3)

  cv2.imwrite(SAVE_IMAGE_PATH + "segment_text1.png", demo_image1)
  cv2.imwrite(SAVE_IMAGE_PATH + "segment_text2.png", demo_image2)
  cv2.imwrite(SAVE_IMAGE_PATH + "segment_text3.png", demo_image3)
  return

def _DEMO_segment_letter():
  SAVE_IMAGE_PATH = 'doc/res/'
  IMAGE_PATH = 'res/images/'
  image_filename = 'simpleR2.png'
  image = cv2.imread(IMAGE_PATH+image_filename,0)
  letter_rect = seg.segmentLetters(image)
  utils.imshow("non segment Input", image)
  demo_image = seg.highlightSegments("demo_letter_image", image, letter_rect)
  return

def main():
  _DEMO_segment_text()
  _DEMO_segment_letter()
  _DEMO_rotate_text()

if __name__ == '__main__':
  main()