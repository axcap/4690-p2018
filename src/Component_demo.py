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


def get_line_image(image):
  """ 
  Note, not the same method used in main project, but same approach. This is only used for demo purpose.
  """
  _, prep_image = cv2.threshold(image, 120.0, 255.0, cv2.THRESH_BINARY_INV)
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
  prep_image = cv2.morphologyEx(prep_image, cv2.MORPH_CLOSE, kernel)

  line_image = image.copy()
  linesHist = utils.find_lines(prep_image)
  lines = utils.segment_lines(prep_image, linesHist)

  y,x = image.shape[:2]
  for l in lines:
    cv2.rectangle(line_image, (0, l[0]), (x, l[1]), 0, thickness = 1)
  cv2.imwrite('res/images/exemple_text_line.png',image[lines[0][0]:lines[0][1], 0:x])
  return line_image

def _DEMO_rotate_text():
  SAVE_IMAGE_PATH = 'doc/res/'
  IMAGE_PATH = 'res/images/'
  image_filename = 'text_skew_region.png'
  image = cv2.imread(IMAGE_PATH+image_filename,0)
  rotated = rotate(image)

  cv2.imwrite(SAVE_IMAGE_PATH + "Rotated.png",rotated)
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

def _DEMO_segment_letter(FileName,SaveFileName):
  SAVE_IMAGE_PATH = 'doc/res/'
  IMAGE_PATH = 'res/images/'
  image_filename = FileName
  image = cv2.imread(IMAGE_PATH+image_filename,0)
  letter_rect = seg.segmentLetters(image)
  utils.imshow("non segment Input", image)
  demo_image = seg.highlightSegments("demo_letter_image", image, letter_rect)
  cv2.imwrite(SAVE_IMAGE_PATH + SaveFileName , demo_image)
  return

def _DEMO_segment_line(FileName,SaveFileName):
  SAVE_IMAGE_PATH = 'doc/res/'
  IMAGE_PATH = 'res/images/'
  image_filename = FileName
  image = cv2.imread(IMAGE_PATH+image_filename,0)

  line_image = get_line_image(image)

  utils.imshow("non segment line Input", image)
  utils.imshow("segment line Input", line_image)
  cv2.imwrite(SAVE_IMAGE_PATH + SaveFileName ,line_image)
  return


def main():
  _DEMO_segment_text()
  _DEMO_rotate_text()
  _DEMO_segment_line('Lorem.png','segment_line.png')
  _DEMO_segment_letter('exemple_text_line.png','segment_letter1.png')
  _DEMO_segment_letter('SimpleR2.png','segment_letter2.png')
  _DEMO_segment_letter('SimpleR2.png','segment_letter2.png')

if __name__ == '__main__':
  main()