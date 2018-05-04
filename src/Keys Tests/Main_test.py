import cv2 
import numpy as np 
import test_find_contour as test 
import text_segmentation as text_seg
import SWT as swt


IMAGE_PATH = '../../res/images/'
image_filename = 'text_example1.png'
image = cv2.imread(IMAGE_PATH+image_filename,0)
text_area = text_seg.detect_text(image)
SWT = swt.stroke_width_transform(text_area[0])


# normalizedImg = cv2.normalize(SWT,  SWT, 0, 255, cv2.NORM_MINMAX)
np.set_printoptions(threshold=np.nan)
print(SWT)

cv2.imshow("SWT", SWT) 
cv2.waitKey()
cv2.destroyAllWindows()


