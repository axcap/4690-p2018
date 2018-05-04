# Source: https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/

# import the necessary packages
import numpy as np
import sys
import cv2

#image_path = "res/images/numbers_"+str(degrees)+"degree.png"
image_path = sys.argv[1]
# load the image from disk
img = cv2.imread(image_path)
img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
# convert the image to grayscale and flip the foreground
# and background to ensure foreground is now "white" and
# the background is "black"
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = cv2.bitwise_not(gray)

# threshold the image, setting all foreground pixels to
# 255 and all background pixels to 0
thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# grab the (x, y) coordinates of all pixel values that
# are greater than zero, then use these coordinates to
# compute a rotated bounding box that contains all
# coordinates
coords = np.column_stack(np.where(thresh > 0))
angle = cv2.minAreaRect(coords)[-1]
print("Original angle: "+str(angle))
# the `cv2.minAreaRect` function returns values in the
# range [-90, 0); as the rectangle rotates clockwise the
# returned angle trends to 0 -- in this special case we
# need to add 90 degrees to the angle
if angle < -45:
	angle = -(90 + angle)
 
# otherwise, just take the inverse of the angle to make
# it positive
else:
	angle = -angle
print("Corrected angle: "+str(angle))        

# rotate the image to deskew it
(h, w) = img.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(img, M, (w, h),
	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

cv2.putText(img, "Original",
	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# draw the correction angle on the image so we can validate it
cv2.putText(rotated, "Rotated by {:.2f} degrees".format(angle),
	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
# show the output image
print("[INFO] angle: {:.3f}".format(angle))
final_frame = cv2.hconcat((img, rotated))
cv2.imshow('Original - Rotated', final_frame)
cv2.waitKey(0)
