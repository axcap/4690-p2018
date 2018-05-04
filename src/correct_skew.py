# Source: https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/

# import the necessary packages
from nn_mnist import NN_MNIST
from matplotlib import pyplot as plt
import utils as utils
import numpy as np
import sys
import cv2

np.set_printoptions(linewidth=1000)
np.set_printoptions(edgeitems=1000)

def imshow(text, img):
    plt.title(text)
    plt.xticks([]), plt.yticks([])
    plt.imshow(img, cmap="gray")
    plt.draw()
    plt.pause(0.1)
    input("<Hit Enter To Continue>")

# Takes an image with rotations (0, 90, 180 or 270)
# and tries to guess correct angle
def final_correct_angle(img):
    y,x = img.shape
    imshow("Input", img)

    nn = NN_MNIST()
    nn.train(None) #no training done in demo

    for angle in [0, 90, 180, 270]:
        M = cv2.getRotationMatrix2D((x/2,y/2), angle, 1)
        dst = cv2.warpAffine(img, M, (x,y))

        imshow("img", dst)

        linesHist = utils.find_lines(dst)
        lines = utils.segment_lines(dst, linesHist)
        print(len(lines))
        for l in lines:
            single_line = dst[l[0]:l[1], 0:x]
            imshow("line", single_line)
            symbolHist = utils.find_symbol(single_line)
            symbols    = utils.segment_symbols(dst, symbolHist)
            for s in symbols:
                single_symbol = dst[l[0]:l[1], s[0]:s[1]]
                imshow("Symbol", single_symbol)
                data = cv2.resize(single_symbol, (28,28))
                print("1")
                r = nn.forward(np.reshape(data, (1, 28*28)))
                print(r)
                index = np.argmax(r)
                print("Class: %d - %.2f%%" % (index, r[0, index]*100))


    return 90.0

#Returns angle with closest to 90degree modulo
def correct_angle(angle):
    if angle < -45:
        angle = -(90 + angle)
    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle
    return angle


#image_path = "res/images/numbers_"+str(degrees)+"degree.png"
image_path = sys.argv[1]

# load the image from disk
img = cv2.imread(image_path)
img = cv2.resize(img, (0,0), fx=0.3, fy=0.3)
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
angle = correct_angle(angle)
print("Corrected angle: "+str(angle))


# rotate the image to deskew it
(h, w) = img.shape[:2]
center = (w // 2, h // 2)

M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(thresh, M, (w, h),
	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


thresh = cv2.threshold(rotated, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
angle2 = final_correct_angle(thresh);
print("Corrected angle2: "+str(angle2))

M = cv2.getRotationMatrix2D(center, angle2, 1.0)
rotated2 = cv2.warpAffine(rotated, M, (w, h),
	flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

cv2.putText(img, "Original",
	    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

cv2.putText(rotated, "Rotated by {:.2f} degrees".format(angle),
	    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

cv2.putText(rotated2, "Rotated2 by {:.2f} degrees".format(angle2),
	    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# show the output image
print("[INFO] angle: {:.3f}".format(angle))
final_frame = np.hstack((gray, rotated, rotated2))
imshow('Original - Rotated - Rotated2', final_frame)
