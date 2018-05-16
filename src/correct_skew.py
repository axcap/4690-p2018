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

nn = NN_MNIST(model_path="res/model/nn_fnist/fnist_demo")
nn.train(None) #no training done in demo


target_angle = -int(sys.argv[2])
print("Target angel: ", target_angle)

def imshow(text, img):
    plt.title(text)
    plt.xticks([]), plt.yticks([])
    plt.imshow(img, cmap="gray")
    plt.draw()
    plt.pause(0.1)
    input("<Hit Enter To Continue>")

#Returns angle with closest to 90degree modulo
def correct_angle(angle):
    if angle < 0:
        angle = -(90 + angle)
        print(1)
        # otherwise, just take the inverse of the angle to make
        # it positive
    else:
        angle = -angle
        print(0)
    return angle

# Takes an image with rotations (0, 90, 180 or 270)
# and tries to guess correct angle
def final_correct_angle(img):
    return 0
    y,x = img.shape
    #imshow("orig", img)
    linesHist = utils.find_lines(img)
    lines = utils.segment_lines(img, linesHist)

    #print("Lines: ", len(lines))
    orientation_arr = []
    for l in lines:
        single_line = img[l[0]:l[1], 0:x]
        #imshow("line", single_line)
        symbolHist = utils.find_symbol(single_line)
        symbols    = utils.segment_symbols(img, symbolHist)

        for idx, s in enumerate(symbols):
            single_symbol = img[l[0]:l[1], s[0]:s[1]]
            single_symbol = cv2.resize(single_symbol, (28,28), interpolation=cv2.INTER_NEAREST)
            y2,x2 = single_symbol.shape

            #imshow("single", single_symbol)
            test = np.zeros(4);
            for angle in [0, 90, 180, 270]:
                M = cv2.getRotationMatrix2D((x2/2,y2/2), angle, 1)
                dst = cv2.warpAffine(single_symbol, M, (x2,y2))

                #imshow("singleR", dst)
                r = nn.forward_raw(np.reshape(dst, (1, 28*28)))[0]
                #print(r)
                index = np.argmax(r)
                #print(index)
                test[angle//90] = \
                                  r[index]
                if(index in [0, 5, 6, 8, 9]):
                    break
                    #print("MATCH")
                    #print(angle)
                    #return angle;
                    pass

            print(test)
            orientation_arr.append(np.argmax(test)*90)

    print(orientation_arr)
    print(orientation_arr.count(0))
    print(orientation_arr.count(90))
    print(orientation_arr.count(180))
    print(orientation_arr.count(270))
    #print("%d%% : Class: %d" % (angle, index))

    return 0.0


#image_path = "res/images/numbers_"+str(degrees)+"degree.png"
image_path = sys.argv[1]

# load the image from disk
img = cv2.imread(image_path)
#img = cv2.resize(img, (0,0), fx=0.3, fy=0.3)
# convert the image to grayscale and flip the foreground
# and background to ensure foreground is now "white" and
# the background is "black"
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)

gray = utils.rotate2target(gray, target_angle)

# threshold the image, setting all foreground pixels to
# 255 and all background pixels to 0
thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# grab the (x, y) coordinates of all pixel values that
# are greater than zero, then use these coordinates to
# compute a rotated bounding box that contains all
# coordinates
coords = np.column_stack(np.where(thresh > 0))
angle = np.around(cv2.minAreaRect(coords)[-1]).astype(int)
print("Original angle: "+str(angle))
# the `cv2.minAreaRect` function returns values in the
# range [-90, 0); as the rectangle rotates clockwise the
# returned angle trends to 0 -- in this special case we
# need to add 90 degrees to the angle
angle2 = correct_angle(angle)
print("Corrected angle2: "+str(angle2))


# rotate the image to deskew it
rotated = utils.rotate2target(thresh, angle2)

thresh = cv2.threshold(rotated, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

angle3 = final_correct_angle(thresh);
print("Corrected angle3: "+str(angle3))

rotated2 = utils.rotate2target(rotated, angle3)

cv2.putText(img, "Original",
	    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

cv2.putText(rotated, "Rotated by {:.2f} degrees".format(angle2),
	    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

cv2.putText(rotated2, "Rotated2 by {:.2f} degrees".format(angle3),
	    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

# show the output image
final_angle = angle3-angle2
print("[INFO] angle: {:.3f}".format(final_angle))

if angle < -45:
    #y = -(90 + x)
    angle = (90 + angle)
    print(angle)
else:
    angle = angle3-angle2
    print(angle)


print("[INFO] angle: {:.3f}".format(final_angle))
space = np.zeros((gray.shape[0], 50), dtype=np.uint8)

final_frame = np.hstack((gray, space, rotated, space, rotated2))
imshow('Original - Rotated - Rotated2', final_frame)
