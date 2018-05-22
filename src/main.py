import classifier as classifier
import segment as segment
import utils as utils
import numpy as np
import cv2

np.set_printoptions(linewidth=150)
np.set_printoptions(edgeitems=150)


def preprocess(img):
    y,x = img.shape

    img[img > 30] = 255
    img = cv2.resize(img, (24, 24))

    back = np.zeros((28, 28))
    back[2:26,2:26] = img

    kernel = np.ones((3,3),np.uint8)
    img = cv2.erode(back, kernel, iterations = 1)
    return img


def extractText(img):
    linesHist = utils.find_lines(seg)
    lines = utils.segment_lines(seg, linesHist)
    for i, l in enumerate(lines):
        single_line = seg[l[0]:l[1], 0:x]
        symbolHist = utils.find_symbol(single_line)
        symbols    = utils.segment_symbols(seg, symbolHist)
        for s in symbols:
            img = seg[l[0]:l[1], s[0]:s[1]]
            #utils.imshow("Single symbol", img)
            img = preprocess(img)
            #single_symbol[single_symbol > 30] = 255
            #single_symbol[single_symbol <= 30] = 0
            digit = nn.forward(img)
            print(digit, end=" ")
            #print(digit)
            #utils.imshow("Final", img)


        print("")


if __name__ == "__main__":

  nn = classifier.Classifier()
  images = utils.imageParser() # create a generator

  for img in images:
    #utils.imshow("Original", img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    y, x = gray.shape
    binary = cv2.bitwise_not(gray)
    #binary = gray
    #utils.imshow("Binary", binary)
    contours = segment.segmentText(binary)
    #utils.imshow("Contours", segment.highlightSegments(img, contours))

    # Contours are presented from burron to top
    # Reverse with [::-1]
    for contour in contours[::-1]:
      seg = utils.extractContour(binary, contour)
      #utils.imshow("Segment", seg)
      extractText(seg)


    print("\n")
    if input("Press 'Enter' to preceed to next image: ") == 'q':
        exit()
