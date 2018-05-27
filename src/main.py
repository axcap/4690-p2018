from matplotlib import pyplot as plt
import classifier as classifier
import segment as segment
import timeit, functools
import utils as utils
import numpy as np
import time
import cv2
import sys

np.set_printoptions(linewidth=150)
np.set_printoptions(edgeitems=150)

def computeSpacing(symbols):
    space_length = 0
    char_length = 0
    for current in symbols:
        char_length += current[2]
    char_length  = char_length //  len(symbols)
    space_length = int(char_length*0.7)
    return space_length


def extractText(image):
    y, x = image.shape
    text_out = ""
    linesHist = utils.find_lines(seg)
    lines = utils.segment_lines(seg, linesHist)
    for i, l in enumerate(lines):
        single_line = image[l[0]:l[1], 0:x]
        #utils.imshow("Line", single_line)
        #symbolHist = utils.find_symbol(single_line)
        #symbols    = utils.segment_symbols(seg, symbolHist)

        symbols = segment.segmentLetters(single_line)
        symbols = np.array(symbols)
        symbols = symbols[symbols[:,0].argsort()]

        single_line = image[l[0]:l[1], 0:x]
        #segment.highlightSegments("Highlight", single_line, symbols)

        # Find average space between chars
        space = computeSpacing(symbols)

        start = time.time()
        line_out = ""
        for idx, s in enumerate(symbols):
            # skip all commas and dots
            if s[1] > 20 or s[3] < 50:
                line_out += " "
                continue

            img = utils.extractContour(single_line, s)

            img_linesHist = utils.find_lines(img)
            img_lines = utils.segment_lines(img, img_linesHist)
            img = img[img_lines[0][0]:img_lines[-1][1],:]
            img = utils.img2data(img)

            digit = nn.forward(np.transpose(img)) # takes 0.3 per run
            digit = digit.lower()
            if digit == 'i' or digit == 'l':
                # See if we can find i's dot/hat by counting individual objects on img
                _, cont, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if np.shape(cont)[0] == 1: digit = 'l'
                else:                      digit = 'i'

            line_out += digit
<<<<<<< HEAD
<<<<<<< HEAD
            # If space between chars > agerage insert 'space char'
            if idx+1 < len(symbols) and symbols[idx+1][0]-(symbols[idx][0] + symbols[idx][2]) > space:
                line_out += " "*((symbols[idx+1][0]-(symbols[idx][0] + symbols[idx][2])) // space)
=======
            # If space between chars > agerage insert x 'space char's
            widht = symbols[idx+1][0]-(symbols[idx][0] + symbols[idx][2])
            if idx+1 < len(symbols) and  widht > space:
                line_out += " " * widht // space
>>>>>>> 968d29c916de6aa01f8830a1c873a695405f7094

            print(".", end="")
            sys.stdout.flush()

<<<<<<< HEAD

=======
            # If space between chars > agerage insert x 'space char's
            widht = symbols[idx+1][0]-(symbols[idx][0] + symbols[idx][2])
            if idx+1 < len(symbols) and  widht > space:
                line_out += " " * widht // space

            print(".", end="")
            sys.stdout.flush()


>>>>>>> 968d29c916de6aa01f8830a1c873a695405f7094
=======

>>>>>>> 968d29c916de6aa01f8830a1c873a695405f7094
        end = time.time()
        text_out += line_out +  "\n"
        print("\n", line_out)
        print("\nTime: ", (end - start)/len(symbols))

        #utils.imshow("Line", single_line)

    print(text_out)


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
    contours = segment.segmentText(binary, (50, 50))
    segment.highlightSegments("Contour", binary, contours)

    # Contours are presented from burron to top
    # Reverse with [::-1]
    for contour in contours[::-1]:
        seg = utils.extractContour(binary, contour)

<<<<<<< HEAD
<<<<<<< HEAD
        coords = np.column_stack(np.where(seg > 0))
        angle  = np.around(cv2.minAreaRect(coords)[-1]).astype(int)
        angle  = utils.correct_angle(angle)
=======
=======
>>>>>>> 968d29c916de6aa01f8830a1c873a695405f7094
        # Simple rotation fix
        coords = np.column_stack(np.where(seg > 0))
        angle  = np.around(cv2.minAreaRect(coords)[-1]).astype(int)
        angle  = utils.correct_angle(angle)

<<<<<<< HEAD
>>>>>>> 968d29c916de6aa01f8830a1c873a695405f7094
=======
>>>>>>> 968d29c916de6aa01f8830a1c873a695405f7094
        seg    = utils.rotate2angle(seg, angle)
        utils.imshow("Segment", seg)

        extractText(seg)


    print("\n")
    if input("Press 'Enter' to preceed to next image: ") == 'q':
        exit()
