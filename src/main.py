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


def extractText(image):
    y, x = image.shape
    text_out = ""
    linesHist = utils.find_lines(seg)
    lines = utils.segment_lines(seg, linesHist)
    for i, l in enumerate(lines):
        single_line = image[l[0]:l[1], 0:x]
        #symbolHist = utils.find_symbol(single_line)
        #symbols    = utils.segment_symbols(seg, symbolHist)

        symbols = segment.segmentLetters(single_line)
        symbols = np.array(symbols)
        symbols = symbols[symbols[:,0].argsort()]

        single_line = image[l[0]:l[1], 0:x]
        #segment.highlightSegments("Highlight", single_line, symbols)

        # Find average space between chars
        summ = 0
        for i in range(len(symbols)-1):
            current = symbols[i]
            nexxt = symbols[i+1]
            summ += nexxt[0] - (current[0]+current[2])
        summ /= len(symbols)-1
        summ += 1
        print(summ)
        
        start = time.time()
        line_out = ""
        for idx, s in enumerate(symbols):
            img = utils.extractContour(single_line, s)

            img_linesHist = utils.find_lines(img)
            img_lines = utils.segment_lines(img, img_linesHist)
            img = img[img_lines[0][0]:img_lines[-1][1],:]
            img = utils.img2data(img)
            
            digit = nn.forward(np.transpose(img)) # takes 0.3 per run

            line_out += digit.lower()
            # If space between chars > agerage insert 'space char'
            if idx+1 < len(symbols) and symbols[idx+1][0]-(symbols[idx][0] + symbols[idx][2]) > summ:
                line_out += " "

            print(".", end="")
            sys.stdout.flush()

        
        end = time.time()
        text_out += line_out +  "\n"
        print("Time: ", (end - start)/len(symbols))
        print(line_out)
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
    contours = segment.segmentText(binary)
    segment.highlightSegments("Contour", img, contours)

    # Contours are presented from burron to top
    # Reverse with [::-1]
    for contour in contours[::-1]:
      seg = utils.extractContour(binary, contour)
      #utils.imshow("Segment", seg)
      extractText(seg)


    print("\n")
    if input("Press 'Enter' to preceed to next image: ") == 'q':
        exit()
