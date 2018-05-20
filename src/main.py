import classifier as classifier
import segment as segment
import utils as utils
import cv2

if __name__ == "__main__":

  nn = classifier.Classifier()
  images = utils.imageParser() # create a generator

  for img in images:
    #utils.imshow("Original", img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    y, x = gray.shape
    binary = cv2.bitwise_not(gray)
    #utils.imshow("Binary", binary)
    contours = segment.segmentText(binary)
    #utils.imshow("Contours", segment.highlightSegments(img, contours))

    # Contours are presented from burron to top
    # Reverse with [::-1]
    for contour in contours[::-1]:
      seg = utils.extractContour(binary, contour)
      #utils.imshow("Segment", seg)

      linesHist = utils.find_lines(seg)
      lines = utils.segment_lines(seg, linesHist)
      for i, l in enumerate(lines):
        single_line = seg[l[0]:l[1], 0:x]
        symbolHist = utils.find_symbol(single_line)
        symbols    = utils.segment_symbols(seg, symbolHist)
        for s in symbols:
            single_symbol = seg[l[0]:l[1], s[0]:s[1]]
            single_symbol = cv2.resize(single_symbol, (28,28))
            single_symbol[single_symbol > 30] = 255
            single_symbol[single_symbol <= 30] = 0
            #utils.imshow("Single symbol", single_symbol)

            digit = nn.forward(single_symbol)
            print(digit, end=" ")

      print("")

    print("\n")
    if input("Press 'Enter' to preceed to next image: ") == 'q':
        exit()
