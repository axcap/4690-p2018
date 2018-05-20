import utils as utils
import segment as segment

if __name__ == "__main__":

  images = utils.imageParser() # create a generator
  for img in images:
    utils.imshow("Original", img)

    contours = segment.segmentText(img)
    utils.imshow("Contours", segment.highlightSegments(img, contours))
