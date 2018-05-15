import cv2 as cv
import numpy as np
import math
import os
import sys

def detect_text(image):
    

    channels = cv.text.computeNMChannels(image)
    vis = image.copy()

    cn = len(channels)-1
    for c in range(0,cn):
        channels.append((255-channels[c]))

    for channel in channels:
        erc1 = cv.text.loadClassifierNM1('trained_classifierNM1.xml')
        er1 = cv.text.createERFilterNM1(erc1,16,0.00015,0.13,0.2,True,0.1)

        erc2 = cv.text.loadClassifierNM2('trained_classifierNM2.xml')
        er2 = cv.text.createERFilterNM2(erc2,0.5)

        regions = cv.text.detectRegions(channel,er1,er2)

        print(channel)
        
        rects = cv.text.erGrouping(image, channel,[r.tolist() for r in regions])
        #rects = cv.text.erGrouping(img,channel,[x.tolist() for x in regions], cv.text.ERGROUPING_ORIENTATION_ANY,'../../GSoC2014/opencv_contrib/modules/text/samples/trained_classifier_erGrouping.xml',0.5)
        #Visualization
        for r in range(0,np.shape(rects)[0]):
            rect = rects[r]
            cv.rectangle(vis, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (0, 0, 0), 2)
            cv.rectangle(vis, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (255, 255, 255), 1)

    #Visualization
    cv.imshow("Text detection result", vis)
    cv.waitKey(0)

    cv.imwrite('detect_text.jpg', image)


def main():
    import my_swt as swt

    IMAGE_PATH = '../../res/images/'
    image_filename = 'text_skew.png'

    image = cv.imread(IMAGE_PATH+image_filename)


    detect_text(image)





if __name__ == '__main__':
    main()