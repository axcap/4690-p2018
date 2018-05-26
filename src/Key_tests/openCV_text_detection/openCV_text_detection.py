import cv2
import numpy as np
import math
import os
import sys

def detect_text(image):

    channels = cv2.text.computeNMChannels(image)
    vis = image.copy()

    cn = len(channels)-1
    print(cn)
    for c in range(0,cn):
        channels.append((255-channels[c]))

    for channel in channels:
        erc1 = cv2.text.loadClassifierNM1('Key_tests/openCV_text_detection/trained_classifierNM1.xml')
        er1 = cv2.text.createERFilterNM1(erc1,16,0.00015,0.1,0.2,True,0.1)

        erc2 = cv2.text.loadClassifierNM2('Key_tests/openCV_text_detection/trained_classifierNM2.xml')
        er2 = cv2.text.createERFilterNM2(erc2,0.5)

        regions = cv2.text.detectRegions(channel,er1,er2)

        # print(regions)
        if regions == []:
            continue
            
        rects = cv2.text.erGrouping(image, channel, [r.tolist() for r in regions])

        for r in range(0,np.shape(rects)[0]):
            rect = rects[r]
            cv2.rectangle(vis, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (0, 0, 0), 2)
            cv2.rectangle(vis, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (255, 255, 255), 1)

    #Visualization
    cv2.imshow("Text detection result", vis)
    cv2.imwrite('OpenCV_Std.png', vis)
    cv2.waitKey(0)
    return vis

def main():

    IMAGE_PATH = '../res/images/'
    image_filename = 'ReceiptSwiss.jpg'

    image = cv2.imread(IMAGE_PATH+image_filename)
    detect_text(image)


if __name__ == '__main__':
    main()