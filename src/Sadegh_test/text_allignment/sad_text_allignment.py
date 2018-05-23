from matplotlib import pyplot as plt
import numpy as np
import cv2

import utils as utils

def load_img(file_path):
    # Load an  image
    # Set last argument to 0 to load in grayscale
    img = cv2.imread(file_path);
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def find_angle(img):
    # Canny edge detection
    # small threshold 50, big threshold 150, apertureSize = kernel size
    edges = cv2.Canny(img,50,150,apertureSize = 3)

    # Hough Trasnform
    minLength = 200
    lines = cv2.HoughLinesP(edges,15,np.pi/180,1,minLength)

    # Find all angles of lines that appear in the img
    N = lines.shape[0]
    angles = []
    for i in range(N):
        x1 = lines[i][0][0]
        y1 = lines[i][0][1]
        x2 = lines[i][0][2]
        y2 = lines[i][0][3]
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),5)


        # count how many times each angle occures
        angle = np.arctan2((y1 - y2), (x1 - x2))
        if angle < 0:
            angle += 2*np.pi

        flag = False # angle already registerd
        for i in range(len(angles)):
            if angle == angles[i][0]:
                flag = True
                angles[i][1] += 1
                break

        # add new angles
        if flag == False:
            angles.append([angle, 1])

    # Find the most occuring angle
    maxProbAngle = []
    for l in angles:
        print(l)
        if maxProbAngle == []:
            maxProbAngle = l
        elif l[1] > maxProbAngle[1]:
            maxProbAngle = l
        else:
            continue

    return maxProbAngle[0]


if __name__ == "__main__":
    file_path = 'numbers_60degree.png'
    img = load_img(file_path)
    angle = np.rad2deg(find_angle(img))
    print(angle)

    N, M = img.shape
    y = N
    x = M
    rotM = cv2.getRotationMatrix2D((M // 2,N // 2), angle, 1)
    img = cv2.warpAffine(img,
        rotM,
        (M, N),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE)



    cv2.imshow('houghlines3.jpg',img)
    cv2.waitKey(0)

    '''

    linesHist = utils.find_lines(img)
    lines = utils.segment_lines(img, linesHist)

    text = ""
    for l in lines:
        single_line = img[l[0]:l[1], 0:x]
        cv2.imshow("line", single_line)
        cv2.waitKey(0)
        symbolHist = utils.find_symbol(single_line)
        symbols    = utils.segment_symbols(img, symbolHist)
        for s in symbols:
            single_symbol = img[l[0]:l[1], s[0]:s[1]]
            cv2.imshow("Symbol", single_symbol)
            cv2.waitKey(0)
            #data = image2data(single_symbol)
            #r = nn.forward(data)
            #index = np.argmax(r)
            #text += str(index)
            #print("Class: %d - %.2f%%" % (index, r[0, index]*100))




    print("Extracted text: %s\n" % text)
    '''
