import numpy as np
import cv2

def rotate2angle(img, angle, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE):
    (h, w) = img.shape[:2]

    edge = max(img.shape[:2])
    img_big = np.zeros((edge, edge), dtype=np.uint8)
    (h_big, w_big) = img_big.shape[:2]

    start_x = (w_big - w) // 2
    start_y = (h_big - h) // 2

    img_big[start_y:start_y+h, start_x:start_x+w] = img

    center = (w_big/2, h_big/2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    final = cv2.warpAffine(img_big, M, (w_big, h_big),
                           flags=flags, borderMode=borderMode)
    return final

def histogramProjection(img, direction = 'vertical'):
    n = img.shape[1] if direction is 'vertical' else img.shape[0]
    sumCols = []
    for i in range(n):
        col = None
        if direction is 'vertical':
            col = img[:,i]
        elif direction is 'horisontal':
            col = img[i,:]

        #black background anticipated
        summ = np.sum(col >= 100)
        sumCols.append(summ)
    return sumCols

#Returns image with rectangles around around each line
def show_lines(img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    linesHist = find_lines(img)
    lines = segment_lines(img, linesHist)
    y,x = img.shape[:2]
    for l in lines:
        cv2.rectangle(img, (0, l[0]), (x, l[1]), color=(255,0,0), thickness = 1)
    return img

#Returns image with rectangles around around each symbol in input line image
def show_symbols(img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    linesHist = find_symbol(img)
    lines = segment_symbols(img, linesHist)
    y,x = img.shape[:2]
    for l in lines:
        cv2.rectangle(img, (l[0], 0), (l[1], y), color=(255,0,0), thickness = 1)
    return img


def find_lines(img):
    hist = histogramProjection(img, direction = 'horisontal')
    return hist

def find_symbol(img):
    hist = histogramProjection(img, direction = 'vertical')
    return hist

def segment_hist(img, hist):
    flag = 0
    skip = 0
    p1 = 0
    p2 = 0
    points = []
    for i in range(len(hist)):
        if skip != 0:
            skip -= 1
            continue

        if hist[i] != 0:
            p1 = i
            for j in range(i, len(hist)):
                #if 'space between hists found' or end reached
                if hist[j] == 0 or j == len(hist)-1:
                    skip = j-i
                    p2 = j
                    points.append((p1, p2))
                    break
    return points


def segment_lines(img, hist):
    return segment_hist(img, hist)

def segment_symbols(img, hist):
    return segment_hist(img, hist)
