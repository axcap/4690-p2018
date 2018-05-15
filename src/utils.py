import numpy as np

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
                if hist[j] == 0:
                    skip = j-i
                    p2 = j
                    points.append((p1, p2))
                    break
    return points


def segment_lines(img, hist):
    return segment_hist(img, hist)

def segment_symbols(img, hist):
    return segment_hist(img, hist)
