from matplotlib import pyplot as plt
import numpy as np
import cv2

mapping_path = "res/datasets/EMNIST_ByMerge/"
mapping = np.loadtxt(mapping_path+"bymerge-mapping.txt", dtype=np.uint8)


def imageParser():
    path = "res/images/sans.png"
    img  = cv2.imread(path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    while True:
      yield np.array(img)

def imshow(text, img):
    plt.title(text)
    plt.xticks([]), plt.yticks([])
    plt.imshow(img, cmap="gray")
    plt.draw()
    plt.pause(0.1)
    return input("<Hit Enter To Continue>")

def class2char(class_n):
    #print(mapping[:,0])
    where = np.argwhere(mapping[:,0] == class_n)
    index = where[0][0]
    char  = mapping[index][1]
    return chr(char)

def char2class(char):
    where = np.argwhere(mapping[:,1] == ord(char))
    if where.size == 0:
        where = np.argwhere(mapping[:,1] == ord(char.upper()))
    class_n = mapping[where[0][0]][0]
    return class_n

def img2data(img):
    max_idx = np.argmax(img.shape)
    max_edge = img.shape[max_idx]
    padding = max_edge // 7
    back = np.zeros((max_edge + 2*padding, max_edge + 2*padding), dtype=np.uint8)

    if max_idx == 0:
        s = (back.shape[1]-img.shape[1])//2
        back[padding:padding+img.shape[0] , s:s+img.shape[1]] = img
    else:
        s = (back.shape[0]-img.shape[0])//2
        back[s:s+img.shape[0], padding:padding+img.shape[1]] = img
    img = back

    #kernel = np.ones((3, 3),np.uint8)
    #img = cv2.erode(img,kernel,iterations = 1)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img[img>127] = 255
    img[img<50]  = 0
    #img = img / 255
    #img = np.transpose(img)
    return img


def extractContour(img, contour):
    x, y, w, h = contour
    return img[y:y+h, x:x+w]

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
