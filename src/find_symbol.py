from matplotlib import pyplot as plt
import numpy as np
import cv2

def verticalProjection(img):
    w,h = img.shape
    sumCols = []
    for j in range(h):
        col = img[:,j]
        summ = np.sum(col != 255)
        sumCols.append(summ)
    return sumCols

def find_symbol(img):
    pHist = verticalProjection(img);
    return pHist

def segment_symbols(img, pHist):
    y = img.shape[0]-1
    flag = 0
    skip = 0
    x1 = 0
    x2 = 0
    for i in range(len(pHist)):        
        if skip != 0:
            skip -= 1
            continue
        
        if pHist[i] != 0:
            x1 = i
            for k in range(i, len(pHist)):
                if pHist[k] == 0:
                    skip = k-i
                    x2 = k
                    cv2.rectangle(img,(x1, 0),(x2, y), color=(255,0,0), thickness=1)
                    break

# Load an  image
# Set last argument to 0 to load in grayscale
img = cv2.imread('res/images/0-9.png');
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

pHist = find_symbol(img_grey)

segment_symbols(img, pHist)

plt.subplot(2, 1, 1)
plt.axis("off")
plt.imshow(img, cmap='gray')

plt.subplot(2, 1, 2)
plt.xlim(xmin=0)
plt.xlim(xmax=len(pHist))
plt.bar(range(len(pHist)), height=pHist)
#plt.axis("off")
plt.show()
    
