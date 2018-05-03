from matplotlib import pyplot as plt
#from nn_mnist import NN_MNIST
import find_symbol as fs
import numpy as np
import cv2

np.set_printoptions(linewidth=9999999)

def horisontalProjection(img):
    w,h = img.shape
    sumCols = []
    for j in range(w):
        col = img[j,:]
        summ = np.sum(col != 255)
        sumCols.append(summ)
    return sumCols

def find_lines(img):
    pHist = horisontalProjection(img);
    return pHist

def segment_lines(img, pHist):
    x = img.shape[1]-1
    flag = 0
    skip = 0
    y1 = 0
    y2 = 0
    points = []
    for i in range(len(pHist)):        
        if skip != 0:
            skip -= 1
            continue
        
        if pHist[i] != 0:
            y1 = i
            for k in range(i, len(pHist)):
                if pHist[k] == 0:
                    skip = k-i
                    y2 = k
                    points.append((y1, y2))
                    break
    return points

if __name__ == "__main__":
    # Load an  image
    # Set last argument to 0 to load in grayscale
    img = cv2.imread('res/images/lorem.png');
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    y,x = img_grey.shape
    cv2.imshow("img_grey", img_grey);        
    cv2.waitKey(0)
    
    #nn = NN_MNIST()
    #nn.train(None, force_retrain=False)
    
    linesHist = find_lines(img_grey)
    #print(linesHist)
    lines = segment_lines(img, linesHist)
    #print(lines)
    
    text = ""
    for l in lines:
        single_line = img_grey[l[0]:l[1], 0:x]
        #data = image2data(crop_img)
        cv2.imshow("line", single_line)
        cv2.waitKey(0)
        #r = nn.forward(data)
        #index = np.argmax(r)
        #text += str(index)
        #print("Class: %d - %.2f%%" % (index, r[0, index]*100))
        #cv2.imshow("cropped", crop_img)        
        #cv2.waitKey(0)
        symbolHist = fs.find_symbol(single_line)
        symbols    = fs.segment_symbols(img, symbolHist)
        for s in symbols:
            single_symbol = img_grey[l[0]:l[1], s[0]:s[1]]
            cv2.imshow("Symbol", single_symbol)
            cv2.waitKey(0)

            
                    
    print("Extracted text: %s\n" % text)

    plt.subplot(1, 2, 1)
    plt.ylim(ymin=0)
    plt.ylim(ymax=len(pHist))

    plt.barh(np.arange(y)[::-1], linesHist)
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.imshow(img, cmap='gray', aspect="auto")

    plt.show()
    
