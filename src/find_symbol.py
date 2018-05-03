from matplotlib import pyplot as plt
from nn_mnist import NN_MNIST
import numpy as np
import cv2

np.set_printoptions(linewidth=9999999)

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
    points = []
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
                    points.append((x1, x2))
                    #cv2.rectangle(img,(x1, 0),(x2, y), color=(255,0,0), thickness=1)
                    break
    return points

def image2data(img):
    y,x = img.shape
    #Convert to grayscale
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #Pad image to fit 28x28 grid
    padding = (y-x)//2 if y>x else (x-y)//2
    img = cv2.copyMakeBorder(img,0,0,padding,padding,cv2.BORDER_CONSTANT, value=255)
    #Fit neural network input size
    img = cv2.resize(img, (28,28))
    #Invert colors and rotate from graph to display coordinates
    img = 255-img
    #Create kernel for convolution
    kernel = np.ones((3,3),np.float32)/25
    #Aply 3x3 kernel to smoothen up our image
    img = cv2.filter2D(img,-1,kernel)
    #Convert from 0-255 to 0-1 range
    #img = ((1/256)*img)
    #Normalize image values after smoothening
    img = img * (255/img.max())
    #Flatten out to feed into network
    img = np.reshape(img, (1, 28*28))    
    return img

if __name__ == "__main__":
    # Load an  image
    # Set last argument to 0 to load in grayscale
    img = cv2.imread('res/images/0-9.png');
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    y,x = img_grey.shape

    nn = NN_MNIST()
    nn.train(None, force_retrain=False)
    
    pHist = find_symbol(img_grey)
    points = segment_symbols(img, pHist)

    text = ""
    for p in points:
        crop_img = img_grey[0:y, p[0]:p[1]]
        data = image2data(crop_img)                    
        r = nn.forward(data)
        index = np.argmax(r)
        text += str(index)
        #print("Class: %d - %.2f%%" % (index, r[0, index]*100))
        #cv2.imshow("cropped", crop_img)        
        #cv2.waitKey(0)
                    
    print("Extracted text: %s\n" % text)

    plt.subplot(2, 1, 1)
    plt.axis("off")
    plt.imshow(img, cmap='gray')
    plt.subplot(2, 1, 2)
    plt.xlim(xmin=0)
    plt.xlim(xmax=len(pHist))
    plt.bar(range(len(pHist)), height=pHist)
    #plt.axis("off")
    plt.show()
    
