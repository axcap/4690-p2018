import utils as utils
import numpy as np
import cv2

np.set_printoptions(linewidth=9999999)


if __name__ == "__main__":
    # Load an  image
    # Set last argument to 0 to load in grayscale
    img = cv2.imread('res/images/lorem.png');
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    y,x = img_grey.shape
    cv2.imshow("img_grey", img_grey);        
    cv2.waitKey(0)
    
    #nn = NN_MNIST()
    #nn.train(None) #no training done in demo
    
    linesHist = utils.find_lines(img_grey)
    lines = utils.segment_lines(img, linesHist)
    
    text = ""
    for l in lines:
        single_line = img_grey[l[0]:l[1], 0:x]
        cv2.imshow("line", single_line)
        cv2.waitKey(0)
        symbolHist = utils.find_symbol(single_line)
        symbols    = utils.segment_symbols(img, symbolHist)
        for s in symbols:
            single_symbol = img_grey[l[0]:l[1], s[0]:s[1]]
            cv2.imshow("Symbol", single_symbol)
            cv2.waitKey(0)
            #data = image2data(single_symbol)
            #r = nn.forward(data)
            #index = np.argmax(r)
            #text += str(index)
            #print("Class: %d - %.2f%%" % (index, r[0, index]*100))


            
                    
    print("Extracted text: %s\n" % text)    
