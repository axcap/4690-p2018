from matplotlib import pyplot as plt
import numpy as np
import cv2

def load_img(file_path):
    # Load an  image
    # Set last argument to 0 to load in grayscale
    img = cv2.imread(file_path);
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_grey

def find_lines(img):
    # small threshold 50, big threshold 150, apertureSize = kernel size
    edges = cv2.Canny(img,50,150,apertureSize = 3)
    #plt.imshow(edges,cmap = 'gray')
    #plt.show()

    lines = cv2.HoughLines(edges,1,np.pi/180,100)
    for x in lines[0]:
        print(x)
    #plt.imshow(lines)
    #plt.show()


    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

    cv2.imwrite('houghlines3.jpg',img)


    '''
    # Probabolistic Hoguh transform

    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    for x1,y1,x2,y2 in lines[0]:
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.imwrite('houghlines5.jpg',img)
    '''


if __name__ == "__main__":
    file_path = 'res/images/numbers_45degree.png'
    img = load_img(file_path)

    find_lines(img)
