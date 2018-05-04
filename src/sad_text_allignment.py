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

    lines = cv2.HoughLinesP(edges,1,np.pi/180,200)

    #print(lines[0][0])
    #for x,y,z,a in lines[0]:
    #    print(x, y, z, a)
    #plt.imshow(lines)
    #plt.show()

    N = lines.shape[0]
    for i in range(N):
        x1 = lines[i][0][0]
        y1 = lines[i][0][1]
        x2 = lines[i][0][2]
        y2 = lines[i][0][3]

        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),10)

    cv2.imwrite('houghlines3.jpg',img)


if __name__ == "__main__":
    file_path = 'res/images/numbers_45degree.png'
    img = load_img(file_path)

    find_lines(img)
