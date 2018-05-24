import cv2
import numpy as np
import math


def swt(image):
    [M,N] = image.shape
    rays = []
    swt = np.full(image.shape, np.inf)

    # canny edge detection
    edges = cv2.Canny(image,100,300)
    sobelx = -1 * cv2.Sobel(image,cv2.CV_64F,1,0,ksize=3)
    sobely = -1 * cv2.Sobel(image,cv2.CV_64F,0,1,ksize=3)

    # orientatio image
    theta = np.arctan2(sobely, sobelx)

    for x in range(N):
        for y in range(M):
            if edges[y,x] > 0:
                step_x = sobelx[y, x]
                step_y = sobely[y, x]
                ray = []

                ray.append((x,y))
                prev_x, prev_y, i = x, y, 0
                while(True):
                    i += 1
                    cur_x = math.floor(x + np.cos(theta[y, x]) * i)
                    cur_y = math.floor(y + np.sin(theta[y, x]) * i)

                    if cur_x != prev_x or cur_y != prev_y:

                        if cur_x < 0 or cur_x >= N or cur_y < 0 or cur_y >= M:
                            break
                        elif edges[cur_y, cur_x] > 0:
                            # found edge,
                            ray.append((cur_x, cur_y))
                            theta_point = theta[y, x]
                            if np.abs(theta[cur_y, cur_x] - np.pi) - theta[cur_y,cur_x] > np.pi/6:
                                thickness = np.round(np.sqrt((cur_x - x) * (cur_x - x) + (cur_y - y) * (cur_y - y)))
                                for (rp_x, rp_y) in ray:
                                    swt[rp_y, rp_x] = min(thickness, swt[rp_y, rp_x])
                                rays.append(ray)
                            break
                        # this is positioned at end to ensure we don't add a point beyond image boundary
                        ray.append((cur_x, cur_y))
                        prev_x = cur_x
                        prev_y = cur_y

    for ray in rays:
        median = np.median([swt[y, x] for (x, y) in ray])
        for (x, y) in ray:
            swt[y, x] = min(median, swt[y, x]) 

    if True:
        cv2.imwrite('swt.jpg', swt * 100)
        cv2.imwrite('canny.jpg', edges)
        cv2.imwrite('angle.jpg', theta)
        
    return swt
              
class Component:
    components_list = None
    label_id = None
    coord = None

    def __init__(self, component_list, label_id, coord):
        self.components_list = component_list
        self.label_id = label_id
        self.coord = coord

def connect_component(image):
    ## One compnent at a time
    ## component is a list of list cordinate of each component
    [M,N] = image.shape
    label_id = 1
    label_image = np.zeros((M,N))
    components = []
    for x in range(N):
        for y in range(M):
            cur_p = image[y,x]
            if label_image[y,x] == 0 and image[y,x] != np.inf: 
                que = []
                que.append((x,y))
                new_component = make_new_component(image, label_image, que, label_id, image[y,x])
                components.append(new_component)
                label_id += 1
    return label_image, components, label_id

def make_new_component(image, label_image, que, label_id, label_value):
    new_component = []
    [x,y] = que[0]
    N,S,E,W = y,y,x,x
    while que:
        [x,y] = que.pop(0)
        new_component.append((x,y))
        if label_image[y,x] == 0 and image[y,x] != np.inf:
            if(y < N):
                N = y
            elif(y > S):
                S = y
            if(x > E):
                E = x
            elif(x < W):
                W = x
            label_image[y,x] = label_id
            add_neighbour(image, label_image, [x,y], que)
    return Component(new_component, label_id, [N,S,E,W])
    
def add_neighbour(image, label_image, coord, que):
    [M,N] = image.shape
    c_x,c_y = coord
    neighboor = [[c_y+1,c_x],[c_y,c_x+1],[c_y-1,c_x],[c_y,c_x-1],[c_y+1,c_x+1],[c_y+1,c_x-1],[c_y-1,c_x+1],[c_y-1,c_x-1]]
    for y,x in neighboor:
        if y >= M or y <= 0 or x >= N or x <= 0:
            continue
        a = np.max([image[y,x], image[c_y, c_x]])
        b = np.min([image[y,x], image[c_y, c_x]])
        if a/b <= 3.0:
            que.append((x,y))

def find_letter(label_image, compoments):
    letters = []
    for i,component in enumerate(compoments):
        [N,S,E,W] = component.coord
        width, height = E - W, S - N
        diameter = np.sqrt(width * width + height * height)
        # if width < 1 or height < 1:
        #     # print(width,height)
        #     label_image[label_image == i] = 0
        #     continue

        # if width / height > 10 or height / width > 10:
        #     label_image[label_image == i] = 0
        #     continue

        letters.append(component)
    return label_image, letters


def remove_lines(label_image, n_label):
    image = np.zeros(label_image.shape)
    # image = image.astype(np.uint8)
    for i in range(n_label):
        img = label_image == i
        img = img.astype(np.uint8)  #convert to an unsigned byte
        kernel = np.ones((3,3),np.uint8)
        img = cv2.erode(img, kernel)
        image += img

    image *= 255
    cv2.imshow("text area", image) 
    cv2.waitKey()
    cv2.destroyAllWindows()

def draw_label(mask, label_id):
    img = mask == label_id
    img = img.astype(np.uint8)  #convert to an unsigned byte
    img*=255
    cv2.imwrite('test.png', img)
    cv2.imshow("text area", img) 
    cv2.waitKey()
    cv2.destroyAllWindows()

def draw_mask(mask):
    bin_image = mask > 0
    bin_image = bin_image.astype(np.uint8)
    bin_image *= 255
    
    kernel = np.ones((3,3),np.uint8)
    bin_image = cv2.erode(bin_image, kernel)

    cv2.imwrite('mask.png', bin_image)
    cv2.imshow("text area", bin_image) 
    cv2.waitKey()
    cv2.destroyAllWindows()
    return bin_image

def main():
    import test_find_contour as test 
    import text_segmentation as text_seg

    IMAGE_PATH = '../res/images/'

    # image_filename = 'lorem.png'
    # image = cv2.imread(IMAGE_PATH+image_filename,0)
    
    image_filename = 'text_skew.png'
    image = cv2.imread(IMAGE_PATH+image_filename,0)

    SWT = swt(image)
    CC,cc_list,n_label = connect_component(SWT)
    # find_letter(CC,cc_list)
    # for i in range(100,500):
    #     draw_label(CC,i)
    draw_mask(CC)
    # remove_lines(CC,n_label)


if __name__ == '__main__':
    main()