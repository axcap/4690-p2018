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
                

def connect_component(image):
    ## One compnent at a time
    [M,N] = image.shape
    label_id = 1
    label_image = np.zeros((M,N))

    for x in range(N):
        for y in range(M):
            cur_p = image[y,x]
            if label_image[y,x] == 0 and image[y,x] != np.inf: 
                que = []
                que.append((x,y))
                connect_from_que(image, label_image, que, label_id, image[y,x])
                label_id += 1
    return label_image


def connect_from_que(image, label_image, que, label_id, label_value):
    while que:
        [x,y] = que.pop(0)
        if label_image[y,x] == 0 and image[y,x] != np.inf:
            a = np.max([image[y,x], label_value])
            b = np.min([image[y,x], label_value])
            if a/b < 3.0:
                label_image[y,x] = label_id
                add_neighbour(image, label_image, [x,y], que)  
    
def add_neighbour(image, label_image, coord, que):
    [M,N] = image.shape
    x,y = coord
    neighboor = [[y+1,x],[y,x+1],[y-1,x],[y,x-1]] # TODO add more neighboors
    for y,x in neighboor:
        if x > 0 and x < N and y > 0 and y < M:
            que.append((x,y))       



def draw_mask(mask, label_id):
    img = mask == label_id
    img = img.astype(np.uint8)  #convert to an unsigned byte
    mask*=255
    cv2.imshow("text area", mask) 
    cv2.waitKey()
    cv2.destroyAllWindows()


def main():
    import test_find_contour as test 
    import text_segmentation as text_seg

    IMAGE_PATH = '../../res/images/'
    image_filename = 'text_example2.png'
    image = cv2.imread(IMAGE_PATH+image_filename,0)
    
    SWT = swt(image)
    CC = connect_component(SWT)
    draw_mask(CC, 30)


if __name__ == '__main__':
    main()