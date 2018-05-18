import cv2
import numpy as np
import math

def swt(image):
    # create empty image, initialized to infinity
    swt = np.full(image.shape, np.inf)
    rays = []

    edges = cv2.Canny(image,100,300)
    sobelx = cv2.Sobel(edges,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(edges,cv2.CV_64F,0,1,ksize=5)   
    theta = np.arctan2(sobely, sobelx)
    
    step_x_g = sobelx
    step_y_g = sobely
    mag_g = np.sqrt(step_x_g * step_x_g + step_y_g * step_y_g)
    np.set_printoptions(threshold=np.nan)

    for x in range(edges.shape[1]):
        for y in range(edges.shape[0]):
            if edges[y, x] > 0:
                step_x = step_x_g[y, x]
                step_y = step_y_g[y, x]
                mag = mag_g[y, x]
                ray = []
                ray.append((x, y))
                prev_x, prev_y, i = x, y, 0
                while True:
                    i += 1
                    cur_x = math.floor(x + np.cos(step_x) * i)
                    cur_y = math.floor(y + np.cos(step_y) * i)

                    if cur_x != prev_x or cur_y != prev_y:
                        # we have moved to the next pixel!
                        try:
                            if edges[cur_y, cur_x] > 0:
                                # found edge,
                                ray.append((cur_x, cur_y))
                                theta_point = theta[y, x]
                                alpha = theta[cur_y, cur_x]
                                # print(theta[cur_y, cur_x] - theta[y,x] , np.pi/2.0)
                                if (theta[cur_y, cur_x] - theta[y,x]) < np.pi/2.0:
                                    thickness = np.sqrt((cur_x - x) * (cur_x - x) + (cur_y - y) * (cur_y - y))
                                    for (rp_x, rp_y) in ray:
                                        swt[rp_y, rp_x] = min(thickness, swt[rp_y, rp_x])
                                    rays.append(ray)
                                break
                            # this is positioned at end to ensure we don't add a point beyond image boundary
                            ray.append((cur_x, cur_y))
                        except IndexError:
                            # reached image boundary
                            break
                        prev_x = cur_x
                        prev_y = cur_y
    # Compute median SWT
    for ray in rays:
        median = np.median([swt[y, x] for (x, y) in ray])
        for (x, y) in ray:
            swt[y, x] = min(median, swt[y, x])
    if True:
        cv2.imwrite('swt.jpg', swt * 100)
    return swt

def main():
    import test_find_contour as test 
    import text_segmentation as text_seg

    IMAGE_PATH = '../../res/images/'
    image_filename = '1.png'
    image = cv2.imread(IMAGE_PATH+image_filename,0)
    text_area = text_seg.detect_text(image)
    cv2.imshow("text area", text_area[0]) 
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    SWT = swt(text_area[0])
    norm = cv2.normalize(SWT,SWT)
    cv2.imshow("SWT", SWT) 
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()