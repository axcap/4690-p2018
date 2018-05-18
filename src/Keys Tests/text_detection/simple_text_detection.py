import cv2 
import numpy as np 
from skimage.filters.rank import median
from skimage.morphology import disk

def detect_text(image):
    """
        input: image
        return: list of text segments images
    """
    # may not need edge, if we do SWT
    edge_img = cv2.Canny(image,100,200)

    ret, thresh_img = cv2.threshold(image, 20, 255, cv2.THRESH_BINARY)
    # component = cv2.connectedComponentsWithStats(thresh_img, )
     
    blur_image = cv2.medianBlur(edge_img, 3)
    kernel = np.ones((5,5),np.uint8)

    # number of iteration need to be bether impremented, currently hardcoded
    morph_img = cv2.dilate(blur_image, kernel, iterations=5)

    img, contours, hier = cv2.findContours(morph_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    component = cv2.connectedComponents(morph_img,connectivity=8)
    text_area_found = []

    vis = image.copy()
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)

        cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 255, 255), 2)
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 0), 1)
        text_area_found.append(image[y+1:y+h, x+1:x+w]) # +1 to remove edge

    cv2.imshow("Image segment", vis) 
    cv2.waitKey()
    cv2.destroyAllWindows()
    return text_area_found

def main():
    IMAGE_RESULT_PATH = '../../../res/images/text_segment_result/'
    IMAGE_PATH = '../../../res/images/'

    image_filename = 'ReceiptSwiss.jpg'
    image = cv2.imread(IMAGE_PATH+image_filename,0)
    text_area = detect_text(image)

    image_filename = 'numbers.png'
    image = cv2.imread(IMAGE_PATH+image_filename,0)
    text_area = detect_text(image)

if __name__ == '__main__':
    main()