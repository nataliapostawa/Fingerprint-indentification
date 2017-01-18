import cv2
import numpy as np
from skimage import img_as_ubyte
from skimage.filters import threshold_otsu, threshold_adaptive

def binarise(img):
    img = cv2.GaussianBlur(img, (3,3), 0)

    #ret,img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    global_thresh = threshold_otsu(img)
    binary_global = img > global_thresh

    # block_size = 49
    # binary = threshold_adaptive(img, block_size, offset=10)

    return img_as_ubyte(binary_global)
