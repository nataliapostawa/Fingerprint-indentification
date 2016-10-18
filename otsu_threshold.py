import cv2
import numpy as np
from PIL import Image

#opencv.blogspot.com

img = cv2.imread('example.png', 0)

#img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 3)

#img = cv2.GaussianBlur(img, (3,3), 0)

ret,img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imwrite('otsu_gaussianblur.png', img)
cv2.waitKey(0)
cv2.destroyAllWindows()