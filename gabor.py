import numpy as np
import cv2

def build_filters():
	filters = []
	ksize = 31
	for theta in np.arange(0, np.pi, np.pi / 16):
		kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
		kern /= 1.5*kern.sum()
		filters.append(kern)
	return filters

def process(img, filters):
	accum = np.zeros_like(img)
	for kern in filters:
		fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
		np.maximum(accum, fimg, accum)
	return accum

img = cv2.imread('example.png', 0)

filters = build_filters()
res = process(img, filters)

for r,row in enumerate(res):
    for c,value in enumerate(row):
        res[r][c] = 255 - value

cv2.imshow('2', res)
cv2.imwrite('2.png', res)
cv2.waitKey(0)
cv2.destroyAllWindows()