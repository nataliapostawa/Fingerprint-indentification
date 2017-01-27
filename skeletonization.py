import cv2
import numpy as np
from skimage import img_as_ubyte
from skimage.morphology import skeletonize as skel

def skeletonize(img):
	# size = np.size(img)
	# skel = np.zeros(img.shape, np.uint8)
	#
	# ret, img = cv2.threshold(img, 127, 255, 0)
	# element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
	# done = False
	#
	# while not done:
	# 	eroded = cv2.erode(img, element)
	# 	temp = cv2.dilate(eroded, element)
	# 	temp = cv2.subtract(img, temp)
	# 	skel = cv2.bitwise_or(skel, temp)
	# 	img = eroded.copy()
	#
	# 	zeros = size - cv2.countNonZero(img)
	# 	if (zeros == size):
	# 	 	done = True
	#
	# for r, row in enumerate(skel):
	#     for c, value in enumerate(row):
	#         skel[r][c] = 255 - value

	return img_as_ubyte(skel(img))
