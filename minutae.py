import numpy as np
import cv2

cells = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]

def minutiae_at(img, i, j):
	values = [img[i + k][j + l] for k, l in cells]

	crossings = 0
	for k in range(0, 8):
		crossings += abs(int(values[k]/255) - int(values[k + 1]/255))
	crossings /= 2
	if img[i][j] == 255:
		if crossings == 1:
			return "ending"
		if crossings == 3:
			return "bifurcation"
	return "none"

def calculate_minutiaes(img):

	(x, y) = img.shape
	colors = { "ending" : (150, 0, 0), "bifurcation" : (0, 0, 255) }
	img2 = img.copy()
	img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)

	for i in range(1, x - 1):
		for j in range(1, y - 1):
			minutiae = minutiae_at(img, i, j)
			if minutiae == "bifurcation":
				img2 = cv2.circle(img2, (j,i), 3, colors[minutiae], 1)

	return img2
