import argparse
import cv2

from gabor import gabor
from threshold import binarise
from skeletonization import skeletonize
from minutae import calculate_minutiaes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", type=str)
    args = parser.parse_args()

    img = cv2.imread(args.file_path, 0)

    gabor_img = gabor(img)
    binarised_img = binarise(gabor_img)
    skeletonized = skeletonize(binarised_img)
    minutiaes = calculate_minutiaes(skeletonized)

    cv2.imshow("gabor", gabor_img)
    cv2.imshow("binarised", binarised_img)
    cv2.imshow("skeletonize", skeletonized)
    cv2.imshow("minutiaes", minutiaes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
