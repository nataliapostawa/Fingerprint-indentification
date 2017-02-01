import argparse
import cv2
from skimage import img_as_ubyte
import numpy as np
from collections import Counter

from gabor import gabor
from threshold import binarise
from skeletonization import skeletonize
from minutae import calculate_minutiaes

from matplotlib import pyplot as plt

def prepare_image(img):
    gabor_img = gabor(img)
    binarised_img = binarise(gabor_img)
    return skeletonize(binarised_img)

def compare_imagesSIFT(img1, img2):
    # Initiate SIFT detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x : x.distance)
    filtered_matches = [ item for item in matches[:10] if item.distance < 25 ]
    print(len(filtered_matches) / 10)

    # Draw first 10 matches.
    return cv2.drawMatches(img1, kp1, img2, kp2, filtered_matches, flags=2,
                           outImg=None)

def compare(img_file1, img_file2):
    img1 = prepare_image(img_file1)
    img2 = prepare_image(img_file2)

    return compare_images(img1, img2)

def get_matrix(list):
    comparison = [ [ 0 for x in range(301) ] for y in range(301) ]
    for item in list:
        comparison[item["x"]][item["y"]] = item["type"];

    return comparison

def get_occurrences(list):
    number = 300 // 8
    a = [ x for x in range(301) if x % number == 0 ]

    results = []
    data = get_matrix(list)
    for i in range(len(a)-1):
        for j in range(len(a)-1):
            dic = Counter()
            window = data[a[i]:a[i+1]][a[j]:a[j+1]]
            for row in range(len(window)):
                for col in range(len(window[row])):
                    if (window[row][col] != 0):
                        dic[window[row][col]] += 1

            results.append(dic)

    print(len(results))
    return results

def simple_matching(list1, list2):
    list1 = get_occurrences(list1)
    list2 = get_occurrences(list2)

    match = [ x == y for (x, y) in zip(list1, list2) if (x and y) ]
    print(match)
    return match.count(True) / len(match)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path1", type=str)
    parser.add_argument("file_path2", type=str)
    args = parser.parse_args()

    img1 = cv2.imread(args.file_path1, 0)
    img2 = cv2.imread(args.file_path2, 0)
    img1 = prepare_image(img1)
    img2 = prepare_image(img2)

    minutiae_img1, minutiae_list1 = calculate_minutiaes(img1)
    minutiae_img2, minutiae_list2 = calculate_minutiaes(img2)

    img = np.concatenate((minutiae_img1, minutiae_img2), axis=1)
    cv2.imshow("img", img)

    print(simple_matching(minutiae_list1, minutiae_list2))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
