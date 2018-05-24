import os
import cv2
from idcard_ocr.settings import BASE_DIR
from matplotlib import pyplot as plt
RESOURCE = os.path.join(BASE_DIR, "django_web/resource")
MASK = os.path.join(RESOURCE,"mask")
IMAGE = os.path.join(RESOURCE,"idcard_img")


def test_orb():
    img1 = get_mask("idcard_mask.jpg")          # queryImage
    img2 = get_idcard(0) # trainImage

    # Initiate SIFT detector
    orb = cv2.ORB()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], flags=2)

    plt.imshow(img3),plt.show()


def get_mask(name):
    file_path = os.path.join(MASK,name)
    img = cv2.imread(file_path,0)
    return img

def get_idcard(num):
    files = os.listdir(IMAGE)
    num = num % len(files)
    file_name = files[num]
    file_path = os.path.join(IMAGE,file_name)
    return cv2.imread(file_path,0)


if __name__ == '__main__':
    test_orb()