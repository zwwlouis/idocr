import numpy as np
import cv2
from matplotlib import pyplot as plt
from idcard_ocr.settings import BASE_DIR
from numpy import ndarray, array
import os
RESOURCE = os.path.join(BASE_DIR, "django_web/resource")
CARD = os.path.join(RESOURCE, "card_locate")
MASK = os.path.join(RESOURCE, "mask")



def testBF():
    img1 = getImg()
    img2 = cv2.imread('box_in_scene.png', 0)  # trainImage

    # Initiate SIFT detector
    orb = cv2.ORB()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)





def getImg(name,path):
    file_path = os.path.join(path,name)
    img = cv2.imread(file_path)
    return img