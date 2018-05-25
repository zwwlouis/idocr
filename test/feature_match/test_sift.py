# -*- coding: utf-8 -*-
import os
import cv2
from idcard_ocr.settings import BASE_DIR
import numpy as np
from django_web.util import img_util as iu
from django_web.util.locate_card import locate_card
import time

RESOURCE = os.path.join(BASE_DIR, "django_web/resource")
MASK = os.path.join(RESOURCE, "mask")
IMAGE = os.path.join(RESOURCE, "card_locate")

MATCH_PIC_WIDTH = 600
MIN_MATCH_COUNT = 10
def get_mask(name):
    file_path = os.path.join(MASK, name)
    img = cv2.imread(file_path, 0)
    return img


def get_idcard(num,flags = None):
    files = os.listdir(IMAGE)
    num = num % len(files)
    file_name = files[num]
    file_path = os.path.join(IMAGE, file_name)
    return cv2.imread(file_path, flags), file_name


def get_idcard_count():
    files = os.listdir(IMAGE)
    return len(files)

def save_match(folder, pic_name, img1, kp1, img2, kp2, matches):
    path = os.path.join("result", folder)
    if not os.path.exists(path):
        os.makedirs(path)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, outImg=None,
                           flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    if pic_name.find(".") < 0:
        pic_name += ".jpg"
    cv2.imwrite(os.path.join(path, pic_name), img3)


def save_img(folder, pic_name, img):
    path = os.path.join("result", folder)
    if not os.path.exists(path):
        os.makedirs(path)
    if pic_name.find(".") < 0:
        pic_name += ".jpg"
    cv2.imwrite(os.path.join(path, pic_name), img)


def orb_test():
    img1 = get_mask("idcard_mask.jpg")  # queryImage
    for i in range(get_idcard_count()):
        img2, img_name = get_idcard(i)  # trainImage
        save_name = "result" + img_name
        img1, _ = iu.img_resize(img1, MATCH_PIC_WIDTH)
        img2, _ = iu.img_resize(img2, MATCH_PIC_WIDTH)
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
        matches = sorted(matches, key=lambda x: x.distance)
        img3 = np.zeros((1000, 1000, 3), dtype=np.uint8)
        # Draw first 10 matches.
        # img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], outImg=img3,flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
        #
        # plt.imshow(img3),plt.show()
        folder = "test_orb"
        save_match(folder, save_name, img1, kp1, img2, kp2, matches[:10])
        print("finish pic %s" % save_name)


sift_time_use = 0
def sift_test(save = False):
    global sift_time_use
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    img1 = get_mask("idcard_mask.jpg")  # queryImage
    for i in range(get_idcard_count()):
        img2, img_name = get_idcard(i)  # trainImage
        save_name = "result" + img_name
        start = time.time()
        img1, _ = iu.img_resize(img1, MATCH_PIC_WIDTH)
        img2, _ = iu.img_resize(img2, MATCH_PIC_WIDTH)

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        end = time.time()
        sift_time_use += end - start
        if save:
            folder = "test_sift"
            save_match(folder, save_name, img1, kp1, img2, kp2, good)
            print("finish pic %s" % save_name)
    print("sift pic num=%d time used = %d"%(get_idcard_count(),int(sift_time_use*1000)))


def sift_test_with_flann(save = False):
    global sift_time_use
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    img1 = get_mask("idcard_mask.jpg")  # queryImage
    for i in range(get_idcard_count()):
        img2, img_name = get_idcard(i)  # trainImage
        save_name = "result" + img_name
        start = time.time()
        # 模板
        img1, _ = iu.img_resize(img1, 600)
        # 目标图片
        img2, _ = iu.img_resize(img2, 600)

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=10)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # Apply ratio test
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)


        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            # 用HomoGraphy计算图像与图像之间映射关系, M为转换矩阵
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            # 使用转换矩阵M计算出img1在img2的对应形状
            h, w = img1.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)

            # 参数
            img_target_gray = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
            # iu.showimg(img_target_gray)

            # 进行图像矫正（透视变换）
            M_r, mask_r = cv2.findHomography(dst, pts, 0, 5.0)
            ratio = 1
            im_r = cv2.warpPerspective(img2, M_r, (int(w*ratio), int(h*ratio)))
            # iu.showimg(im_r)
            save_img("perspective",save_name,im_r)
        else:
            print("身份证匹配度不足 - %d/%d" % (len(good), MIN_MATCH_COUNT))
            matchesMask = None

        end = time.time()
        sift_time_use += end - start
        if save:
            folder = "test_sift_with_flann"
            save_match(folder, save_name, img1, kp1, img2, kp2, good)
            print("finish pic %s" % save_name)
    print("sift pic num=%d time used = %d"%(get_idcard_count(),int(sift_time_use*1000)))


def test_locate_card(save = False):
    global sift_time_use
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    for i in range(get_idcard_count()):
        target_img, img_name = get_idcard(i)  # trainImage
        save_name = "result" + img_name
        start = time.time()
        im_r, img_target_gray = locate_card.locate(target_img)


if __name__ == '__main__':
    print("start test")
    sift_test_with_flann()

