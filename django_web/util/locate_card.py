# -*- coding: utf-8 -*-
import numpy as np
import time
import cv2
import os
from django_web.model import ServiceException
from idcard_ocr.settings import BASE_DIR
from django_web.util import img_util as iu

RESOURCE_PATH = os.path.join(BASE_DIR, "django_web/resource")
MASK = os.path.join(BASE_DIR, "django_web/resource", "mask")

template = os.path.join(MASK, 'idcard_mask.jpg')
img_template = cv2.imread(template, 0)  # queryImage in Gray

XFEATURE_IMG_WIDTH = 400
img_template_small,_ = iu.img_resize(img_template, XFEATURE_IMG_WIDTH)
MIN_MATCH_COUNT = 10


class LocateCard:
    def __init__(self):
        pass

    def locate_with_file(self, file_name):
        # img2 = idocr.hist_equal(img2)
        img_target = cv2.imread(file_name)
        return self.locate(img_target)

    # target为需要识别的图像
    def locate(self, img_target):
        """
        寻找图像中的身份证并进行图像矫正（透视变换）
        :param target:
        :return:
        """
        start = time.time()
        img_target_gray = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)  # trainImage in Gray
        img_target_gray_small,_ = iu.img_resize(img_target_gray, XFEATURE_IMG_WIDTH)

        img_target_gray = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)

        pts, dst = self.findMatchWithXFeature(img_target_gray_small, img_template_small)
        hl, wl = img_target_gray.shape
        h_temp, w_temp = img_template_small.shape
        ratio = wl / XFEATURE_IMG_WIDTH
        pts = pts * ratio
        dst = dst * ratio
        # 参数
        img_target_gray = cv2.polylines(img_target_gray, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        # FIXME
        iu.showimg(img_target_gray)

        # 进行图像矫正（透视变换）
        M_r, mask_r = cv2.findHomography(dst, pts, 0, 5.0)
        im_r = cv2.warpPerspective(img_target, M_r, (int(w_temp*ratio), int(h_temp*ratio)))

        # FIXME
        iu.showimg(im_r)
        time_used = time.time() - start
        start += time_used
        print("match timeUsed = %d ms" % (int(time_used * 1000)))
        # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
        #           singlePointColor = None,
        #           matchesMask = matchesMask, # draw only inliers
        #           flags = 2)
        # img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
        # plt.imshow(img3, 'gray'),plt.show()
        # im_r图像矫正结果
        return im_r, img_target_gray

    def findMatchWithXFeature(self, target, template):
        """
        检测两张图片的相似程度，并找到目标图片中的模板位置
        :param target: 目标图片（灰度图像）
        :param template: 模板图片（灰度图像）
        :return:
        """
        sift = cv2.xfeatures2d.SIFT_create()

        kp1, des1 = sift.detectAndCompute(template, None)
        kp2, des2 = sift.detectAndCompute(target, None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=10)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(des1, des2, k=2)

        # FIXME
        drawImg = None
        # Sort them in the order of their distance.
        # matches = sorted(matches, key=lambda x: x.distance)
        cv2.drawMatches(template,kp1,target,kp2,matches[:10],drawImg)
        cv2.waitKey(0)
        # store all the good matches as per Lowe's ratio test.
        # 两个最佳匹配之间距离需要大于ratio 0.7,距离过于相似可能是噪声点
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        # reshape为(x,y)数组
        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            # 用HomoGraphy计算图像与图像之间映射关系, M为转换矩阵
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            # 使用转换矩阵M计算出img1在img2的对应形状
            h, w = template.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            return pts, dst
        else:
            print("身份证匹配度不足 - %d/%d" % (len(good), MIN_MATCH_COUNT))
            matchesMask = None
            raise ServiceException("身份证匹配度不足 - %d/%d" % (len(good), MIN_MATCH_COUNT))

    def showimg(self, img):
        cv2.namedWindow("contours", 0)
        # cv2.resizeWindow("contours", 1600, 1200);
        cv2.imshow("contours", img)
        cv2.waitKey()


locate_card = LocateCard()

if __name__ == "__main__":
    idfind = LocateCard()
    result, mid = idfind.locate_with_file("D:/git/idocr/django_web/resource/idcard_img/13042519680705345X.jpg")
    idfind.showimg(result)
