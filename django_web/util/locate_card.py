# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
from django_web.model import ServiceException
from idcard_ocr.settings import BASE_DIR
RESOURCE_PATH = os.path.join(BASE_DIR, "django_web/resource")
MASK = os.path.join(BASE_DIR,"django_web/resource","mask")


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
        template = os.path.join(MASK, 'idcard_mask.jpg')
        MIN_MATCH_COUNT = 10
        # idocr = idcardocr.idcardocr()
        img_template = cv2.UMat(cv2.imread(template, 0))  # queryImage in Gray
        img_template = self.img_resize(img_template, 640)
        # self.showimg(img1)

        img_target = self.img_resize(cv2.UMat(img_target), 1920)

        # img1 = idocr.hist_equal(img1)
        img_target_gray = cv2.cvtColor(img_target,cv2.COLOR_BGR2GRAY)  # trainImage in Gray


        #  Initiate SIFT detector


        sift = cv2.xfeatures2d.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img_template, None)
        kp2, des2 = sift.detectAndCompute(img_target_gray, None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=10)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)


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
            h, w = cv2.UMat.get(img_template).shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)


            dst = cv2.perspectiveTransform(pts, M)
            # 参数
            img_target_gray = cv2.polylines(img_target_gray, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
            # self.showimg(img2)
            # 进行图像矫正（透视变换）
            M_r, mask_r = cv2.findHomography(dst, pts, 0, 5.0)
            im_r = cv2.warpPerspective(img_target, M_r, (w, h))
            # self.showimg(im_r)
        else:
            print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
            matchesMask = None
            raise ServiceException("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))

        # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
        #           singlePointColor = None,
        #           matchesMask = matchesMask, # draw only inliers
        #           flags = 2)
        # img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
        # plt.imshow(img3, 'gray'),plt.show()
        # im_r图像矫正结果，img2_中间结果
        return im_r, img_target_gray

    def showimg(self, img):
        cv2.namedWindow("contours", 0)
        # cv2.resizeWindow("contours", 1600, 1200);
        cv2.imshow("contours", img)
        cv2.waitKey()

    def img_resize(self, imggray, dwidth):
        # print 'dwidth:%s' % dwidth
        crop = imggray
        size = crop.get().shape
        height = size[0]
        width = size[1]
        height = int(height * dwidth / width)
        crop = cv2.resize(src=crop, dsize=(dwidth, height), interpolation=cv2.INTER_CUBIC)
        return crop


locate_card = LocateCard()

if __name__ == "__main__":
    idfind = LocateCard()
    result,mid = idfind.locate_with_file("D:/git/idocr/django_web/resource/idcard_img/13042519680705345X.jpg")
    idfind.showimg(result)
