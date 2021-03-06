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
    img = None
    if flags is None:
        img = cv2.imread(file_path)
    else:
        img = cv2.imread(file_path,flags)
    return img, file_name


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

XFEATURE_IMG_WIDTH = 600
template = get_mask("idcard_mask.jpg")
img_template_small,_ = iu.img_resize(template,XFEATURE_IMG_WIDTH)

first_homo_time = 0
second_homo_time = 0
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
        global second_homo_time
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
        # iu.showimg(img_target_gray,"large")

        # 进行图像矫正（透视变换）
        h_time = time.time()
        M_r, mask_r = cv2.findHomography(dst, pts, 0, 5.0)
        second_homo_time += time.time() - h_time
        im_r = cv2.warpPerspective(img_target, M_r, (int(w_temp*ratio), int(h_temp*ratio)))

        # FIXME
        # iu.showimg(im_r,"after",True)
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
        global first_homo_time
        sift = cv2.xfeatures2d.SIFT_create()

        kp1, des1 = sift.detectAndCompute(template, None)
        kp2, des2 = sift.detectAndCompute(target, None)

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
            f_time = time.time()
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            first_homo_time += time.time()-f_time
            matchesMask = mask.ravel().tolist()
            # 使用转换矩阵M计算出img1在img2的对应形状
            h, w = template.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            # 参数
            # FIXME
            # target_border = cv2.polylines(target, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
            # iu.showimg(target_border,"small")
            return pts, dst
        else:
            print("身份证匹配度不足 - %d/%d" % (len(good), MIN_MATCH_COUNT))
            matchesMask = None
            raise ValueError("身份证匹配度不足 - %d/%d" % (len(good), MIN_MATCH_COUNT))



locate_card = LocateCard()

sift_time_use = 0
sift_count = 0
def test_locate_card(save = False):
    global sift_time_use
    global sift_count
    # Initiate SIFT detector
    for i in range(get_idcard_count()):
        try:
            target_img, img_name = get_idcard(i)  # trainImage
            save_name = "result" + img_name
            start = time.time()
            im_r, img_target_gray = locate_card.locate(target_img)
            end = time.time()
            sift_time_use += end - start
            sift_count += 1
        except Exception as error:
            print(error)
    print("图片处理%d张 总用时 %.2f s"%(sift_count,sift_time_use))
    print("first homo spend = %d ms"%(int(first_homo_time*1000)))
    print("second homo spend = %d ms"%(int(second_homo_time*1000)))

if __name__ == '__main__':
    print("start test")
    test_locate_card()

