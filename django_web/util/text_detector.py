import sys

import cv2
import numpy as np
import os
from django_web.util import face_detector as fd
from django_web.util import img_util as iu
from idcard_ocr.settings import BASE_DIR
# from matplotlib import pyplot as plt
from django_web.util import text_recognize as tr
from django_web.model import *

RESOURCE = os.path.join(BASE_DIR, "django_web/resource")


def img_binary(gray):
    """图像二值化函数"""
    shape = gray.shape
    # 计算图像的平均灰度值
    grayf = np.array(gray, dtype=np.float64)
    ave = sum(sum(grayf)) / shape[0] / shape[1]
    print("average gray is %.2f" % ave)
    ret, binary = cv2.threshold(gray, ave / 1.5, 255, cv2.THRESH_BINARY_INV)
    return binary


def get_card_area(img, debug=False, save_folder="result"):
    """
    剪裁出身份证文字区域
    :param img:
    :param debug:
    :param save_folder:
    :return:
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 1, ksize=3)
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    # 3. 膨胀和腐蚀操作的核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))

    # 4. 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations=1)

    # 5. 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
    erosion = cv2.erode(dilation, element1, iterations=1)

    # 6. 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(erosion, element2, iterations=3)

    # 查找和筛选文字区域
    box = find_max_region(sobel)
    # 用绿线画出这些找到的轮廓
    cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
    if debug:
        # 存储中间图片
        iu.write_middle_result(binary, "1_binary.png", save_folder)
        iu.write_middle_result(dilation, "2_dilation.png", save_folder)
        iu.write_middle_result(erosion, "3_erosion.png", save_folder)
        iu.write_middle_result(dilation2, "4_dilation2.png", save_folder)
        iu.write_middle_result(img, "contour.jpg", save_folder)
        cv2.waitKey()


def preprocess(gray, debug, save_folder):
    # 1. Sobel算子，x方向求梯度
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    # 2. 二值化
    # TODO 关于二值化阈值的选择，或者二值化阈值算法的选择THRESH_TRIANGLE/THRESH_OTSU留待之后进行试验。目前来看THRESH_OTSU算法更适合当前情况下的图片
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    # binary = img_binary(gray)
    # 3. 膨胀和腐蚀操作的核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))

    # 4. 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations=1)

    # 5. 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
    erosion = cv2.erode(dilation, element1, iterations=1)

    # 6. 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(erosion, element2, iterations=3)
    if debug:
        # 7. 存储中间图片
        iu.write_middle_result(binary, "1_binary.png", save_folder)
        iu.write_middle_result(dilation, "2_dilation.png", save_folder)
        iu.write_middle_result(erosion, "3_erosion.png", save_folder)
        iu.write_middle_result(dilation2, "4_dilation2.png", save_folder)
    return dilation2


def findTextRegion(img, debug, save_path):
    region = []

    # 1. 查找轮廓
    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 2. 筛选那些面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        area = cv2.contourArea(cnt)

        # 面积小的都筛选掉
        if area < 1000:
            continue

        # 轮廓近似，作用很小
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 找到最小的矩形，该矩形可能有方向
        # rect属性 point2f 矩形中心(x,y)   point2f 矩形宽高（w,h） angle 矩阵旋转角度(>0为顺时针)
        rect = cv2.minAreaRect(cnt)
        # # 统一旋转30度查看效果
        # rect = (rect[0],rect[1],rect[2]+10)
        # if debug:
        #     print("rect is: ")
        #     print(rect)

        # box是四个点的坐标
        # box属性 [point2f,point2f,point2f,point2f] 矩形的四个顶点（x,y）
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])

        # 筛选那些太细的矩形，留下扁的
        if (height > width * 1.2):
            continue
        region.append(box)

    return region


def find_max_region(img):
    # 1. 查找轮廓
    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_cnt = None
    # 2. 筛选那些面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            max_cnt = cnt
    # 轮廓近似，作用很小
    epsilon = 0.001 * cv2.arcLength(max_cnt, True)
    approx = cv2.approxPolyDP(max_cnt, epsilon, True)
    # 找到最小的矩形，该矩形可能有方向
    # rect属性 point2f 矩形中心(x,y)   point2f 矩形宽高（w,h） angle 矩阵旋转角度(>0为顺时针)
    rect = cv2.minAreaRect(max_cnt)
    # box是四个点的坐标
    # box属性 [point2f,point2f,point2f,point2f] 矩形的四个顶点（x,y）
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    return box


def detect(img, debug=False, save_folder="result"):
    # 1.  转化成灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 提取人脸区域 face_window = [x,y,w,h]
    face_window = fd.find_face(img)

    if face_window is None:
        raise ServiceException("未识别到人脸")

    # TODO 暂时松弛边界写为固定，之后可以先扣取身份证区域，根据身份证大小决定边界大小，或者将身份证区域处理成统一大小
    shape = gray.shape
    gray = fd.remove_face(gray, face_window, relax_border=np.intp([shape[0] / 15, shape[1] / 12]))
    fd.draw_face_box(img, face_window)

    # 2. 形态学变换的预处理，得到可以查找矩形的图片
    dilation = preprocess(gray, debug, save_folder)

    # 3. 查找和筛选文字区域
    region = findTextRegion(dilation, debug, save_folder)
    # 识别文字
    result_text = ""

    image_array = []
    text_array = []

    for box in region:
        # 识别框内文字
        x_min, y_min = np.min(box, 0)
        x_max, y_max = np.max(box, 0)
        text_window = [x_min, y_min, x_max - x_min, y_max - y_min]
        if not region_filter(img, text_window, face_window):
            continue
        image_part, text = tr.ocr_result(img, text_window)
        result_text = text + "\n" + result_text
        image_array.append(image_part)
        text_array.append(text)
    print(result_text)
    print("-" * 10)

    # 4. 用绿线画出这些找到的轮廓
    for box in region:
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
        if debug:
            # 带轮廓的图片
            iu.write_middle_result(img, "contours.png", save_folder)
            cv2.waitKey(0)
    return img, image_array, text_array


def region_filter(img, text_window, face_window):
    """
    根据窗口大小和窗口和人像的相对位置来判断是否为正确的窗口
    :param img:图片
    :param text_window: [x,y,w,h]
    :param face_window: [x,y,w,h]
    :return:
    """
    # 文字框的中点应当位于头像左侧
    if (text_window[0] + text_window[2] / 2) > face_window[0]:
        return False
    shape = img.shape
    # 文字框的高度应当小于总高度的1/3
    if text_window[3] > shape[0] / 3:
        return False
    return True


def get_tess_time():
    return tr.time_cvt, tr.time_tess


if __name__ == '__main__':
    # 读取文件
    imagePath = sys.argv[1]
    img = cv2.imread(imagePath)
    detect(img)
