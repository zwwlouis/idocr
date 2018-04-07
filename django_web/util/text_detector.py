import sys

import cv2
import numpy as np
import os
from django_web.util import face_detector as fd
from django_web.util import img_util as iu
from idcard_ocr.settings import BASE_DIR
RESOURCE = os.path.join(BASE_DIR, "django_web/resource")
MIDDLE_RESULT = "middle_result"

def preprocess(gray,debug):
    # 1. Sobel算子，x方向求梯度
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize = 3)
    # 2. 二值化
    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)

    # 3. 膨胀和腐蚀操作的核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))

    # 4. 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations = 1)

    # 5. 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
    erosion = cv2.erode(dilation, element1, iterations = 1)

    # 6. 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(erosion, element2, iterations = 3)
    if debug:
        # 7. 存储中间图片
        iu.write_img(binary,MIDDLE_RESULT,"binary.png")
        iu.write_img(dilation, MIDDLE_RESULT, "dilation.png")
        iu.write_img(erosion, MIDDLE_RESULT, "erosion.png")
        iu.write_img(dilation2, MIDDLE_RESULT, "dilation2.png")
    return dilation2


def findTextRegion(img,debug):
    region = []

    # 1. 查找轮廓
    _, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 2. 筛选那些面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        area = cv2.contourArea(cnt)

        # 面积小的都筛选掉
        if(area < 1000):
            continue

        # 轮廓近似，作用很小
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(cnt)
        if debug:
            print("rect is: ")
            print(rect)

        # box是四个点的坐标
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])

        # 筛选那些太细的矩形，留下扁的
        if(height > width * 1.2):
            continue

        region.append(box)

    return region


def detect(img, debug=False):
    # 1.  转化成灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #    window = fd.find_face(img)
    # if window is not None:
    #     gray = fd.remove_face(gray,window)
    #     fd.box_face(img,window)

    # 2. 形态学变换的预处理，得到可以查找矩形的图片
    dilation = preprocess(gray,debug)

    # 3. 查找和筛选文字区域
    region = findTextRegion(dilation,debug)

    # 4. 用绿线画出这些找到的轮廓
    for box in region:
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

    if debug:
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.imshow("img", img)

        # 带轮廓的图片
        iu.write_img(img, MIDDLE_RESULT, "contours.png")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return img


if __name__ == '__main__':
    # 读取文件
    imagePath = sys.argv[1]
    img = cv2.imread(imagePath)
    detect(img)