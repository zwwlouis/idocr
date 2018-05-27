# -*- coding: utf-8 -*-
import os
import cv2
from idcard_ocr.settings import BASE_DIR
from numpy import ndarray, array
import numpy as np
import sys

RESOURCE = os.path.join(BASE_DIR, "django_web/resource")
TEMP = os.path.join(BASE_DIR, "django_web/temp")
SHOW_IMG = sys.platform == "win32"


def get_imgs_from_path(file_dir):
    if not os.path.exists(file_dir):
        return []
    return os.listdir(file_dir)


def read_img(file_path, width=1000, flags=1):
    if not os.path.exists(file_path):
        return None
    image = cv2.imread(file_path, flags=flags)
    shape = image.shape
    h_after_resize = int(shape[0] / shape[1] * width)
    image = cv2.resize(image, (width, h_after_resize))
    return image


def get_mask(file_name, flags=1):
    file_path = os.path.join(RESOURCE, "mask", file_name)
    image = cv2.imread(file_path, flags=flags)
    return image;


def write_img(img, file_path, file_name):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    cv2.imwrite(os.path.join(file_path, file_name), img)


def write_middle_result(img, file_name, folder="result"):
    result_file_path = os.path.join(TEMP, folder)
    if not os.path.exists(result_file_path):
        os.makedirs(result_file_path)
    cv2.imwrite(os.path.join(result_file_path, file_name), img)


def img_resize(img, dwidth):
    """
    :param img:
    :param dwidth:
    :return:
    """
    if isinstance(img, ndarray):
        size = img.shape
    elif isinstance(img, list):
        imgArray = array(img)
        size = imgArray.size
        if len(size) < 2:
            return None
    else:
        return None
    height = size[0]
    width = size[1]
    scale = dwidth / width
    dheight = int(height * scale)
    nImg = cv2.resize(img, dsize=(dwidth, dheight), interpolation=cv2.INTER_CUBIC)
    return nImg, scale


max_win_width = 1000
max_win_height = 800


def showimg(img, win_name="test", wait_flag=False):
    # 非windows环境不画图
    if not SHOW_IMG:
        return
    size = img.shape
    # print size
    height = size[0]
    width = size[1]
    if width > max_win_width:
        height = int(height / width * max_win_width)
        width = max_win_width
    if height > max_win_height:
        width = int(width / height * max_win_height)
        height = max_win_height
    cv2.namedWindow(win_name, 0)
    cv2.resizeWindow(win_name, width, height)
    cv2.imshow(win_name, img)
    if wait_flag:
        cv2.waitKey()


def img_joint(img_turple, axis=0):
    """
    横向拼接图片元组生成一张大图
    :param img_turple:
    :param axis: 0-纵向拼接  1-横向拼接
    :return:
    """
    if len(img_turple) < 1:
        raise ValueError("no pic param")
    mask = np.ones((3,), dtype=np.int32)
    mask[axis] = 0
    final_img = None
    for img in img_turple:
        shape = img.shape
        if len(shape) < 2:
            raise ValueError("pic shape error %s" % str(shape))
        if len(shape) < 3:
            img = img[:, np.newaxis]
            # 将灰度图转成伪彩图
            img = np.concatenate((img, img, img), axis=2)
        if final_img is None:
            # 初始化拼接图
            final_shape = img.shape * mask
            final_img = np.zeros(final_shape, dtype=np.uint8)
        shape_dif = (np.asarray(final_img.shape) - np.asarray(img.shape)) * mask
        # 补全尺寸
        for i in range(3):
            dif_i = shape_dif[i]
            if dif_i > 0:
                img = enlarge(img, len=dif_i, axis=i)
            else:
                final_img = enlarge(final_img, len=-dif_i, axis=i)
        # 拼接图片
        final_img = np.concatenate((final_img, img), axis=axis)
    return final_img


def enlarge(img, len=0, axis=0):
    """
    将图片沿指定坐标轴方向伸展
    :param img:
    :param len:
    :param axis:
    :return:
    """
    if len == 0:
        return img
    len_up = int(len / 2)
    len_down = len - len_up
    shape = np.asarray(img.shape, dtype=np.int32)
    shape[axis] = len_up
    up_part = np.zeros(shape, dtype=np.uint8)
    img = np.concatenate((up_part, img), axis=axis)
    shape[axis] = len_down
    down_part = np.zeros(shape, dtype=np.uint8)
    img = np.concatenate((img, down_part), axis=axis)
    return img


if __name__ == '__main__':
    file_dir = get_imgs_from_path(os.path.join(RESOURCE, "idcard_img"))
    file_path_1 = os.path.join(RESOURCE, "idcard_img", file_dir[0])
    image_1 = read_img(file_path_1)
    file_path_2 = os.path.join(RESOURCE, "idcard_img", file_dir[1])
    image_2 = read_img(file_path_2)
    file_path_3 = os.path.join(RESOURCE, "idcard_img", file_dir[2])
    image_3= read_img(file_path_3)
    img = img_joint((image_1, image_2,image_3), axis=2)
    # cv2.imwrite("test.jpg", img)
    showimg(img, wait_flag=True)
