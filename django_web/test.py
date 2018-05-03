from django_web.util import text_detector as td
from django_web.util import img_util as iu, locate_card as lc
import os
from idcard_ocr.settings import BASE_DIR
import cv2
import time
from django_web.model import *
from django_web.util import subplot_util as su
from matplotlib import pyplot as plt

RESOURCE_PATH = os.path.join(BASE_DIR, "django_web/resource")
RESULT_PATH = "ocr_result"


def group_test_detect():
    path = os.path.join(RESOURCE_PATH, 'idcard_img')
    files = iu.get_imgs_from_path(path)

    image_array = []
    text_array = []
    for file_name in files:
        print("识别图片 %s" % file_name)
        file_path = os.path.join(path, file_name)
        if not os.path.isfile(file_path):
            continue
        img = iu.read_img(file_path,width=1000)
        try:
            contour_img, imgs, txts = td.detect(img)
            image_array += imgs
            text_array += txts
            # 保存绿框结果
            iu.write_middle_result(contour_img, file_name, "final_result")
        except ServiceException as exp:
            print("识别出错 %s" % str(exp))




def group_test_card_cut():
    path = os.path.join(RESOURCE_PATH, 'test_card_cut')
    files = iu.get_imgs_from_path(path)
    i = 0
    for file_name in files:
        i += 1
        file_path = os.path.join(path, file_name)
        if not os.path.isfile(file_path):
            continue
        img = iu.read_img(file_path)
        td.get_card_area(img, debug=True, save_folder=os.path.join(path, "middle_result%d" % i))


def single_test_detect():
    path = os.path.join(RESOURCE_PATH, 'test3')
    files = iu.get_imgs_from_path(path)
    image_array = []
    text_array = []
    for file_name in files:
        print("识别图片 %s" % file_name)
        file_path = os.path.join(path, file_name)
        if not os.path.isfile(file_path):
            continue
        img = iu.read_img(file_path, width=1000)
        try:
            contour_img, imgs, txts = td.detect(img, debug=True, save_folder=os.path.join(path, "middle_result"))
        except ServiceException as exp:
            print("识别出错 %s" % exp.message)
        image_array += imgs
        text_array += txts
        # 保存绿框结果
        iu.write_middle_result(contour_img, file_name, "final_result")
    for img in image_array:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    su.auto_subplt(image_array, 5, [plt.imshow], title=text_array)

def test_card_locater(offset=0,limit = 100):
    """
    测试身份证定位函数，从图片中找到倾斜或是旋转的身份证，对其进行重新放置
    :return:
    """
    path = os.path.join(RESOURCE_PATH, 'card_locate')
    files = iu.get_imgs_from_path(path)

    image_array = []
    text_array = []
    locate_card = lc.LocateCard()
    i = -1
    for file_name in files:
        i += 1
        if i >= (offset+limit) or i < offset:
            continue
        print("识别图片 %s" % file_name)
        file_path = os.path.join(path, file_name)
        if not os.path.isfile(file_path):
            continue
        try:
            t1 = round(time.time() * 1000)
            result, mid = locate_card.locate(file_path)
            image_array.append(mid)
            text_array.append(file_name + "_mid")
            image_array.append(result)
            text_array.append(file_name + "_result")

            t2 = round(time.time() * 1000)
            print('time cost:%s ms' % (t2 - t1))
        except ServiceException as exp:
            print("识别出错 %s" % str(exp))
    su.auto_subplt(image_array,col=2,funcs=[plt.imshow],params=[{"cmap":"gray"},{}],title=text_array)




if __name__ == '__main__':
    # group_test_detect()
    # time_cvt, time_tess = td.get_tess_time()
    # print("cvt time=%.3f   tess time=%.3f" % (time_cvt, time_tess))
    test_card_locater(offset=8,limit=1)
    # test_card_locater(offset=0, limit=12)
