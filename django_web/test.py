from django_web.util import text_detector as td
from django_web.util import img_util as iu
import os
from idcard_ocr.settings import BASE_DIR
import cv2
from django_web.model import *
from django_web.util import subplot_util as su
from matplotlib import pyplot as plt

RESOURCE_PATH = os.path.join(BASE_DIR, "django_web/resource")
RESULT_PATH = "ocr_result"

def group_test_detect():
    path = os.path.join(RESOURCE_PATH, 'idcard_img')
    files = iu.get_imgs_from_path(path)

    image_array  = []
    text_array = []
    for file_name in files:
        print("识别图片 %s"%file_name)
        file_path = os.path.join(path, file_name)
        if not os.path.isfile(file_path):
            continue
        img = iu.read_img(file_path)
        try:
            contour_img, imgs, txts = td.detect(img)
        except ServiceException as exp:
            print("识别出错 %s"%exp.message)
        image_array += imgs
        text_array += txts
        # 保存绿框结果
        iu.write_img(contour_img, os.path.join(path, "final_result"), file_name)



def group_test_binary():
    path = os.path.join(RESOURCE_PATH, 'test_binary')
    files = iu.get_imgs_from_path(path)
    i = 0
    for file_name in files:
        i+=1
        file_path = os.path.join(path, file_name)
        if not os.path.isfile(file_path):
            continue
        img = iu.read_img(file_path)
        img = td.detect(img,debug=True, save_path=os.path.join(path,"middle_result%d"%i))
        iu.write_img(img, os.path.join(path, "final_result"), file_name)

def group_test_card_cut():
    path = os.path.join(RESOURCE_PATH, 'test_card_cut')
    files = iu.get_imgs_from_path(path)
    i = 0
    for file_name in files:
        i+=1
        file_path = os.path.join(path, file_name)
        if not os.path.isfile(file_path):
            continue
        img = iu.read_img(file_path)
        td.get_card_area(img,debug=True, save_path=os.path.join(path,"middle_result%d"%i))


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
        img = iu.read_img(file_path,width=1000)
        try:
            contour_img, imgs, txts = td.detect(img,debug=True, save_path=os.path.join(path,"middle_result"))
        except ServiceException as exp:
            print("识别出错 %s" % exp.message)
        image_array += imgs
        text_array += txts
        # 保存绿框结果
        iu.write_img(contour_img, os.path.join(path, "final_result"), file_name)
    for img in image_array:
        cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    su.subplt(image_array,2,5,[plt.imshow],title=text_array)



if __name__ == '__main__':
    single_test_detect()
    time_cvt,time_tess = td.get_tess_time()
    print("cvt time=%.3f   tess time=%.3f"%(time_cvt,time_tess))
