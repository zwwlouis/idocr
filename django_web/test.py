
from django_web.util import text_detector as td
from django_web.util import img_util as iu
import os
from idcard_ocr.settings import BASE_DIR
import cv2

RESOURCE_PATH = os.path.join(BASE_DIR,"django_web/resource")
RESULT_PATH = "ocr_result"

def group_test_detect():
    file_path = os.path.join(RESOURCE_PATH,'idcard_img')
    file_dir = iu.get_imgs_from_path(file_path)
    for file_name in file_dir:
        img = iu.read_img(os.path.join(file_path,file_name))
        img = td.detect(img)
        iu.write_img(img, RESULT_PATH, file_name)

def single_test_detect():
    file_path = os.path.join(RESOURCE_PATH, 'idcard_img')
    file_dir = iu.get_imgs_from_path(file_path)
    file_name =file_dir[0]
    img = iu.read_img(os.path.join(file_path, file_name))
    img = td.detect(img,debug=True)
    cv2.imshow("img",img)
    cv2.waitKey()

if __name__ == '__main__':
    group_test_detect()