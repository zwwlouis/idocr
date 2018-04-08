from django_web.util import text_detector as td
from django_web.util import img_util as iu
import os
from idcard_ocr.settings import BASE_DIR
import cv2

RESOURCE_PATH = os.path.join(BASE_DIR, "django_web/resource")
RESULT_PATH = "ocr_result"


def group_test_detect():
    path = os.path.join(RESOURCE_PATH, 'idcard_img')
    files = iu.get_imgs_from_path(path)
    for file_name in files:
        file_path = os.path.join(path, file_name)
        if not os.path.isfile(file_path):
            continue
        img = iu.read_img(file_path)
        img = td.detect(img)
        iu.write_img(img, os.path.join(path, "final_result"), file_name)


def single_test_detect():
    file_path = os.path.join(RESOURCE_PATH, 'test2')
    file_dir = iu.get_imgs_from_path(file_path)
    file_name = file_dir[0]
    img = iu.read_img(os.path.join(file_path, file_name))
    img = td.detect(img, debug=True, save_path=os.path.join(file_path, "middle_result"))
    cv2.imshow("img", img)
    cv2.waitKey()


if __name__ == '__main__':
    single_test_detect()
