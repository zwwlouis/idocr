# -*- coding: utf-8 -*-
import os

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(BASE_DIR)

from ctypes import *
import cv2
from PIL import Image
import ctypes
import time
from django_web.model import *
from django_web.util.constants import *
from idcard_ocr.settings import BASE_DIR
from numpy import array
import logging
import threading

logger = logging.getLogger('django_logger')
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESOURCE = os.path.join(BASE_DIR, "django_web", "resource")
TESSDATA_PATH = os.path.join(RESOURCE, "local", "share", "tessdata")
LIBTESS_SO_PATH = os.path.join(RESOURCE, "local", "lib", "libtesseract.so")
file_path = os.path.join(RESOURCE, "card_locate", "8.jpg")
logger = logging.getLogger('django_logger')


class TessLinuxLink(object):
    def __init__(self):
        self.id = int(time.time())
        self.tesseract = None
        self.api_group = dict()
        self.lock_group = dict()
        pass

    def init_apis(self):
        pass

    def __del__(self):
        print("start __del__()")
        api_group = self.api_group
        for key in api_group:
            api = api_group[key]
            self.tesseract.TessBaseAPIEnd(api)

    def create_new_api(self, label, lang="chi_sim", path=TESSDATA_PATH, psm=PSM_AUTO):
        """
        crate new api
        :param lang: language traindata for api
        :param path: path of traindata
        :param psm:
        :return:
        """
        # print("create new api path = %s" % path)
        if self.tesseract is None:
            logger.info("loadLibrary tesseract path=%s" % LIBTESS_SO_PATH)
            self.tesseract = cdll.LoadLibrary(LIBTESS_SO_PATH)

        tesseract = self.tesseract
        if not isinstance(path, bytes):
            path = path.encode("utf-8")
        if not isinstance(lang, bytes):
            lang = lang.encode("utf-8")
        api = tesseract.TessBaseAPICreate()
        logger.info("TessBaseAPIInit3 api=%d label=%s lang=%s  psm=%d" % (api, label, lang, psm))
        c_path = c_char_p(path)
        c_lang = c_char_p(lang)
        # print("TessBaseAPIInit3 api=%d path=%s  lang=%s" % (api, c_path, c_lang))
        rc = tesseract.TessBaseAPIInit3(api, c_path, c_lang)
        # print(type(rc))
        # print("init complete rc=%d"%(rc))
        if rc:
            tesseract.TessBaseAPIDelete(api)
            logger.error("Could not initialize tesseract. path=%s lang=%s" % (TESSDATA_PATH, lang))
        else:
            tesseract.TessBaseAPISetPageSegMode(api, psm)
            if isinstance(lang, bytes):
                lang = lang.decode("utf-8")
            self.api_group[label] = api
            self.lock_group[label] = threading.Lock()
            # print("create api successful lang=%s" % lang)

    def image_to_string(self, img, label):
        """
        recognize img to word
        :param img:
        :param lang: language
        :param psm:
        :return:
        """
        img = array(img, dtype="uint8")
        # print("start recognize")
        tesseract = self.tesseract
        api_group = self.api_group
        if label not in api_group:
            raise ServiceException("text ocr failed label=%s not exists" % label)
        lock = self.lock_group[label]
        # 锁的过期时间为5秒
        lock.acquire(5)
        try:
            api = self.api_group[label]
            shape = img.shape
            # print("image shape=%s"% str(shape))
            cchar_for_so = c_char_p(img.tobytes())
            tesseract.TessBaseAPISetImage(api, cchar_for_so, shape[1], shape[0], 1, shape[1])
            tesseract.TessBaseAPIGetUTF8Text.restype = ctypes.c_char_p
            text_out = tesseract.TessBaseAPIGetUTF8Text(api)
            result_text = ctypes.string_at(text_out)
            result_text = result_text.decode("utf-8")
        except Exception as error:
            raise error
        finally:
            # 保证释放锁
            lock.release()
        return result_text


class TessPool(object):
    def __init__(self):
        self.api_list_group = dict()

    def create_new_api(self, label, lang="chi_sim", path=TESSDATA_PATH, psm=PSM_AUTO, num=3):
        """
        创建一组api
        :param label:
        :param lang:
        :param path:
        :param psm:
        :param num: api数量
        :return:
        """
        api_list = list()
        for i in range(num):
            api_link = TessLinuxLink()
            api_link.create_new_api(label, lang, path, psm)
            api_list.append(api_link)
        self.api_list_group[label] = api_list

    def image_to_string(self, img, label):
        api_list = self.api_list_group[label]
        while len(api_list) < 1:
            time.sleep(0.5)
        # 取出一个api
        api = api_list.pop()
        result_text = ""
        try:
            result_text = api.image_to_string(img, label)
        except Exception as error:
            raise error
        finally:
            # 归还api
            api_list.append(api)
        return result_text


tess_pool = TessPool()

if __name__ == '__main__':
    pass
# img = cv2.imread(file_path, 0)
# tess_linux_link.create_new_api("name", "chi_sim", TESSDATA_PATH, PSM_AUTO)
# text = tess_linux_link.image_to_string(img, "name")
# print("ocr result = %s" % (text))
# # tesseract.TessBaseAPIEnd(api)
