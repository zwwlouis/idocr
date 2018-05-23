from ctypes import *
import os
import cv2
from PIL import Image
import ctypes
import sys
import threading

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCE = os.path.join(BASE_DIR, "django_web/resource")
LIBTESS_SO_PATH = "/usr/lib64/libtesseract.so.3"
file_dir = os.path.join(RESOURCE, "card_locate")
file_path = os.path.join(file_dir, "8.jpg")
TESSDATA_PREFIX = os.path.join(RESOURCE,"local","share","tessdata")
print("prefix = "+TESSDATA_PREFIX)
if not TESSDATA_PREFIX:
    TESSDATA_PREFIX = "../"

def init(tesseract, path, lang):
    # print(sys.path)
    c_path = c_char_p(path)
    c_lang = c_char_p(lang)
    print("TessBaseAPIInit3 api=%d path=%s  lang=%s" % (api, path, lang))
    rc = tesseract.TessBaseAPIInit3(api, c_path, c_lang)
    print("initialize result rc = %s" % (rc))

tesseract = cdll.LoadLibrary(LIBTESS_SO_PATH)
api = tesseract.TessBaseAPICreate()
init(tesseract, TESSDATA_PREFIX.encode("utf-8"), b"chi_sim")


def recognize(img):
    shape = img.shape
    print(shape)
    cchar_for_so = c_char_p(img.astype('uint8').tobytes())
    print("TessBaseAPISetImage api=%d" % (api))
    tesseract.TessBaseAPISetImage(api, cchar_for_so, shape[1], shape[0], 1, shape[1])
    text_out = tesseract.TessBaseAPIGetUTF8Text(api)
    result_text = ctypes.string_at(text_out)
    result_text = result_text.decode("utf-8")
    print("ocr result = %s" % (result_text))
    return result_text


def threadfun(filename, _):
    # print("start recognize file=%s" % filename)
    f_path = os.path.join(file_dir, filename)
    img = cv2.imread(f_path, 0)
    # # text = recognize(img)
    # # print("file=%s  result=%s" % (filename, text))
    # tesseract = cdll.LoadLibrary(LIBTESS_SO_PATH)
    # t_api = tesseract.TessBaseAPICreate()
    # path = TESSDATA_PATH.encode("utf-8")
    # lang = b"chi_sim"
    # c_path = c_char_p(path)
    # c_lang = c_char_p(lang)
    # print("TessBaseAPIInit3 api=%d path=%s  lang=%s" % (t_api,path,lang))
    # rc = tesseract.TessBaseAPIInit3(t_api, c_path, c_lang)
    # print("initialize result rc = %s" % (rc))
    #
    # shape = img.shape
    # print(shape)
    # cchar_for_so = c_char_p(img.astype('uint8').tobytes())
    # print("TessBaseAPISetImage api=%d" % (t_api))
    # tesseract.TessBaseAPISetImage(t_api, cchar_for_so, shape[1], shape[0], 1, shape[1])
    # text_out = tesseract.TessBaseAPIGetUTF8Text(t_api)
    # result_text = ctypes.string_at(text_out)
    # result_text = result_text.decode("utf-8")

    result_text = recognize(img)
    # print("ocr result = %s" % (result_text))





# try:
#     thread1 = threading.Thread(target=threadfun, args=("8.jpg", 1))
#     # thread2 = threading.Thread(target=threadfun, args=("9.jpg", 2))
#     thread1.start()
#     # thread2.start()
# except:
#     print
#     "Error: unable to start thread"
#
# while 1:
#     pass

threadfun("8.jpg",1)
threadfun("9.jpg",2)
if __name__ == '__main__':
    print("start main")
    # img = cv2.imread(file_path,0)
    # print(type(img))
    # print(img)
    # text = recognize(img)

    # try:
    #     threading.Thread(target=threadfun, args=("8.jpg"))
    #     threading.Thread(target=threadfun, args=("9.jpg"))
    # except:
    #     print
    #     "Error: unable to start thread"
    #
    # while 1:
    #     pass
