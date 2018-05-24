import os
import sys
from django_web.util.constants import *
from idcard_ocr.settings import BASE_DIR
import logging
logger = logging.getLogger('django_logger')
import cv2
from django_web import idcardocr
RESOURCE_PATH = os.path.join(BASE_DIR, "django_web/resource")
LANG_PAC = dict()
LANG_PAC["name"] = dict(lang="chi_sim", psm=PSM_SINGLE_LINE)
LANG_PAC["nation"] = dict(lang="nation", psm=PSM_SINGLE_LINE)
LANG_PAC["address"] = dict(lang="chi_sim", psm=PSM_AUTO)
LANG_PAC["idnum"] = dict(lang="idnum", psm=PSM_SINGLE_LINE)
LANG_PAC["chi_sim"] = dict(lang="chi_sim", psm=PSM_AUTO)

lib_tesseract = None

logger.info("start tess link init")
if sys.platform == "win32":
    from django_web.tess_ocr import tess_win32_link
    logger.info("win32 env")
    lib_tesseract = tess_win32_link
    # libname = libpath_w + "libtesseract302.dll"
    # libname_alt = "libtesseract302.dll"
    # os.environ["PATH"] += os.pathsep + libpath_w
    # TODO for win32 tesserocr is used
    pass
else:
    from django_web.tess_ocr import tess_linux_link
    logger.info("linux env")
    # libname = libpath + "libtesseract.so.3.0.2"
    # libname_alt = "libtesseract.so.3"
    lib_tesseract = tess_linux_link.tess_linux_link

if lib_tesseract is not None:
    for key in LANG_PAC:
        LANG = LANG_PAC[key]
        lib_tesseract.create_new_api(label=key, lang=LANG["lang"], psm=LANG["psm"])
else:
    logger.error("lib tesseract init failed!")


def image_to_string(img, label,lang):
    if lib_tesseract is None:
        return ""
    return lib_tesseract.image_to_string(img, label)
    # return ""
    # return test_so.recognize(img)


# file_path = os.path.join(RESOURCE_PATH, 'card_locate',"9.jpg")
# img_target = cv2.imread(file_path)
# # image_to_string(img_target,"name","chi_sim")
# result = idcardocr.idcardocr(img_target)
# print(result)

if __name__ == '__main__':
    print("hello world")
