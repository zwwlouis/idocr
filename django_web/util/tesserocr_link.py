import os
from idcard_ocr.settings import BASE_DIR
import ctypes
import tesserocr as tessocr
from tesserocr import PSM, PyTessBaseAPI

# TESSDATA_PATH = os.path.join(BASE_DIR,"django_web","resource","tessdata")
TESSDATA_PATH = "./"

LANG_PAC = dict()
try:
    LANG_PAC["name"] = dict(api=PyTessBaseAPI(lang="chi_sim"))
    LANG_PAC["nation"] = dict(api=PyTessBaseAPI(lang="nation"))
    LANG_PAC["address"] = dict(api=PyTessBaseAPI(lang="chi_sim"))
    LANG_PAC["idnum"] = dict(api=PyTessBaseAPI(lang="idnum"))
    LANG_PAC["chi_sim"] = dict(api=PyTessBaseAPI(lang="chi_sim"))
    # LANG_PAC["name"] = dict(api=PyTessBaseAPI())
    # LANG_PAC["nation"] = dict(api=PyTessBaseAPI())
    # LANG_PAC["address"] = dict(api=PyTessBaseAPI())
    # LANG_PAC["idnum"] = dict(api=PyTessBaseAPI())
    # LANG_PAC["chi_sim"] = dict(api=PyTessBaseAPI())
except Exception as e:
    print("init error")


def image_to_string(img, lang="chi_sim",psm=0):
    # text = tessocr.image_to_text(img, lang=lang)
    tesseract = LANG_PAC[lang]["api"]
    tesseract.SetImage(img)
    text = tesseract.GetUTF8Text()
    return text
