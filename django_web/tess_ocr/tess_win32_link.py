import os
from idcard_ocr.settings import BASE_DIR
import tesserocr as tessocr
from tesserocr import PSM, PyTessBaseAPI
import logging
from PIL import Image
logger = logging.getLogger('django_logger')
TESSDATA_PATH = os.path.join(BASE_DIR, "django_web", "resource","local","share","tessdata")
# TESSDATA_PATH = "./"

LANG_PAC = dict()
# try:
#     """
#     3 = Fully automatic page segmentation, but no OSD. (Default)
#     7 = Treat the image as a single text line.
#     8 = Treat the image as a single word.
#     10 = Treat the image as a single character
#     """
#     LANG_PAC["name"] = dict(api=PyTessBaseAPI(path=TESSDATA_PATH, lang="chi_sim", psm=PSM.SINGLE_CHAR))
#     LANG_PAC["nation"] = dict(api=PyTessBaseAPI(path=TESSDATA_PATH, lang="nation", psm=PSM.SINGLE_LINE))
#     LANG_PAC["address"] = dict(api=PyTessBaseAPI(path=TESSDATA_PATH, lang="chi_sim", psm=PSM.AUTO))
#     LANG_PAC["idnum"] = dict(api=PyTessBaseAPI(path=TESSDATA_PATH, lang="idnum", psm=PSM.SINGLE_LINE))
#     LANG_PAC["chi_sim"] = dict(api=PyTessBaseAPI(path=TESSDATA_PATH, lang="chi_sim", psm=PSM.AUTO))
# except Exception as e:
#     print("init error")


def create_new_api(label, lang="chi_sim", psm=PSM.AUTO):
    if label is None:
        return
    try:
        LANG_PAC[label] = dict(api=PyTessBaseAPI(path=TESSDATA_PATH, lang=lang, psm=psm))
        logger.info("tesserocr init label=%s lang=%s  psm=%d" % (label, lang, psm))
    except Exception as e:
        logger.error("init error label=%s lang=%s psm=%d"%(label,lang,psm))

def image_to_string(img, label='name', lang="chi_sim"):
    # text = tessocr.image_to_text(img, lang=lang)
    pil_img = Image.fromarray(img)
    tesseract = LANG_PAC[label]["api"]
    tesseract.SetImage(pil_img)
    text = tesseract.GetUTF8Text()
    return text


def image_to_string_exe(img, lang="chi_sim", psm=PSM.AUTO):
    text = tessocr.image_to_text(img,lang=lang,psm=psm)
    return text
