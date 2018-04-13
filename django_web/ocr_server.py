from django_web.util import text_detector as td
from django_web.util import img_util as iu
import os
from idcard_ocr.settings import BASE_DIR
import cv2

RESOURCE_PATH = os.path.join(BASE_DIR, "django_web/resource")
RESULT_PATH = "ocr_result"

def card_detect():
   pass