import pytesseract
import re
from PIL import Image
import cv2
from numpy import array
import time

time_tess = 0
time_cvt = 0


def number_find(content):
    re_words = re.compile("[X0-9]+")
    result = re.findall(re_words, content)
    result = "".join(result)
    return result


def ocr_result(image, window):
    """
    通过tesseract识别图片中的文字
    :param image_part:原始图片
    :param window:截取窗口[x,y,w,h]
    :return:
    """
    global time_cvt, time_tess
    window = array(window)
    window[window < 0] = 0
    image_part = image[window[1]:window[1] + window[3], window[0]:window[0] + window[2], :]
    image_part_gray = img_preprocess(image_part)
    # 计时
    start = time.time()
    image_text = Image.fromarray(image_part_gray)
    time_cvt += time.time() - start
    start = time.time()
    text = pytesseract.image_to_string(image_text, lang='chi_sim')
    text = txt_process(text)
    time_tess += time.time() - start
    return image_part, text


def txt_process(txt):
    """
    文字识别后处理
    :param txt:
    :return:
    """
    return txt.replace(" ","")


def img_preprocess(image):
    """
    将彩色图片处理成黑白分明的图片
    :param image:
    :return:
    """
    # 先处理成灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 进行二值化处理
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    print("img_preprocess OTSU threshold is %d" % ret)
    # cv2.imshow("binary",binary)
    # cv2.waitKey()
    return binary
