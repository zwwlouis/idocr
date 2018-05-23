# -*- coding: utf-8 -*-
from PIL import Image
import cv2
import numpy as np
import re
import time
from django_web.util import img_util as iu
from django_web.tess_ocr import tess_link as tl
import logging
logger = logging.getLogger('django_logger')
x = 1280.00 / 3840.00
pixel_x = int(x * 3840)
print(x, pixel_x)

# debug模式下保存图片处理的中间结果
DEBUG = False

name_mask = iu.get_mask('name_mask_%s.jpg' % pixel_x, 0)
nation_mask = iu.get_mask('nation_mask_%s.jpg' % pixel_x, 0)
sex_mask = iu.get_mask('sex_mask_%s.jpg' % pixel_x, 0)
birth_mask = iu.get_mask('birth_mask_%s.jpg' % pixel_x, 0)
address_mask = iu.get_mask('address_mask_%s.jpg' % pixel_x, 0)
idnum_mask = iu.get_mask('idnum_mask_%s.jpg' % pixel_x, 0)

target_list = list()
target_list.append(dict(label="name", template=name_mask, shape=(300, 700), lang='chi_sim', config='-psm 3'))
# region_list.append(dict(label="sex",template=sex_mask, shape=(300, 300), lang='sex', config='-psm 10'))
target_list.append(dict(label="nation", template=nation_mask, shape=(300, 500), lang='nation', config='-psm 7'))
# region_list.append(dict(label="birth",template=birth_mask, shape=(300, 1500), lang='', config='-psm 3'))
target_list.append(dict(label="address", template=address_mask, shape=(500, 1700), lang='chi_sim', config='-psm 3'))
target_list.append(dict(label="idnum", template=idnum_mask, shape=(300, 2300), lang='idnum', config='-psm 7'))


def idcardocr(img):
    # generate_mask(x)
    img_data_gray, img_org = img_resize_gray(img)
    # TODO 判断身份证的朝向和风险情况
    result_dict = dict(direction=0, riskType=0)
    # 记录文字的列表
    text_dict = {}
    result_dict["textResult"] = text_dict

    for target in target_list:
        label = target['label']
        region = find_region(img_data_gray, img_org, target['template'], target['shape'],label=label)
        text_dict[label] = text_ocr(region, label, target['lang'], target['config'])

    # start = time.time()
    # name_pic = find_name(img_data_gray, img_org)
    # text_dict['name'] = get_name(name_pic)
    # time_used = time.time() - start
    # start += time_used
    # print("name timeUsed = %d ms" % (int(time_used * 1000)))
    #
    # nation_pic = find_nation(img_data_gray, img_org)
    # text_dict['nation'] = get_nation(nation_pic)
    # time_used = time.time() - start
    # start += time_used
    # print("nation timeUsed = %d ms" % (int(time_used * 1000)))
    #
    # address_pic = find_address(img_data_gray, img_org)
    # text_dict['address'] = get_address(address_pic)
    # time_used = time.time() - start
    # start += time_used
    # print("address timeUsed = %d ms" % (int(time_used * 1000)))
    #
    # idnum_pic = find_idnum(img_data_gray, img_org)
    # text_dict['idnum'] = get_idnum(idnum_pic)

    idnum = text_dict['idnum']["text"]
    # TODO idcard_util 检测身份证号的有效性
    if len(idnum) == 18:
        birth = idnum[6:14]
        text_dict['birth'] = dict(label="birth", text=birth, location={})
        sex_num = int(idnum[16])
        if sex_num % 2 == 0:
            sex = "女"
        else:
            sex = "男"
        text_dict['sex'] = dict(label="sex", text=sex, location={})

    else:
        pass
        # year_pic, month_pic, day_pic = find_birth(img_data_gray, img_org)
        # text_dict['birth'] = get_birth(year_pic, month_pic, day_pic)
        #
        # sex_pic = find_sex(img_data_gray, img_org)
        # # showimg(sex_pic)
        # # print 'sex'
        # text_dict['sex'] = get_sex(sex_pic)
        # # print result_dict['sex']

    result_dict['textCount'] = len(text_dict)
    return result_dict



def find_region(img_gray, img_rgb, template, shape, label=''):
    """
    查找图片中和标题模板相对应的位置，并提取模板右方的文字区域
    :param img_gray: 灰度图片，用于模板匹配
    :param img_rgb:  彩色图片，实际的剪切操作
    :param template: 模板图片
    :param shape:    文字区域的尺寸 turple（h,w）
    :return:
    """
    start = time.time()
    h, w = template.shape
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    # 找到矩阵中全局最大值和最小值的位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = (max_loc[0] + w, max_loc[1] - int(20 * x))
    bottom_right = (top_left[0] + int(shape[1] * x), top_left[1] + int(shape[0] * x))
    result = img_rgb[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    cv2.rectangle(img_gray, top_left, bottom_right, 255, 2)
    # showimg(result)
    time_used = time.time() - start
    logger.info("find %s timeUsed = %d ms" % (label,int(time_used * 1000)))
    return result


def text_ocr(img, label, lang, config='-psm 3'):
    start = time.time()
    _, _, red = cv2.split(img)
    red = hist_equal(red)
    red = cv2.adaptiveThreshold(red, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 151, 50)
    red = img_resize(red, 150)
    red = red.astype('uint8')
    # print(type(red))
    # print(red)
    # img = Image.fromarray(red.astype('uint8'))
    text = tl.image_to_string(red, label=label, lang=lang).replace(" ", "").replace("\n","")
    ocr_text = dict(label=label, text=text, location={})
    time_used = time.time() - start
    logger.info("recognize %s timeUsed = %d ms" % (label, int(time_used * 1000)))
    return ocr_text








def generate_mask(x):
    name_mask_pic = cv2.UMat(cv2.imread('name_mask.jpg'))
    sex_mask_pic = cv2.UMat(cv2.imread('sex_mask.jpg'))
    nation_mask_pic = cv2.UMat(cv2.imread('nation_mask.jpg'))
    birth_mask_pic = cv2.UMat(cv2.imread('birth_mask.jpg'))
    year_mask_pic = cv2.UMat(cv2.imread('year_mask.jpg'))
    month_mask_pic = cv2.UMat(cv2.imread('month_mask.jpg'))
    day_mask_pic = cv2.UMat(cv2.imread('day_mask.jpg'))
    address_mask_pic = cv2.UMat(cv2.imread('address_mask.jpg'))
    idnum_mask_pic = cv2.UMat(cv2.imread('idnum_mask.jpg'))
    name_mask_pic = img_resize_x(name_mask_pic)
    sex_mask_pic = img_resize_x(sex_mask_pic)
    nation_mask_pic = img_resize_x(nation_mask_pic)
    birth_mask_pic = img_resize_x(birth_mask_pic)
    year_mask_pic = img_resize_x(year_mask_pic)
    month_mask_pic = img_resize_x(month_mask_pic)
    day_mask_pic = img_resize_x(day_mask_pic)
    address_mask_pic = img_resize_x(address_mask_pic)
    idnum_mask_pic = img_resize_x(idnum_mask_pic)
    cv2.imwrite('name_mask_%s.jpg' % pixel_x, name_mask_pic)
    cv2.imwrite('sex_mask_%s.jpg' % pixel_x, sex_mask_pic)
    cv2.imwrite('nation_mask_%s.jpg' % pixel_x, nation_mask_pic)
    cv2.imwrite('birth_mask_%s.jpg' % pixel_x, birth_mask_pic)
    cv2.imwrite('year_mask_%s.jpg' % pixel_x, year_mask_pic)
    cv2.imwrite('month_mask_%s.jpg' % pixel_x, month_mask_pic)
    cv2.imwrite('day_mask_%s.jpg' % pixel_x, day_mask_pic)
    cv2.imwrite('address_mask_%s.jpg' % pixel_x, address_mask_pic)
    cv2.imwrite('idnum_mask_%s.jpg' % pixel_x, idnum_mask_pic)


# 用于生成模板
def img_resize_x(imggray):
    # print 'dheight:%s' % dheight
    crop = imggray
    size = crop.shape
    dheight = int(size[0] * x)
    dwidth = int(size[1] * x)
    crop = cv2.resize(src=crop, dsize=(dwidth, dheight), interpolation=cv2.INTER_CUBIC)
    return crop


# idcardocr里面resize以高度为依据, 用于get部分
def img_resize(imggray, dheight):
    # print 'dheight:%s' % dheight
    crop = imggray
    size = crop.shape
    height = size[0]
    width = size[1]
    width = int(width * dheight / height)
    crop = cv2.resize(src=crop, dsize=(width, dheight), interpolation=cv2.INTER_CUBIC)
    return crop


def img_resize_gray(imgorg):
    # imgorg = cv2.imread(imgname)
    size = imgorg.shape
    # print size
    height = size[0]
    width = size[1]
    # 参数是根据3840调的
    height = int(height * 3840 * x / width)
    # print height
    crop = cv2.resize(src=imgorg, dsize=(int(3840 * x), height), interpolation=cv2.INTER_CUBIC)
    return hist_equal(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)), crop



def showimg(img):
    cv2.namedWindow("contours", 0)
    cv2.resizeWindow("contours", 1280, 720)
    cv2.imshow("contours", img)
    cv2.waitKey()


# psm model:
#  0    Orientation and script detection (OSD) only.
#  1    Automatic page segmentation with OSD.
#  2    Automatic page segmentation, but no OSD, or OCR.
#  3    Fully automatic page segmentation, but no OSD. (Default)
#  4    Assume a single column of text of variable sizes.
#  5    Assume a single uniform block of vertically aligned text.
#  6    Assume a single uniform block of text.
#  7    Treat the image as a single text line.
#  8    Treat the image as a single word.
#  9    Treat the image as a single word in a circle.
#  10    Treat the image as a single character.
#  11    Sparse text. Find as much text as possible in no particular order.
#  12    Sparse text with OSD.
#  13    Raw line. Treat the image as a single text line,
# 			bypassing hacks that are Tesseract-specific



def punc_filter(str):
    temp = str
    xx = u"([\u4e00-\u9fff0-9]+)"
    pattern = re.compile(xx)
    results = pattern.findall(temp)
    string = ""
    for result in results:
        string += result
    return string


# 这里使用直方图拉伸，不是直方图均衡
def hist_equal(img):
    # clahe_size = 8
    # clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(clahe_size, clahe_size))
    # result = clahe.apply(img)
    # test

    # result = cv2.equalizeHist(img)
    if isinstance(img, cv2.UMat):
        image = img.get()  # UMat to Mat
    else:
        image = img
    # result = cv2.equalizeHist(image)
    lut = np.zeros(256, dtype=image.dtype)  # 创建空的查找表
    # lut = np.zeros(256)
    hist = cv2.calcHist([image],  # 计算图像的直方图
                        [0],  # 使用的通道
                        None,  # 没有使用mask
                        [256],  # it is a 1D histogram
                        [0, 256])
    minBinNo, maxBinNo = 0, 255
    # 计算从左起第一个不为0的直方图柱的位置
    for binNo, binValue in enumerate(hist):
        if binValue != 0:
            minBinNo = binNo
            break
    # 计算从右起第一个不为0的直方图柱的位置
    for binNo, binValue in enumerate(reversed(hist)):
        if binValue != 0:
            maxBinNo = 255 - binNo
            break
    # print minBinNo, maxBinNo
    # 生成查找表
    for i, v in enumerate(lut):
        if i < minBinNo:
            lut[i] = 0
        elif i > maxBinNo:
            lut[i] = 255
        else:
            lut[i] = int(255.0 * (i - minBinNo) / (maxBinNo - minBinNo) + 0.5)
    # 计算,调用OpenCV cv2.LUT函数,参数 image --  输入图像，lut -- 查找表
    # print lut
    result = cv2.LUT(image, lut)
    # print type(result)
    # showimg(result)
    return result


if __name__ == "__main__":
    idocr = idcardocr(cv2.imread("testimages/1.jpg"))
    print(idocr)
    # idocr.idcardocr(cv2.imread('9.jpg'))
