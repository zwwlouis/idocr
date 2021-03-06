from django_web.util import text_detector as td
from django_web.util import img_util as iu
import os
import json
import time
import logging
import traceback
from idcard_ocr.settings import BASE_DIR
import cv2
import numpy as np
from PIL import Image
from django_web.util.locate_card import locate_card
from django_web import idcardocr
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from django.utils.datastructures import MultiValueDictKeyError
from django_web.model import *

RESOURCE_PATH = os.path.join(BASE_DIR, "django_web/resource")
RESULT_PATH = "ocr_result"
logger = logging.getLogger('django_logger')


@csrf_exempt
def idcard_ocr(request):
    start = time.time()
    ori_time = start
    # 初始化返回对象
    response_data = dict(code=20001, message="", result=None)
    if request.method == 'POST':
        # 0-身份证正面  1-反面
        card_side = request.POST.get('side')
        if not card_side:
            # 默认检测正面
            card_side = 0
        # 是否有伪造风险，如果有则开启风险检测
        detect_risk = request.POST.get('risk')
        if detect_risk is None:
            # 默认无风险检测
            detect_risk = False
        # 是否检测身份证旋转角度
        detect_direction = request.POST.get('direction')
        if detect_direction is None:
            detect_direction = False
        time_used = time.time() - start
        start += time_used
        logger.info("prehandle timeUsed = %d ms" % (int(time_used * 1000)))

        try:
            card_file = request.FILES['image']
            image = Image.open(card_file, mode="r").convert("RGB")
            imgArray = np.asarray(image)
            logger.info("get a image shape = %s" % str(imgArray.shape))
            img_mat = cv2.cvtColor(imgArray, cv2.COLOR_RGB2BGR)
            img_mat, scale = iu.img_resize(img_mat, 1920)
            time_used = time.time() - start
            start += time_used
            logger.info("file load timeUsed = %d ms" % (int(time_used * 1000)))

            img_full, _ = locate_card.locate(img_mat)
            time_used = time.time() - start
            start += time_used
            logger.info("card location timeUsed = %d ms" % (int(time_used * 1000)))

            result_dict = idcardocr.idcardocr(img_full)
            time_used = time.time() - ori_time
            logger.info("total procession timeUsed = %d ms" % (int(time_used * 1000)))
            response_data = dict(code=0, message="ok", result=result_dict)
        except (MultiValueDictKeyError, OSError) as error:
            logger.error("图片参数错误" + traceback.format_exc())
            response_data["message"] = "图片参数错误"
        except ServiceException as error:
            logger.error(str(error))
            response_data["message"] = str(error)
    else:
        response_data["message"] = "请求服务方式错误"
    ret_str = json.dumps(response_data)
    logger.info("respond = %s" % response_data)
    return HttpResponse(ret_str, content_type='application/json')
