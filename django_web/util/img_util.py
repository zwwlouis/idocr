import os
import cv2
from idcard_ocr.settings import BASE_DIR
from numpy import ndarray, array

RESOURCE = os.path.join(BASE_DIR, "django_web/resource")
TEMP = os.path.join(BASE_DIR, "django_web/temp")


def get_imgs_from_path(file_dir):
    if not os.path.exists(file_dir):
        return []
    return os.listdir(file_dir)


def read_img(file_path, width=1000, flags=1):
    if not os.path.exists(file_path):
        return None
    image = cv2.imread(file_path, flags=flags)
    shape = image.shape
    h_after_resize = int(shape[0] / shape[1] * width)
    image = cv2.resize(image, (width, h_after_resize))
    return image


def get_mask(file_name, flags=1):
    file_path = os.path.join(RESOURCE, "mask", file_name)
    image = cv2.imread(file_path, flags=flags)
    return image;


def write_img(img, file_path, file_name):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    cv2.imwrite(os.path.join(file_path, file_name), img)


def write_middle_result(img, file_name, folder="result"):
    result_file_path = os.path.join(TEMP, folder)
    if not os.path.exists(result_file_path):
        os.makedirs(result_file_path)
    cv2.imwrite(os.path.join(result_file_path, file_name), img)


def img_resize(img, dwidth):
    """
    兼容UMat
    :param img:
    :param dwidth:
    :return:
    """
    if isinstance(img, cv2.UMat):
        size = img.get().shape
    elif isinstance(img, ndarray):
        size = img.shape
    elif isinstance(img, list):
        imgArray = array(img)
        size = imgArray.size
        if len(size) < 2:
            return None
    else:
        return None
    height = size[0]
    width = size[1]
    dheight = int(height * dwidth / width)
    nImg = cv2.resize(img, dsize=(dwidth, dheight), interpolation=cv2.INTER_CUBIC)
    return nImg


if __name__ == '__main__':
    file_dir = get_imgs_from_path(os.path.join(RESOURCE, "idcard_img"))
    file_path = os.path.join(RESOURCE, "idcard_img", file_dir[11])
    image = read_img(file_path)
    cv2.imshow("img", image)
    cv2.waitKey()
