import os
import cv2
from idcard_ocr.settings import BASE_DIR

RESOURCE = os.path.join(BASE_DIR, "django_web/resource")
RESULT = os.path.join(RESOURCE, "result")


def get_imgs_from_path(file_dir):
    if not os.path.exists(file_dir):
        return []
    return os.listdir(file_dir)

def read_img(file_path, width = 400):
    if not os.path.exists(file_path):
        return None
    image = cv2.imread(file_path)
    shape = image.shape
    h_after_resize = int(shape[0]/shape[1]*width)
    image = cv2.resize(image,(width,h_after_resize))
    return image

def write_img(img,folder,file_name):
    file_path = os.path.join(RESULT,folder)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    cv2.imwrite(os.path.join(file_path,file_name),img)


if __name__ == '__main__':
    file_dir = get_imgs_from_path(os.path.join(RESOURCE,"idcard_img"))
    file_path = os.path.join(RESOURCE, "idcard_img",file_dir[11])
    image = read_img(file_path)
    cv2.imshow("img",image)
    cv2.waitKey()
    write_img(image,os.path.join(RESOURCE, "test/test/test"),file_dir[11])


