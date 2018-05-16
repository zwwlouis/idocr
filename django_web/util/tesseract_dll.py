import os
from idcard_ocr.settings import BASE_DIR
import ctypes


RESOURCE = os.path.join(BASE_DIR, "django_web/resource")
TESS_DLL_PATH = os.path.join(RESOURCE, 'dll', 'libtesseract-4.dll')
TESSDATA_PREFIX = os.path.join(RESOURCE, 'tessdata')
print(TESS_DLL_PATH)
tesseract = ctypes.CDLL(TESS_DLL_PATH)
api = tesseract.TessBaseAPICreate()
rc = tesseract.TessBaseAPIInit3(api, TESSDATA_PREFIX, 'chi_sim')
if rc:
    tesseract.TessBaseAPIDelete(api)
    print('Could not initialize tesseract.\n')
    exit(3)


def from_file(path):
    tesseract.TessBaseAPIProcessPages(api, path, None, 0, None)
    text_out = tesseract.TessBaseAPIGetUTF8Text(api)
    return ctypes.string_at(text_out)

if __name__ == '__main__':

    image_file_path = b'./test.jpg'
    result = from_file(image_file_path)
    print(result)

