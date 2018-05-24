import cv2
from PIL import Image
import PIL
import numpy as np
img = cv2.imread("123.png")
print(img.shape)
img_pil = Image.open("123.png").convert('RGB')
print(img_pil)
img_pil = np.asarray(img_pil,dtype=np.uint8)
print(img_pil.shape)
img_mat = cv2.cvtColor(img_pil, cv2.COLOR_RGB2BGR)
cv2.imshow('Image',img_mat)
cv2.waitKey(10000)

