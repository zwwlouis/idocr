# -*- coding: utf-8 -*-
"""
Created on Mon Jun 05 15:27:42 2017

@author: bob.lee
"""
import cv2
import os
from idcard_ocr.settings import BASE_DIR
import numpy as np
from django_web.model import ServiceException

xml_path = os.path.join(BASE_DIR, 'django_web/resource/xml/haarcascade_frontalface_default.xml')

def find_face(image):
    face_path = xml_path
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(face_path)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(5, 5),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # content = sorted(faces, key=faces[2]*faces[3], reverse=True)
    area = []
    for (x, y, w, h) in faces:
        area.append(w * h)
        # cv2.rectangle(image,(x,y),(x+w,y+w),(0,255,0),2)
        # cv2.circle(image, (int((x + x + w) / 2), int((y + y + h) / 2)), int(w / 2), (0, 255, 0), 2)
    face = None
    if len(faces) != 0:
        face = faces[area.index(max(area))]
        # image = image[faces[1]:faces[1] + faces[3], faces[0]:faces[0] + faces[2], :]
    return face

def box_face(image,window,color=(0, 255, 0)):
    box = []
    # convert window to four box point
    box.append([window[0],window[1]])
    box.append([window[0]+window[2], window[1]])
    box.append([window[0]+window[2], window[1]]+window[3])
    box.append([window[0], window[1]]+window[3])
    box = np.intp(box)
    cv2.drawContours(image, [box], 0, color, 2)
    return image

def remove_face(image,window,relax_border = [30,30]):
    """
    replace face with average color from image
    :param image:
    :param window: [x,y,w,h] mark the region of face
    :param relax_border: [x_border,y_border] cut more area to ensure removing the face
    :return:
    """
    image_float = np.array(image,dtype=np.float64)
    shape = image_float.shape
    if len(shape) < 2:
        raise ServiceException("wrong size of image")
    ave = sum(sum(image_float))/(shape[0]*shape[1])
    ave = np.array(ave,dtype=np.uint8)
    top = window[1] - relax_border[1]
    bottom = window[1]+window[3] + relax_border[1]
    left = window[0] - relax_border[0]
    right = window[0] + window[2] + relax_border[0]
    if len(shape) == 3:
        image[top:bottom,left:right,:] = ave
    else:
        image[top:bottom,left:right] = ave
    return image
