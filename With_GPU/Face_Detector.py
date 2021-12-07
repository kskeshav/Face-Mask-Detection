from PIL import Image
import cv2
import os
import pickle
import torch
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import pandas as pd
import matplotlib.pyplot as plt


def Face_Detection(faceCascade, human_bbox_frame):

    gray = cv2.cvtColor(human_bbox_frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                        scaleFactor=1.1,
                                        minNeighbors=5,
                                        minSize=(60, 60),
                                        flags=cv2.CASCADE_SCALE_IMAGE)
    face_bbox_frame = None
    for (x,y,w,h) in faces:
        # cv2.rectangle(human_bbox_frame, (x, y), (x + w, y + h),(0,255,0), 2)
        # face_bbox_frame = human_bbox_frame[y:y+h, x:x+w]
        return (x, y) ,(x + w, y + h)
    return face_bbox_frame