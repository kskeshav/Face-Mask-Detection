from __future__ import division
import time
import numpy as np
import cv2 
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import argparse
import pickle as pkl
from Face_Detector import Face_Detection
import pickle
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
from Nose_Detector import Nose_Detection

# Torch related imports
import torch 
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms, models

test_transforms = transforms.Compose([transforms.RandomRotation(5),
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor()])


cascPath_Mouth = "../haarcascade_mcs_mouth"
cascPath_Nose = "../haarcascade_mcs_nose.xml"
cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPathface)
mouthCascade = cv2.CascadeClassifier(cascPath_Mouth)
noseCascade = cv2.CascadeClassifier(cascPath_Nose)


def drawLabel(img, text, c1, c2, bg_color):
    # scale = 0.4
    color = (0, 0, 0)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, text, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)


def prep_image(img, inp_dim):
    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = (164, 80, 133) # random.choice(colors)
    # print(color)
    cv2.rectangle(img, c1, c2,color, 4)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, 4)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img

def arg_parse():
    parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "160", type = str)
    return parser.parse_args()

if __name__ == '__main__':
    cfgfile = "cfg/yolov3.cfg"
    weightsfile = "yolov3.weights"
    num_classes = 80

    logreg_filename = '../finalized_classifier_model.sav'
    logreg = pickle.load(open(logreg_filename, 'rb'))
    model_ft = models.alexnet(pretrained=True)

    args = arg_parse()
    confidence = 0.25 # float(args.confidence)
    nms_thesh = 0.4 # float(args.nms_thresh)
    start = 0
    CUDA = torch.cuda.is_available()
    
    num_classes = 80
    bbox_attrs = 5 + num_classes
    
    model = Darknet(cfgfile)
    model.load_weights(weightsfile)
    
    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    if CUDA:
        model.cuda()
            
    model.eval()
    
    cap = cv2.VideoCapture(0)
    
    assert cap.isOpened(), 'Cannot capture source'
    
    frames = 0
    start = time.time()    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            img, orig_im, dim = prep_image(frame, inp_dim)
            im_dim = torch.FloatTensor(dim).repeat(1,2)                        
            
            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()
            output = model(Variable(img), CUDA)
            output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)
            if type(output) == int:
                frames += 1
                print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
                cv2.imshow("frame", orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue
                
            if not output == None and len(output) == 0:
                print("No person found")
            else:
                output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))/inp_dim            
                output[:,[1,3]] *= frame.shape[1]
                output[:,[2,4]] *= frame.shape[0]            
                classes = load_classes('data/coco.names')
                list(map(lambda x: write(x, orig_im), output))

                human_bbox = []
                # print("Total number of people detected ", len(output))
                # if len(output) > 0:
                try:
                    for bbox in output:
                        c1 = tuple(bbox[1:3].int())
                        c2 = tuple(bbox[3:5].int())
                        human_bbox.append([c1, c2])
                        human_frame = frame[c1[1] : c2[1], c1[0] : c2[0]]
                        face_c1, face_c2 = Face_Detection(faceCascade, human_frame)
                        face_frame = human_frame[face_c1[1] : face_c2[1], face_c1[0] : face_c2[0]]
                        face_img = Image.fromarray(face_frame)
                        # plt.imshow(face_img)
                        # plt.show()
                        faceTensor = test_transforms(face_img)
                        faceTensor = faceTensor.unsqueeze(0) # batch size 1
                        features = model_ft.features(faceTensor)
                        features = features.view(-1, 6*6*256)
                        feat_df = pd.DataFrame(features.detach().numpy(), columns=[f'img_feature_{n}' for n in range(features.size(-1))])
                        prediction = logreg.predict(feat_df)
                        if prediction == 0:
                            print("Masked face")
                            flag = Nose_Detection(noseCascade = noseCascade, face_bbox_frame = face_frame)
                            if flag:
                                print("Mask not worn properly")
                                cv2.rectangle(human_frame, face_c1, face_c2 ,(255, 0, 0), 2)
                                drawLabel(human_frame, 'Wear properly!', face_c1, face_c2, (0, 255, 0))
                            else:
                                drawLabel(human_frame, 'Good Job!', face_c1, face_c2, (255, 0, 0))
                                cv2.rectangle(human_frame, face_c1, face_c2 ,(0,255,0), 2)
                                print("Worn correctly")
                        else:
                            print("No mask!")
                            cv2.rectangle(human_frame, face_c1, face_c2 ,(0, 0, 255), 3)
                            drawLabel(human_frame, 'Wear a mask!', face_c1, face_c2, (0, 0, 255))
                except:
                    pass
            cv2.imshow("frame", orig_im)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
        else:
            break
        # time.sleep(1)