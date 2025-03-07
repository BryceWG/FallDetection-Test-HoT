from __future__ import division
import time
import torch
import numpy as np
import cv2
import os
import sys
import random
import pickle as pkl
import argparse

from lib.yolov3.util import *
from lib.yolov3.darknet import Darknet
from lib.yolov3 import preprocess

cur_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.join(cur_dir, '../../../')
chk_root = os.path.join(project_root, 'checkpoint/')
data_root = os.path.join(project_root, 'data/')


sys.path.insert(0, project_root)
sys.path.pop(0)


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """
    ori_img = img
    dim = ori_img.shape[1], ori_img.shape[0]
    img = cv2.resize(ori_img, (inp_dim, inp_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, ori_img, dim


def write(x, img, colors):
    x = [int(i) for i in x]
    c1 = tuple(x[0:2])
    c2 = tuple(x[2:4])

    label = 'People {}'.format(0)
    color = (0, 0, 255)
    cv2.rectangle(img, c1, c2, color, 2)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    return img


def get_default_args():
    """
    返回默认参数而不是解析命令行
    """
    class Args:
        def __init__(self):
            self.confidence = 0.70
            self.nms_thresh = 0.4
            self.reso = 416
            self.weight_file = 'demo/lib/checkpoint/yolov3.weights'
            self.cfg_file = cur_dir + '/cfg/yolov3.cfg'
            self.animation = False
            self.video = 'camera'
            self.image = cur_dir + '/data/dog-cycle-car.png'
            self.num_person = 1
            self.gpu = '0'
            self.fix_z = False
    
    return Args()


def arg_parse():
    """
    返回默认参数而不是解析命令行
    """
    return get_default_args()


def load_model(args=None, CUDA=None, inp_dim=416):
    if args is None:
        args = get_default_args()

    if CUDA is None:
        CUDA = torch.cuda.is_available()

    # Set up the neural network
    model = Darknet(args.cfg_file)
    model.load_weights(args.weight_file)
    # print("YOLOv3 network successfully loaded")

    model.net_info["height"] = inp_dim
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()

    # Set the model in evaluation mode
    model.eval()

    return model


def yolo_human_det(img, model=None, reso=416, confidence=0.70):
    args = get_default_args()
    inp_dim = reso
    num_classes = 80

    CUDA = torch.cuda.is_available()
    if model is None:
        model = load_model(args, CUDA, inp_dim)

    if type(img) == str:
        assert os.path.isfile(img), 'The image path does not exist'
        img = cv2.imread(img)

    img, ori_img, img_dim = preprocess.prep_image(img, inp_dim)
    img_dim = torch.FloatTensor(img_dim).repeat(1, 2)

    with torch.no_grad():
        if CUDA:
            img_dim = img_dim.cuda()
            img = img.cuda()
        output = model(img, CUDA)
        output = write_results(output, confidence, num_classes, nms=True, nms_conf=args.nms_thresh, det_hm=True)

        if len(output) == 0:
            return None, None

        img_dim = img_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(inp_dim / img_dim, 1)[0].view(-1, 1)

        output[:, [1, 3]] -= (inp_dim - scaling_factor * img_dim[:, 0].view(-1, 1)) / 2
        output[:, [2, 4]] -= (inp_dim - scaling_factor * img_dim[:, 1].view(-1, 1)) / 2
        output[:, 1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, img_dim[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, img_dim[i, 1])

    bboxs = []
    scores = []
    for i in range(len(output)):
        item = output[i]
        bbox = item[1:5].cpu().numpy()
        # conver float32 to .2f data
        bbox = [round(i, 2) for i in list(bbox)]
        score = item[5].cpu().numpy()
        bboxs.append(bbox)
        scores.append(score)
    scores = np.expand_dims(np.array(scores), 1)
    bboxs = np.array(bboxs)

    return bboxs, scores
