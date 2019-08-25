#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import os
import getpass
import argparse
import logging
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pdb

import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
from .net.pspnet import PSPNet

import cv2
import numpy as np
from PIL import Image, ImageOps
from PIL import ImageDraw, ImageFont

# import onnx
# from onnx_tf.backend import prepare
# import tensorflow as tf

import glob

expose_arm = 1

models = {
    'squeezenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet'),
    'densenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet'),
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}

#parser = argparse.ArgumentParser(description="Pyramid Scene Parsing Network")
#parser.add_argument('--data-path', type=str, default='./LIP', help='Path to dataset folder')
#parser.add_argument('--models-path', type=str, default='./checkpoints', help='Path for storing model snapshots')
#parser.add_argument('--backend', type=str, default='densenet', help='Feature extractor')
#parser.add_argument('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
#parser.add_argument('--batch-size', type=int, default=1, help="Number of images sent to the network in one step.")
#parser.add_argument('--num-classes', type=int, default=20, help="Number of classes.")
#
#args = parser.parse_args()
#
args = dict()
args['gpu'] = False
args['models_path'] = '/data/parsesite/parse/parse_model/checkpoints'
args['backend'] = 'densenet'
args['num_classes'] = 20


def equalizeHistOp(image):
    """

    :param image:   3d channel image
    :return:
    """
    (b, g, r) = cv2.split(image)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    result = cv2.merge((bH, gH, rH))
    return result


def gammaTrans(image, gamma=1.0):
    # 具体做法先归一化到1，然后gamma作为指数值求出新的像素值再还原
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    # 实现映射用的是Opencv的查表函数
    return cv2.LUT(image, gamma_table)


def brightCheck(image):
    """

    :param image: 3-d image   bgr
    :return: k, da
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.reshape(gray, (-1))
    da = np.mean(gray) - 127.5
    D = abs(da)
    recounted = Counter(gray)
    Ma = 0
    for i in range(256):
        Ma += abs(i - 127.5 - da) * recounted[i]
    Ma = Ma / len(gray)
    M = abs(Ma)
    k = D / M
    return k, da


def build_network(snapshot, backend):
    epoch = 0
    backend = backend.lower()
    net = models[backend]()
    net = nn.DataParallel(net)
    if snapshot is not None:
        _, epoch = os.path.basename(snapshot).split('_')
        if not epoch == 'last':
            epoch = int(epoch)
        # model_dict = net.state_dict()
        # checkpoint = torch.load(snapshot)

        # singlegpu_checkpoint = {k[7:]: v for k, v in checkpoint.items()}

        # model_dict.update(singlegpu_checkpoint)
        net.load_state_dict(torch.load(snapshot, map_location='cpu'))
        logging.info(
            "Snapshot for epoch {} loaded from {}".format(epoch, snapshot))

    if args['gpu']:
        net = net.cuda()
    '''
    example = torch.ones([1, 3, 256, 256]).cuda()
    # print(example.size())
    torch.onnx.export(net, example, './model_simple.onnx', input_names=['input'], output_names=['output'])
    model_onnx = onnx.load('./model_simple.onnx')
    model_onnx.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '?'
    tf_rep = prepare(model_onnx, strict=True)
    print(tf_rep.tensor_dict)
    tf_rep.export_graph('./human_expose_detector.pb')
    '''
    return net, epoch


def show_image_eval(img, pred):
    fig, axes = plt.subplots(1, 2)
    ax0, ax1 = axes
    ax0.get_xaxis().set_ticks([])
    ax0.get_yaxis().set_ticks([])
    ax1.get_xaxis().set_ticks([])
    ax1.get_yaxis().set_ticks([])

    classes = np.array(('Background',  # always index 0
                        'Hat', 'Hair', 'Glove', 'Sunglasses',
                        'UpperClothes', 'Dress', 'Coat', 'Socks',
                        'Pants', 'Jumpsuits', 'Scarf', 'Skirt',
                        'Face', 'Left-arm', 'Right-arm', 'Left-leg',
                        'Right-leg', 'Left-shoe', 'Right-shoe',))
    colormap = [(0, 0, 0),
                (1, 0.25, 0), (0, 0.25, 0), (0.5, 0, 0.25), (1, 1, 1),
                (1, 0.75, 0), (0, 0, 0.5), (0.5, 0.25, 0), (0.75, 0, 0.25),
                (1, 0, 0.25), (0, 0.5, 0), (0.5, 0.5, 0), (0.25, 0, 0.5),
                (1, 0, 0.75), (0, 0.5, 0.5), (0.25, 0.5, 0.5), (1, 0, 0),
                (1, 0.25, 0), (0, 0.75, 0), (0.5, 0.75, 0), ]
    cmap = matplotlib.colors.ListedColormap(colormap)
    bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
              11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    h, w, _ = pred.shape

    def denormalize(img, mean, std):
        c, _, _ = img.shape
        for idx in range(c):
            img[idx, :, :] = img[idx, :, :] * std[idx] + mean[idx]
        return img

    img = denormalize(img[0].numpy(), [0.485, 0.456, 0.406], [
                      0.229, 0.224, 0.225])
    img = img.transpose(1, 2, 0).reshape((h, w, 3))
    pred = pred.reshape((h, w))

    # show image
    ax0.set_title('img')
    ax0.imshow(img)
    ax1.set_title('pred')
    mappable = ax1.imshow(pred, cmap=cmap, norm=norm)

    # colorbar legend
    cbar = plt.colorbar(mappable, ax=axes, shrink=0.7,
                        )  # orientation='horizontal'
    cbar.ax.get_yaxis().set_ticks([])
    for j, lab in enumerate(classes):
        cbar.ax.text(1.3, (j + 0.45) / 20.0, lab,
                     ha='left', va='center', )  # fontsize=7
    # cbar.ax.get_yaxis().labelpad = 5

    # plt.savefig(fname="result.jpg")
    plt.show()


def test(frame_path):
    frame = cv2.imread(frame_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # pdb.set_trace()

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    mean = np.mean(gray)
    h, w = gray.shape
    nums = len(np.where(gray < 50)[0])
    #print(mean, nums / h / w)
    if mean < 100:
        frame = gammaTrans(frame, 0.8)
    # frame = equalizeHistOp(frame)
    # print(nums /h/w)
    elif mean > 140 and nums / h/w < 0.15:
        frame = gammaTrans(frame, 2)

    flag = False
    # --------------- model --------------- #
    # --------------- model --------------- #
    snapshot = os.path.join(args['models_path'], args['backend'], 'PSPNet_last')
    net, starting_epoch = build_network(snapshot, args['backend'])
    net.eval()

    # ------------ data loader ------------ #
    img_transform = transforms.Compose([
        transforms.Resize((256, 256), 3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # video_capture = cv2.VideoCapture("1.mp4")
    # frames_cnt=0
    # while (frames_cnt < 1):
    # ret, frame = video_capture.read()
    # frame = cv2.imread('timg1.jpg')

    img = Image.fromarray(frame)
    img_pt = img_transform(img)
    img_pt = Variable(img_pt[None, :, :, :])
    with torch.no_grad():
        pred_seg, pred_cls = net(img_pt.cuda() if args['gpu'] else img_pt)
        pred_seg = pred_seg[0]
        pred = pred_seg.cpu().numpy().transpose(1, 2, 0)
        pred = np.asarray(np.argmax(pred, axis=2),
                          dtype=np.uint8).reshape((256, 256, 1))

    # classes = np.array(('Background',  # always index 0
    #                     'Hat', 'Hair', 'Glove', 'Sunglasses',
    #                     'UpperClothes', 'Dress', 'Coat', 'Socks',
    #                     'Pants', 'Jumpsuits', 'Scarf', 'Skirt',
    #                     'Face', 'Left-arm', 'Right-arm', 'Left-leg',
    #                     'Right-leg', 'Left-shoe', 'Right-shoe',))

    img_show = np.asarray(pred * 10)
    (_, thresh_face) = cv2.threshold(img_show, 120, 255, cv2.THRESH_TOZERO)
    (_, thresh_face) = cv2.threshold(thresh_face, 130, 255, cv2.THRESH_TOZERO_INV)
    contours_face, hierarchy_face = cv2.findContours(
        thresh_face, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if (len(contours_face) > 0):
        for i in range(0, len(contours_face)):
            x_face, y_face, w_face, h_face = cv2.boundingRect(contours_face[i])

    (_, thresh0) = cv2.threshold(img_show, 40, 255, cv2.THRESH_TOZERO)
    (_, thresh0) = cv2.threshold(thresh0, 70, 255, cv2.THRESH_TOZERO_INV)

    contours0, hierarchy0 = cv2.findContours(
        thresh0, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    h, w = thresh0.shape

    r2High = h
    h_clothes_up = h
    h_clothes_down = 0
    sum_colthes = 0
    area = 0
    percent = 0.0
    if (len(contours0) > 0):
        for i in range(0, len(contours0)):
            x0, y0, w0, h0 = cv2.boundingRect(contours0[i])
            if (cv2.contourArea(contours0[i]) > 100):
                sum_colthes += cv2.contourArea(contours0[i])
            area += w0 * h0
            if (h_clothes_up > y0):
                h_clothes_up = y0
            if (h_clothes_down < y0 + h0):
                h_clothes_down = y0 + h0

        if (sum_colthes > 7000):
            flag = False
        else:
            percent = 1 - 1.1 * (sum_colthes / area)
            flag = True

    else:
        flag = True
        percent = 1

    if (expose_arm):

        (_, thresh1) = cv2.threshold(img_show, 130, 255, cv2.THRESH_TOZERO)
        (_, thresh1) = cv2.threshold(thresh1, 150, 255, cv2.THRESH_TOZERO_INV)
        # cv2.imshow('thresh', thresh1)
        contours1, hierarchy1 = cv2.findContours(
            thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        ## expose arm ##
        for i in range(0, len(contours1)):
            x1, y1, w1, h1 = cv2.boundingRect(contours1[i])
            if (r2High > h1):
                r2High = h1
            #cv2.rectangle(img_show, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 255), 5)

        ## expose arm ##
        if r2High > (h_clothes_down - h_clothes_up) * 1 / 2:
            flag_arm = True
        else:
            flag_arm = False

    try:
        thresh1 = thresh0[0:255, y_face + h_face:y_face + 2 * h_face]
        if (cv2.countNonZero(thresh1) < 5000):
            flag = True
            percent = 1

        flag |= flag_arm

        #cv2.imshow('Contours', img_show)
        if 1:
            show_image_eval(img_pt, pred)

        # frames_cnt += 1
    except:
        flag = True
    return flag, percent


if __name__ == '__main__':
    if 1:
        paths = glob.glob(os.path.join('samples', '*.jpg'))
        # paths = ['/run/media/sky/a5f86773-a7e2-4f0f-a16e-1fa289d496bd/CV_WORK/single_human_paring/img/woman/unexpose/img_001.jpg']
        # paths = ['/run/media/sky/a5f86773-a7e2-4f0f-a16e-1fa289d496bd/CV_WORK/single_human_paring/img/woman/unexpose/img_002.jpg']
        for path in paths:
            name, _ = os.path.splitext(path)
            # frame = cv2.imread('/run/media/sky/a5f86773-a7e2-4f0f-a16e-1fa289d496bd/CV_WORK/facerecognition/datasets/imgs/lightcnn1/1.jpg')
            # image_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            # 直方图均衡化
            # image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 2])
            # 显示效果
            # frame = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)

            frame = cv2.imread(path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            flag, percent = test(frame)
