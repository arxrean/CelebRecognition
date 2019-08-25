#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import sys
import os
# import argparse
import logging

import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
import cv2

import numpy as np
from PIL import Image
from django.conf import settings

sys.path.append(os.path.join(settings.BASE_DIR,
                             'parse/parse_zt_model/bareness/single_human_paring'))
from net.pspnet import PSPNet

# classes = np.array(('Background',  # always index 0
#                         'Hat', 'Hair', 'Glove', 'Sunglasses',
#                         'UpperClothes', 'Dress', 'Coat', 'Socks',
#                         'Pants', 'Jumpsuits', 'Scarf', 'Skirt',
#                         'Face', 'Left-arm', 'Right-arm', 'Left-leg',
#                         'Right-leg', 'Left-shoe', 'Right-shoe',))

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

# parser = argparse.ArgumentParser(description="Pyramid Scene Parsing Network")
# parser.add_argument('--data-path', type=str, default='./LIP', help='Path to dataset folder')
# parser.add_argument('--models-path', type=str, default='./single_human_paring/checkpoints', help='Path for storing model snapshots')
# parser.add_argument('--backend', type=str, default='densenet', help='Feature extractor')
# parser.add_argument('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
# parser.add_argument('--batch-size', type=int, default=1, help="Number of images sent to the network in one step.")
# parser.add_argument('--num-classes', type=int, default=20, help="Number of classes.")

args = dict()
args['gpu'] = False
args['models_path'] = os.path.join(
    settings.BASE_DIR, 'parse/parse_zt_model/bareness/single_human_paring/checkpoints')
args['backend'] = 'densenet'
args['num_classes'] = 20


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
    net = net.cuda() if args['gpu'] else net
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


class test():
    def __init__(self):
        snapshot = os.path.join(
            args['models_path'], args['backend'], 'PSPNet_last')
        self.net, self.starting_epoch = build_network(
            snapshot, args['backend'])
        self.net.eval()
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256), 3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def pred(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img_pt = self.img_transform(img)
        img_pt = Variable(img_pt[None, :, :, :])
        with torch.no_grad():
            pred_seg, pred_cls = self.net(img_pt.cuda() if args['gpu'] else img_pt)
            pred_seg = pred_seg[0]
            pred = pred_seg.cpu().numpy().transpose(1, 2, 0)
            pred = np.asarray(np.argmax(pred, axis=2),
                              dtype=np.uint8).reshape((256, 256, 1))
        return pred
