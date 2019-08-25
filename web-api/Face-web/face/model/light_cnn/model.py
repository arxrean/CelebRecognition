from __future__ import print_function
import argparse
import os
import shutil
import time
import math
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.nn.parameter import Parameter
import numpy as np
import cv2


class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(
                in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])


class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv = mfm(in_channels, out_channels,
                        kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x


class resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resblock, self).__init__()
        self.conv1 = mfm(in_channels, out_channels,
                         kernel_size=3, stride=1, padding=1)
        self.conv2 = mfm(in_channels, out_channels,
                         kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + res
        return out

#OUT_NUM     = 93957


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class network_29layers_v3(nn.Module):
    def __init__(self, block, layers, num_classes=79077):
        super(network_29layers_v3, self).__init__()
        self.out_num = num_classes
        self.weights_arc = Parameter(torch.Tensor(256, num_classes))
        torch.nn.init.kaiming_uniform_(self.weights_arc, a=math.sqrt(5))
        self.conv1 = mfm(1, 48, 5, 1, 2)
        self.block1 = self._make_layer(block, layers[0], 48, 48)
        self.group1 = group(48, 96, 3, 1, 1)
        self.block2 = self._make_layer(block, layers[1], 96, 96)
        self.group2 = group(96, 192, 3, 1, 1)
        self.block3 = self._make_layer(block, layers[2], 192, 192)
        self.group3 = group(192, 128, 3, 1, 1)
        self.block4 = self._make_layer(block, layers[3], 128, 128)
        self.group4 = group(128, 128, 3, 1, 1)
        self.fc = nn.Linear(7*7*128, 256)

    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x, target_var=None):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block1(x)
        x = self.group1(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block2(x)
        x = self.group2(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block3(x)
        x = self.group3(x)
        x = self.block4(x)
        x = self.group4(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = x.view(x.size(0), -1)
        fc = self.fc(x)
        if target_var is None:
            out = []
            fc = l2_norm(fc)
        else:
            fc = l2_norm(fc)
            out = self.arcface_loss(fc, target_var, self.out_num, s=64, m=0.1)
        return out, fc

    def arcface_loss(self, embedding, labels, out_num, s=64., m=0.5):
        #print("label:", labels)
        labels_v = labels.view(labels.shape[0], 1)
        #print(labels_v.shape, labels.shape)
        cos_m = math.cos(m)
        sin_m = math.sin(m)
        mm = sin_m * m  # issue 1
        threshold = math.cos(math.pi - m)
        embedding_norm = torch.norm(embedding, p=2, dim=1, keepdim=True)
        embedding = torch.div(embedding, embedding_norm)
        weights_norm = torch.norm(self.weights_arc, p=2, dim=0, keepdim=True)
        weights_arc_normed = torch.div(self.weights_arc, weights_norm)

        cos_t = torch.mm(embedding, weights_arc_normed)
        # print(embedding)
        embedding_cpu = embedding.cpu()
        embedding_np = embedding_cpu.data.numpy()
        cos_t2 = torch.mul(cos_t, cos_t)
        sin_t2 = 1. - cos_t2
        sin_t = torch.sqrt(sin_t2)
        cos_mt = s * (torch.mul(cos_t, cos_m) - torch.mul(sin_t, sin_m))

        cond_v = cos_t - threshold
        cond = nn.ReLU(inplace=True)

        keep_val = s*(cos_t - mm)
        # print(cond)
        cos_mt_temp = torch.where(cond(cond_v) > 0, cos_mt, keep_val)

        #mask = torch.one_hot(labels, depth=out_num, name='one_hot_mask')
        mask = torch.zeros(labels_v.shape[0], out_num)
        #print('mask before:',mask.device)

        mask.scatter_(1, labels_v, 1)
        inv_mask = 1. - mask

        s_cos_t = torch.mul(s, cos_t)

        output = torch.add(torch.mul(s_cos_t, inv_mask),
                           torch.mul(cos_mt_temp, mask))
        return output


def LightCNN_29Layers_v3(**kwargs):
    model = network_29layers_v3(resblock, [1, 2, 3, 4], **kwargs)
    return model
