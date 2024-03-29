#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 09:57:19 2019

@author: taozhu
"""
import os
import numpy as np
from sklearn.model_selection import KFold
import math
from scipy import interpolate
from sklearn import metrics
import torch
import cv2
import argparse
from scipy.optimize import brentq
from PIL import Image
from torchvision import transforms

from model import LightCNN_29Layers_v3
from model import l2_norm


def p_args():
    parser = argparse.ArgumentParser(description='lightcnn_train_gt',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--session_name', default='light_no_arc')
    parser.add_argument(
        '--csv_path', default='./save/csv2/valid_data_done.csv')
    parser.add_argument('--pretrained_weights_path', default='./save/weights')

    # train
    parser.add_argument('--pretrained', default=True, type=bool)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--cuda', default=False, type=bool)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_memory', default=True, type=bool)
    parser.add_argument('--epoch', default=1000, type=int)
    parser.add_argument('--trainval_split', default=0.1, type=int)

    # adjust
    parser.add_argument('--lr', default=1e-6, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)

    args = parser.parse_args()
    return args


def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)


def add_extension(path):
    if os.path.exists(path+'.jpg'):
        return path+'.jpg'
    elif os.path.exists(path+'.png'):
        return path+'.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)


def get_paths(lfw_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    for pair in pairs:
        if len(pair) == 3:
            path0 = add_extension(os.path.join(
                lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(
                lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
            issame = True
        elif len(pair) == 4:
            path0 = add_extension(os.path.join(
                lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(
                lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
            issame = False
        # Only add the pair if both paths exist
        if os.path.exists(path0) and os.path.exists(path1):
            path_list += (path0, path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs, flush=True)

    return path_list, issame_list


def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric == 0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)
    elif distance_metric == 1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * \
            np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric

    return dist


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(
        predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp+fn == 0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn == 0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc


def calculate_val_far(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(
        predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    val = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)
    return val, far


def calculate_val(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10, distance_metric=0, subtract_mean=False):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    val = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate(
                [embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
            mean = 0.0
        dist = distance(embeddings1-mean, embeddings2-mean, distance_metric)

        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = calculate_val_far(
                threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        val[fold_idx], far[fold_idx] = calculate_val_far(
            threshold, dist[test_set], actual_issame[test_set])

    val_mean = np.mean(val)
    far_mean = np.mean(far)
    val_std = np.std(val)
    return val_mean, val_std, far_mean


def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        if subtract_mean:
            mean = np.mean(np.concatenate(
                [embeddings1[train_set], embeddings2[train_set]]), axis=0)
        else:
            mean = 0.0
        dist = distance(embeddings1-mean, embeddings2-mean, distance_metric)

        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(
                threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(
                threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])

        tpr = np.mean(tprs, 0)
        fpr = np.mean(fprs, 0)
    print("best_threshold:", thresholds[best_threshold_index], flush=True)
    return tpr, fpr, accuracy


def lfw_evaluate(embeddings, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds, embeddings1, embeddings2,
                                       np.asarray(actual_issame), nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = calculate_val(thresholds, embeddings1, embeddings2,
                                      np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    return tpr, fpr, accuracy, val, val_std, far


def lfw_evaluate_acc(embeddings, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = calculate_roc(thresholds, embeddings1, embeddings2,
                                       np.asarray(actual_issame), nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)

    return accuracy


def merge_eval(model_name_list):
    args = p_args()
    print('lfw test', flush=True)
    lfw_pairs = "/mnt/lustre/jiangsu/dlar/home/zyk17/data/LFW/pairs.txt"
    lfw_dir = "/mnt/lustre/jiangsu/dlar/home/zyk17/data/LFW/lfw_insight"

    model_list = []
    for model_name in model_name_list:
        model = LightCNN_29Layers_v3(num_classes=8642, args=args)
        pre_dict = torch.load(os.path.join(
            './save/weights', model_name), map_location='cpu')['state_dict']
        if model_name == 'lightCNN_62_checkpoint.pth.tar':
            pre_dict = {k[7:]: v for k, v in pre_dict.items(
            ) if 'fc2' not in k and 'arc' not in k}
        else:
            pre_dict = {k: v for k, v in pre_dict.items(
            ) if 'fc2' not in k and 'arc' not in k}
        model_dict = model.state_dict()
        model_dict.update(pre_dict)
        model.load_state_dict(model_dict)

        if args.cuda:
            model = model.cuda()

        model_list.append(model)

    trans = transforms.Compose([
        transforms.ToTensor()
    ])

    pairs = read_pairs(os.path.expanduser(lfw_pairs))
    paths, actual_issame = get_paths(os.path.expanduser(lfw_dir), pairs)
    embeddings = []
    for path in paths:
        img = Image.open(path).convert('L')
        img = trans(img)
        if args.cuda:
            img = img.cuda()
        fc = np.mean(np.concatenate([model(img.unsqueeze(0))[
                     1].detach().cpu().numpy() for model in model_list], 0), axis=0, keepdims=True)
        # norm?
        # fc = l2_norm(fc)
        embeddings.append(fc)
    embeddings = np.concatenate(embeddings, 0)

    tpr, fpr, accuracy, val, val_std, far = lfw_evaluate(
        embeddings, actual_issame, nrof_folds=10, distance_metric=1, subtract_mean=False)
    print('Accuracy: %2.5f+-%2.5f' %
          (np.mean(accuracy), np.std(accuracy)), flush=True)
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' %
          (val, val_std, far), flush=True)

    auc = metrics.auc(fpr, tpr)
    print('Area Under Curve (AUC): %1.3f' % auc, flush=True)
    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    print('Equal Error Rate (EER): %1.3f' % eer, flush=True)


def single_eval():
    args = p_args()
    batch_size = 100
    print('lfw test', flush=True)
    lfw_pairs = "/mnt/lustre/jiangsu/dlar/home/zyk17/data/LFW/pairs.txt"
    lfw_dir = "/mnt/lustre/jiangsu/dlar/home/zyk17/data/LFW/lfw_insight"
    model = LightCNN_29Layers_v3(num_classes=8642, args=args)
    pre_dict = torch.load(os.path.join(
        './save/weights', 'lightCNN_62_checkpoint.pth.tar'), map_location='cpu')['state_dict']
    pre_dict = {k[7:]: v for k, v in pre_dict.items(
    ) if 'fc2' not in k and 'arc' not in k}
    model_dict = model.state_dict()
    model_dict.update(pre_dict)
    model.load_state_dict(model_dict)

    if args.cuda:
        model = model.cuda()

    trans = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor()
    ])

    pairs = read_pairs(os.path.expanduser(lfw_pairs))
    paths, actual_issame = get_paths(os.path.expanduser(lfw_dir), pairs)
    embeddings = []
    for path in paths:
        img = Image.open(path).convert('L')
        img = trans(img)
        if args.cuda:
            img = img.cuda()
        _, fc = model(img.unsqueeze(0))
        # norm?
        # fc = l2_norm(fc)
        embeddings.append(fc.detach().cpu().numpy())
    embeddings = np.concatenate(embeddings, 0)

    tpr, fpr, accuracy, val, val_std, far = lfw_evaluate(
        embeddings, actual_issame, nrof_folds=10, distance_metric=1, subtract_mean=False)
    print('Accuracy: %2.5f+-%2.5f' %
          (np.mean(accuracy), np.std(accuracy)), flush=True)
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' %
          (val, val_std, far), flush=True)

    auc = metrics.auc(fpr, tpr)
    print('Area Under Curve (AUC): %1.3f' % auc, flush=True)
    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    print('Equal Error Rate (EER): %1.3f' % eer, flush=True)


if __name__ == '__main__':
    # merge_eval(['506.pth.tar', 'lightCNN_62_checkpoint.pth.tar'])
    single_eval()