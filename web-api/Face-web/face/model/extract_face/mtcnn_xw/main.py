import json
import cv2
import numpy as np
from PIL import Image
import datetime
import os
from django.conf import settings

from .mtcnn import Mtcnn
from .align_trans import get_reference_facial_points, warp_and_crop_face


def detect_by_mtcnn(model, configs, img_list):
    align_img_list = []
    align_img_rgb_list = []
    for img in img_list:
        img, img_fd, ratio = read_img_from_opencv(img, configs)
        bboxes, landmarks = model.detect_face(img_fd)
        if len(bboxes) == 0:
            return 'no face is detected!'
        margin_bboxes = margin_operation_opencv(
            bboxes, img_fd.shape[0:2], configs)

        bboxes_ori = margin_bboxes * ratio
        landmarks_ori = landmarks * ratio

        batch_imgs = align_multi(
            img, landmarks_ori.transpose())
        for img in batch_imgs:
            cropped_img = Image.fromarray(
                img.astype('uint8')).convert('L')
            align_img_list.append(cropped_img)

    return align_img_list


def read_img_from_opencv(img_path, configs):
    try:
        img_path = os.path.join(settings.MEDIA_URL, img_path)
        img = cv2.imread(img_path)
        if img.ndim < 2:
            raise Exception('invalid dim of input img!')
        if img.ndim == 2:
            img = utils.to_rgb(img)
        elif img.ndim == 3:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)

        detect_w = configs['config']['detect_w']
        if img.shape[1] > detect_w:
            detect_h = int(img.shape[0] * detect_w / img.shape[1])
            img_fd = cv2.resize(img, (detect_w, detect_h),
                                interpolation=cv2.INTER_LINEAR)
            ratio = img.shape[1] / detect_w

            return img, img_fd, ratio
        else:
            img_fd = img.copy()
            ratio = 1

            return img, img_fd, ratio
    except Exception as e:
        print(e)


def margin_operation_opencv(bounding_boxes, img_size, configs):
    margin_bboxes = []

    if configs['para']['margin'] >= 1:
        for bbox in bounding_boxes:
            bbox[0] = np.maximum(bbox[0] - configs['para']['margin'], 0)
            bbox[1] = np.maximum(bbox[1] - configs['para']['margin'], 0)
            bbox[2] = np.minimum(
                bbox[2] + configs['para']['margin'], img_size[1])
            bbox[3] = np.minimum(
                bbox[3] + configs['para']['margin'], img_size[0])
            margin_bboxes.append(bbox)
    else:
        for bbox in bounding_boxes:
            margin_w = configs['para']['margin'] * abs(bbox[0] - bbox[2])
            margin_h = configs['para']['margin'] * abs(bbox[1] - bbox[3])
            box_margin = bbox.copy()
            box_margin[0] = np.maximum(bbox[0]-margin_w, 0)
            box_margin[1] = np.maximum(bbox[1]-margin_h, 0)
            box_margin[2] = np.minimum(bbox[2]+margin_w, img_size[1])
            box_margin[3] = np.minimum(bbox[3]+margin_h, img_size[0])
            margin_bboxes.append(box_margin)

    return np.array(margin_bboxes)


def align_multi(img, landmarks, crop_size=(112, 112)):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = []
    for landmark in landmarks:
        facial5points = [[landmark[j], landmark[j+5]] for j in range(5)]
        warped_face = warp_and_crop_face(img_gray, facial5points, get_reference_facial_points(
            default_square=True), crop_size=crop_size)
        faces.append(warped_face)
    return faces
