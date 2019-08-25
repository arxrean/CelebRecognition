# coding=utf-8
import os
import shutil
import sys
import time
from django.conf import settings

import cv2
import numpy as np
import tensorflow as tf

from .nets import model_train as model
from .utils.rpn_msr.proposal_layer import proposal_layer
from .utils.text_connector.detectors import TextDetector
from PIL import Image
from glob import glob
import copy

test_data_path = 'src/detection/ctpn/data/demo/'
output_path = 'src/detection/ctpn/data/res/'
gpu = '0'
checkpoint_path = 'src/detection/ctpn/checkpoints_mlt/'


def get_images():
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def resize_image(img):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    # im_scale = float(600) / float(im_size_min)
    # if np.round(im_scale * im_size_max) > 1200:
    #     im_scale = float(1200) / float(im_size_max)
    # new_h = int(img_size[0] * im_scale)
    # new_w = int(img_size[1] * im_scale)
    #
    # new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    # new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    im_scale = float(1200) / float(im_size_max)
    if int(im_scale * im_size_min) <=16:
        im_scale = float(16) / float(im_size_min)

    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h % 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w % 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return re_im, (new_h / img_size[0], new_w / img_size[1])


class ctpn():
    def __init__(self, checkpoint_path = checkpoint_path):
        self.checkpoint_path = checkpoint_path

        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

        with tf.get_default_graph().as_default():
            self.input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
            self.input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')
            global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
            self.bbox_pred, self.cls_pred, self.cls_prob = model.model(self.input_image)
            variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
            saver = tf.train.Saver(variable_averages.variables_to_restore())

            sess_config = tf.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=sess_config)

            ckpt_state = tf.train.get_checkpoint_state(self.checkpoint_path)
            model_path = os.path.join(self.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(self.sess, model_path)


    def predict(self, im):
        if im.shape[0] >=8:
            img, (rh, rw) = resize_image(im)
            h, w, c = img.shape
            im_info = np.array([h, w, c]).reshape([1, 3])
            bbox_pred_val, cls_prob_val = self.sess.run([self.bbox_pred, self.cls_prob],
                                                        feed_dict={self.input_image: [img]})
                                                                   #self.input_im_info: im_info})

            textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)
            scores = textsegs[:, 0]
            textsegs = textsegs[:, 1:5]

            textdetector = TextDetector(DETECT_MODE='O')
            boxes = textdetector.detect(textsegs, scores[:, np.newaxis], img.shape[:2])
            boxes = np.array(boxes, dtype=np.int)

            text_recs_resize = boxes[:, :8]

            text_recs_resize = sorted(text_recs_resize, key=lambda x: sum([x[1], x[3], x[5], x[7]]))
        else:
            text_recs_resize = []
            img = im
        return text_recs_resize, img


    def get_roi(self, ims):
        nums = len(ims)
        imgs = []
        im_infos = []
        text_recs_resize_all = []
        for im in ims:
            img, (rh, rw) = resize_image(im)
            h, w, c = img.shape
            imgs.append(img)
            im_infos.append(np.array([h, w, c]).reshape([1, 3]))

        bbox_pred_vals, cls_prob_vals = self.sess.run([self.bbox_pred, self.cls_prob],
                                                    feed_dict={self.input_image: imgs})
        textdetector = TextDetector(DETECT_MODE='O')
        for i in range(nums):
            cls_prob_val = cls_prob_vals[i:i+1,:,:,:]
            bbox_pred_val = bbox_pred_vals[i:i+1,:,:,:]
            im_info = im_infos[i]
            textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)
            scores = textsegs[:, 0]
            textsegs = textsegs[:, 1:5]

            boxes = textdetector.detect(textsegs, scores[:, np.newaxis], imgs[i].shape[:2])
            boxes = np.array(boxes, dtype=np.int)

            text_recs_resize = boxes[:, :8]

            text_recs_resize = sorted(text_recs_resize, key=lambda x: sum([x[0], x[2], x[4], x[6]]))
            text_recs_resize_all.append(text_recs_resize)
        return text_recs_resize_all, imgs


# if __name__ == '__main__':
#
#     image_files = glob('test_disk/*.*')
#     c = ctpn()
#     for image_file in sorted(image_files):
#         image = cv2.imread(image_file)[:, :, ::-1]
#         text_recs, image_framed, img = c.predict(image, image_file)
