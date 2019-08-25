import tensorflow as tf
from tensorflow.contrib import slim
from django.conf import settings
import sys
import os

from . import vgg
sys.path.append(os.path.join(settings.BASE_DIR,"ocr/custom_ocr_module/src/detection/ctpn"))
from utils.rpn_msr.anchor_target_layer import anchor_target_layer as anchor_target_layer_py


def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)


def make_var(name, shape, initializer=None):
    return tf.get_variable(name, shape, initializer=initializer)


def Bilstm(net, input_channel, hidden_unit_num, output_channel, scope_name):
    # width--->time step
    # reshape [NH, W,C]
    # BLSTM   [NH,W, 2*hidden_unit_num]
    # reshape [NHW, 2*hidden_unit_num]
    # FC      [NHW, output_channel]
    # reshape [N, H, W, output_channel]
    with tf.variable_scope(scope_name) as scope:
        shape = tf.shape(net)
        N, H, W, C = shape[0], shape[1], shape[2], shape[3]
        net = tf.reshape(net, [N * H, W, C])
        net.set_shape([None, None, input_channel])

        lstm_fw_cell = tf.contrib.rnn.LSTMCell(hidden_unit_num, state_is_tuple=True)
        lstm_bw_cell = tf.contrib.rnn.LSTMCell(hidden_unit_num, state_is_tuple=True)

        lstm_out, last_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, net, dtype=tf.float32)
        lstm_out = tf.concat(lstm_out, axis=-1)

        lstm_out = tf.reshape(lstm_out, [N * H * W, 2 * hidden_unit_num])

        init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
        init_biases = tf.constant_initializer(0.0)
        weights = make_var('weights', [2 * hidden_unit_num, output_channel], init_weights)
        biases = make_var('biases', [output_channel], init_biases)

        outputs = tf.matmul(lstm_out, weights) + biases

        outputs = tf.reshape(outputs, [N, H, W, output_channel])
        return outputs


def lstm_fc(net, input_channel, output_channel, scope_name):
    #         [N, H, W, C]
    # reshape [NHW, C]
    # FC      [NHW, output_channel]
    # reshape [N, H, W, output_channel]
    with tf.variable_scope(scope_name) as scope:
        shape = tf.shape(net)
        N, H, W, C = shape[0], shape[1], shape[2], shape[3]
        net = tf.reshape(net, [N * H * W, C])

        init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
        init_biases = tf.constant_initializer(0.0)
        weights = make_var('weights', [input_channel, output_channel], init_weights)
        biases = make_var('biases', [output_channel], init_biases)

        output = tf.matmul(net, weights) + biases
        output = tf.reshape(output, [N, H, W, output_channel])
    return output


def model(image):
    image = mean_image_subtraction(image)                 # subtract the mean  N ,H, W, C
    with slim.arg_scope(vgg.vgg_arg_scope()):
        conv5_3 = vgg.vgg_16(image)                       # N ,H/16, W/16, 512

    rpn_conv = slim.conv2d(conv5_3, 512, 3)               # N ,H/16, W/16, 512

    lstm_output = Bilstm(rpn_conv, 512, 128, 512, scope_name='BiLSTM')      # [N, H/16, W/16, output_channel]

    # rpn  2 branch
    bbox_pred = lstm_fc(lstm_output, 512, 10 * 4, scope_name="bbox_pred")   # [N, H/16, W/16, 40]
    cls_pred = lstm_fc(lstm_output, 512, 10 * 2, scope_name="cls_pred")     # [N, H/16, W/16, 20]

    # transpose: (1, H, W, A x d) -> (1, H, WxA, d)
    cls_pred_shape = tf.shape(cls_pred)
    cls_pred_reshape = tf.reshape(cls_pred, [cls_pred_shape[0], cls_pred_shape[1], -1, 2])     # [N, H/16, W/16*10, 2]

    cls_pred_reshape_shape = tf.shape(cls_pred_reshape)
    cls_prob = tf.reshape(tf.nn.softmax(tf.reshape(cls_pred_reshape, [-1, cls_pred_reshape_shape[3]])),            # softmax [NHW/16/16*10, 2]
                          [-1, cls_pred_reshape_shape[1], cls_pred_reshape_shape[2], cls_pred_reshape_shape[3]],   # reshape [N, H/16, W/16, 2]
                          name="cls_prob")

    return bbox_pred, cls_pred, cls_prob           # [N, H/16, W/16, 40]     [N, H/16, W/16, 20]     [N, H/16, W/16, 2]


def anchor_target_layer(cls_pred, bbox, im_info, scope_name):
    # [N, H/16, W/16, 20]    [N, 5],  [N, 3]
    with tf.variable_scope(scope_name) as scope:
        # 'rpn_cls_score', 'gt_boxes', 'im_info'
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
            tf.py_func(anchor_target_layer_py,
                       [cls_pred, bbox, im_info, [16, ], [16]],
                       [tf.float32, tf.float32, tf.float32, tf.float32])

        rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels, tf.int32),
                                          name='rpn_labels')
        rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets,
                                                name='rpn_bbox_targets')
        rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights,
                                                       name='rpn_bbox_inside_weights')
        rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights,
                                                        name='rpn_bbox_outside_weights')

        return [rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights]


def smooth_l1_dist(deltas, sigma2=9.0, name='smooth_l1_dist'):
    with tf.name_scope(name=name) as scope:
        deltas_abs = tf.abs(deltas)
        smoothL1_sign = tf.cast(tf.less(deltas_abs, 1.0 / sigma2), tf.float32)
        return tf.square(deltas) * 0.5 * sigma2 * smoothL1_sign + \
               (deltas_abs - 0.5 / sigma2) * tf.abs(smoothL1_sign - 1)


def loss(bbox_pred, cls_pred, bbox, im_info):
    # [N, H/16, W/16, 40]     [N, H/16, W/16, 20]    [N, 5],  [N, 3]
    rpn_data = anchor_target_layer(cls_pred, bbox, im_info, "anchor_target_layer")           # rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights
    # ((1, H/16, W/16, A), (1, H/16, W/16, 4A), (1, H/16, W/16, 4A), (1, H/16, W/16, 4A))

    # classification loss
    # transpose: (1, H, W, A x d) -> (1, H, WxA, d)
    cls_pred_shape = tf.shape(cls_pred)
    cls_pred_reshape = tf.reshape(cls_pred, [cls_pred_shape[0], cls_pred_shape[1], -1, 2])          # [N, H/16, W/16 * 10, 2]
    rpn_cls_score = tf.reshape(cls_pred_reshape, [-1, 2])                                           # [N * H/16 * W/16 * 10, 2]
    rpn_label = tf.reshape(rpn_data[0], [-1])                                                       # [H/16 * W/16 * A]
    # ignore_label(-1)
    fg_keep = tf.equal(rpn_label, 1)
    rpn_keep = tf.where(tf.not_equal(rpn_label, -1))                                           # 只保留前景和背景
    rpn_cls_score = tf.gather(rpn_cls_score, rpn_keep)                                         # 选取前景和背景的 cls score
    rpn_label = tf.gather(rpn_label, rpn_keep)                                                 # 选取前景和背景的 label (0/1)
    rpn_cross_entropy_n = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=rpn_label, logits=rpn_cls_score)      # 计算分类交叉熵  (?,1)

    # box loss
    rpn_bbox_pred = bbox_pred                     # [N, H/16, W/16, 40]
    rpn_bbox_targets = rpn_data[1]                # (1, H/16, W/16, 4A)
    rpn_bbox_inside_weights = rpn_data[2]         # (1, H/16, W/16, 4A)
    rpn_bbox_outside_weights = rpn_data[3]        # (1, H/16, W/16, 4A)

    rpn_bbox_pred = tf.gather(tf.reshape(rpn_bbox_pred, [-1, 4]), rpn_keep)  # shape (N, 4)                  # 选取前景和背景的bbox pred (dx,dy,dw,dh)   [N * H/16 * W/16 * 10, 1, 4]
    rpn_bbox_targets = tf.gather(tf.reshape(rpn_bbox_targets, [-1, 4]), rpn_keep)                            # 选取前景和背景的anchor和gt的(dx,dy,dw,dh)  [H/16 * W/16 * 10, 1, 4]
    rpn_bbox_inside_weights = tf.gather(tf.reshape(rpn_bbox_inside_weights, [-1, 4]), rpn_keep)
    rpn_bbox_outside_weights = tf.gather(tf.reshape(rpn_bbox_outside_weights, [-1, 4]), rpn_keep)

    rpn_loss_box_n = tf.reduce_sum(rpn_bbox_outside_weights * smooth_l1_dist(
        rpn_bbox_inside_weights * (rpn_bbox_pred - rpn_bbox_targets)), reduction_indices=[1])            # (?, 4)

    rpn_loss_box = tf.reduce_sum(rpn_loss_box_n) / (tf.reduce_sum(tf.cast(fg_keep, tf.float32)) + 1)
    rpn_cross_entropy = tf.reduce_mean(rpn_cross_entropy_n)

    model_loss = rpn_cross_entropy + rpn_loss_box

    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n(regularization_losses) + model_loss

    tf.summary.scalar('model_loss', model_loss)
    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('rpn_cross_entropy', rpn_cross_entropy)
    tf.summary.scalar('rpn_loss_box', rpn_loss_box)

    return total_loss, model_loss, rpn_cross_entropy, rpn_loss_box
