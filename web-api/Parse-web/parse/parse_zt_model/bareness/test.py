import pandas as pd
import cv2
import numpy as np
import sys
import os


def m2(image, model1, model2, threshold1=0.6, threshold2=0.45):
    """
    model2把图片分割为背景0、裸露+手臂255、不关注127（腿、脸、头发）、衣物, 计算裸露/衣物的比例，与threshold相比
    若小于threshold，利用model1计算背景+裸露与model2计算的背景+裸露作比较,取并集
    :param image:
    :param threshold:
    :return:   1/0  1 represents expose, 0 represents unexpose  0.5 part unexpose
    """

    pred2 = model2.pred(image)
    pred2[np.where(pred2 == 10)] = 255          # expose
    pred2[np.where(pred2 == 15)] = 255          # arms
    pred2[np.where(pred2 == 17)] = 127          # legs
    pred2[np.where(pred2 == 13)] = 127          # face
    pred2[np.where(pred2 == 2)] = 127           # hair
    bg = len(np.where(pred2 == 0)[0])
    expose = len(np.where(pred2 == 255)[0])
    clothes = 320*320 - expose - bg - len(np.where(pred2 == 127)[0])
    if clothes != 0:
        # print(expose / clothes)
        if expose / clothes >= threshold1:
            flags = 1
        else:
            pred1 = model1.pred(image)
            pred1 = cv2.resize(pred1, (320, 320))
            # cv2.imwrite("pred1.jpg", pred1)
            bg1 = np.where(pred1 == 0)       # model1 bg & expose
            nums = len(bg1[0])
            for i in range(nums):
                # pixel that model1 is 0
                mask = pred2[bg1[0][i], bg1[1][i]]
                # model1 predict 0 however model2  predict clothes
                if mask != 0 and mask != 255 and mask != 127:
                    pred2[bg1[0][i], bg1[1][i]] = 255
            bg = len(np.where(pred2 == 0)[0])
            expose = len(np.where(pred2 == 255)[0])
            clothes = 320 * 320 - expose - bg - len(np.where(pred2 == 127)[0])
            if clothes != 0:
                # print(expose / clothes)
                if expose / clothes >= threshold1:
                    flags = 1
                elif expose / clothes >= threshold2:
                    flags = 0.5
                else:
                    flags = 0
            else:
                flags = 1

    else:
        flags = 1

     #
    #cv2.imwrite("pred2.jpg", pred2)
    return flags


def detect(img_path, test1, test2):
    image = cv2.imread(img_path)
    return m2(image, test1, test2)  # 0 NORMAL 0.5 PART 1 FULL

# path = "./single_human_paring/img/woman/expose"
# #path = "./res"
# imgs = os.listdir(path)
# for im in imgs:
#     imname = os.path.join(path, im)
#     image = cv2.imread(imname)
#     flags = m2(image)
#     print(imname, flags)
