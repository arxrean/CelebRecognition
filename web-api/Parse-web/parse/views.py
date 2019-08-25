from .parse_zt_model.bareness.test import detect
from .parse_zt_model.bareness.Look_Into_Person.pred import test as test2
from .parse_zt_model.bareness.single_human_paring.pred import test as test1
from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import numpy as np
import pandas as pd
import json
import torch
import os
from django.conf import settings
import json

import pdb
import dlib
import sys

import tensorflow as tf
from keras import backend as K
from keras.models import load_model

from .parse_model.test import test as detect_input_image
from .age_gender_model.demo import obtain_age_gender
from .age_gender_model.wide_resnet import WideResNet

# for face detection
detector = dlib.get_frontal_face_detector()

age_model = WideResNet(64, depth=16, k=8)()
age_model.load_weights(
    '/data/parsesite/parse/age_gender_model/pretrained_models/weights.28-3.73.hdf5')
age_model._make_predict_function()

# sys.path.append(os.path.join(settings.BASE_DIR,'parse'))
# zt
model1 = test1()
model2 = test2()


def index(request):
    return HttpResponse('parse man')


@csrf_exempt
def parse_single_man(request):
    # try:
    #     image = request.FILES.get('image')
    #     if image is None:
    #         return HttpResponse('{}'.format('no image received!'))
    #     path = default_storage.save(
    #         'images/{}'.format(image.name), ContentFile(image.read()))

    #     flag,frac = detect_input_image(os.path.join(settings.MEDIA_URL, path))
    #     age_gender_res = obtain_age_gender(
    #         os.path.join(settings.MEDIA_URL, path), age_model, detector)

    #     item = {}
    #     item['bare'] = None
    #     if flag and frac==1:
    #         item['bare']='FULL'
    #     elif flag and frac<1:
    #         item['bare']='PART'
    #     else:
    #         item['bare']='NORMAL'
    #     item['age'] = age_gender_res[0] if age_gender_res else None
    #     item['gender'] = age_gender_res[1] if age_gender_res else None

    #     res = json.dumps(item)

    #     return HttpResponse(res, content_type="application/json")
    # except Exception as e:
    #     return HttpResponse(e)
    return HttpResponse('None')


@csrf_exempt
def parse_single_man_zt(request):
    try:
        image = request.FILES.get('image')
        if image is None:
            return HttpResponse('{}'.format('no image received!'))
        path = default_storage.save(
            'images/{}'.format(image.name), ContentFile(image.read()))

        # pdb.set_trace()
        frac = detect(os.path.join(settings.MEDIA_URL, path), model1, model2)
        age_gender_res = obtain_age_gender(
            os.path.join(settings.MEDIA_URL, path), age_model, detector)

        item = {}
        item['bare'] = None
        if frac == 1:
            item['bare'] = 'FULL'
        elif frac == 0.5:
            item['bare'] = 'PART'
        else:
            item['bare'] = 'NORMAL'
        item['age'] = age_gender_res[0] if age_gender_res else None
        item['gender'] = age_gender_res[1] if age_gender_res else None

        res = json.dumps(item)

        return HttpResponse(res, content_type="application/json")
    except Exception as e:
        return HttpResponse(e)


@csrf_exempt
def error_upload(request):
    try:
        image = request.FILES.get('image')
        if image is None:
            return HttpResponse('{}'.format('no image received!'))
        print('receive image success!')
        path = default_storage.save(
            'err_img/{}'.format(image.name), ContentFile(image.read()))

        return HttpResponse('receive err image success!', content_type="application/json")

    except Exception as e:
        return HttpResponse(e)
