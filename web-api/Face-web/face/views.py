from django.shortcuts import render
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
import numpy as np
import pandas as pd
import json
import torch
import os
import pdb

from .models import StarInfo
from .model.extract_face.mtcnn_xw.main import detect_by_mtcnn
from .model.light_cnn.main import detect_star_id
from .model.light_cnn.model import LightCNN_29Layers_v3
from .model.extract_face.mtcnn_xw.mtcnn import Mtcnn

# load mtcnn
configs = {}
configs['config'] = json.load(
    open(os.path.join(settings.BASE_DIR, 'face/model/extract_face/mtcnn_xw/global_config.json'), 'r'))['fd_config']
configs['para'] = json.load(open(
    os.path.join(settings.BASE_DIR, 'face/model/extract_face/mtcnn_xw/global_para.json'), 'r'))['fd_para']
mtcnn_model = Mtcnn(config=configs['config'], para=configs['para'])

# load base model
base_model = LightCNN_29Layers_v3(num_classes=8642)
pre_dict = torch.load(os.path.join(
    settings.BASE_DIR, 'face/model/weights/lightCNN_62_checkpoint.pth.tar'), map_location='cpu')['state_dict']
pre_dict = {k[7:]: v for k, v in pre_dict.items(
) if 'fc2' not in k and 'arc' not in k}
model_dict = base_model.state_dict()
model_dict.update(pre_dict)
base_model.load_state_dict(model_dict)

# load mean feat
mean_feat_id = []
mean_feat = []
for item in os.listdir(os.path.join(settings.BASE_DIR, 'face/register/train_avg/baseline_valid')):
    mean_feat_id.append(item[:-4])
    mean_feat.append(np.load(os.path.join(
        settings.BASE_DIR, 'face/register/train_avg/baseline_valid', item)))
mean_feat_id = np.asarray(mean_feat_id)
mean_feat = np.concatenate(mean_feat, axis=0)


@csrf_exempt
def index(request):
    return HttpResponse("Please input parameters for face recognition")


@csrf_exempt
def model(request, model):
    return HttpResponse('You are using model:{}'.format(model))


@csrf_exempt
def model_repo(request, model, repo):
    return HttpResponse('You are using model:{} repo:{}'.format(model, repo))


@csrf_exempt
def recognition(request):
    try:
        image = request.FILES.get('image')
        if image is None:
            return HttpResponse('{}'.format('no image received!'))
        path = default_storage.save(
            'images/{}'.format(image.name), ContentFile(image.read()))
        detect_patch_list = detect_by_mtcnn(
            mtcnn_model, configs, [path])
        if isinstance(detect_patch_list, str):
            return HttpResponse('{}'.format(detect_patch_list))

        paras = request.build_absolute_uri().split('/')
        id_list = detect_star_id(
            img_patch=detect_patch_list, model=base_model, repo=[mean_feat_id, mean_feat])

        res = []

        for idx, id in enumerate(id_list):
            entry = StarInfo.objects.get(star_id=int(id[0]))
            star_info = {}
            star_info['id'] = entry.star_id
            star_info['path'] = os.path.join(
                settings.MEDIA_URL,'key/', str(entry.star_id)+'.jpg')
            star_info['name'] = entry.name
            star_info['desc'] = entry.desc
            star_info['works'] = entry.reels
            if entry.sex == 1:
                star_info['gender'] = 'male'
            else:
                star_info['gender'] = 'female'
            star_info['similarity'] = str('{:.4f}'.format(id[1]))
            res.append(star_info)

        res = json.dumps(res)

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
            'err/{}'.format(image.name), ContentFile(image.read()))

        return HttpResponse('receive err image success!', content_type="application/json")

    except Exception as e:
        return HttpResponse(e)
