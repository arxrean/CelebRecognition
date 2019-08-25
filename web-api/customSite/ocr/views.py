from .custom_ocr_module.src.detection.ctpn.text_detect import ctpn
from .custom_ocr_module.struction import run

from django.shortcuts import render
from django.shortcuts import render_to_response
from django.http import HttpResponse
from django.http import JsonResponse
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
import sys

from django.shortcuts import render_to_response
from django.conf import settings

import tensorflow as tf
from keras import backend as K
from keras.models import load_model

ctpn_model = ctpn(checkpoint_path=os.path.join(
    settings.BASE_DIR, 'ocr/custom_ocr_module/src/detection/ctpn/checkpoints_mlt/'))


def index(request):
    return render_to_response('index.html')


@csrf_exempt
def ocr_main(request):
    files = request.FILES.getlist('image')

    path_list = []
    for i, image in enumerate(files):
        path = default_storage.save(
            'images/{}'.format(image.name), ContentFile(image.read()))

        path_list.append(path)

    df = run(path_list, ctpn_model)

    return HttpResponse(df.to_json(orient='index'), content_type='application/json')
