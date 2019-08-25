import cv2 as cv
import numpy as np
from django.conf import settings
import os

from .config import num_classes
from .model import build_model
from .data_generator import to_bgr


class test():
    def __init__(self):
        model_weights_path = os.path.join(
            settings.BASE_DIR, 'parse/parse_zt_model/bareness/Look_Into_Person/models/model.11-0.8409.hdf5')
        self.model = build_model()
        self.model.load_weights(model_weights_path)
        self.model._make_predict_function()
        self.img_rows, self.img_cols = 320, 320

    def pred(self, image):
        image = cv.resize(
            image, (self.img_rows, self.img_cols), cv.INTER_CUBIC)
        x_test = np.array([image], dtype=np.float32)
        x_test = x_test / 255.
        out = self.model.predict(x_test)
        out = np.reshape(out, (self.img_rows, self.img_cols, num_classes))
        out = np.argmax(out, axis=2)
        #out = to_bgr(out)
        return out
