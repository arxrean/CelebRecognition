import torch
import os
import numpy as np
from torchvision import transforms
import scipy.spatial as sci


def detect_star_id(img_patch, model, repo):
    res = []
    feat_id = repo[0]
    feat = repo[1]

    trans = transforms.Compose([
        transforms.ToTensor()
    ])

    for img in img_patch:
        _, fc = model(trans(img).unsqueeze(0))
        fc = fc.squeeze().detach().numpy()
        # sims = np.array([sci.distance.cosine(fc, x) for x in feat])
        # res.append([feat_id[np.argmin(sims)], 1-np.min(sims)])

        up = feat.dot(np.transpose(fc))
        down_1 = np.sum((feat**2), axis=1)**0.5
        down_2 = np.sum((fc**2))**0.5
        sims = up/(down_1*down_2)
        res.append([feat_id[np.argmax(sims)], np.max(sims)])

    return res
