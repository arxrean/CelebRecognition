import os
import cv2
from .utils import preprocessing

import pandas as pd
import pytesseract
import numpy as np
import Levenshtein



def run(images, ctpn_model):
    lineImgs = []
    for image in images:
        img = cv2.imread(image)
        rotate = preprocessing.rotateImg(img)
        # """
        # detect the rows with opencv
        # """
        lineImgs.extend(preprocessing.detect_rows_cv(rotate))

    """
    refine the rows with ctpn
    """
    lineRefineImgs, startpoints = preprocessing.ctpnRefineRows(lineImgs, ctpn_model)
    nums = len(lineRefineImgs)
    structionInvoice = []
    invoice = []
    for i in range(nums):
        lineimg = lineRefineImgs[i]
        startpoint = startpoints[i]
        if startpoint <= 0.12:
            if invoice != []:
                structionInvoice.append(invoice)
            invoice = []
        invoice.append(lineimg)
    structionInvoice.append(invoice)

    contents = []
    lineOneBase = [0.0981053, 0.1672427, 0.38371045, 0.41128038, 0.49240859, 0.55988408, 0.66523756, 0.68642909,
                   0.81079926]
    NetWeightBase = [0.1670214, 0.20022902]
    NetWeightBase_right = [0.19051984, 0.25435351, 0.56661325, 0.60052368]
    CtyBase = [0.16666667, 0.23407407, 0.25731482, 0.40939815]

    title = ["Material", "Bill.qty", "SU", "Net Value", "Curr", "Net Weight", "Origin Country"]
    config = ("--psm 7")
    for invoice in structionInvoice:
        d = [None, None, None, None, None, None, None]
        lines = len(invoice)
        netdis = []
        ctydis = []
        texts = []
        lefts = []
        rights = []
        for i in range(lines):
            lineimg = invoice[i]
            h, w, c = lineimg.shape
            pad = np.ones([h + 10, w, c], dtype=np.uint8) * 255
            pad[5:h + 5, :, :] = lineimg
            data = pytesseract.image_to_data(pad, config=config)
            data = data.split("\n")
            left = []
            text = []
            right = []
            data = data[1:]
            for line_ in data:
                line = line_.split("\t")
                if len(line) == 12:
                    if line[-1] != "":
                        left.append(int(line[-6]) / w)
                        text.append(line[-1])
                        right.append((int(line[-6]) + int(line[-4])) / w)
            texts.append(text)
            left = np.array(left)
            lefts.append(left)
            right = np.array(right)
            rights.append(right)
            if i == 0:
                if len(text) >= 9:
                    dis = abs(left - lineOneBase[1])
                    idx = np.argmin(dis)
                    d[0] = text[idx]

                    dis = abs(left - lineOneBase[2])
                    idx = np.argmin(dis)
                    d[1] = text[idx]

                    dis = abs(left - lineOneBase[3])
                    idx = np.argmin(dis)
                    d[2] = text[idx]

                    dis = abs(left - lineOneBase[8])
                    idx = np.argmin(dis)
                    val = text[idx]
                    val1 = val[:-3]
                    val2 = val[-2:]
                    s = ""
                    for item in val1:
                        if item in "0123456789":
                            s += item
                    s += "."
                    s += val2
                    d[3] = s

                    dis = abs(left - lineOneBase[5])
                    idx = np.argmin(dis)
                    d[4] = text[idx].upper()
            else:
                str1 = ""
                dis = abs(left - NetWeightBase[0])
                idx = np.argmin(dis)
                str1 += text[idx]
                dis = abs(left - NetWeightBase[1])
                idx = np.argmin(dis)
                str1 += text[idx]

                netdis.append(Levenshtein.distance(str1, "Netweight:"))

                str2 = ""
                dis = abs(left - CtyBase[0])
                idx = np.argmin(dis)
                str2 += text[idx]
                dis = abs(left - CtyBase[1])
                idx = np.argmin(dis)
                str2 += text[idx]
                dis = abs(left - CtyBase[2])
                idx = np.argmin(dis)
                str2 += text[idx]

                ctydis.append(Levenshtein.distance(str2, "Countryoforigin:"))

        netidx = np.argmin(netdis)
        if netdis[netidx] <= 3:
            text = texts[netidx + 1]
            r = rights[netidx + 1]
            dis = abs(r - NetWeightBase_right[2])
            idx = np.argmin(dis)
            val = text[idx]
            val1 = val[:-4]
            val2 = val[-3:]
            s = ""
            for item in val1:
                if item in "0123456789":
                    s += item
            s += "."
            s += val2
            try:
                s = float(s)
                dis = abs(r - NetWeightBase_right[3])
                idx = np.argmin(dis)
                unit = text[idx]
                if unit == "KG":
                    d[5] = s
                elif unit == "G":
                    d[5] = s / 1000
                else:
                    d[5] = s
            except:
                continue

        ctyidx = np.argmin(ctydis)
        if ctydis[ctyidx] <= 4:
            text = texts[ctyidx + 1]
            l = lefts[ctyidx + 1]
            dis = abs(l - CtyBase[3])
            idx = np.argmin(dis)
            d[6] = text[idx].upper()

        contents.append(d)

    df = pd.DataFrame(contents)

    df.columns = title

    return df


