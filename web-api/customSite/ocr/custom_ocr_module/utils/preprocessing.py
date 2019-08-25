import cv2
import numpy as np
import math

from .union_box import union_rbox


def getHProjection(binary):
    hProjection = np.zeros(binary.shape, np.uint8)
    (h, w) = binary.shape
    h_ = [0] * h
    for y in range(h):
        for x in range(w):
            if binary[y, x] == 0:
                h_[y] += 1
    for y in range(h):
        for x in range(h_[y]):
            hProjection[y, x] = 255

    return h_

def getVProjection(binary):
    vProjection = np.zeros(binary.shape, np.uint8)
    h, w = binary.shape
    w_ = [0]*w
    for x in range(w):
        for y in range(h):
            if binary[y, x] == 0:
                w_[x] += 1

    for x in range(w):
        for y in range(w_[x]):
            vProjection[y, x] = 255

    return w_



def rotateImg(image):
    """
    use fft to rotate image
    :param image:   Input image
    :return:        Rotated image
    """
    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # IMREAD_COLOR, IMREAD_UNCHANGED
    h, w = im_gray.shape

    binary = cv2.threshold(im_gray, 30, 255, cv2.THRESH_BINARY)
    binary = 255 - binary[1]

    new_h = cv2.getOptimalDFTSize(h)
    new_w = cv2.getOptimalDFTSize(w)
    new_im = np.zeros((new_h, new_w))
    new_im[0:h, 0:w] = binary

    dft = cv2.dft(np.float32(new_im), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    mag = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    magnitude_spectrum = 20 * np.log(mag)

    ms_min, ms_max = magnitude_spectrum.min(), magnitude_spectrum.max()
    spect_norm = (magnitude_spectrum - ms_min) / (ms_max - ms_min)

    spect = np.clip(255 * spect_norm, 0, 255)
    spect = np.uint8(spect)
    spect_bin = cv2.threshold(spect, 180, 255, cv2.THRESH_BINARY)[1]

    res = []
    split = 2
    while len(res) == 0:
        lines = cv2.HoughLines(spect_bin, 1, np.pi / 360, int(new_h / split))
        if lines is not None:
            res = [line for line in lines if (line[0][1] < np.pi / 30 or line[0][1] > np.pi * 29 / 30)]
        split += 1
    angle = []
    for item in res:
        if item[0][1] > np.pi / 2:
            item[0][1] -= np.pi
        angle.append(item[0][1])

    angle = np.mean(angle)
    real_angle = math.atan(new_h * math.tan(angle) / new_w) / np.pi * 180
    center = (w // 2, h // 2)
    Mat = cv2.getRotationMatrix2D(center, real_angle, 1.0)
    rotated = cv2.warpAffine(image, Mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated




def detect_rows_cv(image):
    """

    :param image: rotated images
    :return: list  contains line images
    """
    lineImgs = []
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 白底黑字
    img = cv2.medianBlur(img, 3)
    _, thresh = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)
    h_ = getHProjection(thresh)
    height = len(h_)
    begin = False
    division_start = []
    division_end = []
    for y in range(0, height):
        if (h_[y] > 1):
            begin = True
            if len(division_end) == len(division_start):
                division_start.append(y)
        elif (begin):
            division_end.append(y)
            begin = False
        if y == height-1 and len(division_end) != len(division_start):
            division_end.append(height)

    parts = len(division_start)
    for i in range(parts):

        start = max(division_start[i] -1, 0)
        end = min(division_end[i] + 1, height)
        partImg = image[start: end]
        lineImgs.append(partImg)
    return lineImgs


def ctpnRefineRows(lineImgs, model):
    """
    refine the lineimgs
    :param lineImgs: list contain cv detected line imgs
    :return:
    """
    lineImgsRefine = []
    startpoints = []
    for lineimg in lineImgs:
        text_recs_resize, im = model.predict(lineimg)
        h,w = im.shape[:2]
        if len(text_recs_resize) > 0:
            boxes = union_rbox(text_recs_resize)
            if len(boxes) == 1:
                lineImgsRefine.append(im)
                box = boxes[0]
                startpoints.append(int(min(box[0::2])) / w)
            else:
                for box in boxes:
                    start = int(max(0, min(box[1::2])))
                    end = int(max(box[1::2]))
                    lineImgsRefine.append(im[start:end+1, :, :])
                    startpoints.append(int(min(box[0::2])) / w)

    return lineImgsRefine, startpoints


def splitBlanks(image):
    """
    split line imgs according to the blanks
    :param image: line imgs
    :return: list, contains words-images
    """
    res = []
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 白底黑字
    _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)

    v_ = getVProjection(thresh)
    height, width = image.shape[:2]
    begin = False
    division_start = []
    division_end = []
    for y in range(0, width):
        if (v_[y] > 0):
            begin = True
            if len(division_end) == len(division_start):
                division_start.append(y)
        elif (begin):
            division_end.append(y)
            begin = False
        if y == width-1 and len(division_end) != len(division_start):
            division_end.append(width)

    parts = len(division_start)
    blanks = []
    for i in range(parts - 1):
        blanks.append(division_start[i+1] - division_end[i])
    blanks = np.array(blanks)
    blanks = blanks / width
    idx = np.where(blanks >= 0.01)[0]
    numbreaks = len(idx) + 1
    for i in range(numbreaks):
        if i == 0:
            start = division_start[0] * 9 // 10
        else:
            start = division_start[idx[i-1] + 1] - (division_start[idx[i-1] + 1] - division_end[idx[i-1]]) //5
        if i !=numbreaks - 1:
            end = division_end[idx[i]] + (division_start[idx[i] + 1] - division_end[idx[i]]) // 5
        else:
            end = (width + division_end[-1]) // 2
        start = max(start, 0)
        end = min(end, width)

        res.append(image[:, start: end + 1])

    return res
