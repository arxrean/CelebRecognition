import cv2
import numpy as np
from collections import Counter
import os
def equalizeHistOp(image):
    """
    
    :param image:   3d channel image 
    :return: 
    """
    (b, g, r) = cv2.split(image)
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    result = cv2.merge((bH, gH, rH))
    return result

def gammaTrans(image, gamma=1.0):
    # 具体做法先归一化到1，然后gamma作为指数值求出新的像素值再还原
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    # 实现映射用的是Opencv的查表函数
    return cv2.LUT(image, gamma_table)

def  brightCheck(image):
    """
    
    :param image: 3-d image   bgr
    :return: k, da
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.reshape(gray, (-1))
    da = np.mean(gray) - 127.5
    D = abs(da)
    recounted = Counter(gray)
    Ma = 0
    for i in range(256):
        Ma += abs(i-127.5-da) * recounted[i]
    Ma = Ma / len(gray)
    M = abs(Ma)
    k = D/M
    return k, da

def color_test(img):
    b, g, r = cv2.split(img)
    m, n, z = img.shape
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(img_lab)
    d_a, d_b, M_a, M_b = 0, 0, 0, 0
    for i in range(m):
        for j in range(n):
            d_a = d_a + a[i][j]
            d_b = d_b + b[i][j]
    d_a, d_b = (d_a / (m * n)) - 128, (d_b / (n * m)) - 128
    D = np.sqrt((np.square(d_a) + np.square(d_b)))

    for i in range(m):
        for j in range(n):
            M_a = np.abs(a[i][j] - d_a - 128) + M_a
            M_b = np.abs(b[i][j] - d_b - 128) + M_b


    M_a, M_b = M_a / (m * n), M_b / (m * n)
    M = np.sqrt((np.square(M_a) + np.square(M_b)))
    k = D / M
    print('偏色值:%f' % k)
    return k



img = cv2.imread("res/NXD0000008363.jpg")
res = equalizeHistOp(img)
gamma = gammaTrans(img, 0.5)
# cv2.imshow("img", img)
# cv2.waitKey()
# cv2.imshow("eq", res)
# cv2.waitKey()
# cv2.imshow("gamma", gamma)
# cv2.waitKey()

path = "./res"
imnames = os.listdir(path)
for imname in imnames:
    im = cv2.imread(os.path.join(path, imname))
    k,da = brightCheck(im)
    print(k, k*da)
    h = color_test(im)
    cv2.imshow("1111", im)
    cv2.waitKey()


