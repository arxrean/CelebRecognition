import numpy as np


# 生成基础anchor box
def generate_basic_anchors(sizes, base_size=16):
    base_anchor = np.array([0, 0, base_size - 1, base_size - 1], np.int32)
    anchors = np.zeros((len(sizes), 4), np.int32)         # (10, 4)
    index = 0
    for h, w in sizes:
        anchors[index] = scale_anchor(base_anchor, h, w)
        index += 1
    return anchors


# 根据base anchor和设定的anchor的高度和宽度进行设定的anchor生成
def scale_anchor(anchor, h, w):
    x_ctr = (anchor[0] + anchor[2]) * 0.5
    y_ctr = (anchor[1] + anchor[3]) * 0.5
    scaled_anchor = anchor.copy()
    scaled_anchor[0] = x_ctr - w / 2  # xmin
    scaled_anchor[2] = x_ctr + w / 2  # xmax
    scaled_anchor[1] = y_ctr - h / 2  # ymin
    scaled_anchor[3] = y_ctr + h / 2  # ymax
    return scaled_anchor


# 生成anchor box
# 此处使用的是宽度固定，高度不同的anchor设置
def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2 ** np.arange(3, 6)):
    heights = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
    widths = [16]
    sizes = []
    for h in heights:
        for w in widths:
            sizes.append((h, w))                  # [(11, 16), (16, 16), (23, 16), (33, 16), ...]
    return generate_basic_anchors(sizes)


if __name__ == '__main__':
    import time

    t = time.time()
    a = generate_anchors()
    print(time.time() - t)
    print(a)
    from IPython import embed;

    embed()
