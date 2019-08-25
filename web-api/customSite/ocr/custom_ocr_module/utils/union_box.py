import numpy as np


def diff(box1, box2):
    """
    计算box1,box2之间的距离
    """
    cy1 = (box1[1] + box1[3] + box1[5] + box1[7]) / 4
    cy2 = (box2[1] + box2[3] + box2[5] + box2[7]) / 4
    h1 = max(box1[1], box1[3], box1[5], box1[7]) - min(box1[1], box1[3], box1[5], box1[7])
    h2 = max(box2[1], box2[3], box2[5], box2[7]) - min(box2[1], box2[3], box2[5], box2[7])

    return abs(cy1 - cy2) / max(0.01, min(h1 / 2, h2 / 2))


def combine_group_box(boxes):
    """
    对box进行排序, 并合并box
    """
    boxes = sorted(boxes, key=lambda x: sum([x[0], x[2], x[4], x[6]]))
    n = len(boxes)
    box4 = np.zeros((n, 8))
    for i in range(n):
        x1, y1, x2, y2, x3, y3, x4, y4 = boxes[i]
        box4[i] = [x1, y1, x2, y2, x3, y3, x4, y4]

    x1 = box4[:, 0].min()
    y1 = box4[:, 1].min()
    x2 = box4[:, 2].max()
    y2 = box4[:, 3].min()
    x3 = box4[:, 4].max()
    y3 = box4[:, 5].max()
    x4 = box4[:, 6].min()
    y4 = box4[:, 7].max()
    return np.array([x1, y1, x2, y2, x3, y3, x4, y4])

def sort_group_box(boxes):
    """
    对box进行排序, 并合并box
    """
    boxes = sorted(boxes, key=lambda x: sum([x[0], x[2], x[4], x[6]]))
    return boxes


def union_rbox(text_recs, alpha=0.4):
    """
    按行合并box
    """
    newBox = []
    for line in text_recs:
        if len(newBox) == 0:
            newBox.append([line])
        else:
            check = False
            for box in newBox[-1]:
                if diff(line, box) > alpha:
                    check = True

            if not check:
                newBox[-1].append(line)
            else:
                newBox.append([line])
    newBox = [combine_group_box(bx) for bx in newBox]
    return newBox


def sort_rbox(text_recs, alpha=0.4):
    """
    按行合并box
    """

    newBox = []
    for line in text_recs:
        if len(newBox) == 0:
            newBox.append([line])
        else:
            check = False
            for box in newBox[-1]:
                if diff(line, box) > alpha:
                    check = True

            if not check:
                newBox[-1].append(line)
            else:
                newBox.append([line])
    nums = []
    sortBox = []
    for bx in newBox:
        sortBox.extend(sort_group_box(bx))
        nums.append(len(sort_group_box(bx)))
    return sortBox, nums
