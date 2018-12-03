"""
Bounding Box util functions
"""
import numpy as np
from numba import jit


@jit
def calc_iou(box1, box2):
    """Calculate Intersection over Union between 2 boxes. 
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    Returns:
        iou: float
    """
    x1 = np.maximum(box1[0], box2[0])
    y1 = np.maximum(box1[1], box2[1])
    x2 = np.minimum(box1[2], box2[2])
    y2 = np.minimum(box1[3], box2[3])

    if x1 <= x2 or y1 <= y2:
        return 0.

    intersect = (x2 - x1) * (y2 - y1)
    box1_size = (box1[2]-box1[0])*(box1[3]-box1[0])
    box2_size = (box2[2]-box2[0])*(box2[3]-box2[0])
    iou = intersect / (box1_size + box2_size + intersect)
    return iou


def scale_box(box, image):
    x1, y1, x2, y2 = box
    height, width, _ = image.shape
    scaled = [x1 * width, y1 * height, x2 * width, y2 * height]
    scaled = [int(s) for s in scaled]
    return scaled


def clip_box(box):
    x1, y1, x2, y2 = box
    x1 = np.maximum(0, x1)
    y1 = np.maximum(0, y1)
    x2 = np.minimum(1, x2)
    y2 = np.minimum(1, y2)
    return np.asarray([x1, y1, x2, y2])


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return np.asarray([x1, y1, x2 - x1, y2 - y1])


def xywh_to_xyxy(box):
    x, y, w, h = box
    return np.asarray([x, y, x + w, y + h])


def normalize_box(box, image_shape):
    (H, W) = image_shape
    return np.array(box) / np.array([W, H, W, H]).astype(np.float)


