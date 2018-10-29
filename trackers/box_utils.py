"""
Bounding Box util functions
"""

def scale_box(box, image):
    x1, y1, x2, y2 = box
    height, width, _ = image.shape
    scaled = [x1 * width, y1 * height, x2 * width, y2 * height]
    scaled = [int(s) for s in scaled]
    return scaled
    
def clip_box(box):
    x1, y1, x2, y2 = box
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(1, x2)
    y2 = min(1, y2)
    return [x1, y1, x2, y2]

def get_box_size(box):
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    return w * h 

def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]

def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]

def calc_iou(box1, box2):
    """Calculate Intersection over Union between 2 boxes. 
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    Returns:
        iou: float
    """
    intersect_box = get_intersect(box1, box2)
    intersect = get_box_size(intersect_box)
    union = get_box_size(box1) + get_box_size(box2) - intersect
    iou = intersect / float(union)
    return iou

def get_intersect(box1, box2):
    """Get intersection between 2 boxes. 
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    Returns:
        box: [x1, y1, x2, y2]
    """
    ax1, ay1, ax2, ay2 = box1
    bx1, by1, bx2, by2 = box2
    
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    if (x1 > x2) or (y1 > y2):
        return [0, 0, 0, 0] # no intersect
    
    return [x1, y1, x2, y2]