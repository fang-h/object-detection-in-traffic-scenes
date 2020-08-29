import paddle.fluid as fluid
import numpy as np


def sigmoid(inx):
    return 1.0/(1+np.exp(-inx))


def point_form(boxes):
    """convert [cx, cy, w, h] to [xmin, ymin, xmax, ymax]
        Args:
            boxes:[:, 4]
    """
    return fluid.layers.concat([boxes[:, 0:2] - boxes[:, 2:] / 2,
                               boxes[:, 0:2] + boxes[:, 2:] / 2], axis=1)
  
 
def numpy_point_form(boxes):
    """convert [cx, cy, w, h] to [xmin, ymin, xmax, ymax]
        Args:
            boxes:[:, 4]
    """
    return np.concatenate([boxes[:, 0:2] - boxes[:, 2:] / 2,
                               boxes[:, 0:2] + boxes[:, 2:] / 2], axis=1) 
                               

def center_form(boxes):
    """convert [xmin, ymin, xmax, ymax] to [cx, cy, w, h]
        Args:
            boxes:[:, 4]
    """
    return fluid.layers.concat([(boxes[:, 0:2] + boxes[:, 2:]) / 2,
                               boxes[: 2:] - boxes[:, 0:2]], dim=1)
   
                               
def numpy_center_form(boxes):
    """convert [xmin, ymin, xmax, ymax] to [cx, cy, w, h]
        Args:
            boxes:[:, 4]
    """
    return np.concatenate([(boxes[:, 0:2] + boxes[:, 2:]) / 2,
                               boxes[: 2:] - boxes[:, 0:2]], dim=1)


def intersect(boxes_1, boxes_2):
    """
    compute the intersect of box1 in boxes_1 and box2 in boxes_2
    :param box1:[num, 4], the second dim is [xmin, ymin, xmax, ymax]
    :param box2:[num, 4], the second dim is [xmin, ymin, xmax, ymax]
    :return:[num1, num2]
    """
    # num1 = boxes_1.shape[0]
    # num2 = boxes_2.shape[0]

    # max_xy = fluid.layers.elementwise_min(fluid.layers.expand(fluid.layers.unsqueeze(boxes_1[:, 2:], axes=1), (1, num2, 1)),
    #                                       fluid.layers.expand(fluid.layers.unsqueeze(boxes_2[:, 2:], axes=0), (num1, 1, 1)))
    # min_xy = fluid.layers.elementwise_max(fluid.layers.expand(fluid.layers.unsqueeze(boxes_1[:, :2], axes=1), (1, num2, 1)),
    #                                       fluid.layers.expand(fluid.layers.unsqueeze(boxes_2[:, :2], axes=0), (num1, 1, 1)))
    max_xy = fluid.layers.elementwise_min(boxes_1[:, 2:], boxes_2[:, 2:])
    min_xy = fluid.layers.elementwise_max(boxes_1[:, :2], boxes_2[:, :2])
    inter = fluid.layers.clip((max_xy - min_xy), min=0.0, max=1.0)
    inter_area = inter[:, 0] * inter[:, 1]
    return inter_area              
    

def numpy_intersect(boxes_1, boxes_2):
    """
    compute the intersect of box1 in boxes_1 and box2 in boxes_2
    :param box1:[num1, 4], the second dim is [xmin, ymin, xmax, ymax]
    :param box2:[num2, 4], the second dim is [xmin, ymin, xmax, ymax]
    :return:[num1, num2]
    """
    num1 = boxes_1.shape[0]
    num2 = boxes_2.shape[0]

    max_xy = np.minimum(np.tile(np.expand_dims(boxes_1[:, 2:], axis=1), (1, num2, 1)),
                        np.tile(np.expand_dims(boxes_2[:, 2:], axis=0), (num1, 1, 1)))
    min_xy = np.maximum(np.tile(np.expand_dims(boxes_1[:, :2], axis=1), (1, num2, 1)),
                        np.tile(np.expand_dims(boxes_2[:, :2], axis=0), (num1, 1, 1)))
    inter = np.clip((max_xy - min_xy), a_min=0.0, a_max=1.0)
    inter_area = inter[:, :, 0] * inter[:, :, 1]
    return inter_area


def compute_iou(boxes_1, boxes_2):
    """
    :param boxes_1: [num, 4], the second dim is [xmin, ymin, xmax, ymax]
    :param boxes_2: [num, 4], the second dim is [xmin, ymin, xmax, ymax]
    :return:
    """
    inter_area = intersect(boxes_1, boxes_2)
    # print(inter_area)
    # area_1 = fluid.layers.expand_as(fluid.layers.unsqueeze((boxes_1[:, 2] - boxes_1[:, 0]) * (boxes_1[:, 3] - boxes_1[:, 1]), axes=1), inter_area)
    # area_2 = fluid.layers.expand_as(fluid.layers.unsqueeze((boxes_2[:, 2] - boxes_2[:, 0]) * (boxes_2[:, 3] - boxes_2[:, 1]), axes=0), inter_area)
    area_1 = (boxes_1[:, 2] - boxes_1[:, 0]) * (boxes_1[:, 3] - boxes_1[:, 1])
    area_2 = (boxes_2[:, 2] - boxes_2[:, 0]) * (boxes_2[:, 3] - boxes_2[:, 1])    
    iou = inter_area / (area_1 + area_2 - inter_area)
    iou = fluid.layers.clip(iou, min=1e-10, max=1.0)
    return iou


def numpy_compute_iou(boxes_1, boxes_2):
    """
    :param boxes_1: [num1, 4], the second dim is [xmin, ymin, xmax, ymax]
    :param boxes_2: [num2, 4], the second dim is [xmin, ymin, xmax, ymax]
    :return:
    """
    inter_area = numpy_intersect(boxes_1, boxes_2)
    shape = inter_area.shape
    area_1 = np.tile(np.expand_dims((boxes_1[:, 2] - boxes_1[:, 0]) * (boxes_1[:, 3] - boxes_1[:, 1]), axis=1), (1, shape[-1]))
    area_2 = np.tile(np.expand_dims((boxes_2[:, 2] - boxes_2[:, 0]) * (boxes_2[:, 3] - boxes_2[:, 1]), axis=0), (shape[0], 1))
    iou = inter_area / (area_1 + area_2 - inter_area)
    return iou
    

def matrix_iof(a, b):
    """a是未裁剪图片中的所有框，[num_object, 4]
    b是裁剪图片的范围，[[xmin,ymin,xmax,ymax]]"""
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])
    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)  # 保留裁剪后的有效框
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    return area_i / np.maximum(area_a[:, np.newaxis], 1)


def numpy_encode(matched, default_anchors, variances=[0.1, 0.2]):
    delta_xy = (matched[:, :2] - default_anchors[:, :2]) / (variances[0] * default_anchors[:, 2:])
    delta_wh = np.log(matched[:, 2:] / default_anchors[:, 2:]) / variances[1]
    return np.concatenate([delta_xy, delta_wh], axis=-1)


def encode(matched, default_anchors, variances=[0.1, 0.2]):
    delta_xy = (matched[:, :2] - default_anchors[:, :2]) / (variances[0] * default_anchors[:, 2:])
    delta_wh = fluid.layers.log(matched[:, 2:] / default_anchors[:, 2:]) / variances[1]
    return fluid.layers.concat([delta_xy, delta_wh], axis=-1)


def yolo_decode(loc_p, matched_anchors, matched_normal):
    box_x = (fluid.layers.sigmoid(loc_p[:, 0:1]) - 0.5) * matched_normal[:, 0:1] + matched_anchors[:, 0:1]
    box_y = (fluid.layers.sigmoid(loc_p[:, 1:2]) - 0.5) * matched_normal[:, 1:2] + matched_anchors[:, 1:2]
    box_xy = fluid.layers.concat((box_x, box_y), axis=-1)
    box_wh = fluid.layers.exp(loc_p[:, 2:]) * matched_anchors[:, 2:]
    boxes = fluid.layers.concat([box_xy, box_wh], axis=1)
    return boxes


def numpy_yolo_decode(loc_p, pos_anchors, pos_normal):
    box_x = (sigmoid(loc_p[:, 0:1]) - 0.5) * pos_normal[:, 0:1] + pos_anchors[:, 0:1]
    box_y = (sigmoid(loc_p[:, 1:2]) - 0.5) * pos_normal[:, 1:2] + pos_anchors[:, 1:2]
    box_xy = np.concatenate((box_x, box_y), axis=-1)
    box_wh = np.exp(loc_p[:, 2:]) * pos_anchors[:, 2:]
    boxes = np.concatenate([box_xy, box_wh], axis=1)
    return numpy_point_form(boxes)


def numpy_decode(loc_p, default_boxes,variances=[0.1, 0.2]):
    box_xy = loc_p[:, :2] * variances[0] * default_boxes[:, 2:] + default_boxes[:, :2]
    # box_xy = loc_p[:, :2]  * default_boxes[:, 2:] + default_boxes[:, :2]
    box_wh = np.exp(loc_p[:, 2:] * variances[1]) * default_boxes[:, 2:]
    # box_wh = np.exp(loc_p[:, 2:]) * default_boxes[:, 2:]
    boxes = np.concatenate([box_xy, box_wh], axis=1)
    boxes = numpy_point_form(boxes)
    return boxes
    

def nms(boxes, scores, overlap=0.3):
    keep = []
    if len(boxes) == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idx = scores.argsort()[::-1]
    while len(idx) > 0:
        i = idx[0]
        keep.append(i)
        if len(idx) == 1:
            break
        xx1 = np.maximum(x1[i], x1[idx[1:]])
        yy1 = np.maximum(y1[i], y1[idx[1:]])
        xx2 = np.minimum(x2[i], x2[idx[1:]])
        yy2 = np.minimum(y2[i], y2[idx[1:]])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        interface = w * h
        iou = interface / (area[i] + area[idx[1:]] - interface)
        index = np.where(iou <= overlap)[0]
        idx = idx[index + 1]
    return keep


# def numpy_yolo_encode(matched, default_anchors, variances=[0.1, 0.2]):
#     """
#         从anchor的[cx, cy, Pw, Ph]和gt_box的[bx, by, bw, bh]到模型输出[tx, ty, tw, th]
#         Args：
#             matched：shape[number of default_anchors, 4],对应着[bx, by, bw, bh]
#             default_anchors:shape[number of default_anchors], 对应着[cx, cy, Pw, Ph]
#         return:
#             shape[number of default_anchors, 4], 对应着[tx, ty, tw, th]
#     """
#     delta_xy = np.log((matched[:, 0:2] - default_anchors[:, 0:2] + 0.5) / (default_anchors[:, 0:2] - matched[:, 0:2] + 0.5))
#     delta_xy /= variances[0]
#     delta_wh = np.log(matched[:, 2:] / default_anchors[:, 2:]) / variances[1]
#     return np.concatenate([delta_xy, delta_wh], axis=-1)