import numpy as np
import math
from itertools import product as product
import paddle.fluid as fluid


class YoloAnchors(object):
    
    def __init__(self, img_size=(640, 1152)):
        self.img_size = img_size  # h, w
        self.steps = [8.0, 16.0, 32.0]
        self.default_anchors = [[[11.6, 12.5], [25.0, 16.5], [15.4, 30.1], [35.2, 31.5]],
                                [[61.1, 24.1], [26.9, 62.1], [64.7, 49.8], [58.2, 129.1]],
                                [[113.6, 74.0], [177.4, 126.5], [258.6, 220.3],[409.3, 351.3]]]  # w, h
        self.feature_map_size = [[math.ceil(self.img_size[0] / step), math.ceil(self.img_size[1] / step)] for step in self.steps]
    
    def get_anchors(self):
        anchors = []
        for k, f_size in enumerate(self.feature_map_size):
            anchor = np.zeros((f_size[0], f_size[1], 4, 4))
            for i, j in product(range(f_size[0]), range(f_size[1])):
                cx = (j + 0.5) * self.steps[k] / self.img_size[1]
                cy = (i + 0.5) * self.steps[k] / self.img_size[0]
                for idx, anchor_size in enumerate(self.default_anchors[k]):
                    w = anchor_size[0] / self.img_size[1]
                    h = anchor_size[1] / self.img_size[0]
                    anchor[i, j, idx, :] = [cx, cy, w, h]
                    # anchors.append([cx, cy, w, h])
            anchors.append(anchor)
        # output = np.reshape(anchors, (-1, 4))
        # output = np.asarray(output, dtype='float32')
        return anchors
