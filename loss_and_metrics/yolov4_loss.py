import paddle.fluid as fluid
import paddle.fluid.layers as layers
import numpy as np
from utils.box_utils import numpy_compute_iou, compute_iou, numpy_point_form, point_form,  yolo_decode


def match(gt_box, gt_label, default_anchors, threshold_for_pos, loc_t, conf_t, label_t, batch_idx):
    conf = [np.zeros((80, 144, 4)), np.zeros((40, 72, 4)), np.zeros((20, 36, 4))]
    label = [np.zeros((80, 144, 4)), np.zeros((40, 72, 4)), np.zeros((20, 36, 4))]
    boxes = [np.zeros((80, 144, 4, 4)), np.zeros((40, 72, 4, 4)), np.zeros((20, 36, 4, 4))]
    # 对于每一个box
    for i, box in enumerate(gt_box):
        # 处理该box没有和相应位置任意一个的anchor的iou大于阈值的情况
        match_flag = False
        match_x = []  # 用来记录每一个map上的匹配情况
        for idx, size in enumerate([[80, 144], [40, 72], [20, 36]]):
            cx, cy = box[0], box[1]
            # 该点落在的位置
            cx, cy = int(cx * size[1]), int(cy * size[0])
            # 从default_anchors中找到对应的先验框,每一个位置3个anchors
            anchors = default_anchors[idx][cy, cx]
            # 计算该box和该位置的3个anchors的iou，以确定和哪个anchor进行匹配
            iou = numpy_compute_iou(numpy_point_form(np.expand_dims(box, 0)), numpy_point_form(anchors))[0]
            max_idx = np.argmax(iou)
            match_x.append([idx, cx, cy, max_idx, iou[max_idx]])
            # 匹配
            if iou[max_idx] >= threshold_for_pos:
                match_flag = True  # 匹配成功
                conf[idx][cy, cx, max_idx] = 1.0
                label[idx][cy, cx, max_idx] = gt_label[i]
                boxes[idx][cy, cx, max_idx] = box
        # 如果未正常匹配成功，则把三个特征图中该位置最大的iou anchor和box进行匹配
        if match_flag == False:
            match_x_iou = np.asarray(match_x)[:, -1]
            match_idx = np.argmax(match_x_iou)
            idx = match_x[match_idx][0]
            cx, cy = match_x[match_idx][1], match_x[match_idx][2]
            max_idx = match_x[match_idx][3]
            conf[idx][cy, cx, max_idx] = 1.0
            label[idx][cy, cx, max_idx] = gt_label[i]
            boxes[idx][cy, cx, max_idx] = box
    
    conf = np.concatenate([np.reshape(x, [-1]) for x in conf], axis=0)
    label = np.concatenate([np.reshape(x, [-1]) for x in label], axis=0)
    boxes = np.concatenate([np.reshape(x, [-1, 4]) for x in boxes], axis=0)
    loc_t[batch_idx] = boxes
    label_t[batch_idx] = label
    conf_t[batch_idx] = conf
   

class YoloV4Loss(object):
    def __init__(self, config):
        self.config = config
    
    def get_loss(self, predictions, gt_boxes, gt_labels, default_anchors):
        loc_p, conf_p, label_p = predictions
        batch_size = len(gt_boxes)
        num_default_anchors = conf_p.shape[1]
        loc_t = np.zeros((batch_size, num_default_anchors, 4), dtype='float32')
        conf_t = np.zeros((batch_size, num_default_anchors))
        label_t = np.zeros((batch_size, num_default_anchors))
        for batch_idx in range(batch_size):
            gt_box = gt_boxes[batch_idx]
            gt_label = gt_labels[batch_idx]
            default = default_anchors  
            match(gt_box, gt_label, default, self.config.threshold_for_pos, loc_t, conf_t, label_t, batch_idx)
        
        loc_t = fluid.dygraph.to_variable(loc_t)  # 包含的是每个位置的真实gt_box
        conf_t = layers.cast(fluid.dygraph.to_variable(conf_t), 'float32')
        label_t = layers.cast(fluid.dygraph.to_variable(label_t), 'int64')

        pos = conf_t > 0
        neg = conf_t == 0
        num_pos = layers.reduce_sum(layers.cast(pos, 'float32')).numpy()
        pos = layers.where(pos)
        neg = layers.where(neg)
        
        # 计算位置损失，采用IoU损失
        loc_t = layers.gather_nd(loc_t, pos)
        loc_p = layers.gather_nd(loc_p, pos)
        # 准备匹配的anchors和用于归一化的normal
        anchors = np.concatenate([np.reshape(x, [-1, 4]) for x in default_anchors], axis=0)
        anchors = np.tile(np.expand_dims(anchors, 0), [batch_size, 1, 1])
        matched_anchors = fluid.layers.gather_nd(fluid.layers.cast(fluid.dygraph.to_variable(anchors), 'float32'), pos)
        normal_h = [np.ones((80, 144, 4)) / 80, np.ones((40, 72, 4)) / 40, np.ones((20, 36, 4)) / 20]
        normal_w = [np.ones((80, 144, 4)) / 144, np.ones((40, 72, 4)) / 72, np.ones((20, 36, 4)) / 36]
        normal_h = np.concatenate([np.reshape(x, [-1, 1]) for x in normal_h], axis=0)
        normal_w = np.concatenate([np.reshape(x, [-1, 1]) for x in normal_w], axis=0)
        normal_h = layers.unsqueeze(layers.gather_nd(layers.cast(fluid.dygraph.to_variable(normal_h), 'float32'), pos), 1)
        normal_w = layers.unsqueeze(layers.gather_nd(layers.cast(fluid.dygraph.to_variable(normal_w), 'float32'), pos), 1)
        matched_normal = fluid.layers.concat((normal_w, normal_h), axis=-1)

        # 由loc_p, matched_anchors, matched_normal计算[bx, by, bw, bh]
        b_xywh = yolo_decode(loc_p, matched_anchors, matched_normal)
        # 计算iou损失
        iou = compute_iou(point_form(b_xywh), point_form(loc_t))
        # print(matched_anchors.numpy())
        # print(loc_t.numpy())
        # print(iou.numpy())
        # exit()
        iou_loss = 2 * layers.reduce_mean(1 - iou)
        # 计算xywh损失
        # print(layers.abs(b_xywh / loc_t - 1).numpy())
        # exit()
        xywh_loss = 2 * layers.reduce_mean(layers.reduce_sum(layers.abs(b_xywh / loc_t - 1), -1))
        

        # 计算分类损失，采用交叉熵损失(cross entropy loss)
        label_p = fluid.layers.gather_nd(label_p, pos)
        label_t = fluid.layers.gather_nd(label_t, pos)
        label_loss = fluid.layers.reduce_mean(fluid.layers.cross_entropy(fluid.layers.softmax(label_p, axis=-1), label_t))
        
        # 计算置信度损失，采用BCE损失
        conf_p = fluid.layers.sigmoid(conf_p, [-1])
        conf_p_pos = fluid.layers.gather_nd(conf_p, pos)
        conf_t_pos = fluid.layers.gather_nd(conf_t, pos)
        conf_p_pos = fluid.layers.reshape(conf_p_pos, (-1, 1))
        conf_t_pos = fluid.layers.reshape(conf_t_pos, (-1, 1))
        pos_conf_loss = fluid.layers.reduce_mean(fluid.layers.log_loss(conf_p_pos, conf_t_pos))
        # print(np.sort(conf_p_pos.numpy(), axis=0))

        # conf_p_neg = fluid.layers.gather_nd(conf_p, neg)
        # conf_t_neg = fluid.layers.gather_nd(conf_t, neg)
        # 找困难样本，转化为numpy
        conf_p_numpy = np.reshape(conf_p.numpy(), (-1, 1))
        conf_p_numpy[np.reshape((conf_t == 1.0).numpy(), (-1, 1))] = 0   # 去掉正样本
        conf_p_numpy =np.reshape(conf_p_numpy, (batch_size, -1))
        index = np.argsort(-conf_p_numpy, -1)  # 按降序排列
        num_pos = fluid.layers.reduce_sum(fluid.layers.cast(conf_t == 1, 'int32'), -1).numpy()
        neg = np.asarray([[0, x] for x in index[0][: 20 * num_pos[0]]])
        for i in range(1, batch_size):
            neg = np.concatenate((neg, [[i, x] for x in index[i][: 20 * num_pos[i]]]))

        # 转化为tensor
        neg = fluid.dygraph.to_variable(neg)
        conf_p_neg = fluid.layers.gather_nd(conf_p, neg)
        conf_t_neg = fluid.layers.gather_nd(conf_t, neg)
        conf_p_neg = fluid.layers.reshape(conf_p_neg, (-1, 1))
        conf_t_neg = fluid.layers.reshape(conf_t_neg, (-1, 1))
        # print(np.sort(conf_p_neg.numpy(), axis=0)[::-1][:20])
        # print(conf_p_neg.numpy())
        neg_conf_loss = fluid.layers.reduce_sum(fluid.layers.log_loss(conf_p_neg, conf_t_neg)) / np.sum(num_pos)
        return xywh_loss, iou_loss, label_loss, pos_conf_loss, neg_conf_loss
        
        


        
        
        
        
        
        
        
        
        
        
        
        