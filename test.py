import cv2
import paddle
import paddle.fluid as fluid
import numpy as np
from tqdm import tqdm as tqdm
# backbone
from backbone.DarkNet53 import DarkNet53
from backbone.CSPDarkNet53 import CSPDarkNet53
# model 
from model.yolov3 import YoloV3
from model.yolov4 import YoloV4
# anchor
from utils.anchors import YoloAnchors
# reader
from utils.reader import TrainDataReader
from utils.reader import ValDataReader
# loss
from loss_and_metrics.yolov3_loss import YoloV3Loss
# config
import config
# preprocess
from utils.preprocess import pad_image, resize_image
# decode
from utils.box_utils import numpy_yolo_decode, nms


yolov3_path = '/home/aistudio/work/code/logs/yolov3_15.pdparams'


def yolov3_test():
    with fluid.dygraph.guard(fluid.CUDAPlace(0)):
        model = YoloV3()
        model_dict, _ = fluid.dygraph.load_dygraph(yolov3_path)
        model.load_dict(model_dict)
        # data_reader = TrainDataReader(image_size=(640, 1152))
        data_reader = ValDataReader(image_size=(640, 1152))
        # yolov3_loss = YoloV3Loss(config)
        default_anchors = YoloAnchors().get_anchors()
        anchors = np.concatenate([np.reshape(x, [-1, 4]) for x in default_anchors], axis=0)
        normal_h = [np.ones((80, 144, 4)) / 80, np.ones((40, 72, 4)) / 40, np.ones((20, 36, 4)) / 20]
        normal_w = [np.ones((80, 144, 4)) / 144, np.ones((40, 72, 4)) / 72, np.ones((20, 36, 4)) / 36]
        normal_h = np.concatenate([np.reshape(x, [-1, 1]) for x in normal_h], axis=0)
        normal_w = np.concatenate([np.reshape(x, [-1, 1]) for x in normal_w], axis=0)
        # normal_h = np.expand_dims((normal_h), 1)
        # normal_w = np.expand_dims((normal_w), 1)
        normal = np.concatenate((normal_w, normal_h), axis=-1)
        model.eval()
        for i in range(1):
            data_reader.shuffle()
            images = []
            gt_boxes = []
            gt_labels = []
            image, gt_box, gt_label = data_reader.get_items(100)
            # image_show = image_show = np.transpose(np.asarray(image * 255, 'uint8'), (1, 2, 0))  # image
            image_show = image.copy()
            # 进行预处理
            image = pad_image(image)
            height, width, _ = image.shape
            image = resize_image(image)
            image = image / 255.0
            image = image.transpose(2, 0, 1)
            images.append(image)
            # gt_boxes.append(gt_box)
            # gt_labels.append(gt_label)
            images = np.asarray(images, dtype='float32')
            images = fluid.dygraph.to_variable(images)
            loc_p, conf_p, label_p = model(images)
            # xywh_loss, iou_loss, label_loss, pos_conf_loss, neg_conf_loss = yolov3_loss.get_loss((loc_p, conf_p, label_p), gt_boxes, gt_labels, default_anchors)
            # print(xywh_loss.numpy(), iou_loss.numpy(), label_loss.numpy(), pos_conf_loss.numpy(), neg_conf_loss.numpy())

            # image = cv2.imread(os.path.join(os.getcwd(), 'test_images/test17.jpg'))
            # image_show = image.copy()
            # height, width, _ = image.shape
            # long_side = max(height, width)
            # image = _pad_to_square_and_resize(image, size=416)
            # image = image / 255.0
            # image = image.transpose(2, 0, 1)
            # image = np.asarray(image, dtype='float32')
            # image = fluid.layers.unsqueeze(fluid.dygraph.to_variable(image), 0)
            # loc_p, conf_p, label_p = model(image)
            # 计算conf_scores
            conf_scores = fluid.layers.sigmoid(conf_p, [-1]).numpy()[0]  #  sigmoid(conf_p.numpy()[0])
            pos = conf_scores > 0.2
            pos = np.squeeze(pos, 1)  
            conf_scores = conf_scores[pos]
            # print(conf_scores)

            # 计算class_scores以及获得类别
            label = fluid.layers.softmax(label_p).numpy()[0]  # 去掉batch维度
            label = label[pos]
            classes = np.argmax(label, axis=-1)
            x_axis_index=np.tile(np.arange(len(label)), (np.expand_dims(classes, -1).shape[1],1)).transpose()
            class_scores = label[x_axis_index, np.expand_dims(classes, -1)]
            # 由conf_scores和class_scores计算scores
            scores = np.reshape(conf_scores * class_scores, [-1])  # 维度改为1

            # 计算boxes
            # boxes = numpy_decode(loc_p.numpy()[0][pos], default_anchors[pos])
            pos_anchors = anchors[pos]
            pos_normal = normal[pos]            
            boxes = numpy_yolo_decode(loc_p.numpy()[0][pos], pos_anchors, pos_normal)
            boxes[:, 0::2] = boxes[:, 0::2] * width      
            boxes[:, 1::2] = boxes[:, 1::2] * height
            # 根据scores排序
            order = scores.argsort()[::-1][:200]
            boxes = boxes[order]
            scores = scores[order]
            classes = classes[order]
            # nms per class
            # class_set = np.unique(classes)
            # boxes_t = []
            # scores_t = []
            # classes_t = []
            # for cla in class_set:
            #     idx = classes == cla
            #     # print(idx)
            #     per_class = classes[idx]
            #     per_scores = scores[idx]
            #     per_boxes = boxes[idx]
            #     # print(per_class)
            #     keep = nms(per_boxes, per_scores, 0.3)
            #     per_class = per_class[keep]
            #     per_scores = per_scores[keep]
            #     per_boxes = per_boxes[keep]
            #     classes_t.append(per_class)
            #     scores_t.append(per_scores)
            #     boxes_t.append(per_boxes)
            # classes_t = np.concatenate([x for x in classes_t])
            # scores_t = np.concatenate([x for x in scores_t])
            # boxes_t = np.concatenate([x for x in boxes_t])
            # print(scores_t)
            # nms
            keep = nms(boxes, scores, 0.3)
            boxes_t = boxes[keep]
            scores_t = scores[keep]
            classes_t = classes[keep]
    
            # plot and show
            for det in zip(boxes_t, scores_t, classes_t):
                box = det[0:-1][0]
                score = det[-2]
                cla = det[-1]
                if score < 0.16:
                    continue
                # # 记录每一个框
                # xmin = box[0]
                # ymin = box[1]
                # xmax = box[2]
                # ymax = box[3]
                # fw.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(image_name, score, xmin, ymin, xmax, ymax))
                # 可视化
                text = data_reader.class_name[cla] #+ "/" + "{:.2f}".format(score)
                box = list(map(int, box))
                cv2.rectangle(image_show, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)
                cx = box[0]
                cy = box[1] + 12
                cv2.putText(image_show, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
            cv2.imwrite('infer.jpg', image_show) 

            
if __name__ == '__main__':
    yolov3_test()
    
   

    
    