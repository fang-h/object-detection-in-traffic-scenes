import os
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
from utils.reader import VideoReader
# config
import config
# preprocess
from utils.preprocess import pad_image, resize_image
# decode
from utils.box_utils import numpy_yolo_decode, nms


yolov3_path = '/home/aistudio/work/code/logs/yolov3_70.pdparams'
vis_path = '/home/aistudio/work/code/vis_for_video'


def yolov3_on_video():
    with fluid.dygraph.guard(fluid.CUDAPlace(0)):
        model = YoloV3()
        model_dict, _ = fluid.dygraph.load_dygraph(yolov3_path)
        model.load_dict(model_dict)
        video_reader = VideoReader()
        default_anchors = YoloAnchors().get_anchors()
        anchors = np.concatenate([np.reshape(x, [-1, 4]) for x in default_anchors], axis=0)
        normal_h = [np.ones((80, 144, 4)) / 80, np.ones((40, 72, 4)) / 40, np.ones((20, 36, 4)) / 20]
        normal_w = [np.ones((80, 144, 4)) / 144, np.ones((40, 72, 4)) / 72, np.ones((20, 36, 4)) / 36]
        normal_h = np.concatenate([np.reshape(x, [-1, 1]) for x in normal_h], axis=0)
        normal_w = np.concatenate([np.reshape(x, [-1, 1]) for x in normal_w], axis=0)
        normal = np.concatenate((normal_w, normal_h), axis=-1)
        model.eval()
        # 帧计算
        cur_frame_counter = 0
        while video_reader.not_finished(cur_frame_counter):
            print(cur_frame_counter)
            images = []
            image, image_show = video_reader.get_frame()
            # 进行预处理
            image = pad_image(image)
            height, width, _ = image.shape
            image = resize_image(image)
            image = image / 255.0
            image = image.transpose(2, 0, 1)
            images.append(image)
            images = np.asarray(images, dtype='float32')
            images = fluid.dygraph.to_variable(images)
            loc_p, conf_p, label_p = model(images)
            # 计算conf_scores
            conf_scores = fluid.layers.sigmoid(conf_p, [-1]).numpy()[0]  #  sigmoid(conf_p.numpy()[0])
            pos = conf_scores > 0.4
            pos = np.squeeze(pos, 1)  
            conf_scores = conf_scores[pos]

            # 计算class_scores以及获得类别
            label = fluid.layers.softmax(label_p).numpy()[0]  # 去掉batch维度
            label = label[pos]
            classes = np.argmax(label, axis=-1)
            x_axis_index=np.tile(np.arange(len(label)), (np.expand_dims(classes, -1).shape[1],1)).transpose()
            class_scores = label[x_axis_index, np.expand_dims(classes, -1)]
            # 由conf_scores和class_scores计算scores
            scores = np.reshape(conf_scores * class_scores, [-1])  # 维度改为1

            # 计算boxes
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
                if score < 0.3:
                    continue
                # 可视化
                text = video_reader.class_name[cla] #+ "/" + "{:.2f}".format(score)
                box = list(map(int, box))
                cv2.rectangle(image_show, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 1)
                cx = box[0]
                cy = box[1] + 12
                cv2.putText(image_show, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255))
            if cur_frame_counter % 10 == 0 and cur_frame_counter != 0:
                cv2.imwrite(os.path.join(vis_path, str(cur_frame_counter)+'.jpg'), image_show) 
            cur_frame_counter += 1

            
if __name__ == '__main__':
    yolov3_on_video()
   

    
    