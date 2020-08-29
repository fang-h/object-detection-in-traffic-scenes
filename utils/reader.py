import os
import numpy as np
import random
import json
import cv2

from utils.data_augment import DataAug


train_image_path = '/home/aistudio/data/data17100/bdd100k/bdd100k/images/100k/train/trainA'
test_image_path = '/home/aistudio/data/data17100/bdd100k/bdd100k/images/100k/train/testA'
train_data_path = '/home/aistudio/work/code/data/train_ground_truth.json'
test_data_path = '/home/aistudio/work/code/data/test_ground_truth.json'
video_path = '/home/aistudio/work/code/test_image/traffic2.mp4'


class TrainDataReader(object):
    
    def __init__(self, image_size, train_data_path=train_data_path, train_image_path=train_image_path):
        self.image_size = image_size
        with open(train_data_path, 'r') as f1:
            self.train_data = json.load(f1)
        self.class_name =class_name = ['car', 'traffic sign', 'traffic light', 'person',
                                       'truck', 'bus', 'bike', 'rider', 'motor']
        self.image_name = list(self.train_data.keys())

    def get_lengths(self):
        return len(self.image_name)
        
    def shuffle(self):
        random.shuffle(self.image_name)

    def get_items(self, idx):
        image_name = self.image_name[idx]
        image = cv2.imread(os.path.join(train_image_path, image_name))
        bbox_and_catid = np.asarray(self.train_data[image_name])
        bbox = bbox_and_catid[:, :-1]  # [xmin, ymin, xmax, ymax]格式
        class_label = bbox_and_catid[:, -1]
        class_label = np.asarray(class_label, dtype='float32') 
        sample = (image, bbox, class_label)
        image, bbox, class_label = DataAug(self.image_size)(sample)  # bbox为[cx, cy, w, h]格式 
        return np.transpose(image,[2, 0, 1]), bbox, class_label     


class ValDataReader(object):
    
    def __init__(self, image_size, test_data_path=test_data_path):
        self.image_size = image_size
        with open(test_data_path, 'r') as f1:
            self.test_data = json.load(f1)
        self.image_name = list(self.test_data.keys())
        self.class_name = ['car', 'traffic sign', 'traffic light', 'person',
                           'truck', 'bus', 'bike', 'rider', 'motor']
                           
    def get_lengths(self):
        return len(self.image_name)
        
    def shuffle(self):
        random.shuffle(self.image_name)

    def get_items(self, idx):
        image_name = self.image_name[idx]
        image = cv2.imread(os.path.join(test_image_path, image_name))
        bbox_and_catid = np.asarray(self.test_data[image_name])
        bbox = bbox_and_catid[:, :-1]  # [xmin, ymin, xmax, ymax]格式
        class_label = bbox_and_catid[:, -1]
        class_label = np.asarray(class_label, dtype='float32') 
        return image, bbox, class_label    


class VideoReader(object):

    def __init__(self, video_path = video_path):
        self.video_in = cv2.VideoCapture()
        self.video_in.open(video_path)
        self.frame_width = int(self.video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_fps = int(self.video_in.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.finish_frame_nums = 400
        self.class_name = ['car', 'traffic sign', 'traffic light', 'person',
                           'truck', 'bus', 'bike', 'rider', 'motor']

    def not_finished(self, cut_frame):
        if self.video_in.isOpened():
            if self.finish_frame_nums == 0:
                return True
            if cut_frame < self.finish_frame_nums:
                return True
            else:
                return False
        else:
            print('Video is not opened')
            return False

    def get_frame(self):
        ret, frame = self.video_in.read()
        if ret is False:
            print('Video is done')
            exit()
        frame_show = frame.copy()
        return frame, frame_show

    def end(self):
        self.video_in.release()
        

