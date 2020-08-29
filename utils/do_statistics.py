import os
import json
import numpy as np
import cv2


all_label_path = '/home/aistudio/data/data17100/bdd100k_labels_release/bdd100k/labels/bdd100k_labels_images_train.json'
train_image_path = '/home/aistudio/data/data17100/bdd100k/bdd100k/images/100k/train/trainA'
test_image_path = '/home/aistudio/data/data17100/bdd100k/bdd100k/images/100k/train/testA'
train_data_path = '/home/aistudio/work/code/data/train_ground_truth.json'


def get_train_data():
    train_image_name = []
    for name in os.listdir(train_image_path):
        train_image_name.append(name)
    with open(all_label_path, 'r') as load_f:
        label = json.load(load_f)
    class_name = ['car', 'traffic sign', 'traffic light', 'person', 'truck', 'bus', 'bike', 'rider', 'motor']
    # 先把label中的关于detection的label提取出来为字典，以key=image_name， values=bbox|catid的格式存储
    all_ground_truth = {}
    # class_name = []
    for image_id in label:
        image_name = image_id['name']
        image_label = image_id['labels']
        image_gt = []
        for instance in image_label:
            if 'box2d' in instance.keys():
                cla_name = instance['category']
                box2d = list(instance['box2d'].values())
                if cla_name in class_name:
                    instance_gt = box2d + [class_name.index(cla_name)]
                    image_gt.append(instance_gt)
                else:
                    print(cla_name)
                    print(image_name)
        if len(image_gt) != 0:
            all_ground_truth[image_name] = image_gt

    print("XXXXXXXX")
    train_ground_truth = {}
    for image_name in train_image_name:
        if image_name not in all_ground_truth.keys():
            print(image_name)
        else:
            train_ground_truth[image_name] = all_ground_truth[image_name]
    print(len(train_ground_truth.keys()))
    with open('/home/aistudio/work/code/data/train_ground_truth.json', 'w') as f:
        json.dump(train_ground_truth, f)


def get_test_data():
    test_image_name = []
    for name in os.listdir(test_image_path):
        test_image_name.append(name)
    with open(all_label_path, 'r') as load_f:
        label = json.load(load_f)
    class_name = ['car', 'traffic sign', 'traffic light', 'person', 'truck', 'bus', 'bike', 'rider', 'motor']
    # 先把label中的关于detection的label提取出来为字典，以key=image_name， values=bbox|catid的格式存储
    all_ground_truth = {}
    # class_name = []
    for image_id in label:
        image_name = image_id['name']
        image_label = image_id['labels']
        image_gt = []
        for instance in image_label:
            if 'box2d' in instance.keys():
                cla_name = instance['category']
                box2d = list(instance['box2d'].values())
                if cla_name in class_name:
                    instance_gt = box2d + [class_name.index(cla_name)]
                    image_gt.append(instance_gt)
                else:
                    print(cla_name)
                    print(image_name)
        if len(image_gt) != 0:
            all_ground_truth[image_name] = image_gt
    print("XXXXXXXX")
    test_ground_truth = {}
    for image_name in test_image_name:
        if image_name not in all_ground_truth.keys():
            print(image_name)
        else:
            test_ground_truth[image_name] = all_ground_truth[image_name]
    print(len(test_ground_truth.keys()))
    with open('/home/aistudio/work/code/data/test_ground_truth.json', 'w') as f:
        json.dump(test_ground_truth, f)



def num_of_cls():
    """统计数据集中每一个类别共有多少个实例"""
    with open(train_data_path, 'r') as f:
        train_data = json.load(f)
    class_name = ['car', 'traffic sign', 'traffic light', 'person', 'truck', 'bus', 'bike', 'rider', 'motor']
    res = {}
    image_id = list(train_data.keys())
    for i in range(len(image_id)):
        print(i)
        id = image_id[i]
        bbox_and_catid = np.asarray(train_data[id])
        # print(bbox_and_catid)
        catid = bbox_and_catid[:, -1]
        class_label = []
        for i in range(len(catid)):
            # print(int(catid[i]))
            cla_name = class_name[int(catid[i])]
            if cla_name in res.keys():
                res[cla_name] += 1
            else:
                res[cla_name] = 1
    res = sorted(res.items(),key = lambda x:x[1],reverse = True)
    print(res) #[('car', 406290), ('traffic sign', 134045), ('traffic light', 90745), ('person', 65288), ('truck', 22074), ('bus', 8468), ('bike', 5024), ('rider', 3268), ('motor', 2059)]


def prepare_wh_for_kmeans():
    with open(train_data_path, 'r') as f:
        train_data = json.load(f)
    image_id = list(train_data.keys())
    image_name = image_id[0]
    image = cv2.imread(os.path.join(train_image_path, image_name))
    h, w, _ = image.shape
    print(h, w)
    bbox_and_catid = np.asarray(train_data[image_name])
    bbox = bbox_and_catid[:, :-1]
    wh_set = bbox[:, 2:] - bbox[:, 0:2]
    wh_set[:, 0] = wh_set[:, 0] / w * 1152
    wh_set[:, 1] = wh_set[:, 1] / h * 640

    for i in range(1, len(image_id)):
        # print(i)
        image_name = image_id[i]
        image = cv2.imread(os.path.join(train_image_path, image_name))
        h, w, _ = image.shape
        if h != 720 or w != 1280:
            print("XXXXXXXXXX")
        bbox_and_catid = np.asarray(train_data[image_name])
        bbox = bbox_and_catid[:, :-1]
        wh = bbox[:, 2:] - bbox[:, 0:2]
        wh[:, 0] = wh[:, 0] / w * 1152
        wh[:, 1] = wh[:, 1] / h * 640
        wh_set = np.vstack([wh_set, wh])
    np.save('/home/aistudio/work/code/data/wh_for_kmeans.npy', wh_set)

    
class KMeansOnBdd(object):

    def __init__(self, k, max_iter):
        self.k = k  # 聚类个数
        self.max_iter = max_iter  # 最大的循环次数
        self.wh = np.load('/home/aistudio/work/code/data/wh_for_kmeans.npy')  # 所有ins的wh
        self.length = len(self.wh) # ins个数

    def init_centroids(self):
        centroids = self.wh[np.random.choice(self.length, self.k, replace=False)]
        return centroids

    def compute_iou(self, wh, centroids):
        # 以原点为中心
        xy_min_1 = 0 - wh / 2.0
        xy_max_1 = 0 + wh / 2.0

        xy_min_2 = 0 - centroids / 2.0
        xy_max_2 = 0 + centroids / 2.0
        
        xy_min = np.maximum(xy_min_1, xy_min_2)
        xy_max = np.minimum(xy_max_1, xy_max_2)

        inter_wh = xy_max - xy_min
        intersection = inter_wh[:, 0] * inter_wh[:, 1]
        wh_area = wh[:, 0] * wh[:, 1]
        centroids_area = centroids[:, 0] * centroids[:, 1]
        iou = intersection / (wh_area + centroids_area - intersection)
        return iou

    def new_centroids(self, nearest):
        centroids = []
        for i in range(self.k):
            ith_wh = self.wh[nearest == i]
            centroids.append(np.mean(ith_wh, axis=0))
        centroids = np.vstack([x for x in centroids])
        return centroids

    def k_means(self):
        # 初始化聚类中心
        centroids = self.init_centroids()
        # 记录距离
        distances = np.zeros((self.length, self.k))
        print(self.length)
        for _ in range(self.max_iter):
            print(_)
            for i in range(self.length):
                wh = self.wh[i:i+1]
                iou = self.compute_iou(wh, centroids)
                distances[i] = 1 - iou
            nearest = np.argmin(distances, axis=1)
            centroids = self.new_centroids(nearest)
        print(centroids)
        np.save('/home/aistudio/work/code/data/anchors1.npy', centroids)


def avg_iou():
    """计算wh和聚类得到的anchors的avg_iou"""
    """9个anchor:0.645
       12个anchor:0.684"""
    all_wh = np.load('/home/aistudio/work/code/data/wh_for_kmeans.npy')  # 所有ins的wh
    anchors = np.load('/home/aistudio/work/code/data/anchors_12.npy')
    avg_iou = 0
    for i in range(len(all_wh)):
        # 以原点为中心
        wh = all_wh[i:i+1]
        xy_min_1 = 0 - wh / 2.0
        xy_max_1 = 0 + wh / 2.0

        xy_min_2 = 0 - anchors / 2.0
        xy_max_2 = 0 + anchors / 2.0
        
        xy_min = np.maximum(xy_min_1, xy_min_2)
        xy_max = np.minimum(xy_max_1, xy_max_2)

        inter_wh = xy_max - xy_min
        intersection = inter_wh[:, 0] * inter_wh[:, 1]
        wh_area = wh[:, 0] * wh[:, 1]
        anchors_area = anchors[:, 0] * anchors[:, 1]
        iou = intersection / (wh_area + anchors_area - intersection)
        max_iou = np.max(iou)
        avg_iou += max_iou
    print(avg_iou / len(all_wh))
    

if __name__ == '__main__':
    # get_train_data()
    get_test_data()
    # num_of_cls()
    # prepare_wh_for_kmeans()
    # KMeansOnBdd(12, 50).k_means()
    # avg_iou()
    # anchors = np.load('/home/aistudio/work/code/data/anchors_12.npy')
    # s = anchors[:, 0] * anchors[:, 1]
    # index = np.argsort(s)
    # anchors = anchors[index]
    # print(anchors)
    
