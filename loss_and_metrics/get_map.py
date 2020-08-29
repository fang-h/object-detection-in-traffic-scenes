import numpy as np
import os


test_anno_path = '/home/aistudio/work/code/result_file/test_anno.txt'
yolov3_det_path = '/home/aistudio/work/code/result_file/yolov3_0.3_0.24_det.txt'
# yolov4_det_path = /home/aistudio/


def get_anno(anno_path):
    anno_images = []  # 所属图片
    anno_boxes = []  # 标注的每一个框
    anno_labels = []  # 每一个框对应的类别
    f = open(anno_path)
    for line in f.readlines():
        anno = line.strip('\n').split(' ')
        if len(anno) == 5:
            anno_images.append([anno[0]])
            anno_boxes.append([float(anno[1]), float(anno[2]), float(anno[3]), float(anno[4])])
            anno_labels.append([0])
        elif len(anno) == 6:
            anno_images.append([anno[0]])
            anno_boxes.append([float(anno[1]), float(anno[2]), float(anno[3]), float(anno[4])])
            anno_labels.append([float(anno[-1])])
    return np.asarray(anno_images), np.asarray(anno_boxes), np.asarray(anno_labels)


def get_det(det_path):
    det_images = []  # 所属图片
    det_scores = []  # 检测到框的分数
    det_boxes = []  # 检测到的每一个框
    det_labels = []  # 每一个框对应的类别
    f = open(det_path)
    for line in f.readlines():
        det = line.strip('\n').split(' ')
        if len(det) == 6:
            det_images.append([det[0]])
            det_scores.append([float(det[1])])
            det_boxes.append([float(det[2]), float(det[3]), float(det[4]), float(det[5])])
            det_labels.append([0])
        elif len(det) == 7:
            det_images.append([det[0]])
            det_scores.append([float(det[1])])
            det_boxes.append([float(det[2]), float(det[3]), float(det[4]), float(det[5])])
            det_labels.append([float(det[-1])])
    return np.asarray(det_images), np.asarray(det_scores), np.asarray(det_boxes), np.asarray(det_labels)


def get_intersection(set1, set2):
    lower_bounds = np.maximum(set1[:, :2], set2[:, :2])
    upper_bounds = np.minimum(set1[:, 2:], set2[:, 2:])
    intersection_dims = np.clip(upper_bounds - lower_bounds, a_min=0, a_max=1e10)
    return intersection_dims[:, 0] * intersection_dims[:, 1]


def get_overlap(set1, set2):
    # Find intersection
    intersection = get_intersection(set1, set2)
    # find ares of each box in both sets
    areas_set1 = (set1[:, 2] - set1[:, 0]) * (set1[:, 3] - set1[:, 1])
    areas_set2 = (set2[:, 2] - set2[:, 0]) * (set2[:, 3] - set2[:, 1])
    # Find union
    union = areas_set1 + areas_set2 - intersection
    return intersection / union


def calculate_map(anno_images, anno_boxes, anno_labels,
                  det_images, det_scores, det_boxes, det_labels,
                  num_classes):
    # 存储每一个类别的precision
    average_precisions = np.zeros(num_classes, dtype='float32')
    # 存储每一个类别的recall
    average_recalls = np.zeros(num_classes, dtype='float32')
    for c in range(num_classes):
        # 提取出当前分类的ground truth
        anno_class_images = anno_images[anno_labels == c]
        anno_class_boxes = np.reshape(anno_boxes[np.tile(anno_labels == c, [1, 4])], (-1, 4))
        # 用来记录该类别下的每一个anno box是否被检测到
        anno_class_boxes_detected = np.zeros(len(anno_class_boxes), dtype='uint8')
        # 提取出当前分类的detected结果
        det_class_images = det_images[det_labels == c]
        det_class_boxes = np.reshape(det_boxes[np.tile(det_labels == c, [1, 4])], (-1, 4))
        det_class_scores = det_scores[det_labels == c]
        # 该类中被检测到的目标数量
        num_class_detections = len(det_class_boxes)
        if num_class_detections == 0:
            continue
        # 按照detected评分按降序排列
        sort_ind = np.argsort(det_class_scores, axis=0)[::-1]
        # det_class_scores = np.sort(det_class_scores, axis=0)[::-1]
        det_class_images = det_class_images[sort_ind]
        det_class_boxes = det_class_boxes[sort_ind]
        # 用于记录真正例和假正例
        true_positives = np.zeros(num_class_detections, dtype='float32')
        false_positives = np.zeros(num_class_detections, dtype='float32')
        # 对于该类中每一个被检测到的目标
        for d in range(num_class_detections):
            this_detection_box = np.expand_dims(det_class_boxes[d], axis=0)
            # 该目标属于的图片
            this_image = det_class_images[d]
            # 该目标所属图片的所有标注
            this_image_anno_boxes = anno_class_boxes[anno_class_images == this_image]
            # 如果本图片中无物体，则d视为一个false positive
            if len(this_image_anno_boxes) == 0:
                false_positives[d] = 1
                continue
            # 计算该目标和所属图片所有标注的iou
            overlaps = get_overlap(this_detection_box, this_image_anno_boxes)
            max_overlap = np.max(overlaps)
            max_ind = np.argmax(overlaps)
            # max_ind所对应的anno box在anno_class_images中的位置
            original_ind = np.asarray((range(len(anno_class_boxes))))[anno_class_images == this_image][max_ind]
            # 计算0.5和0.75
            if max_overlap > 0.75:
                # 该标注未被皮匹配
                if anno_class_boxes_detected[original_ind] == 0:
                    true_positives[d] = 1
                    # 将该标注改为被匹配过
                    anno_class_boxes_detected[original_ind] = 1
                else:
                    false_positives[d] = 1
            else:
                false_positives[d] = 1

        # precision: TP/(TP+FP)
        precision = sum(true_positives) / num_class_detections
        print(precision)
        # recall: TP/(TP+FN)
        recall = sum(true_positives) / len(anno_class_boxes)
        print(recall)
        average_precisions[c] = precision  # 针对每一个class的均值
        average_recalls[c] = recall
    print(average_precisions)
    print(average_recalls)
    




if __name__ == '__main__':
    anno_images, anno_boxes, anno_labels = get_anno(test_anno_path)
    det_images, det_scores, det_boxes, det_labels = get_det(yolov3_det_path)
    calculate_map(anno_images, anno_boxes, anno_labels,
                  det_images, det_scores, det_boxes, det_labels,
                  num_classes=9)
    





