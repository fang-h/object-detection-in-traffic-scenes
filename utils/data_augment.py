import cv2
from PIL import Image, ImageEnhance  # 用于像素增强中的一些操作，需要cv2格式转PIL格式
import numpy as np
import random
from utils.box_utils import matrix_iof


def flip(image, boxes):
    height, width, _ = image.shape
    if random.uniform(0, 1) > 0.5:
        if random.uniform(0, 1) < 0.8:
            # 水平翻转
            image = cv2.flip(image, 1)
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        else:
            # 垂直翻转
            image = cv2.flip(image, 0)
            boxes = boxes.copy()
            boxes[:, 1::2] = height - boxes[:, 3::-2]
    return image, boxes


def pixel_aug(image):
    """基于像素点的aug"""
    if np.random.uniform(0, 1) < 0.8:
        # 高斯模糊
        if np.random.uniform(0, 1) < 0.2:
            kernel_set = [(3, 3), (5, 5), (7, 7), (9, 9)]
            sigma_set = [1, 2, 3, 4]
            kernel = kernel_set[random.randrange(4)]
            sigma = sigma_set[random.randrange(4)]
            image = cv2.GaussianBlur(image, kernel, sigma)
        # 颜色增强
        elif np.random.uniform(0, 1) < 0.4:
            # cv2转PIL
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            random_factor = np.random.randint(0, 51) / 10.
            image = ImageEnhance.Color(image).enhance(random_factor)  
            image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        # 亮度
        elif np.random.uniform(0, 1) < 0.6:
            # cv2转PIL
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            random_factor = np.random.randint(5, 16) / 10.  
            image = ImageEnhance.Brightness(image).enhance(random_factor)  
            image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        # 对比度
        elif np.random.uniform(0, 1) < 0.8:
            # cv2转PIL
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            random_factor = np.random.randint(5, 16) / 10.  
            image = ImageEnhance.Contrast(image).enhance(random_factor)  
            image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        # 锐化
        else:
            # cv2转PIL
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            random_factor = np.random.randint(0, 51) / 10.  
            image = ImageEnhance.Sharpness(image).enhance(random_factor)  
            image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return image


def random_erasing(image):
    image_t = image.copy()
    if np.random.uniform(0, 1) < 0.3:
        num_patch = np.random.randint(1, 10)
        h, w, _ = image_t.shape
        short_side = min(h, w)
        for i in range(num_patch):
            erasing_size = np.asarray([np.random.uniform(0.05, 0.1) * short_side,
                                       np.random.uniform(0.05, 0.1) * short_side])
            cut_size_half = erasing_size // 2
            center_x_min, center_x_max = cut_size_half[0], w - cut_size_half[0]
            center_y_min, center_y_max = cut_size_half[1], h - cut_size_half[1]
            center_x, center_y = np.random.randint(center_x_min, center_x_max), np.random.randint(center_y_min,
                                                                                                  center_y_max)
            x_min, y_min = center_x - int(cut_size_half[0]), center_y - int(cut_size_half[1])
            x_max, y_max = x_min + int(erasing_size[0]), y_min + int(erasing_size[1])
            x_min, y_min, x_max, y_max = max(0, x_min), max(0, y_min), min(w, x_max), min(h, y_max)
            image_t[y_min:y_max, x_min:x_max] = 0
    return image_t
        

def _pad(image, size=(720, 1296)):
    """
    输入的图像大小都是720x1280，为了能够整除2的倍数，将图像pad成720x1296
    """
    height, width, _ = image.shape
    image_t = np.empty((720, 1296, 3), dtype=image.dtype)
    image_t[:, :] = 0 
    image_t[0:0 + height, 0:0 + width] = image
    return image_t


def _resize(image, size=(640, 1152)):
    """将720x1296resize为640x1152"""
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    image = cv2.resize(image, (size[1], size[0]), interpolation=interp_method)
    return image


class DataAug(object):
    def __init__(self, img_dim=(640, 1152)):
        self.img_dim = img_dim

    def __call__(self, sample):
        image, boxes, labels = sample
        assert len(labels) > 0, "this image does not have object"
        image_t, boxes_t = flip(image, boxes)
        image_t = pixel_aug(image_t)
        image_t = random_erasing(image_t)
        image_t = _pad(image_t)
        height, width, _ = image_t.shape
        image_t = _resize(image_t, self.img_dim)
        boxes_t[:, 0::2] = boxes_t[:, 0::2] / width      
        boxes_t[:, 1::2] = boxes_t[:, 1::2] / height
        # for i, box in enumerate(boxes_t):
        #     cv2.rectangle(image_t, (int(box[0] * 1152), int(box[1] * 640)), (int(box[2] * 1152), int(box[3] * 640)), (0, 0, 255), 1)
        #     cx = int(box[0] * 1152)
        #     cy = int(box[1] * 640 + 12)
        #     cv2.putText(image_t, str(labels[i]), (cx, cy),
        #                 cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        # cv2.imwrite('inwwwwwwwwfer.jpg', image_t) 
        # exit()
        # 转化为[cx, cy, w, h]
        boxes_t[:, 2:] = boxes_t[:, 2:] - boxes_t[:, 0:2]  
        boxes_t[:, 0:2] = boxes_t[:, 0:2] + boxes_t[:, 2:] / 2.0
        image_t = image_t / 255.0 
        return image_t, np.clip(boxes_t, a_max=1, a_min=0), labels


