import numpy as np
import cv2
import random


def pad_image(image, pad_size=(720, 1296)):
    """
    输入的图像大小都是720x1280，为了能够整除2的倍数，将图像pad成720x1296
    """
    height, width, _ = image.shape
    image_t = np.empty((pad_size[0], pad_size[1], 3), dtype=image.dtype)
    image_t[:, :] = 0 
    image_t[0:0 + height, 0:0 + width] = image
    return image_t


def resize_image(image, resize_size=(640, 1152)):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    image_t = cv2.resize(image, (resize_size[1], resize_size[0]), interpolation=interp_method)
    return image_t

    
    