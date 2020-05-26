
import cv2
import torch
import numpy as np

from torchvision import transforms

class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, img_size=512, padding=True):
        self.img_size = img_size
        self.padding = padding

    def __call__(self, image, boxes, labels):
        height, width, _ = image.shape

        boxes = boxes.astype(np.float)

        if self.padding:
            if height > width:
                scale = self.img_size / height
                resized_height = self.img_size
                resized_width = int(width * scale)
            else:
                scale = self.img_size / width
                resized_height = int(height * scale)
                resized_width = self.img_size

            image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

            new_image = np.zeros((self.img_size, self.img_size, 3))
            new_image[0:resized_height, 0:resized_width] = image

            boxes *= scale
        
        else:
            new_image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            boxes[:, 0::2] *= (self.img_size / width)
            boxes[:, 1::2] *= (self.img_size / height)
        
        return new_image, boxes, labels


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, flip_x=0.5):
        self.flip_x = flip_x

    def __call__(self, image, boxes, labels):
        if np.random.rand() < self.flip_x:
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = boxes[:, 0].copy()
            x2 = boxes[:, 2].copy()

            x_tmp = x1.copy()

            boxes[:, 0] = cols - x2
            boxes[:, 2] = cols - x_tmp

        return image, boxes, labels


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, image, boxes, labels):
        return (image.astype(np.float32) - self.mean) / self.std, boxes, labels


class TrainTransformer(object):
    def __init__(self, mean, std, crop_size):
        self.compose = [
            Normalizer(mean=mean, std=std),
            Augmenter(),
            Resizer(crop_size)]
    
    def __call__(self, img, boxes, labels):
        for fn in self.compose:
            img, boxes, labels = fn(img, boxes, labels)
        img = img.transpose(2, 0, 1)
        return img, boxes, labels


class TestTransformer(object):
    def __init__(self, mean, std, crop_size):
        self.compose = [
            Normalizer(mean=mean, std=std),
            Resizer(crop_size, False)
        ]
    
    def __call__(self, img, boxes, labels):
        for fn in self.compose:
            img, boxes, labels = fn(img, boxes, labels)
        img = img.transpose(2, 0, 1)
        return img, boxes, labels