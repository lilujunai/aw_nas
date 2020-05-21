
import cv2
import torch
import numpy as np

from torchvision import transforms

class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, image, boxes, labels):
        height, width, _ = image.shape
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
        
        return torch.from_numpy(new_image).to(torch.float32), torch.from_numpy(boxes).to(torch.float32), torch.from_numpy(labels).to(torch.long)


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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
        img = img.permute(2, 1, 0)
        return img, boxes, labels


class TestTransformer(object):
    def __init__(self, mean, std, crop_size):
        self.compose = transforms.Compose([
            Normalizer(mean=mean, std=std),
            # Augmenter(),
            Resizer(crop_size)]
        )
    
    def __call__(self, img, boxes, labels):
        for fn in self.compose:
            img, boxes, labels = fn(img, boxes, labels)
        img = img.permute(2, 1, 0)
        return img, boxes, labels