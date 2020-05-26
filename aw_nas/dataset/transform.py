import torch

from aw_nas.dataset.voc_data_augmentation import Preproc
from aw_nas.utils.box_utils import match

class TrainAugmentation:
    def __init__(self, size, mean=(104, 117, 123), std=1.0):
        """
        Args:
            size: the size the of final image.
            mean: mean pixel value per channel.
        """
        self.mean = mean
        self.size = size
        self.std = std
        self.preproc = Preproc(size, mean, std, 0.6)

    def __call__(self, img, boxes, labels):
        """
        Args:
            img: the output of cv.imread in RGB layout.
            boxes: boundding boxes in the form of (x1, y1, x2, y2).
            labels: labels of boxes.
        """
        return self.preproc(img, boxes, labels)


class TestTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.preproc = Preproc(size, mean, std, -1)

    def __call__(self, image, boxes, labels):
        return self.preproc(image, boxes, labels)


class TargetTransform(object):
    def __init__(self, iou_threshold, variance):
        self.threshold = iou_threshold
        self.variance = variance

    def __call__(self, boxes, labels, priors):

        num_priors = priors.size(0)
        loc_t = torch.Tensor(1, num_priors, 4)
        conf_t = torch.LongTensor(1, num_priors)
        match(self.threshold, torch.tensor(boxes).float(), priors, self.variance, torch.tensor(labels),
                loc_t, conf_t, 0)
        loc_t = loc_t.squeeze(0)
        conf_t = conf_t.squeeze(0)
        return conf_t, loc_t