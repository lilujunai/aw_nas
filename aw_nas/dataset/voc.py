import numpy as np
import logging
import pathlib
import xml.etree.ElementTree as ET
import cv2
import os
import os
from datetime import datetime
import six
import numpy as np
from torchvision import (datasets, transforms)
from aw_nas.utils.torch_utils import Cutout
from aw_nas.utils.box_utils import *
from aw_nas.utils.voc_eval import write_voc_results_file, do_python_eval

from aw_nas.dataset.base import BaseDataset
from aw_nas.dataset.prior_model import PriorBox
from aw_nas.dataset.transform import *

CLASSES = (
    '__background__',
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)


class VOCDataset(object):
    def __init__(self, data_dir, image_sets=[("VOC2007", "trainval"), ("VOC2012", "trainval")], transform=None, target_transform=None, is_test=False, keep_difficult=False, label_file=None):
        """Dataset for VOC data.
        Args:
            root: the root of the VOC2007 or VOC2012 dataset, the directory contains the following sub-directories:
                Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
        """
        self.data_dir = data_dir
        self.root = pathlib.Path(self.data_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.anno_path = os.path.join(data_dir, '%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join(data_dir, '%s', 'JPEGImages', '%s.jpg')
        self.ids = []

        for sub_dir in image_sets:
            self.image_sets_file = self.root / ("%s/ImageSets/Main/%s.txt" % sub_dir)
            self.ids.extend(VOCDataset._read_image_ids(self.image_sets_file, sub_dir[0]))
        self.keep_difficult = keep_difficult
        self.is_test = is_test

        # if the labels file exists, read in the class names
        label_file_name = self.root / "labels.txt"

        if os.path.isfile(label_file_name):
            class_string = ""
            with open(label_file_name, 'r') as infile:
                for line in infile:
                    class_string += line.rstrip()

            # classes should be a comma separated list

            classes = class_string.split(',')
            classes  = [elem.replace(" ", "") for elem in classes]
            self.class_names = tuple(classes)
            logging.info("VOC Labels read from file: " + str(self.class_names))

        else:
            logging.info("No labels file, using default VOC classes.")
            self.class_names = CLASSES

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image, target, height, width = self._getitem(index)
        if self.is_test:
            return index, image, target, height, width
        return index, image, target

    def _getitem(self, index):
        img_id, (boxes, labels, is_difficult) = self.get_annotation(index)
        image = self._read_image(img_id)
        height, width, channels = image.shape

        if self.transform is not None:
            image, boxes, labels = self.transform(image, boxes, labels)

        if self.target_transform is not None:
            (boxes, labels) = self.target_transform(boxes, labels)
        target = (boxes, labels)
        return torch.tensor(image), target, height, width

    def get_image(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_id = self.ids[index]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file, year):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append((year, line.rstrip()))
        return ids

    def _get_annotation(self, image_id):
        annotation_file = self.anno_path % (image_id)
        objects = ET.parse(annotation_file).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for object in objects:
            class_name = object.find('name').text.lower().strip()
            # we're only concerned with clases in our list
            if class_name in self.class_dict:
                bbox = object.find('bndbox')

                # VOC dataset format follows Matlab, in which indexes start from 0
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1
                boxes.append([x1, y1, x2, y2])

                labels.append(self.class_dict[class_name])
                is_difficult_str = object.find('difficult').text
                is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def _read_image(self, image_id):
        image_file = self._imgpath % image_id
        image = cv2.imread(str(image_file))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image


class VOC(BaseDataset):
    NAME = "voc"

    def __init__(self, load_train_only=False, class_name_file=None,
                 random_choose=False, random_seed=123,
                 train_sets=[("VOC2007", "trainval"), ("VOC2012", "trainval")],
                 test_sets=[("VOC2007", "test")],
                 normalize=False, normalize_box=True,
                 train_crop_size=300, test_crop_size=300, image_mean=(103.94, 116.78, 123.68), image_std=1.,
                 iou_threshold=0.5, keep_difficult=False, relative_dir=None):
        super(VOC, self).__init__(relative_dir)
        self.load_train_only = load_train_only
        self.train_data_dir = os.path.join(self.data_dir, "train")
        self.class_name_file = class_name_file
        self.normalize = normalize
        self.normalize_box = normalize_box

        train_transform = TrainAugmentation(train_crop_size, np.array(image_mean), np.array(image_std), normalize, normalize_box)
        test_transform = TestTransform(test_crop_size, np.array(image_mean), np.array(image_std), normalize, normalize_box)

        self.datasets = {}
        self.datasets['train'] = VOCDataset(self.train_data_dir, train_sets, train_transform)
        self.grouped_annotation = {}

        if not self.load_train_only:
            self.test_data_dir = os.path.join(self.data_dir, "test")
            self.datasets['test'] = VOCDataset(self.test_data_dir, test_sets, test_transform, is_test=True)

    def splits(self):
        return self.datasets

    @classmethod
    def data_type(cls):
        return "image"

    @staticmethod
    def _read_names_from_file(path):
        with open(path, "r") as f:
            return f.read().strip().split("\n")

    @staticmethod
    def _write_names_to_file(names, path):
        with open(path, "w") as f:
            f.write("\n".join(names))

    def evaluate_detections(self, box_list, output_dir):
        dataset = self.datasets["test"]
        write_voc_results_file(output_dir, box_list, dataset)
        return do_python_eval(dataset.anno_path % ("VOC2007", "%s"), dataset.image_sets_file, dataset.class_names, output_dir)
