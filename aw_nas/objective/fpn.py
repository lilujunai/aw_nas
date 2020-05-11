# -*- coding: utf-8 -*-
import itertools

import numpy as np

import torch
import torch.nn.functional as F

from torch import nn
from torch.autograd import Variable
from torchvision.ops.boxes import batched_nms

from aw_nas.dataset.prior_model import PriorBox
# from aw_nas.final.det_model import PredictModel
from aw_nas.dataset.transform import TargetTransform
from aw_nas.objective.base import BaseObjective
from aw_nas.utils.torch_utils import accuracy
from aw_nas.utils import box_utils

class FPNObjective(BaseObjective):
    NAME = "fpn_detection"

    def __init__(self, search_space, num_classes=21, 
                 pyramid_levels=[3, 4, 5, 6, 7],
                 crop_size=300,
                 scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],
                 ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
                 nms_threshold=0.5):
        super(FPNObjective, self).__init__(search_space)
        self.num_classes = num_classes

        self.anchors = Anchors(pyramid_levels=pyramid_levels, scales=scales, ratios=ratios).forward((crop_size, crop_size))
        self.focal_loss = FocalLoss(self.anchors)
        # self.predictor = PredictModel(num_classes, 0, 200, 0.01, nms_threshold, variance=(1.0, 1.0), priors=self.anchors)
        self.predictor = PredictModel(self.anchors.unsqueeze(0), num_classes, 200, 0.05, nms_threshold, crop_size)

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def perf_names(self):
        return ["acc"]

    def get_perfs(self, inputs, outputs, targets, cand_net):
        """
        Get top-1 acc.
        """
        return [float(accuracy(outputs, targets)[0]) / 100]

    def get_reward(self, inputs, outputs, targets, cand_net):
        return self.get_perfs(inputs, outputs, targets, cand_net)[0]

    def get_loss(self, inputs, outputs, targets, cand_net,
                 add_controller_regularization=True, add_evaluator_regularization=True):
        """
        Get the cross entropy loss *tensor*, optionally add regluarization loss.

        Args:
            outputs: logits
            targets: labels
        """
        raise NotImplementedError

    def _criterion(self, outputs, annotations):
        return self.focal_loss(outputs, annotations)


class PredictModel(nn.Module):
    def __init__(self, anchors, num_classes, top_k=200, confidence_thresh=0.05, nms_thresh=0.5, crop_size=300):
        super(PredictModel, self).__init__()
        self.anchors = anchors 
        self.num_classes = num_classes
        self.top_k = top_k
        self.confidence_thresh = confidence_thresh
        self.nms_thresh = nms_thresh
        self.crop_size = crop_size

    def forward(self, confidences, regressions):
        # anchor is not normalized
        self.anchors = self.anchors.to(confidences.device)
        y_centers_a = (self.anchors[..., 0] + self.anchors[..., 2]) / 2
        x_centers_a = (self.anchors[..., 1] + self.anchors[..., 3]) / 2
        ha = self.anchors[..., 2] - self.anchors[..., 0]
        wa = self.anchors[..., 3] - self.anchors[..., 1]

        num = confidences.shape[0]
        confidences = confidences[:, :, 1:].sigmoid()
        scores = torch.max(confidences, dim=2, keepdim=True)[0]
        scores_over_thresh = (scores > self.confidence_thresh)[:, :, 0]

        output = torch.zeros(num, self.num_classes + 1, self.top_k, 5)
        
        # decode boxes
        # regression = regressions[i]
        w = regressions[..., 3].exp() * wa
        h = regressions[..., 2].exp() * ha

        y_centers = regressions[..., 0] * ha + y_centers_a
        x_centers = regressions[..., 1] * wa + x_centers_a

        ymin = y_centers - h / 2.
        xmin = x_centers - w / 2.
        ymax = y_centers + h / 2.
        xmax = x_centers + w / 2.

        decoded_boxes = torch.stack([xmin, ymin, xmax, ymax], dim=2)

        # clamp size
        decoded_boxes[:, :, 0] = torch.clamp(decoded_boxes[:, :, 0], min=0) / self.crop_size
        decoded_boxes[:, :, 1] = torch.clamp(decoded_boxes[:, :, 1], min=0) / self.crop_size

        decoded_boxes[:, :, 2] = torch.clamp(decoded_boxes[:, :, 2], max=self.crop_size - 1) / self.crop_size
        decoded_boxes[:, :, 3] = torch.clamp(decoded_boxes[:, :, 3], max=self.crop_size - 1) / self.crop_size

        for i in range(num):
            scores_over_t = scores_over_thresh[i, :]
            if scores_over_t.sum() == 0:
                continue
            classification_per = confidences[i, scores_over_t, ...].permute(1, 0)
            decoded_boxes_per = decoded_boxes[i, scores_over_t, ...]
            scores_per = scores[i, scores_over_t, ...]
            scores_, classes_ = classification_per.max(dim=0)
            
            anchors_nms_idx = batched_nms(decoded_boxes_per, scores_per[:, 0], classes_, iou_threshold=self.nms_thresh)

            if anchors_nms_idx.shape[0] != 0:
                classes_ = classes_[anchors_nms_idx]
                scores_ = scores_[anchors_nms_idx]
                boxes_ = decoded_boxes_per[anchors_nms_idx, :]
                for idx in range(self.num_classes):
                    cls_idx = classes_ == idx
                    output[i, idx + 1, :cls_idx.sum()] = torch.cat(
                        (scores_[cls_idx].unsqueeze(1), boxes_[cls_idx]),
                        1
                    )
        return output
                



class FocalLoss(nn.Module):
    def __init__(self, anchors, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha 
        self.gamma = gamma
        self.anchors = anchors

    def forward(self, predict, annotations):
        alpha = self.alpha
        gamma = self.gamma
        regressions, classifications = predict
        classifications = classifications.sigmoid()
        device = classifications.device

        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = self.anchors.to(device) # assuming all image sizes are the same, which it is
        dtype = self.anchors.dtype

        anchor_widths = anchor[:, 3] - anchor[:, 1]
        anchor_heights = anchor[:, 2] - anchor[:, 0]
        anchor_ctr_x = anchor[:, 1] + 0.5 * anchor_widths
        anchor_ctr_y = anchor[:, 0] + 0.5 * anchor_heights

        loc_t = []
        conf_t = []
        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            boxes, labels = annotations[j]
            bbox_annotation = torch.from_numpy(np.concatenate((boxes, labels.reshape(-1, 1)), axis=1)).to(dtype).to(device)
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)
            
            if bbox_annotation.shape[0] == 0:
                alpha_factor = torch.ones_like(classification) * alpha
                alpha_factor = 1. - alpha_factor
                focal_weight = classification
                focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
                
                bce = -(torch.log(1.0 - classification))
                
                cls_loss = focal_weight * bce
                
                conf_t.append(torch.from_numpy(np.zeros_like(classification)).to(dtype).to(device).unsqueeze(0))
                loc_t.append([])
                regression_losses.append(torch.tensor(0).to(dtype).to(device))
                classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.to(dtype), min=1.0))
 
                continue
            
            IoU = box_utils.calc_iou(anchor[:, :], bbox_annotation[:, :4])

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)

            # compute the loss for classification
            targets = torch.ones_like(classification) * -1

            targets[torch.lt(IoU_max, 0.4), :] = 0

            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            alpha_factor = torch.ones_like(targets) * alpha

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            cls_loss = focal_weight * bce

            zeros = torch.zeros_like(cls_loss)

            cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, zeros)
            conf_t.append(targets.argmax(1).unsqueeze(0))

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.to(dtype).to(device), min=1.0))

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                # efficientdet style
                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dy, targets_dx, targets_dh, targets_dw))
                targets = targets.t()

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
                loc_t.append(targets)
            else:
                regression_losses.append(torch.tensor(0).to(dtype).to(device))
                loc_t.append(np.array([0, 4]))

        conf_t = torch.stack(conf_t, 0)
        return torch.stack(regression_losses).mean(dim=0, keepdim=True), \
            torch.stack(classification_losses).mean(dim=0, keepdim=True), \
            (loc_t, conf_t)


class Anchors(nn.Module):
    """
    adapted and modified from https://github.com/google/automl/blob/master/efficientdet/anchors.py by Zylo117
    """

    def __init__(self, anchor_scale=4., pyramid_levels=None, **kwargs):
        super().__init__()
        self.anchor_scale = anchor_scale

        if pyramid_levels is None:
            pyramid_levels = [3, 4, 5, 6, 7]
        self.pyramid_levels = pyramid_levels

        self.strides = kwargs.get("strides", [2 ** x for x in self.pyramid_levels])
        scales = kwargs.get('scales', [0., 1., 2.])
        self.scales = np.array([2 ** (s / 3.) for s in scales])
        self.ratios = kwargs.get("ratios", [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)])

        self.last_anchors = {}
        self.last_shape = None
        self.dtype = np.float32

    def forward(self, image_shape, dtype=torch.float32):
        """Generates multiscale anchor boxes.

        Args:
          image_size: integer number of input image size. The input image has the
            same dimension for width and height. The image_size should be divided by
            the largest feature stride 2^max_level.
          anchor_scale: float number representing the scale of size of the base
            anchor to the feature stride 2^level.
          anchor_configs: a dictionary with keys as the levels of anchors and
            values as a list of anchor configuration.

        Returns:
          anchor_boxes: a numpy array with shape [N, 4], which stacks anchors on all
            feature levels.
        Raises:
          ValueError: input size must be the multiple of largest feature stride.
        """

        # if image_shape == self.last_shape and image.device in self.last_anchors:
        #     return self.last_anchors[image.device]

        # if self.last_shape is None or self.last_shape != image_shape:
        #     self.last_shape = image_shape

        if dtype == torch.float16:
            self.dtype = np.float16
        else:
            self.dtype = np.float32

        boxes_all = []
        step = None
        for stride in self.strides:
            if step is None:
                step = np.ceil(image_shape[0] / stride), np.ceil(image_shape[1] / stride)
            else:
                step = np.ceil(step[0] / 2), np.ceil(step[1] / 2)
            boxes_level = []
            for scale, ratio in itertools.product(self.scales, self.ratios):
                # if image_shape[1] % stride != 0:
                #     raise ValueError('input size must be divided by the stride.')
                base_anchor_size = self.anchor_scale * stride * scale
                anchor_size_x_2 = base_anchor_size * ratio[0] / 2.0
                anchor_size_y_2 = base_anchor_size * ratio[1] / 2.0

                # TODO: 不知道是否正确？
                x = np.arange(stride / 2, step[1] * stride, stride)
                y = np.arange(stride / 2, step[0] * stride, stride)
                xv, yv = np.meshgrid(x, y)
                xv = xv.reshape(-1)
                yv = yv.reshape(-1)
                # y1,x1,y2,x2
                # TODO: anchor没有归一化？怎么办？归一化anchor 还是把boxes变回来？
                boxes = np.vstack((yv - anchor_size_y_2, xv - anchor_size_x_2,
                                   yv + anchor_size_y_2, xv + anchor_size_x_2))
                boxes = np.swapaxes(boxes, 0, 1)
                # boxes[:, ::2] /= image_shape[1]
                # boxes[:, 1::2] /= image_shape[0]

                boxes_level.append(np.expand_dims(boxes, axis=1))
            # concat anchors on the same level to the reshape NxAx4
            boxes_level = np.concatenate(boxes_level, axis=1)
            boxes_all.append(boxes_level.reshape([-1, 4]))

        anchor_boxes = np.vstack(boxes_all)

        anchor_boxes = torch.from_numpy(anchor_boxes.astype(self.dtype))
        # anchor_boxes = anchor_boxes.unsqueeze(0)

        # save it for later use to reduce overhead
        # self.last_anchors[image.device] = anchor_boxes
        return anchor_boxes
