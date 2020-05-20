# -*- coding: utf-8 -*-
import itertools

import numpy as np

import torch
import torch.nn.functional as F

from torch import nn
from torch.autograd import Variable
from torchvision.ops.boxes import batched_nms

from aw_nas.objective.base import BaseObjective
from aw_nas.utils.torch_utils import accuracy, unique
from aw_nas.utils import box_utils

import timeit

class FPNObjective(BaseObjective):
    NAME = "fpn_detection"

    SCHEDULABLE_ATTRS = ["soft_loss_coeff"]

    def __init__(self, search_space, num_classes=21, 
                 pyramid_levels=[3, 4, 5, 6, 7],
                 crop_size=300,
                 top_k=200,
                 scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],
                 ratios=[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)],
                 confidence_thresh=0.05,
                 nms_threshold=0.5, soft_loss_coeff=0., schedule_cfg=None):
        super(FPNObjective, self).__init__(search_space, schedule_cfg)
        self.num_classes = num_classes

        self.anchors = Anchors(pyramid_levels=pyramid_levels, scales=scales, ratios=ratios)
        self.target_transform = TargetTransform(0.5, num_classes)
        self.focal_loss = FocalLoss(self.anchors)
        self.predictor = PredictModel(self.anchors, num_classes, top_k, confidence_thresh, nms_threshold, crop_size)
        self.soft_loss_coeff = soft_loss_coeff

        self.matched_target = []

        # background label: 0
        # FPN includes no backgound label in predictions
        self.all_boxes = [{} for _ in range(self.num_classes + 1)]

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def batch_transform(self, inputs, outputs, annotations):
        """
        annotations: [-1, 4 + 1 + 1 + 2] boxes + labels + ids + shapes
        annotations: boxes, labels, index, height, width
        """

        for _id, ret in self.matched_target:
            if id(annotations) == _id:
                return ret

        device = inputs.device
        img_shape = inputs.shape[-1]
        batch_size = inputs.shape[0]
        anchors = self.anchors.forward(img_shape).to(device)
        num_priors = anchors.shape[0]
        location_t = []
        classification_t = torch.zeros([batch_size, num_priors, self.num_classes]).to(device)

        shapes = []
        for i, (boxes, labels, _id, height, width) in enumerate(annotations):
            labels = labels - 1
            assert all(labels >= 0) and all(labels <= 89)
            conf_t, loc_t = self.target_transform(boxes.to(device), labels.to(device), anchors)
            classification_t[i] = conf_t
            location_t.append(loc_t.to(device))
            shapes.append([_id, height, width])


        ret = classification_t.long().to(device), location_t, shapes
        if len(self.matched_target) >= 1:
            del self.matched_target[0]
            
        self.matched_target.append((id(annotations), ret))
        return ret

    def perf_names(self):
        return ["acc"]

    def get_acc(self, inputs, outputs, targets, cand_net):
        conf_t, loc_t, shapes = self.batch_transform(inputs, outputs, targets)
        """
        target: [batch_size, anchor_num, 5], boxes + labels
        """
        positive_indices = conf_t.sum(-1) > 0
        confidences, regressions = outputs
        confidences = confidences[positive_indices]
        conf_t = conf_t[positive_indices].argmax(-1)

        return [float(a) for a in accuracy(confidences, conf_t, topk=(1, 5))]

    def get_mAP(self, inputs, outputs, targets, cand_net):
        """
        Get mAP.
        """
        confidences, regression = outputs
        # ids, indices = unique(targets[:, 5], True)
        # shapes = targets[indices, -2:].cpu().detach().numpy()
        # heights, widths = shapes[:, 0], shapes[:, 1]
        
        detections = self.predictor(confidences, regression, inputs.shape[-1])
        # for batch_id, (_id, h, w) in enumerate(zip(ids.to(torch.long).tolist(), heights, widths)):
        for batch_id, (_, _, _id, h, w) in enumerate(targets):
            _id = int(_id)
            h = int(h)
            w = int(w)
            for j in range(self.num_classes):
                dets = detections[batch_id][j]
                if len(dets) == 0:
                    continue
                dets[:, 0::2] *= w
                dets[:, 1::2] *= h
                self.all_boxes[j + 1][_id] = dets.cpu().detach().numpy()
        return 0.

    def get_perfs(self, inputs, output, target, cand_net):
        # t0 = timeit.default_timer()
        acc = self.get_acc(inputs, output, target, cand_net)
        # print("\nelapse acc: ", timeit.default_timer() - t0)

        # t0 = timeit.default_timer()
        self.get_mAP(inputs, output, target, cand_net)
        # print("elapse mAP: ", timeit.default_timer() - t0)
        return acc

    def get_reward(self, inputs, outputs, targets, cand_net):
        return self.get_acc(inputs, outputs, targets, cand_net)[0]

    def get_loss(self, inputs, outputs, targets, cand_net,
                 add_controller_regularization=True, add_evaluator_regularization=True):
        """
        Get the cross entropy loss *tensor*, optionally add regluarization loss.

        Args:
            outputs: logits
            targets: labels
        """
        # t0 = timeit.default_timer()
        loss = sum(self._criterion(inputs, outputs, targets, cand_net))
        # print("\nelapse loss: ", timeit.default_timer() - t0)
        return loss

    def _criterion(self, inputs, outputs, annotations, model):
        conf_t, loc_t, shapes = self.batch_transform(inputs, outputs, annotations)
        return self.focal_loss(outputs, (conf_t, loc_t))

    def on_epoch_start(self, epoch):
        super(FPNObjective, self).on_epoch_start(epoch)
        self.search_space.on_epoch_start(epoch)


class TargetTransform(nn.Module):
    def __init__(self, iou_threshold, num_classes):
        super(TargetTransform, self).__init__()
        self.iou_threshold = iou_threshold
        self.num_classes = num_classes

    def forward(self, boxes, labels, anchors):
        num_anchors = anchors.shape[0]

        anchor_widths = anchors[:, 3] - anchors[:, 1]
        anchor_heights = anchors[:, 2] - anchors[:, 0]
        anchor_ctr_x = anchors[:, 1] + 0.5 * anchor_widths
        anchor_ctr_y = anchors[:, 0] + 0.5 * anchor_heights

        if len(boxes) == 0:
            return torch.zeros([num_anchors, self.num_classes]), []
        IoU = box_utils.calc_iou(anchors[:, :], boxes)
        IoU_max, IoU_argmax = torch.max(IoU, dim=1)
        conf_t = torch.ones([num_anchors, self.num_classes]) * -1
        conf_t[torch.lt(IoU_max, 0.4), :] = 0
        positive_indices = torch.ge(IoU_max, 0.5)
        assigned_boxes = boxes[IoU_argmax, :]
        assigned_labels = labels[IoU_argmax]
        conf_t[positive_indices, :] = 0
        conf_t[positive_indices, assigned_labels[positive_indices].long()] = 1

        if positive_indices.sum() > 0:
            assigned_boxes = assigned_boxes[positive_indices, :]
            assigned_labels = assigned_labels[positive_indices]

            anchor_widths_pi = anchor_widths[positive_indices]
            anchor_heights_pi = anchor_heights[positive_indices]
            anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
            anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

            gt_widths = assigned_boxes[:, 2] - assigned_boxes[:, 0]
            gt_heights = assigned_boxes[:, 3] - assigned_boxes[:, 1]
            gt_ctr_x = assigned_boxes[:, 0] + 0.5 * gt_widths
            gt_ctr_y = assigned_boxes[:, 1] + 0.5 * gt_heights

            # efficientdet style
            gt_widths = torch.clamp(gt_widths, min=1)
            gt_heights = torch.clamp(gt_heights, min=1)

            targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
            targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
            targets_dw = torch.log(gt_widths / anchor_widths_pi)
            targets_dh = torch.log(gt_heights / anchor_heights_pi)

            loc_t = torch.stack((targets_dy, targets_dx, targets_dh, targets_dw)).t()
        else:
            loc_t = torch.tensor([])

        return conf_t, loc_t


class PredictModel(nn.Module):
    def __init__(self, anchors, num_classes, top_k=200, confidence_thresh=0.05, nms_thresh=0.5, crop_size=300):
        super(PredictModel, self).__init__()
        self.anchors = anchors 
        self.num_classes = num_classes
        self.top_k = top_k
        self.confidence_thresh = confidence_thresh
        self.nms_thresh = nms_thresh
        self.crop_size = crop_size
        self.bbox_transform = box_utils.BBoxTransform()
        self.cliper = box_utils.ClipBoxes()

    def forward(self, confidences, regressions, img_shape):
        # anchor is not normalized
        anchors = self.anchors(img_shape).to(confidences.device)
        
        num = confidences.shape[0]
        scores = torch.max(confidences, dim=2, keepdim=True)[0]
        scores_over_thresh = (scores > self.confidence_thresh)[:, :, 0]

        output = [[torch.tensor([]) for _ in range(self.num_classes)] for _ in range(num)]
        
        # decode boxes
        decoded_boxes = self.bbox_transform(anchors, regressions)
        decoded_boxes = self.cliper(decoded_boxes, self.crop_size, self.crop_size) / self.crop_size

        for i in range(num):
            scores_over_t = scores_over_thresh[i]
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
                    output[i][idx] = torch.cat(
                        (boxes_[cls_idx][:self.top_k], scores_[cls_idx].unsqueeze(1)[:self.top_k]),
                        1
                    )
        return output
                


class FocalLoss(nn.Module):
    def __init__(self, anchors, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.anchors = anchors

    def forward(self, predict, targets):
        alpha = self.alpha
        gamma = self.gamma
        classifications, regressions = predict
        conf_t, loc_t = targets
        device = classifications.device

        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        # anchor = self.anchors.to(device) # assuming all image sizes are the same, which it is
        # dtype = self.anchors.dtype
        # anchor_widths = anchor[:, 3] - anchor[:, 1]
        # anchor_heights = anchor[:, 2] - anchor[:, 0]
        # anchor_ctr_x = anchor[:, 1] + 0.5 * anchor_widths
        # anchor_ctr_y = anchor[:, 0] + 0.5 * anchor_heights

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            # boxes, labels = annotations[j]
            # labels = labels - 1
            # boxes = torch.tensor(boxes)
            # labels = torch.tensor(labels.reshape(-1, 1))
            # bbox_annotation = torch.cat([boxes, labels], 1).to(dtype).to(device)
            # # bbox_annotation = annotations[j]
            # bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]
            c_t = conf_t[j].to(torch.float)
            l_t = loc_t[j]
            
            if len(l_t) == 0:
                alpha_factor = torch.ones_like(classification) * alpha
                alpha_factor = 1. - alpha_factor
                focal_weight = classification
                focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
                
                bce = -(torch.log(1.0 - classification))
                
                cls_loss = focal_weight * bce
                
                # conf_t.append(torch.from_numpy(np.zeros_like(classification)).to(dtype).to(device).unsqueeze(0))
                # loc_t.append([])
                regression_losses.append(torch.tensor(0.).to(device))
                classification_losses.append(cls_loss.sum())

                continue
            
            # IoU = box_utils.calc_iou(anchor[:, :], bbox_annotation[:, :4])

            # IoU_max, IoU_argmax = torch.max(IoU, dim=1)

            # compute the loss for classification
            # targets = torch.ones_like(classification) * -1

            # targets[torch.lt(IoU_max, 0.4), :] = 0

            # positive_indices = torch.ge(IoU_max, 0.5)

            # num_positive_anchors = positive_indices.sum()

            # assigned_annotations = bbox_annotation[IoU_argmax, :]

            # targets[positive_indices, :] = 0
            # targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            alpha_factor = torch.ones_like(c_t) * alpha

            alpha_factor = torch.where(torch.eq(c_t, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(c_t, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(c_t * torch.log(classification) + (1.0 - c_t) * torch.log(1.0 - classification))

            cls_loss = focal_weight * bce

            zeros = torch.zeros_like(cls_loss)

            cls_loss = torch.where(torch.ne(c_t, -1.0), cls_loss, zeros)
            # ct = torch.ones([1, targets.shape[0]]).long().to(device) * -1
            # if positive_indices.sum() > 0:
            #     ct[0, positive_indices] = targets[positive_indices].argmax(1)
            # conf_t.append(ct)
            

            classification_losses.append(cls_loss.sum() / torch.clamp(c_t.sum(-1).to(torch.bool).sum().to(device), min=1.0))



            # if positive_indices.sum() > 0:
            
            # assigned_annotations = assigned_annotations[positive_indices, :]

            # anchor_widths_pi = anchor_widths[positive_indices]
            # anchor_heights_pi = anchor_heights[positive_indices]
            # anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
            # anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

            # gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
            # gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
            # gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
            # gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

            # efficientdet style
            # gt_widths = torch.clamp(gt_widths, min=1)
            # gt_heights = torch.clamp(gt_heights, min=1)

            # targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
            # targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
            # targets_dw = torch.log(gt_widths / anchor_widths_pi)
            # targets_dh = torch.log(gt_heights / anchor_heights_pi)

            # targets = torch.stack((targets_dy, targets_dx, targets_dh, targets_dw))
            # targets = targets.t()
            # import ipdb; ipdb.set_trace()
            positive_indices = (c_t.sum(-1) > 0)
            regression_diff = torch.abs(l_t - regression[positive_indices, :])

            regression_loss = torch.where(
                torch.le(regression_diff, 1.0 / 9.0),
                0.5 * 9.0 * torch.pow(regression_diff, 2),
                regression_diff - 0.5 / 9.0
            )
            regression_losses.append(regression_loss.mean())
            # loc_t.append(targets)
            # else:
            #     regression_losses.append(torch.tensor(0).to(dtype).to(device))
            #     loc_t.append(np.array([0, 4]))

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), \
            torch.stack(regression_losses).mean(dim=0, keepdim=True)


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

        self.anchors = {}
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

        if dtype == torch.float16:
            self.dtype = np.float16
        else:
            self.dtype = np.float32

        if image_shape in self.anchors:
            return self.anchors[image_shape]

        boxes_all = []
        step = None
        for stride in self.strides:
            if step is None:
                step = np.ceil(image_shape / stride), np.ceil(image_shape / stride)
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

                boxes_level.append(np.expand_dims(boxes, axis=1))
            # concat anchors on the same level to the reshape NxAx4
            boxes_level = np.concatenate(boxes_level, axis=1)
            boxes_all.append(boxes_level.reshape([-1, 4]))

        anchor_boxes = np.vstack(boxes_all)

        anchor_boxes = torch.from_numpy(anchor_boxes.astype(self.dtype))

        # save it for later use to reduce overhead
        self.anchors[image_shape] = anchor_boxes
        return anchor_boxes

