# -*- coding: utf-8 -*-
import numpy as np

import torch
import torch.nn.functional as F

from torch import nn
from torch.autograd import Variable

from aw_nas.dataset.prior_model import PriorBox
from aw_nas.dataset.transform import TargetTransform
from aw_nas.final.det_model import PredictModel
from aw_nas.objective.base import BaseObjective
from aw_nas.utils.torch_utils import accuracy
from aw_nas.utils import box_utils

class FPNObjective(BaseObjective):
    NAME = "fpn_detection"

    def __init__(self, search_space, num_classes=21, nms_threshold=0.5):
        super(FPNObjective, self).__init__(search_space)
        self.num_classes = num_classes

        min_dim=300
        feature_maps=[19, 10, 5, 3, 2, 1]
        aspect_ratios=[[2, 3], [2,3], [2,3], [2,3], [2, 3], [2, 3]]
        steps=[16, 32, 64, 100, 150, 300]
        scales=[45, 90, 135, 180, 225, 270, 315]
        clip=True
        center_variance=0.1 
        size_variance=0.2
        self.priors = PriorBox(min_dim, aspect_ratios, feature_maps, scales, steps, (center_variance, size_variance), clip).forward()
        # self.target_transform = TargetTransform(self.priors, 0.5,  (center_variance, size_variance))
        self.box_loss = MultiBoxFocalLoss(num_classes, nms_threshold, True, 0, True, 3, 1 - nms_threshold, False)
        self.predictor = PredictModel(num_classes, 0, 200, 0.01, nms_threshold, priors=self.priors)

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

    def _criterion(self, outputs, targets):
        # boxes, labels = targets
        # loc_t, conf_t = self.target_transform(boxes, labels)
        return self.box_loss(outputs, targets)


class MultiBoxFocalLoss(nn.Module):
    """RetinaNet Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Focal loss for classification
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target, variance=(0.1, 0.2), device=None):
        super(MultiBoxFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = variance
        self.device = device

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds
                conf_data   : torch.size(batch_size,num_priors,num_classes)
                loc_data    : torch.size(batch_size,num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch
                conf_t      : torch.size(batch_size,num_priors)
                loc_t       : torch.size(batch_size,num_priors,4)
        """
        loc_data, conf_data = predictions
        loc_t, conf_t = targets
        num = loc_data.size(0)

        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Focal loss for classification
        alpha = 0.25
        gamma = 2
        loss_c = []
        for image in range(num):
            classification = conf_data[image, :, :]
            label = conf_t[image, :]
            pos_image = label > 0
            num_positive_anchors = pos_image.sum()

            # 如果没有前景的话就直接跳过，把loss置为0。这幅图片就没有东西
            if pos_image.sum() == 0:
                if torch.cuda.is_available():
                    loss_c.append(torch.tensor(0).float().cuda())
                else:
                    loss_c.append(torch.tensor(0).float())
                continue
            
            # 此处存疑，repo里是这么写的，没有用sigmoid，看看效果吧.....
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            # 匹配的位置设置为 1，其余为0，targets的大小为num_priors * num_classes
            targets = torch.zeros(classification.shape)
            targets.scatter_(1, label.unsqueeze(label.dim()).long(), 1)
            if torch.cuda.is_available():
                targets = targets.cuda()

            # 计算weights
            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape).cuda() * alpha
            else:
                alpha_factor = torch.ones(targets.shape) * alpha
            
            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))

            cls_loss = focal_weight * bce

            loss_c.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

        N = num_pos.data.sum()
        loss_l /= N
        # loss_c /= N       # focal loss我完全单独算了
        loss_c = torch.stack(loss_c).mean(dim=0, keepdim=True)
        return loss_l, loss_c
