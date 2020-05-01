# -*- coding: utf-8 -*-
"""
A cell-based model whose architecture is described by a genotype.
"""

from __future__ import print_function

import re
from collections import defaultdict

import numpy as np
import six
import torch
import torch.nn.functional as F
from aw_nas import ops, utils
from aw_nas.common import genotype_from_str, group_and_sort_by_to_node
from aw_nas.final.base import FinalModel
from aw_nas.final.ofa_model import OFAGenotypeModel
from aw_nas.ops import MobileNetV3Block
from aw_nas.utils import (RegistryMeta, box_utils, make_divisible, nms, weights_init)
from aw_nas.utils.common_utils import Context, nullcontext
from aw_nas.utils.exception import ConfigException, expect
from torch import nn


def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, relu6=False):
    act_fn = nn.ReLU6 if relu6 else nn.ReLU
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(in_channels),
        act_fn(),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )

def generate_headers(num_classes, feature_channels, expansions=[0.2, 0.25, 0.5, 0.25], channels=[1280, 512, 256, 256, 64], aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]], device=None, **kwargs):
    channels = [feature_channels] + channels
    multi_ratio = [len(r) * 2 + 2 for r in aspect_ratios]
    extras = nn.ModuleList([
        MobileNetV3Block(exp, in_channels, out_channels, stride=2, affine=True, kernel_size=3, activation='relu')
        for exp, in_channels, out_channels in zip(expansions, channels[1:-1], channels[2:])
        ])

    regression_headers = nn.ModuleList([
        SeperableConv2d(in_channels, out_channels=ratio * 4, kernel_size=3, padding=1) for in_channels, ratio in zip(channels, multi_ratio)
    ])

    classification_headers = nn.ModuleList([
        SeperableConv2d(in_channels, out_channels=ratio * num_classes, kernel_size=3, padding=1) for in_channels, ratio in zip(channels, multi_ratio)
    ])

    return extras, regression_headers, classification_headers


class PredictModel(nn.Module):
    def __init__(self, num_classes, background_label, top_k=200, confidence_thresh=0.01, nms_thresh=0.45, variance=(0.1, 0.2)):
        super(PredictModel, self).__init__()
        self.num_classes = num_classes
        self.background_label = background_label
        self.top_k = top_k
        self.confidence_thresh = confidence_thresh
        self.nms_thresh = nms_thresh
        self.variance = variance

    def forward(self, confidences, locations, priors):
        num = confidences.size(0)  # batch size
        num_priors = priors.size(0)
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = confidences.view(num, num_priors,
                                      self.num_classes).transpose(2, 1)

        for i in range(num):
            decoded_boxes = box_utils.decode(locations[i],
                                             priors, self.variance)
            conf_scores = conf_preds[i].clone()

            for cls_idx in range(1, self.num_classes):
                c_mask = conf_scores[cls_idx].gt(self.confidence_thresh)
                scores = conf_scores[cls_idx][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                # ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                box_prob = torch.cat([boxes, scores.view(-1, 1)], 1)
                ids = nms(box_prob.cpu().detach().numpy(),
                          self.nms_thresh, self.top_k)
                output[i, cls_idx, :len(ids)] = \
                    torch.cat((scores[ids].unsqueeze(1),
                               boxes[ids]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output


class SSDHeadModel(nn.Module):

    def __init__(self, device, num_classes=10, extras=None, regression_headers=None, classification_headers=None):
        super(SSDHeadModel, self).__init__()
        self.device = device
        self.num_classes = num_classes

        self.extras = extras
        self.regression_headers = regression_headers
        self.classification_headers = classification_headers
        expect(None not in [extras, regression_headers, classification_headers], 'Extras, regression_headers and classification_headers must be provided, got None instead.', ConfigException)

        self._init_weights()

    def forward(self, features, output):
        expect(isinstance(features, (list, tuple)), 'features must be a series of feature.', ValueError)
        x = output
        for extra in self.extras:
            x = extra(x)
            features.append(x)
        expect(len(features) == len(self.regression_headers) == len(self.classification_headers),
            'features and headers must have the exactly same length, got {}, {}, {} instead.'.format(len(features), len(self.regression_headers), len(self.classification_headers)), ValueError)

        confidences = []
        locations = []
        for feat, l, c in zip(features, self.regression_headers, self.classification_headers):
            locations.append(l(feat).permute(0, 2, 3, 1).contiguous())
            confidences.append(c(feat).permute(0, 2, 3, 1).contiguous())
        locations = torch.cat([t.view(t.size(0), -1) for t in locations], 1).view(x.shape[0], -1, 4)
        confidences = torch.cat([t.view(t.size(0), -1) for t in confidences], 1).view(x.shape[0], -1, self.num_classes)
        return confidences, locations

    def _init_weights(self):
        self.extras.apply(weights_init)
        self.regression_headers.apply(weights_init)
        self.classification_headers.apply(weights_init)


class SSDFinalModel(FinalModel):
    NAME = "ssd_final_model"
    SCHEDULABLE_ATTRS = []

    def __init__(self, search_space, device,
                 backbone_type,
                 backbone_cfg,
                 feature_stages=[4, -1],
                 backbone_state_dict_path=None,
                 head_type='ssd_head_final_model',
                 head_cfg={},
                 num_classes=10,
                 is_test=False,
                 schedule_cfg=None):
        super(SSDFinalModel, self).__init__(schedule_cfg=schedule_cfg)
        self.search_space = search_space
        self.device = device
        self.num_classes = num_classes
        self.feature_stages = feature_stages

        self.final_model = RegistryMeta.get_class('final_model', backbone_type)(search_space, device, num_classes=num_classes, schedule_cfg=schedule_cfg, **backbone_cfg)
        if backbone_state_dict_path:
            self._load_base_net(backbone_state_dict_path)


        first_stage_channel = self.final_model.backbone.channels[feature_stages[0] + 1]

        extras, regression_headers, classification_headers = generate_headers(num_classes, first_stage_channel, device=device, **head_cfg)
        self.norm = ops.L2Norm(first_stage_channel, 20)
        self.head = SSDHeadModel(device, num_classes=num_classes,
                                            extras=extras, regression_headers=regression_headers, classification_headers=classification_headers)


        self.search_space = search_space
        self.device = device
        self.num_classes = num_classes
        self.is_test = is_test
        self.to(self.device)

        # for flops calculation
        self.total_flops = 0
        self._flops_calculated = False
        self.set_hook()


    def _load_base_net(self, backbone_state_dict_path):
        state_model = torch.load(backbone_state_dict_path, map_location=torch.device('cpu'))
        if 'classifier.weight' in state_model:
            del state_model['classifier.weight']
        if 'classifier.bias' in state_model:
            del state_model['classifier.bias']
        res = self.final_model.backbone.load_state_dict(state_model, strict=False)
        print('load base_net weight successfully.', res)

    def _load_head(self, head_state_dict_path):
        state_model = torch.load(head_state_dict_path, map_location=torch.device('cpu'))
        res = self.head.load_state_dict(state_model, strict=True)
        print('load head_net weight successfully.', res)

    def set_hook(self):
        for name, module in self.named_modules():
            module.register_forward_hook(self._hook_intermediate_feature)

    def _hook_intermediate_feature(self, module, inputs, outputs):
        if not self._flops_calculated:
            if isinstance(module, nn.Conv2d):
                self.total_flops += 2* inputs[0].size(1) * outputs.size(1) * \
                                    module.kernel_size[0] * module.kernel_size[1] * \
                                    outputs.size(2) * outputs.size(3) / module.groups
            elif isinstance(module, nn.Linear):
                self.total_flops += 2 * inputs[0].size(1) * outputs.size(1)
        else:
            pass

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def forward(self, inputs): #pylint: disable=arguments-differ
        # features, output = self.final_model.get_det_features(inputs, self.feature_stages)
        features = []
        backbone = self.final_model.backbone
        feature = backbone.stem(inputs)

        for i, cell in enumerate(backbone.cells):
            for j, block in enumerate(cell):
                feature = block(feature)
            if i == 4:
                features.append(feature)
        feature = backbone.conv_head(feature)
        features.append(feature)
        features[0] = self.norm(features[0])
        confidences, locations = self.head(features, feature)

        return confidences, locations
