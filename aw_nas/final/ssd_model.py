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

from torch import nn

from aw_nas import ops, utils
from aw_nas.common import genotype_from_str, group_and_sort_by_to_node
from aw_nas.final.base import FinalModel
from aw_nas.final.det_model import HeadModel, PredictModel
from aw_nas.final.ofa_model import OFAGenotypeModel
from aw_nas.ops import MobileNetV3Block
from aw_nas.utils import (RegistryMeta, box_utils, make_divisible, nms, feature_level_to_stage_index)
from aw_nas.utils.common_utils import Context, nullcontext
from aw_nas.utils.exception import ConfigException, expect



def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, relu6=False):
    act_fn = nn.ReLU6 if relu6 else nn.ReLU
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(in_channels),
        act_fn(),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )

def generate_headers(num_classes, feature_channels, expansions=[0.2, 0.25, 0.5, 0.25], channels=[512, 256, 256, 64], aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]], device=None, **kwargs):
    channels = feature_channels + channels
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

class SSDHeadFinalModel(FinalModel):
    NAME = "ssd_head_final_model"

    def __new__(cls, device, num_classes, 
                 feature_channels,
                 expansions,
                 channels,
                 aspect_ratios,
                 pretrained_path=None,
                 schedule_cfg=None):
        extras, regression_headers, classification_headers = generate_headers(num_classes, feature_channels, expansions, channels, aspect_ratios, device=device)
        norm = ops.L2Norm(feature_channels[0], 20)
        expect(None not in [extras, regression_headers, classification_headers], 'Extras, regression_headers and classification_headers must be provided, got None instead.', ConfigException)
        head = HeadModel(device, num_classes=num_classes,
                                            extras=extras, regression_headers=regression_headers, classification_headers=classification_headers, norm=norm)
        if pretrained_path:
            head.load_state_dict(torch.load(pretrained_path, "cpu"), strict=False)
        return head


    @classmethod
    def supported_data_types(cls):
        return ["image"]


class SSDFinalModel(FinalModel):
    NAME = "ssd_final_model"
    SCHEDULABLE_ATTRS = []

    def __init__(self, search_space, device,
                 backbone_type,
                 backbone_cfg,
                 feature_levels=[4, 5],
                 supernet_state_dict=None,
                 head_type='ssd_head_final_model',
                 head_cfg={},
                 num_classes=10,
                 is_test=False,
                 schedule_cfg=None):
        super(SSDFinalModel, self).__init__(schedule_cfg=schedule_cfg)
        self.search_space = search_space
        self.device = device
        self.num_classes = num_classes
        self.feature_levels = feature_levels

        genotypes = backbone_cfg.pop("genotypes")
        self.backbone = RegistryMeta.get_class('final_model', backbone_type)(search_space, device, **backbone_cfg)

        feature_channels = self.backbone.get_feature_channel_num(feature_levels)
        self.head = SSDHeadFinalModel(device, num_classes, feature_channels, **head_cfg)

        if supernet_state_dict:
            self.load_supernet_state_dict(supernet_state_dict)
        self.finalize(genotypes)

        self.search_space = search_space
        self.device = device
        self.num_classes = num_classes
        self.is_test = is_test
        self.to(self.device)

        # for flops calculation
        self.total_flops = 0
        self._flops_calculated = False
        self.set_hook()

    def finalize(self, genotypes): 
        self.backbone.finalize(genotypes)
        return self

    def load_supernet_state_dict(self, supernet_state_dict, strict=True):
        model_state = torch.load(supernet_state_dict, "cpu")
        backbone_state = {k[9:]: v for k, v in model_state.items() if k.startswith("backbone")}
        head_state = {k[5:]: v for k, v in model_state.items() if k.startswith("head")}
        self.backbone.load_supernet_state_dict(backbone_state, filter_regex=r"*classifier*")
        self.head.load_state_dict(head_state, strict=strict)
        return self.backbone, self.head
        
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

    def forward(self, inputs):
        features, feature = self.backbone.extract_features(inputs, [4, 5])
        confidences, locations = self.head(features)
        return confidences, locations
