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
from aw_nas.final.det_model import EfficientDetHeadModel, PredictModel
from aw_nas.final.ofa_model import OFAGenotypeModel
from aw_nas.ops import MobileNetV3Block
from aw_nas.utils import (RegistryMeta, box_utils, make_divisible, nms)
from aw_nas.utils.common_utils import Context, nullcontext
from aw_nas.utils.exception import ConfigException, expect
from .ssd_bifpn import BiFPN, Conv2dStaticSamePadding


# TODO: 确认是否要去掉bn
def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, relu6=False):
    act_fn = nn.ReLU6 if relu6 else nn.ReLU
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
               groups=in_channels, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(in_channels),
        act_fn(),
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
    )

class SeparableConvBlock(nn.Module):
    """
    created by Zylo117
    """

    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # Q: whether separate conv
        #  share bias between depthwise_conv and pointwise_conv
        #  or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels,
                                                      kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation
        if self.activation:
            self.swish = MemoryEfficientSwish() if not onnx_export else Swish()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x

class Classifier(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes, num_layers, onnx_export=False):
        super(Classifier, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.conv_list = nn.ModuleList([
                # SeperableConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, relu6=True) for i in range(num_layers)
                SeparableConvBlock(in_channels, in_channels, norm=False, activation=False) for i in range(num_layers)
            ])

        self.bn_list = nn.ModuleList([
            nn.ModuleList([
                nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3) for i in range(num_layers)
            ]) 
            for j in range(5)
        ])
        # self.header = SeperableConv2d(in_channels, num_classes * num_anchors, kernel_size=3, stride=1, padding=1, relu6=True)
        self.header = SeparableConvBlock(in_channels, num_anchors * num_classes, norm=False, activation=False)
        self.swish = ops.get_op("h_swish")(inplace=True) 

    def forward(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for i, bn, conv in zip(range(self.num_layers), bn_list, self.conv_list):
                feat = conv(feat)
                feat = bn(feat)
                feat = self.swish(feat)
            feat = self.header(feat)
            feats.append(feat)

        return feats


def generate_headers_bifpn(num_classes, in_channels, bifpn_out_channels, attention=True, repeat=3, num_layers=4, device=None, **kwargs):
    '''
    Args:
        num_class:          分类类别的数量
        in_channels:        输入BiFPN的特征维度，应该是分别是[p3, p4, p5]
        bifpn_out_channels: BiFPN输出的5个feature map的channel数量
        aspect_ratios:      BiFPN因为有共享权重，所以只有一个
        attention:          BiFPN是否使用feature map加权
        repeat:             BiFPN级联的次数
        device:
    '''
    # extras 可以级联的BiFPN, first_time参数会控制in_channels是否有用, True的时候有用否则没用
    extras = nn.Sequential(
        *[BiFPN(bifpn_out_channels, in_channels, num == 0, epsilon=1e-4, attention=attention) for num in range(repeat)]
    )

    # regression & classification 共享权重，因此只有一个
    # TODO: 他们共享权重，但是应该不共享bn？
    ratios = 9

    regression_headers = Classifier(bifpn_out_channels, ratios, 4, num_layers)
    classification_headers = Classifier(bifpn_out_channels, ratios, num_classes, num_layers)

    return extras, regression_headers, classification_headers


class EfficientDetHeadFinalModel(FinalModel):
    NAME = "efficientdet_head_final_model"

    def __new__(self, device, num_classes, 
                 feature_channels,
                 channels,
                 attention,
                 repeat,
                 num_layers,
                 schedule_cfg=None):
        extras, regression_headers, classification_headers = generate_headers_bifpn(num_classes, feature_channels, channels, attention, repeat, num_layers, device=device)
        self.device = device
        self.num_classes = num_classes

        self.extras = extras
        self.regression_headers = regression_headers
        self.classification_headers = classification_headers
        expect(None not in [extras, regression_headers, classification_headers], 'Extras, regression_headers and classification_headers must be provided, got None instead.', ConfigException)
        return EfficientDetHeadModel(device, num_classes=num_classes,
                                            extras=extras, regression_headers=regression_headers, classification_headers=classification_headers)

    @classmethod
    def supported_data_types(cls):
        return ["image"]

class EfficientDetFinalModel(FinalModel):
    NAME = "efficientdet_final_model"
    SCHEDULABLE_ATTRS = []

    def __init__(self, search_space, device,
                 backbone_type,
                 backbone_cfg,
                 feature_levels=[3, 4, 5],
                 backbone_state_dict_path=None,
                 head_type='efficientdet_head_final_model',
                 head_cfg={},
                 num_classes=10,
                 is_test=False,
                 schedule_cfg=None):
        super(EfficientDetFinalModel, self).__init__(schedule_cfg=schedule_cfg)
        '''
        Args:
            head_cfg:   需要匹配head的内容，可能还有需要改动的
        '''
        self.search_space = search_space
        self.device = device
        self.num_classes = num_classes
        self.feature_levels = feature_levels

        self.final_model = RegistryMeta.get_class('final_model', backbone_type)(search_space, device, num_classes=num_classes, schedule_cfg=schedule_cfg, **backbone_cfg)
        if backbone_state_dict_path:
            self._load_base_net(backbone_state_dict_path)

        # 这里要给出所有的backbone中需要做BiFPN的特征维数，按照[p3, p4, p5]的顺序给出
        backbone_stage_channel = self.final_model.get_feature_channel_num(feature_levels)
        self.head = EfficientDetHeadFinalModel(device, num_classes, backbone_stage_channel, **head_cfg)


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
        # 这一句话就可以表示中间层的feature？？我默认了
        features, _ = self.final_model.get_features(inputs, self.feature_levels)
        
        # 这里的输入是(p3, p4, p5)的feature map
        confidences, locations = self.head(features)
        confidences = confidences.sigmoid()

        return confidences, locations
