# -*- coding: utf-8 -*-
"""
A cell-based model whose architecture is described by a genotype.
"""

from __future__ import print_function

import torch
from torch import nn

from aw_nas.common import genotype_from_str
from aw_nas.final.base import FinalModel
from aw_nas.weights_manager.ofa_backbone import BaseBackboneArch


class OFAGenotypeModel(FinalModel):
    NAME = "ofa_final_model"
    def __init__(self, search_space, device, genotypes,
                 backbone_type="mbv2_backbone",
                 backbone_cfg=None,
                 ofa_state_dict=None,
                 num_classes=10,
                 schedule_cfg=None):
        super(OFAGenotypeModel, self).__init__(schedule_cfg)

        self.search_space = search_space
        self.device = device
        self.ofa_state_dict = ofa_state_dict
        assert isinstance(genotypes, str)
        self.genotypes = list(genotype_from_str(genotypes, self.search_space)._asdict().values())

        self.num_classes = num_classes
        self.backbone_cfg = backbone_cfg or {}

        self.depth, self.width, self.kernel = self.parse(self.genotypes)
        self.backbone_type = backbone_type
        self.backbone_cfg = backbone_cfg
        self.backbone = self.load_geno_state_dict(
            ofa_state_dict, self.depth, self.width, self.kernel)

        self.to(self.device)

        # for flops calculation
        self.total_flops = 0
        self._flops_calculated = False
        self.set_hook()

    def forward(self, inputs):
        res = self.backbone(inputs)
        if not self._flops_calculated:
            self.logger.info("FLOPS: flops num = %d M", self.total_flops/1.e6)
            self._flops_calculated = True
        return res

    def get_features(self, inputs, p_levels=(4, 5)):
        return self.backbone.get_features(inputs, p_levels)
    
    def get_feature_channel_num(self, p_level):
        return self.backbone.get_feature_channel_num(p_level)
    
    def load_state_dict(self, model, strict=True):
        keys = model.keys()
        for key in keys:
            if key.startswith('backbone'):
                return super().load_state_dict(model, strict)
            else:
                return self.backbone.load_state_dict(model, strict)

    def load_ofa_state_dict(self, ofa_state_dict, strict=True):
        """
        ofa_state_dict includes all params and weights of FlexibileArch
        """
        flexible_backbone = BaseBackboneArch.get_class_(self.backbone_type)(
            device=self.device, **self.backbone_cfg)
        if ofa_state_dict is not None:
            state_dict = torch.load(ofa_state_dict, map_location="cpu")
            new_state_dict = {}
            for k, v in state_dict.items():
                if ".flex_bn." in k:
                    k = k.replace(".flex_bn.", ".")
                new_state_dict[k] = v
            state_dict = state_dict.get("weights_manager", state_dict)
            flexible_backbone.load_state_dict(state_dict, strict=strict)
        return flexible_backbone

    def load_geno_state_dict(self, ofa_state_dict, depth, width, kernel, strict=True):
        flexible_backbone = self.load_ofa_state_dict(ofa_state_dict, strict)
        backbone = flexible_backbone.finalize(depth, width, kernel)
        return backbone

    def set_hook(self):
        for _, module in self.named_modules():
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

    def parse(self, genotype):
        depth = genotype[:len(self.search_space.num_cell_groups)]
        width = []
        kernel = []
        ind = len(self.search_space.num_cell_groups)
        for i, max_depth in zip(depth, self.search_space.num_cell_groups):
            width_list = []
            kernel_list = []
            for j in range(max_depth):
                if j < i:
                    try:
                        width_list.append(genotype[ind][0])
                        kernel_list.append(genotype[ind][1])
                    except Exception:
                        width_list.append(genotype[ind])
                        kernel_list.append(3)
                ind += 1
            width.append(width_list)
            kernel.append(kernel_list)
        return depth, width, kernel

    def layer_idx_to_named_modules(self, idx):
        stage_idx, block_idx = idx
        prefix = f"backbone.cells.{stage_idx}.{block_idx}"
        m = self
        for name in prefix.split('.'):
            m = getattr(m, name)
        for n, sub_m in m.named_modules():
            if not n:
                yield prefix
            yield '.'.join([prefix, n])
