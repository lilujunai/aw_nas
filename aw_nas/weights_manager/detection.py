"""
Super net for detection tasks.
"""

import torch
from torch import nn

from aw_nas.common import assert_rollout_type
from aw_nas.final.base import FinalModel
from aw_nas.ops import *
from aw_nas.utils import data_parallel
from aw_nas.utils.common_utils import make_divisible
from aw_nas.utils.exception import expect, ConfigException
from aw_nas.utils import DistributedDataParallel
from aw_nas.weights_manager.base import BaseWeightsManager, CandidateNet
from aw_nas.weights_manager.ofa_backbone import BaseBackboneArch

try:
    from aw_nas.utils.SynchronizedBatchNormPyTorch.sync_batchnorm import (
        convert_model as convert_sync_bn,
    )
except ImportError:
    convert_sync_bn = lambda m: m

__all__ = ["DetectionBackboneSupernet"]

class DetectionBackboneSupernet(BaseWeightsManager, nn.Module):
    NAME = "ofa_supernet"

    def __init__(
        self,
        search_space,
        device,
        rollout_type,
        search_backbone_type="ofa_supernet",
        search_backbone_cfg={},
        head_type="ssd_final_model",
        head_cfg={},
        num_classes=21,
        multiprocess=False,
        gpus=tuple(),
        schedule_cfg=None,
    ):
        super(DetectionBackboneSupernet, self).__init__(
            search_space, device, rollout_type, schedule_cfg
        )
        nn.Module.__init__(self)
        self.backbone = BaseWeightsManager.get_class_(search_backbone_type)(
            search_space, device, rollout_type, 
            num_classes=num_classes, 
            multiprocess=False, 
            gpus=gpus, 
            **search_backbone_cfg
        )
        
        # TODO: update SSDHeadModel in ssd_model to be able to accept yaml config
        self.head = FinalModel.get_class_(head_type)(
            device,
            num_classes,
            **head_cfg
        )
    
    def _parallelize(self):
        if self.multiprocess:
            net = convert_sync_bn(self).to(self.device)
            object.__setattr__(
                self, "parallel_model", DistributedDataParallel(net, self.gpus)
            )
    
    def forward(self, inputs, rollout=None):
        features, out = self.backbone.forward_rollout(inputs, rollout)
        return self.head(features, out)

    # ---- APIs ----
    def assemble_candidate(self, rollout):
        return self.backbone.assemble_candidate(self, rollout)

    @classmethod
    def supported_rollout_types(cls):
        return [assert_rollout_type("ofa")]

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def save(self, path):
        torch.save(
        {
            "epoch": self.epoch,
            "state_dict": self.state_dict(),
            # "norms": self.norms
        },
        path,
    )

    def load(self, path):
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        self.load_state_dict(checkpoint["state_dict"])
        self.on_epoch_start(checkpoint["epoch"])

    def step(self, gradients, optimizer):
        self.zero_grad()  # clear all gradients
        named_params = dict(self.named_parameters())
        for k, grad in gradients:
            named_params[k].grad = grad
        # apply the gradients
        optimizer.step()

    def step_current_gradients(self, optimizer):
        optimizer.step()

    def set_device(self, device):
        self.device = device
        self.to(device)


class DetectionBackboneCandidateNet(CandidateNet):
    def __init__(self, super_net, rollout, gpus=tuple()):
        super(DetectionBackboneCandidateNet, self).__init__()
        self.super_net = super_net
        self._device = self.super_net.device
        self.gpus = gpus
        self.multiprocess = self.super_net.multiprocess
    
    def get_device(self):
        return self._device

    def _forward(self, inputs):
        return self.super_net(inputs, self.rollout)

    def forward(self, inputs, single=False):
        if single or not self.gpus or len(self.gpus) == 1:
            return self._forward(inputs)
        if self.multiprocess:
            out = self.super_net.parallel_model.forward(inputs, self.rollout)
        elif len(self.gpus) > 1:
            out = data_parallel(
                self, (inputs,), self.gpus, module_kwargs={"single": True}
            )
        return out