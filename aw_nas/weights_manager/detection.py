"""
Super net for detection tasks.
"""

import six
import torch
from torch import nn

from aw_nas import utils
from aw_nas.common import assert_rollout_type
from aw_nas.final.base import FinalModel
from aw_nas.ops import *
from aw_nas.utils import data_parallel
from aw_nas.utils.common_utils import make_divisible, nullcontext
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
    NAME = "det_supernet"

    def __init__(
        self,
        search_space,
        device,
        rollout_type,
        feature_levels=[3, 4, 5],
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
            num_classes=1000,
            multiprocess=False,
            gpus=gpus,
            **search_backbone_cfg
        )
        self.multiprocess = multiprocess
        
        # TODO: update SSDHeadModel in ssd_model to be able to accept yaml config
        self.feature_levels = feature_levels
        backbone_stage_channel = self.backbone.backbone.get_feature_channel_num(feature_levels)
        cfg_channels = head_cfg.get("feature_channels", backbone_stage_channel)
        self.glue_conv = nn.ModuleList()
        for c1, c2 in zip(backbone_stage_channel, cfg_channels):
            if c1 != c2:
                self.glue_conv.append(nn.Conv2d(c1, c2, 1))
            else:
                self.glue_conv.append(nn.Identity())

        self.head = FinalModel.get_class_(head_type)(
            device,
            num_classes,
            cfg_channels,
            **head_cfg
        )

        self.to(device)
    
    def _parallelize(self):
        if self.multiprocess:
            net = convert_sync_bn(self).to(self.device)
            object.__setattr__(
                self, "parallel_model", DistributedDataParallel(net, self.gpus, self.gpus[0]).to(self.device)
            )
    
    def forward(self, inputs, rollout=None):
        features, out = self.backbone.get_features(inputs, self.feature_levels, rollout)
        features = [conv(f) for f, conv in zip(features, self.glue_conv)]
        confidences, regression = self.head(features)
        confidences = confidences.sigmoid()
        return confidences, regression

    # ---- APIs ----
    def assemble_candidate(self, rollout):
        return DetectionBackboneCandidateNet(self, rollout)


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
        self.rollout = rollout
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

    def gradient(self, data, criterion=lambda i, l, t: nn.CrossEntropyLoss()(l, t),
                     parameters=None, eval_criterions=None, mode="train",
                            zero_grads=True, return_grads=True, **kwargs):
        if eval_criterions:
            eval_criterions = eval_criterions[1:]
        return super(DetectionBackboneCandidateNet, self).gradient(data, criterion, parameters, eval_criterions, mode=mode,
                            zero_grads=zero_grads, return_grads=return_grads, **kwargs)

    def eval_queue(self, queue, criterions, steps=1, mode="eval", **kwargs):
        self._set_mode(mode)

        average_ans = None
        context = torch.no_grad if self.eval_no_grad else nullcontext
        with context():
            for _ in range(steps):
                data = next(queue)
                # print("{}/{}\r".format(i, steps), end="")
                data = (data[0].to(self.get_device()), data[1])
                outputs = self.forward_data(data[0], **kwargs)
                ans = utils.flatten_list([c(data[0], outputs, data[1]) for c in criterions])
                if average_ans is None:
                    average_ans = ans
                else:
                    average_ans = [s + a for s, a in zip(average_ans, ans)]
        return [s / steps for s in average_ans]