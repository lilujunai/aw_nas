# -*- coding: utf-8 -*-
import timeit

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from aw_nas.objective.base import BaseObjective
from aw_nas.utils.torch_utils import accuracy


class OFAClassificationObjective(BaseObjective):
    NAME = "ofa_classification"

    SCHEDULABLE_ATTRS = ["soft_loss_coeff"]

    def __init__(self, search_space, label_smooth=None, soft_loss_coeff=1.0, latency_coeff=1., reward="add", expect_latency=30, punishment="soft", latency_file=None, schedule_cfg=None):
        super(OFAClassificationObjective, self).__init__(search_space, schedule_cfg)
        self.label_smooth = label_smooth
        self.soft_loss_coeff = soft_loss_coeff
        self.loss_soft = SoftCrossEntropy()
        self.latency_coeff = latency_coeff
        self.latency_file = latency_file
        self.latency_table = []
        self.expect_latency = expect_latency
        self.reward = reward
        self.punishment = punishment
        decode = lambda x:[int(x[0]), int(x[1]), int(x[2]), int(x[3]), float(x[4])]
        if self.latency_file is not None:
            with open(self.latency_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    self.latency_table.append(decode(line.split(" ")))
        self._criterion = nn.CrossEntropyLoss() if not self.label_smooth \
                          else CrossEntropyLabelSmooth(self.label_smooth)

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def perf_names(self):
        return ["acc"]

    def get_perfs(self, inputs, outputs, targets, cand_net):
        """
        Get top-1 acc.
        """

        return float(accuracy(outputs, targets)[0]) / 100, 

    def get_addition_reward(self, perf):
        latency_coeff = self.latency_coeff
        if self.punishment == "hard" and self.expect_latency > perf[2]:
            return perf[0] + self.expect_latency / (self.expect_latency + 1) * latency_coeff
        return perf[0] + self.expect_latency / (perf[2] + 1) * latency_coeff

    def get_mult_reward(self, perf, log=False):
        latency_coeff = self.latency_coeff
        if self.punishment == "hard" and self.expect_latency > perf[2]:
            latency_coeff = 0

        if not log:
            return perf[0] * ((self.expect_latency / perf[2]) ** latency_coeff)
        return perf[0] * ((self.expect_latency / np.log(1 + perf[2])) ** latency_coeff)

    def get_reward(self, inputs, outputs, targets, cand_net):
        perf = self.get_perfs(inputs, outputs, targets, cand_net)
        return perf[0]

    def get_loss(self, inputs, outputs, targets, cand_net,
                 add_controller_regularization=True, add_evaluator_regularization=True):
        """
        Get the cross entropy loss *tensor*, optionally add regluarization loss.

        Args:
            outputs: logits
            targets: labels
        """
        loss = self._criterion(outputs, targets)
        if self.soft_loss_coeff > 0:
            with torch.no_grad():
                outputs_all = cand_net.super_net(inputs).detach()
            
            soft = self.loss_soft(outputs, outputs_all)
            loss2 = loss + soft * self.soft_loss_coeff
            return loss2
        return loss

    def on_epoch_start(self, epoch):
        super(OFAClassificationObjective, self).on_epoch_start(epoch)
        self.search_space.on_epoch_start(epoch)

class SoftCrossEntropy(nn.Module):
    def __init__(self):
        super(SoftCrossEntropy, self).__init__()

    def forward(self, inputs, targets):
        log_likelihood = -F.log_softmax(inputs, dim=1)
        likelihood = F.softmax(targets, dim=1)
        sample_num, class_num = targets.shape
        loss = torch.sum(torch.mul(log_likelihood, likelihood)) / sample_num
        return loss

class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        num_classes = int(inputs.shape[-1])
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss
