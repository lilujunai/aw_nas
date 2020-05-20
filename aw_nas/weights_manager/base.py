# -*- coding: utf-8 -*-
"""Base class definition of WeightsManager, CandidateNet."""

import abc
import contextlib

import six
import torch
from torch import nn

from aw_nas import Component, utils
from aw_nas.utils.common_utils import nullcontext
from aw_nas.utils.exception import expect, ConfigException

import timeit

class BaseWeightsManager(Component):
    REGISTRY = "weights_manager"

    def __init__(self, search_space, device, rollout_type, schedule_cfg=None):
        super(BaseWeightsManager, self).__init__(schedule_cfg)

        self.search_space = search_space
        self.device = device
        expect(rollout_type in self.supported_rollout_types(),
               "Unsupported `rollout_type`: {}".format(rollout_type),
               ConfigException) # supported rollout types
        self.rollout_type = rollout_type

    @abc.abstractmethod
    def set_device(self, device):
        """Set the device of the weights manager"""
        pass

    @abc.abstractmethod
    def assemble_candidate(self, rollout):
        """Assemble a candidate net using rollout.
        """
        # eg. return CandidateNet(rollout)

    @abc.abstractmethod
    def step(self, gradients, optimizer):
        """Update the weights manager state using gradients."""

    @abc.abstractmethod
    def save(self, path):
        """Save the state of the weights_manager to `path` on disk."""

    @abc.abstractmethod
    def load(self, path):
        """Load the state of the weights_manager from `path` on disk."""

    @utils.abstractclassmethod
    def supported_rollout_types(cls):
        """Return the accepted rollout-type."""

    @utils.abstractclassmethod
    def supported_data_types(cls):
        """Return the supported data types"""

class CandidateNet(nn.Module):
    def __init__(self, eval_no_grad=True):
        super(CandidateNet, self).__init__()
        self.eval_no_grad = eval_no_grad

    @contextlib.contextmanager
    def begin_virtual(self):
        """Enter a virtual context, in which all the state changes will be reset when exiting.

        For those evaluators that need to conduct virtual update as surrogate steps.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def forward(self, *args, **kwargs): #pylint: disable=arguments-differ
        pass

    @abc.abstractmethod
    def _forward_with_params(self, *args, **kwargs): #pylint: disable=arguments-differ
        pass

    @abc.abstractmethod
    def get_device(self):
        """
        Get the device of the candidate net.
        """

    def _set_mode(self, mode):
        if mode is None:
            return
        if mode == "train":
            self.train()
        elif mode == "eval":
            self.eval()
        else:
            raise Exception("Unrecognized mode: {}".format(mode))

    def forward_with_params(self, inputs, params, mode=None, **kwargs):
        """Forward the candidate net on the data, using the parameters specified in `params`.
        Args:
        Returns:
            output of the last layer.
        """
        self._set_mode(mode)
        return self._forward_with_params(inputs, params, **kwargs)

    def forward_data(self, inputs, targets=None, mode=None, **kwargs):
        """Forward the candidate net on the data.
        Args:
        Returns:
            output of the last layer.
        """
        self._set_mode(mode)

        return self(inputs, **kwargs)

    def forward_queue(self, queue, steps=1, mode=None, **kwargs):
        self._set_mode(mode)

        outputs = []
        for _ in range(steps):
            data = next(queue)
            data = (data[0].to(self.get_device()), data[1].to(self.get_device()))
            outputs.append(self.forward_data(*data, **kwargs))
        return torch.cat(outputs, dim=0)

    def gradient(self, data, criterion=lambda i, l, t: nn.CrossEntropyLoss()(l, t),
                 parameters=None, eval_criterions=None, mode="train",
                 zero_grads=True, return_grads=True, **kwargs):
        """Get the gradient with respect to the candidate net parameters.

        Args:
            parameters (optional): if specificied, can be a dict of param_name: param,
            or a list of parameter name.
        Returns:
            grads (dict of name: grad tensor)
        """
        self._set_mode(mode)

        # t0 = timeit.default_timer()

        if return_grads:
            active_parameters = dict(self.named_parameters())
            if parameters is not None:
                _parameters = dict(parameters)
                _addi = set(_parameters.keys()).difference(active_parameters)
                assert not _addi,\
                    ("Cannot get gradient of parameters that are not active "
                     "in this candidate net: {}")\
                        .format(", ".join(_addi))
            else:
                _parameters = active_parameters
        _, targets = data
        outputs = self.forward_data(*data, **kwargs)
        loss = criterion(data[0], outputs, targets)
        if zero_grads:
            self.zero_grad()
        loss.backward()

        if not return_grads:
            grads = None
        else:
            grads = [(k, v.grad.clone()) for k, v in six.iteritems(_parameters)\
                     if v.grad is not None]

        if eval_criterions:
            eval_res = utils.flatten_list([c(data[0], outputs, targets) for c in eval_criterions])
            return grads, eval_res
        # print("\nelapse gradient: ", timeit.default_timer() - t0)
        return grads

    def train_queue(self, queue, optimizer, criterion=lambda i, l, t: nn.CrossEntropyLoss()(l, t),
                    eval_criterions=None, steps=1, **kwargs):
        assert steps > 0
        # if not steps:
        #     return [None] * len(eval_criterions or [])

        self._set_mode("train")

        average_ans = None
        for _ in range(steps):
            data = next(queue)
            data = (data[0].to(self.get_device()), data[1].to(self.get_device()))
            _, targets = data
            outputs = self.forward_data(*data, **kwargs)
            loss = criterion(data[0], outputs, targets)
            if eval_criterions:
                ans = utils.flatten_list([c(data[0], outputs, targets) for c in eval_criterions])
                if average_ans is None:
                    average_ans = ans
                else:
                    average_ans = [s + a for s, a in zip(average_ans, ans)]
            self.zero_grad()
            loss.backward()
            optimizer.step()

        if eval_criterions:
            return [s / steps for s in average_ans]
        return []

    def eval_queue(self, queue, criterions, steps=1, mode="eval", **kwargs):
        self._set_mode(mode)

        average_ans = None
        context = torch.no_grad if self.eval_no_grad else nullcontext
        with context():
            for _ in range(steps):
                data = next(queue)
                # print("{}/{}\r".format(i, steps), end="")
                data = (data[0].to(self.get_device()), data[1].to(self.get_device()))
                outputs = self.forward_data(data[0], **kwargs)
                ans = utils.flatten_list([c(data[0], outputs, data[1]) for c in criterions])
                if average_ans is None:
                    average_ans = ans
                else:
                    average_ans = [s + a for s, a in zip(average_ans, ans)]
        return [s / steps for s in average_ans]

    def eval_data(self, data, criterions, mode="eval", **kwargs):
        """
        Returns:
           results (list of results return by criterions)
        """
        self._set_mode(mode)

        context = torch.no_grad if self.eval_no_grad else nullcontext
        with context():
            outputs = self.forward_data(data[0], **kwargs)
            return utils.flatten_list([c(data[0], outputs, data[1]) for c in criterions])
