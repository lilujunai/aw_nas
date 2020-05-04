
from __future__ import print_function


import torch

from torch import nn

from aw_nas.utils import  weights_init
from aw_nas.utils.exception import ConfigException, expect


class HeadModel(nn.Module):
    
    def __init__(self, device, num_classes=10, extras=None, regression_headers=None, classification_headers=None):
        super(HeadModel, self).__init__()
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