
from __future__ import print_function


import torch

from torch import nn

from aw_nas.utils import weights_init, nms
from aw_nas.utils import box_utils

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

class EfficientDetHeadModel(nn.Module):
    
    def __init__(self, device, num_classes=10, extras=None, regression_headers=None, classification_headers=None):
        super(EfficientDetHeadModel, self).__init__()
        self.device = device
        self.num_classes = num_classes

        self.extras = extras
        self.regression_headers = regression_headers
        self.classification_headers = classification_headers
        expect(None not in [extras, regression_headers, classification_headers], 'Extras, regression_headers and classification_headers must be provided, got None instead.', ConfigException)

        self._init_weights()

    def forward(self, features):
        '''
        Args:
            features    :这是backbone需要做BiFPN的feature格式————(p3, p4, p5)
        '''
        expect(isinstance(features, (list, tuple)), 'features must be a series of feature.', ValueError)
        output_features = self.extras(features)

        confidences = self.classification_headers(output_features)
        locations = self.regression_headers(output_features)

        confidences = [c.permute(0, 2, 3, 1).contiguous() for c in confidences]
        locations = [l.permute(0, 2, 3, 1).contiguous() for l in locations]
        

        locations = torch.cat([t.view(t.size(0), -1) for t in locations], 1).view(features[0].shape[0], -1, 4)
        confidences = torch.cat([t.view(t.size(0), -1) for t in confidences], 1).view(features[0].shape[0], -1, self.num_classes)
        return confidences, locations

    def _init_weights(self):
        self.extras.apply(weights_init)
        self.regression_headers.apply(weights_init)
        self.classification_headers.apply(weights_init)

class PredictModel(nn.Module):
    def __init__(self, num_classes, background_label, top_k=200, confidence_thresh=0.01, nms_thresh=0.5, variance=(0.1, 0.2), priors=None):
        super(PredictModel, self).__init__()
        self.num_classes = num_classes
        self.background_label = background_label
        self.top_k = top_k
        self.confidence_thresh = confidence_thresh
        self.nms_thresh = nms_thresh
        self.variance = variance
        self.priors = priors
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, confidences, locations, img_shape):
        priors = self.priors(img_shape).to(confidences.device)
        num = confidences.size(0)  # batch size
        num_priors = priors.size(0)
        output = [[torch.tensor([]) for _ in range(self.num_classes)] for _ in range(num)]
        confidences = self.softmax(confidences)
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
                box_prob = torch.cat([boxes, scores.view(-1, 1)], 1)
                ids = nms(box_prob.cpu().detach().numpy(),
                          self.nms_thresh, self.top_k)
                output[i][cls_idx] = torch.cat((boxes[ids], scores[ids].unsqueeze(1)), 1)
                # output[i, cls_idx, :len(ids)] = \
                #     torch.cat((scores[ids].unsqueeze(1),
                #                boxes[ids]), 1)
        return output

