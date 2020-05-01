# -*- coding: utf-8 -*-

import os
import pickle
import timeit

import numpy as np
import six
import torch
import torch.nn.functional as F
from aw_nas import utils
from aw_nas.final.base import FinalTrainer
from aw_nas.final.ofa_trainer import OFAFinalTrainer, _warmup_update_lr
from aw_nas.final.ssd_model import PredictModel
from aw_nas.utils import DataParallel, logger, box_utils
from aw_nas.utils.common_utils import nullcontext
from aw_nas.utils.exception import expect
from aw_nas.utils.voc_eval import evaluate_detections
from torch import nn
from torch.autograd import Variable
from torch.utils.data.distributed import DistributedSampler


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target, variance=(0.1, 0.2), device=None):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = variance
        self.device = device

    def forward(self, predictions, targets, priors):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        loc_data, conf_data = predictions
        loc_t, conf_t = targets
        num = loc_data.size(0)

        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = box_utils.log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        tmp = pos.reshape(loss_c.shape)
        loss_c[tmp] = 0  # filter out pos boxes for now


        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c




class SSDFinalTrainer(OFAFinalTrainer): #pylint: disable=too-many-instance-attributes
    NAME = "ssd_final_trainer"

    def __init__(self, model, dataset, device, gpus, objective,#pylint: disable=dangerous-default-value
                 state_dict_path=None,
                 multiprocess=False, use_gn=False,
                 epochs=600, batch_size=96,
                 optimizer_type="SGD", optimizer_kwargs=None,
                 learning_rate=0.025, momentum=0.9,
                 freeze_base_net=False,
                 base_net_lr=1e-4,
                 warmup_epochs=0,
                 optimizer_scheduler={
                     "type": "CosineAnnealingLR",
                     "T_max": 600,
                     "eta_min": 0.001
                 },
                 weight_decay=3e-4, no_bias_decay=False,
                 grad_clip=5.0,
                 auxiliary_head=False, auxiliary_weight=0.4,
                 add_regularization=False,
                 save_as_state_dict=False,
                 workers_per_queue=2,
                 eval_every=10,
                 eval_no_grad=True,
                 eval_dir=None,
                 schedule_cfg=None):

        self.freeze_base_net = freeze_base_net
        self.base_net_lr = base_net_lr
        super(SSDFinalTrainer, self).__init__(model, dataset, device, gpus, objective, state_dict_path,
                multiprocess, use_gn,
                epochs, batch_size,
                optimizer_type, optimizer_kwargs,
                learning_rate, momentum,
                warmup_epochs,
                optimizer_scheduler,
                weight_decay, no_bias_decay,
                grad_clip,
                auxiliary_head, auxiliary_weight,
                add_regularization,
                save_as_state_dict,
                workers_per_queue,
                eval_no_grad,schedule_cfg)

        self._criterion = MultiBoxLoss(self.model.num_classes, 0.5, True, 0, True, 3, 0.5, False)
        self.eval_every = eval_every
        self.predictor = PredictModel(self.model.num_classes, 0, 200, 0.01, 0.45)
        self.softmax = nn.Softmax(dim=-1)
        self.dataset.priors = self.dataset.priors.to(self.device)

        if eval_dir is None:
            eval_dir = os.environ['HOME']
            pid = os.getpid()
            eval_dir = os.path.join(eval_dir, '.voc_exp', str(pid))
            os.makedirs(eval_dir, exist_ok=True)
        self.eval_dir = eval_dir

    def _init_optimizer(self):
        optim_cls = getattr(torch.optim, self.optimizer_type)
        optim_kwargs = {
            "lr": self.learning_rate,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay
        }
        backbone = self.model.final_model
        head = self.model.head
        if not self.freeze_base_net:
            params = self.model.parameters()
        else:
            params = [
                {
                    "params": backbone.parameters(),
                    "lr": self.base_net_lr
                },
                {
                    "params": head.parameters()
                }
            ]
        optim_kwargs.update(self.optimizer_kwargs or {})
        optimizer = optim_cls(params, **optim_kwargs)

        return optimizer

    def evaluate_split(self, split):
        if len(self.gpus) >= 2:
            self._forward_once_for_flops(self.model)
        assert split in {"train", "test"}
        if split == "test":
            queue = self.valid_queue
        else:
            queue = self.train_queue
        acc, obj, perfs = self.infer_epoch(queue, self.parallel_model,
                                           self._criterion, self.device)
        self.logger.info("acc %f ; obj %f ; performance: %s", acc, obj,
                         "; ".join(
                             ["{}: {:.3f}".format(n, v) for n, v in perfs.items()]))
        return acc, obj

    @classmethod
    def supported_data_types(cls):
        return ["image"]

    def train(self):
        # if len(self.gpus) >= 2:
        #     self._forward_once_for_flops(self.model)
        for epoch in range(self.last_epoch+1, self.epochs+1):
            self.epoch = epoch
            self.on_epoch_start(epoch)

            if epoch < self.warmup_epochs:
                _warmup_update_lr(self.optimizer, epoch, self.learning_rate, self.warmup_epochs)
            else:
                if self.scheduler is not None:
                    self.scheduler.step()
            self.logger.info("epoch %d lr %e", epoch, self.optimizer.param_groups[0]["lr"])

            train_acc, train_obj = self.train_epoch(self.train_queue, self.parallel_model,
                                                    self._criterion, self.optimizer,
                                                    self.device, epoch)
            self.logger.info("train_acc %f ; train_obj %f", train_acc, train_obj)

            if epoch % self.eval_every == 0:
                valid_acc, valid_obj, valid_perfs = self.infer_epoch(self.valid_queue,
                                                                    self.parallel_model,
                                                                    self._criterion, self.device)
                self.logger.info("valid_acc %f ; valid_obj %f ; valid performances: %s",
                                valid_acc, valid_obj,
                                "; ".join(
                                    ["{}: {:.3f}".format(n, v) for n, v in valid_perfs.items()]))

            if self.save_every and epoch % self.save_every == 0:
                path = os.path.join(self.train_dir, str(epoch))
                self.save(path)
            self.on_epoch_end(epoch)

        self.save(os.path.join(self.train_dir, "final"))

    def train_epoch(self, train_queue, model, criterion, optimizer, device, epoch):
        expect(self._is_setup, "trainer.setup should be called first")
        cls_objs = utils.AverageMeter()
        loc_objs = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        model.train()

        results = []
        for step, (ids, inputs, target) in enumerate(train_queue):
            inputs = inputs.to(device)
            target = [t.to(device) for t in target]

            optimizer.zero_grad()
            confidence, locations = model.forward(inputs)
            regression_loss, classification_loss = criterion((locations, confidence), target, self.dataset.priors)
            loss = regression_loss + classification_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            optimizer.step()
            keep = (target[1] > 0).reshape(-1)
            prec1, prec5 = utils.accuracy(confidence.reshape(-1, confidence.shape[-1])[keep], target[1].reshape(-1)[keep], topk=(1, 5))

            n = inputs.size(0)
            cls_objs.update(classification_loss.item(), n)
            loc_objs.update(regression_loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            if step % self.report_every == 0:
                self.logger.info("train %03d %.3f %.3f; %.2f%%; %.2f%%",
                                 step, cls_objs.avg, loc_objs.avg, top1.avg, top5.avg)
        return top1.avg, cls_objs.avg + loc_objs.avg


    def infer_epoch(self, valid_queue, model, criterion, device):
        expect(self._is_setup, "trainer.setup should be called first")
        cls_objs = utils.AverageMeter()
        loc_objs = utils.AverageMeter()
        top1 = utils.AverageMeter()
        top5 = utils.AverageMeter()
        objective_perfs = utils.OrderedStats()
        model.eval()

        context = torch.no_grad if self.eval_no_grad else nullcontext
        with context():
            num_images = len(valid_queue) * valid_queue.batch_size
            all_boxes = [[[] for _ in range(num_images)] for _ in range(self.model.num_classes)]
            for step, (ids, inputs, target, heights, widths) in enumerate(valid_queue):
                inputs = inputs.to(device)
                labels, gt_locations = target = [t.to(device) for t in target]

                confidence, locations = model.forward(inputs)
                regression_loss, classification_loss = criterion((locations, confidence), target, self.dataset.priors)
                keep = (target[1] > 0).reshape(-1)
                perfs = self._perf_func(inputs, confidence.reshape(-1, confidence.shape[-1])[keep], target[1].reshape(-1)[keep], model)
                objective_perfs.update(dict(zip(self._perf_names, perfs)))
                prec1, prec5 = utils.accuracy(confidence.reshape(-1, confidence.shape[-1])[keep], target[1].reshape(-1)[keep], topk=(1, 5))
                n = inputs.size(0)
                cls_objs.update(classification_loss.item(), n)
                loc_objs.update(regression_loss.item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)

                detections = self.predictor(self.softmax(confidence), locations, self.dataset.priors).data
                for batch_id, (_id, h, w) in enumerate(zip(ids, heights, widths)):
                    for j in range(1, detections.size(1)):
                        dets = detections[batch_id, j, :]
                        mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                        dets = torch.masked_select(dets, mask).view(-1, 5)
                        if dets.size(0) == 0:
                            continue
                        boxes = dets[:, 1:]
                        boxes[:, 0] *= w
                        boxes[:, 2] *= w
                        boxes[:, 1] *= h
                        boxes[:, 3] *= h
                        scores = dets[:, 0].cpu().numpy()
                        cls_dets = np.hstack((boxes.cpu().numpy(),
                                                scores[:, np.newaxis])).astype(np.float32,
                                                                                copy=False)
                        all_boxes[j][_id] = cls_dets

                if step % self.report_every == 0:
                    self.logger.info("valid %03d %e %e %f %f %s", step, cls_objs.avg, loc_objs.avg, top1.avg, top5.avg,
                                     "; ".join(["{}: {:.3f}".format(perf_n, v) \
                                                for perf_n, v in objective_perfs.avgs().items()]))
            with open(self.eval_dir + '/all_boxes.pkl', 'wb') as f:
                pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
            t0 = timeit.default_timer()
            out_dir = os.path.join(self.eval_dir, str(self.epoch))
            os.makedirs(out_dir, exist_ok=True)
            mAP = self.dataset.evaluate_detections(all_boxes, out_dir)
            # mAP = evaluate_detections(all_boxes, out_dir, self.dataset.datasets['test'])
            self.logger.info('eval elapse: {}'.format(timeit.default_timer() - t0))
            self.logger.info("valid mAP: {}".format(mAP))
        return top1.avg, cls_objs.avg + loc_objs.avg, objective_perfs.avgs()
