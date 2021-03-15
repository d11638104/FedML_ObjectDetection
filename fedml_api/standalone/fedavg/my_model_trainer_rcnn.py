from __future__ import division, absolute_import
from torch import nn

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer

from utils.utils import *
from utils.datasets import *
from utils.augmentations import *
from utils.transforms import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm
from terminaltables import AsciiTable

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

# rcnn
from collections import namedtuple
from torch.nn import functional as F
from model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator

import torch
from utils import array_tool as at
from torchnet.meter import ConfusionMeter, AverageValueMeter

LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])

def evaluate(model, dataset, iou_thres, conf_thres, nms_thres, img_size, batch_size, device):
    model.eval()

    # Get dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    sample_metrics = []  # List of tuples (TP, confs, pred) (_, imgs, targets)
    for batch_i, (img, bbox_, label_, scale) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        
        if bbox_ is None:
            continue
            
        # build target
        targets = torch.zeros((len(bbox_), 6)).to(device)
        for i in range(len(bbox_)):
            targets[i, 1] = label_[i]
            targets[i, 2] = bbox_[i, 1]
            targets[i, 3] = bbox_[i, 0]
            targets[i, 4] = bbox_[i, 3]
            targets[i, 5] = bbox_[i, 2]

        imgs = Variable(img.type(Tensor), requires_grad=False).to(device)

        with torch.no_grad():
            _bboxes, _labels, _scores = model.predict(imgs, visualize=True,device=device)
            outputs = []
            for i in range(len(_bboxes)):
                if len(_bboxes[i]) == 0:
                    output = None
                else:
                    output = Tensor([_bboxes[i][1], _bboxes[i][0], _bboxes[i][3], _bboxes[i][2], _scores[i], _labels[i]]).to(device)
                outputs.append(output)
            #outputs = Tensor(outputs).to(device)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
    
    if len(sample_metrics) == 0:  # no detections over whole validation set.
        return None
    
    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class

class MyModelTrainer(ModelTrainer):
    def __init__(self, faster_rcnn, args):
        super(MyModelTrainer, self).__init__(faster_rcnn)

        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = 3.
        self.roi_sigma = 1.

        # target creator create gt_bbox gt_label etc as training targets. 
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        self.optimizer = self.faster_rcnn.get_optimizer()

        # indicators for training status
        self.rpn_cm = ConfusionMeter(2)
        self.roi_cm = ConfusionMeter(81)
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss
        self.device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
        
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def forward(self, imgs, bboxes, labels, scale):
        """Forward Faster R-CNN and calculate losses.

        Here are notations used.

        * :math:`N` is the batch size.
        * :math:`R` is the number of bounding boxes per image.

        Currently, only :math:`N=1` is supported.

        Args:
            imgs (~torch.autograd.Variable): A variable with a batch of images.
            bboxes (~torch.autograd.Variable): A batch of bounding boxes.
                Its shape is :math:`(N, R, 4)`.
            labels (~torch.autograd..Variable): A batch of labels.
                Its shape is :math:`(N, R)`. The background is excluded from
                the definition, which means that the range of the value
                is :math:`[0, L - 1]`. :math:`L` is the number of foreground
                classes.
            scale (float): Amount of scaling applied to
                the raw image during preprocessing.

        Returns:
            namedtuple of 5 losses
        """
        #n = bboxes.shape[0]
        #if n != 1:
        #    raise ValueError('Currently only batch size 1 is supported.')

        bboxes = bboxes.view(1, -1, 4)
        labels = labels.view(1, -1)
        _, _, H, W = imgs.shape
        img_size = (H, W)

        features = self.faster_rcnn.extractor(imgs)

        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.faster_rcnn.rpn(features, img_size, scale)

        # Since batch size is one, convert variables to singular form
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        # Sample RoIs and forward
        # it's fine to break the computation graph of rois, 
        # consider them as constant input
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            at.tonumpy(bbox),
            at.tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std)
            
        # NOTE it's all zero because now it only support for batch=1 now
        sample_roi_index = torch.zeros(len(sample_roi)).to(self.device)
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            features,
            sample_roi,
            sample_roi_index)

        # ------------------ RPN losses -------------------#
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            at.tonumpy(bbox),
            anchor,
            img_size)
        gt_rpn_label = at.totensor(gt_rpn_label).long().to(self.device)
        gt_rpn_loc = at.totensor(gt_rpn_loc).to(self.device)
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc,
            gt_rpn_loc,
            gt_rpn_label.data,
            self.rpn_sigma)

        # NOTE: default value of ignore_index is -100 ...
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label, ignore_index=-1)
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label) > -1]
        self.rpn_cm.add(at.totensor(_rpn_score, False), _gt_rpn_label.data.long())

        # ------------------ ROI losses (fast rcnn loss) -------------------#
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4).to(self.device)
        roi_loc = roi_cls_loc[torch.arange(0, n_sample).long(), at.totensor(gt_roi_label).long()]
        gt_roi_label = at.totensor(gt_roi_label).long().to(self.device)
        gt_roi_loc = at.totensor(gt_roi_loc).to(self.device)
        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)

        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.to(self.device))

        self.roi_cm.add(at.totensor(roi_score, False), gt_roi_label.data.long())

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)]

        return LossTuple(*losses)
        
    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()
        
        dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.n_cpu,
            pin_memory=True,
            collate_fn=train_data.collate_fn,
        )

        # train and update
        lr = 1e-3
        params = []
        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': 0.0005}]
        optimizer = torch.optim.SGD(params, momentum=0.9)

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            self.reset_meters()
            for batch_idx, (img, bbox_, label_, scale) in enumerate(dataloader):
                #scale = at.scalar(scale)
                if bbox_ == None:
                    continue
                img, bbox, label = img.cuda().float().to(device), bbox_.cuda().to(device), label_.cuda().to(device)
                optimizer.zero_grad()
                losses = self.forward(img, bbox, label, scale[0])
                losses.total_loss.backward()
                optimizer.step()
                self.update_meters(losses)

                batch_loss.append(losses.total_loss)
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()
        
        metrics_output = evaluate(
                model,
                dataset=test_data,
                iou_thres=0.5,
                conf_thres=0.5,
                nms_thres=0.5,
                img_size=args.img_size,
                batch_size=8,
                device=device
        )
        
        evaluation_metrics = {
            "precision": 0,
            "recall": 0,
            "vmAP": 0,
            "f1": 0,
            }
        
        if metrics_output is not None:
            precision, recall, AP, f1, ap_class = metrics_output
            evaluation_metrics = {
            "precision": precision.mean(),
            "recall": recall.mean(),
            "vmAP": AP.mean(),
            "f1": f1.mean(),
            }
            
            # Print class APs and mAP
            ap_table = [["Index", "Class name", "AP"]]
            for i, c in enumerate(ap_class):
              ap_table += [[c, args.class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
            print(f"---- mAP {AP.mean()}")                
        else:
            print( "---- mAP not measured (no detections found by model)")

        return evaluation_metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False

    def update_meters(self, losses):
        loss_d = {k: at.scalar(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()
        self.roi_cm.reset()
        self.rpn_cm.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}
        
def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = torch.zeros(gt_loc.shape).to(gt_loc.get_device())
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation, 
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float()) # ignore gt_label==-1 for rpn_loss
    return loc_loss
