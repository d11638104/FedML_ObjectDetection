from __future__ import division
import torch
from torch import nn

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer

from fedml_api.model.cv.yolo_models import *
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
import logging

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

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        
        if targets is None:
            continue
        
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False).to(device)

        with torch.no_grad():
            loss, outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
    
    if len(sample_metrics) == 0:  # no detections over whole validation set.
        return None
    
    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class

class MyModelTrainer(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args): # define the customized training procedure, update local model
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
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.zero_grad()
        
        batches_done = 1

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (_, imgs, targets) in enumerate(dataloader):
                if imgs == None:
                    continue
                imgs = Variable(imgs.to(device))
                targets = Variable(targets.to(device), requires_grad=False)
                loss, outputs = model(imgs, targets)
                if type(loss) == int:
                    continue
                loss.backward()

                if batches_done % args.gradient_accumulations == 0:
                    # Accumulates gradient before each step
                    optimizer.step()
                    optimizer.zero_grad()

                    batch_loss.append(loss.item())
                batches_done += 1
            if len(batch_loss) > 0:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                logging.info("epoch : {}, loss: {}".format(epoch, sum(batch_loss) / len(batch_loss)))
            else:
                epoch_loss.append(0)
                logging.info("epoch : {}, loss: {}".format(epoch, 0))

    def test(self, test_data, device, args): # define the customized testing procedure, return local evaluation metrics
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
