from __future__ import division

### determine the model and dataset to use. then launch the training procedure by calling FedAvgAPI.

import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

from fedml_api.standalone.fedavg.fedavg_api import FedAvgAPI
from fedml_api.standalone.fedavg.my_model_trainer_yolo import MyModelTrainer

# yolo import

from fedml_api.model.cv.yolo_models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.augmentations import *
from utils.transforms import *
from utils.parse_config import *

from terminaltables import AsciiTable

import time
import datetime
import tqdm

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # yolo training settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument('--dataset', type=str, default='COCO', metavar='N',
                        help='dataset used for training')
    parser.add_argument('--class_names', type=str, default="",
                        help='class names of dataset')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu')
    parser.add_argument('--client_num_in_total', type=int, default=5, metavar='NN',
                        help='number of workers in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int, default=5, metavar='NN',
                        help='number of workers')
    return parser


def count_data(args): # load COCO dataset, and split it into client_num equal parts
    # Get data configuration
    data_config = parse_data_config(args.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    args.class_names = load_classes(data_config["names"])
    
    # Get data
    train_data_global = ListDataset(list_path=train_path, arg=args) # load complete training data
    test_data_global = ListDataset(list_path=valid_path, arg=args) # load complete testing data
    # allocate data to every client. EX: no.1~no.1000 to client 1, no.1001~no.2000 to client 2, etc.
    train_label_stats = train_data_global.count_label(args.client_num_in_total)
    test_label_stats = test_data_global.count_label(args.client_num_in_total)
    for i in range(args.client_num_in_total):
        print('client %d' % i + ' training')
        for j in range(80):
            print(args.class_names[j] + ': ' + str(train_label_stats[i][j]))
        print('--------------------------------------------------------------')
        print('client %d' % i + ' testing')
        for j in range(80):
            print(args.class_names[j] + ': ' + str(test_label_stats[i][j]))
        print('--------------------------------------------------------------')

if __name__ == "__main__":
    parser = add_args(argparse.ArgumentParser(description='FedAvg-standalone'))
    args = parser.parse_args()
    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # load data
    count_data(args)
