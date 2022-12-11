# Torch and friends import 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
############################
# Modules
import syft as sy
import copy
import numpy as np
import pandas as pd
import time
from ipywidgets import IntProgress

import argparse
import logging
import os
import random
import sys

import wandb

############################
# Helpers 
# from FLDataset import load_dataset, getActualImgs, load_global_test_dataset
# from utils import averageModels, save_df, make_df, make_df_acc
# from cifar10.port import createNN, nn_train, nn_test
# import concurrent.futures
from args_parser import add_args
from data_preprocessing.data_loader import load_data
from models.resnet import resnet56
from models.trainer import ClassificationModelTrainer
from FedAvgApi import FedAvgAPI
from models.centralized_trainer import CentralizedTrainer

if __name__ == "__main__":

    
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    torch.multiprocessing.set_start_method('spawn')
    parser = add_args(argparse.ArgumentParser(description='FedAvg'))
    args = parser.parse_args()
    args.rank = 0
    logger.info(args)
    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    logger.info(device)
    
    wandb.init(
        project="fedml",
        name="Centralized" + str(args.comm_round) + "-e" + str(args.epochs) + "-lr" + str(args.lr),
        config=args
    )

    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True

    dataset = load_data(args, args.dataset)
    model = resnet56(class_num=dataset[7])
    
    logging.info(model)

    single_trainer = CentralizedTrainer(dataset, model, device, args)
    single_trainer.train()

    print("CHECKKING ")


