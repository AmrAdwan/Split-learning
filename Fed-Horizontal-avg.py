# Torch and friends import 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
############################
# Modules
# import syft as sy
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
from models.resnet import resnet56, resnet18
from models.trainer import ClassificationModelTrainer
from FedAvgApi import FedAvgAPI

if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    torch.multiprocessing.set_start_method('spawn')
    parser = add_args(argparse.ArgumentParser(description='FedAvg'))
    args = parser.parse_args()
    args.client_num_per_round = args.client_num_in_total
    

    # Read config file and append configs to args parser
    df = pd.read_csv('./all_runs_config.csv')
    iid_filter = (df['partition_method'] == 0.0)
    non_iid_filter = (df['partition_method'] == 1.0)
    df.loc[iid_filter, 'partition_method'] = 'iid'
    df.loc[non_iid_filter, 'partition_method'] = 'non-iid'

    partition_method, partition_alpha, batch_size, lr, wd, epochs, client_num_in_total, comm_round = list(df.iloc[args.config_id])
    args.partition_method = partition_method
    args.partition_alpha = partition_alpha
    args.batch_size = int(batch_size)
    args.lr = lr
    args.wd = wd
    args.epochs = int(epochs)
    args.client_num_in_total = int(client_num_in_total)
    args.comm_round = int(comm_round)
    logger.info(args)

    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    logger.info(device)

    wandb.init(
        project="fedml",
        name=args.run_name,
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
    # model = resnet56(class_num=dataset[7])
    model = resnet18(class_num=dataset[7])
    
    
    model_trainer = ClassificationModelTrainer(model) 

    logging.info(model)

    fedavgAPI = FedAvgAPI(dataset, device, args, model_trainer)
    fedavgAPI.train()

    print(" DONE ")


