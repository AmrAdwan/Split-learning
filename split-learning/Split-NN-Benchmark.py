import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from uuid import uuid4
from matplotlib import pyplot as plt
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification
from torch.utils.data import Dataset, DataLoader
from utils import TabularDataset, ServerFeedForwardNN, FeedForwardNN
import math
from torch.autograd import Variable
import random
import copy
import openml
import concurrent.futures

from sklearn import datasets
import argparse

import logging
import os
import sys

import wandb
from args_parser import add_args
from sklearn import datasets

# num_features, datasetid
cc18 = [
    # (82, 40966),
    # (20, 40984),
    (1777, 4134),
    # (1559, 40978),
    # (618, 300),
    (562, 1478),
    # (181, 40670),
    (119, 1486),
    (82, 40966),
    # (73, 1487),
    (65, 28),
    (62, 46)
 ]
def check_accuracy(clients, test_dataloader, server_model):
    with torch.no_grad():
        num_correct = 0
        num_samples = 0
        all_inference_output = []
        all_test_dataloader = []
        all_client_models = []

        for client_name, client in clients.items():
            client_test_dataloader = client.get('test_data_loader')
            all_test_dataloader.append(iter(client_test_dataloader))
            model = client.get('model')
            all_client_models.append(model)

        for y, cont_x, cat_x in test_dataloader:
            inference_output = []
            y  = y.to(device)
            for loader, model in zip(all_test_dataloader, all_client_models):
                cont_x, cat_x = next(loader)
                cat_x = cat_x.to(device)
                cont_x = cont_x.to(device)
                activation = model(cont_x, cat_x)
                inference_output.append(activation)




            if(server_nn_type == 'stack'):
                server_input = torch.cat(inference_output, dim=1).detach().clone()
            else:
                server_input = torch.mean(torch.stack(inference_output), dim=0).detach().clone()
                
            server_input = Variable(server_input, requires_grad=False)
            outputs = server_model(server_input)

            predictions = outputs.max(1)[1]
            y = y.reshape(1,-1)[0].type(torch.long)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print('Acc:: ',float(num_correct)/float(num_samples))
        return float(num_correct)/float(num_samples)

def remove_from_array(arr, items):
    return [item for item in arr if item not in items]

def partition_data(n_feature, n_nets, alpha, columns):
    n_feature = n_feature
    feature_index = [i for i in range(n_feature)]
    proportions = np.random.dirichlet(np.repeat(alpha, n_nets))

    res = np.array([int(p * n_feature) for p in proportions])
    
    while res.sum() != n_feature:
        res[res.argmin()] += 1
    for i, part in enumerate(res):
        if part < 2:
            res[res.argmax()] -= 1
            res[i] += 1
    final_res = []
    print(res)
    cols = np.array(columns)
    for i in res:
        if len(cols) == 0:
            break
        batched = random.sample(list(cols), i)
        final_res.append(batched)
        cols = remove_from_array(cols, batched)
    return final_res  


def distribute_features(dataframe, num_clients, alpha, ignore_columns = []):
    
    features = list(filter(lambda x: x not in ignore_columns, list(dataframe.columns)))
    clients_columns = partition_data(len(features), num_clients, alpha, features)
    dfs = []
    for selected_columns in clients_columns:        
        dfs.append(dataframe[list(selected_columns)])
    return dfs

if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    torch.multiprocessing.set_start_method('spawn')
    parser = add_args(argparse.ArgumentParser(description='SplitLearning'))
    args = parser.parse_args()
    
    # Read config file and append configs to args parser
    df = pd.read_csv('./run-configs/SL_ALL_VARIENT_A22.csv')

    partition_alpha,batch_size,lr,wd,epochs,client_num_in_total,cut_layer,num_ln,agg_type,ln_upscale,random_seed,db_id,config_id = list(df.iloc[args.config_id])
    args.config_id = config_id
    args.partition_alpha = partition_alpha
    args.batch_size = int(batch_size)
    args.lr = lr
    args.wd = wd
    args.epochs = int(epochs)
    args.client_num_in_total = int(client_num_in_total)
    args.cut_layer = int(cut_layer)
    args.num_ln = int(num_ln)
    args.random_seed = int(random_seed)
    num_clients = args.client_num_in_total
    args.agg_type = 'stack' if int(agg_type) == 0 else 'average'
    args.ln_upscale = int(ln_upscale)

    args.desc = 'All varient. nLn fixed. Aggregation check. 1 datasets (200 runs). Seed is fixed.'
    args.dataset_index_id = int(db_id)
    dataset_index_id = args.dataset_index_id
    (_, dataset_id) = cc18[dataset_index_id]
    args.dataset_id = dataset_id
    print(" dataset_id ", dataset_id)

    clients = {}
    batchsize = args.batch_size
    config = { "cut_layer": args.cut_layer }

    num_linear_layers = args.num_ln
    ln_upscale_ce = args.ln_upscale
    alpha = args.partition_alpha
    

    server_nn_type = args.agg_type #'stack' # stack, average



    logger.info(args)

    device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    logger.info(device)
    print(' GPU =========================== ', torch.cuda.is_available())
    wandb.init(
        project="fedml",
        name=args.run_name + '_Config_' + str(args.config_id) + '_DS_' + str(args.dataset_id) + '_Alice_22',
        config=args
    )

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True


    openml_dataset = openml.datasets.get_dataset(
        dataset_id, 
        download_data=True # Do not download the dataset.
        )

    target_label = openml_dataset.default_target_attribute
    ignore_attributes = openml_dataset.ignore_attribute or []
    id_index_attribute = openml_dataset.row_id_attribute
    feature_obj = openml_dataset.features

    ignore_categorical = [id_index_attribute] + ignore_attributes
    categorical_features = [feature.name for _, feature in feature_obj.items() if feature.data_type == 'nominal' and feature.name not in ignore_categorical]
    numerical_features = [feature.name for _, feature in feature_obj.items() if feature.data_type == 'numeric' and feature.name not in ignore_categorical]

    categorical_features.append(target_label)

    # df = pd.read_csv('./40966.csv')
    df = openml_dataset.get_data()[0]
    complete_df = df.copy()


    columns = list(df.columns)

    for drop_col in ignore_categorical:
        if drop_col in columns:
            print('Droping: ', drop_col)
            df.drop(columns=[drop_col], inplace=True)
            columns = list(df.columns)


    # Load data and pre-clean
    df.replace("?", float("NaN"), inplace=True)
    df.dropna(axis=1, inplace=True)
    df.dropna(axis=0, inplace=True)


    numerical_features = [col for col in numerical_features if col in df.columns]

    n_train_samples, n_features = df.shape
    df = sklearn.utils.shuffle(df)
    df_test = df.iloc[:math.floor(n_train_samples * 0.2)]
    df = df.iloc[math.floor(n_train_samples * 0.2):]

    n_train_samples, n_features = df.shape
    n_test_samples, _ = df_test.shape


    # Pre-proccess and initlize test and train datasets
    training_labels = df[target_label]
    training_labels = LabelEncoder().fit_transform(training_labels)
    training_labels = torch.from_numpy(training_labels).type(torch.long)

    testing_labels = df_test[target_label]
    testing_labels = LabelEncoder().fit_transform(testing_labels)
    testing_labels = torch.from_numpy(testing_labels).type(torch.long)


    df.drop(columns=[target_label], inplace=True)
    distributed_dataframes = distribute_features(df, num_clients, alpha, ignore_categorical)

    n_output_classes = len(training_labels.unique())

    # Pre-process test data
    output_feature = [target_label]
    label_encoders = {}
    for cat_col in categorical_features:
        label_encoders[cat_col] = LabelEncoder()
        df_test[cat_col] = label_encoders[cat_col].fit_transform(df_test[cat_col])


    ## Setup client data and params


    clients = {}
    for i, raw_df in enumerate(distributed_dataframes):
        raw_df = raw_df.copy()
        columns = raw_df.columns
        cat_cols =[cat_col for cat_col in columns if cat_col in categorical_features and cat_col not in ignore_categorical]
        
        label_encoders = {}
        for cat_col in cat_cols:
            label_encoders[cat_col] = LabelEncoder()
            raw_df[cat_col] = label_encoders[cat_col].fit_transform(raw_df[cat_col])

        dataset = TabularDataset(data=raw_df, cat_cols=cat_cols, is_client=True)

    #     # Using complete_df here bacause we must set emb layer for all types of a feature.     
        cat_dims = [int(complete_df[col].nunique()) for col in cat_cols]
        emb_dims = [(x, min(50, (x + 1) // 2)) for x in cat_dims]

        cont_columns = [column for column in columns if column not in cat_cols]


        if(server_nn_type == 'stack'):
            ln_dim = len(columns) * ln_upscale_ce
        else:
            ln_dim = len(df.columns)

        model = FeedForwardNN(emb_dims, no_of_cont=len(cont_columns), lin_layer_sizes=np.repeat(ln_dim, num_linear_layers),
                            output_size=n_output_classes, emb_dropout=0.04,
                            lin_layer_dropouts=np.repeat(0.001, num_linear_layers), config=config).to(device)
        dataloader = DataLoader(dataset, batchsize, shuffle=False, num_workers=1)
        iter_dataloader = iter(DataLoader(dataset, batchsize, shuffle=False, num_workers=1))
        client_optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)


        client_test_data = df_test[list(raw_df.columns)]
        client_test_dataset = TabularDataset(data=client_test_data, cat_cols=cat_cols, is_client=True)
        client_test_dataloader = DataLoader(client_test_dataset, batchsize, shuffle=False, num_workers=1)

        clients['client_' + str(i)] = {
            'raw_df': raw_df,
            'dataset': dataset,
            'dataloader': dataloader,
            'iter_dataloader': iter_dataloader,
            'model': model,
            'optimizer': client_optimizer,
            'features': columns,
            'order': i,
            'cat_cols': cat_cols,
            'cont_columns': cont_columns,
            'test_data_loader': client_test_dataloader,
            'ln_dim': ln_dim,
            'activation': None
        }


    # Test data prerequisits

    test_dataset = TabularDataset(data=df_test, cat_cols=categorical_features,
                                output_col=output_feature)
    test_dataloader = DataLoader(test_dataset, batchsize, shuffle=False, num_workers=1)


    # Intialize central server model
    ln_size = 0
    client_net_ln_map = []

    for _, client in clients.items():
        ln_size += client.get('ln_dim')
        client_net_ln_map.append(client.get('ln_dim'))

    # ln_size = num_clients * 100
    if(server_nn_type == 'stack'):
        server_model = ServerFeedForwardNN(input_size=n_features, lin_layer_sizes=np.repeat(ln_size, num_linear_layers),
                                output_size=n_output_classes, emb_dropout=0.04,
                                lin_layer_dropouts=np.repeat(0.001, num_linear_layers), config=config).to(device)
    else:
        ln_size = client.get('ln_dim')
        server_model = ServerFeedForwardNN(input_size=n_features, lin_layer_sizes=np.repeat(ln_size, num_linear_layers),
                            output_size=n_output_classes, emb_dropout=0.001,
                            lin_layer_dropouts=np.repeat(0.001, num_linear_layers), config=config).to(device)



    #  Training
    no_of_epochs = args.epochs
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(server_model.parameters(), lr=args.lr, weight_decay=args.wd)


    for epoch in range(no_of_epochs):
        
        for batch_pointer in range(math.ceil(n_train_samples/batchsize)):        
            activations = []

            for (client_name, client) in clients.items():            
                client_model = client.get('model')
                dataloader = client.get('dataloader')

                
                cont_x, cat_x = dataloader.dataset[batch_pointer * batchsize:(batch_pointer +1) * batchsize]
                client_model.train()
                client.get('optimizer').zero_grad()
                
                cat_x = torch.from_numpy(cat_x).to(device)
                cont_x = torch.from_numpy(cont_x).to(device)
                activation = client_model(cont_x, cat_x)
                
                activations.append(activation)

            # Combine activations and pass it to the server_model
            optimizer.zero_grad()

            if(server_nn_type == 'stack'):
                server_inputs = torch.cat(activations, dim=1).detach().clone()
            else:
                server_inputs = torch.mean(torch.stack(activations), dim=0).detach().clone()

            server_inputs = Variable(server_inputs, requires_grad=True)
            outputs = server_model(server_inputs)

            labels = training_labels[batch_pointer * batchsize:(batch_pointer +1) * batchsize]
            labels = labels.to(device)
            loss = criterion(outputs, labels)            
            
            loss.backward()
            optimizer.step()
            prev_pointer = 0
            for activation, (client_name, client) in zip(activations, clients.items()):
                nn_partition = client.get('ln_dim')

                if(server_nn_type == 'stack'):
                    activation.backward(server_inputs.grad[:,prev_pointer:nn_partition + prev_pointer])
                else:
                    activation.backward(server_inputs.grad)
                prev_pointer += nn_partition
                client.get('optimizer').step()
                
            # print("\r Epoch: {} , Loss {}".format(epoch, loss.item()), end="")
            # Calculate test accuracy after each epoch
        accuracy = check_accuracy(clients, test_dataloader, server_model)
        wandb.log({"Test/Acc": accuracy, "epoch": epoch})
        wandb.log({"Train/Loss": loss.item(), "epoch": epoch})
        print("Epoch: {}, Accuracy: {} ".format(epoch, accuracy), end="")

