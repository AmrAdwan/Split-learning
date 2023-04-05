import os
import pandas as pd
from fanova import fANOVA
from pyrfr import regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter, UniformIntegerHyperparameter
from fanova import visualizer
from fanova.visualizer import Visualizer as Visualizer
import numpy as np

# # Read the data from CSV file
# df_acc = pd.read_csv("./run-configs/accuracy_values.csv")

# # Assuming these are the input feature columns you want to use
# column_names = ['batch_size', 'cut_layer', 'num_ln', 'ln_upscale', 'partition_alpha', 'agg_type', 'lr', 'wd']
# X = df_acc[column_names]
# y = df_acc['accuracy']

# # Filter out columns with NaN values
# filtered_column_names = [col for col in column_names if not np.isnan(X[col]).any()]

# # Update X with filtered columns
# X = df_acc[filtered_column_names]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# scaler = MinMaxScaler()
# y_train_normalized = scaler.fit_transform(y_train.to_numpy().reshape(-1, 1)).reshape(-1)

# cs = ConfigurationSpace()

# # Automatically adjust hyperparameter ranges based on your dataset
# for col in column_names:
#     if np.isnan(X[col]).any():
#         print(f"Skipping hyperparameter '{col}' due to NaN values.")
#         continue

#     if X[col].dtype == np.float64:
#         # Check if min and max are the same for float columns
#         if X[col].min() != X[col].max():
#             cs.add_hyperparameter(UniformFloatHyperparameter(col, X[col].min(), X[col].max()))
#     elif X[col].dtype == np.int64:
#         # Check if min and max are the same for integer columns
#         if X[col].min() != X[col].max():
#             cs.add_hyperparameter(UniformIntegerHyperparameter(col, X[col].min(), X[col].max()))



# # Perform fANOVA test
# fanova = fANOVA(X_train.to_numpy(), y_train_normalized, config_space=cs)

# importances = {}

# for i, feature in enumerate(column_names):
#     importance = fanova.quantify_importance((i,))
#     importances[feature] = importance[(i,)]['total importance']

# # print("Feature importances:")
# for feature, importance in importances.items():
#     print(f"Feature {feature}: {importance}")

# output_dir = './output/directory/'
# os.makedirs(output_dir, exist_ok=True)

# vis = Visualizer(fanova, cs, output_dir)
# vis.create_all_plots()



df_acc = pd.read_csv("./run-configs/accuracy_values2.csv")

column_names = ['batch_size','client_num_in_total', 'cut_layer', 'partition_alpha']
X = df_acc[column_names]
y = df_acc['accuracy']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = MinMaxScaler()
y_train_normalized = scaler.fit_transform(y_train.to_numpy().reshape(-1, 1)).reshape(-1)

cs = ConfigurationSpace()

pa_min = df_acc['partition_alpha'].min()
pa_max = df_acc['partition_alpha'].max()
batch_size_min = df_acc['batch_size'].min()
batch_size_max = df_acc['batch_size'].max()
cut_layer_min = df_acc['cut_layer'].min()
cut_layer_max = df_acc['cut_layer'].max()
client_num_in_total_min = df_acc['client_num_in_total'].min()
client_num_in_total_max = df_acc['client_num_in_total'].max()

partition_alpha = UniformFloatHyperparameter("partition_alpha", pa_min, pa_max, log=True)
# batch_size = UniformIntegerHyperparameter("batch_size", 8, 128)
batch_size = UniformIntegerHyperparameter("batch_size", batch_size_min, batch_size_max)
# client_num_in_total = UniformIntegerHyperparameter("client_num_in_total", 1, 19) 
cut_layer = UniformIntegerHyperparameter("cut_layer", cut_layer_min, cut_layer_max)
client_num_in_total = UniformIntegerHyperparameter("client_num_in_total", client_num_in_total_min, client_num_in_total_max)

cs.add_hyperparameters([batch_size, client_num_in_total, cut_layer, partition_alpha])

# f = fANOVA(X_train.to_numpy(), y_train.to_numpy(), config_space=cs)

# Reorder the columns in X_train to match the order of hyperparameters in the configuration space
ordered_hyperparameters = [hp.name for hp in cs.get_hyperparameters()]
X_train = X_train[ordered_hyperparameters]

f = fANOVA(X_train.to_numpy(), y_train_normalized, config_space=cs)

importances = {}

for i, feature in enumerate(column_names):
    importance = f.quantify_importance((i,))
    importances[feature] = importance[(i,)]['total importance']

# print("Feature importances:")
for feature, importance in importances.items():
    print(f"Feature {feature}: {importance}")

output_dir = './output/directory/'
os.makedirs(output_dir, exist_ok=True)
constant = [0,1]
vis = visualizer.Visualizer(f, cs, output_dir)
vis.create_all_plots()