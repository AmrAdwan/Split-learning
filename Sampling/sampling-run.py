import numpy as np
import random 
import os 
import sys
import ast
# dataset_type = [0, 1]
# partition_alpha = [0, 0.5, 1]
# batch_size = [64, 32, 128, 16]
# lr = [0.001, 0.01, 0.1]
# wd = [0.001, 0.01, 0.1]
# epochs = [5, 10]
# client_num_in_total = [1, 3, 5, 10, 20]
# comm_round = [10, 20]

# parameters = [[i, j, k, l, m, n , o, p] for i in dataset_type 
#                  for j in partition_alpha
#                  for k in batch_size
#                  for l in lr
#                  for m in wd
#                  for n in epochs
#                  for o in client_num_in_total
#                  for p in comm_round
#       ]


# all_runs = np.array([random.choice(parameters) for _ in range(100)])
# a_file = open("./all_runs_config.txt", "w")
# # np.savetxt(a_file, np.array(["dataset_type", "partition_alpha", "batch_size", "lr", "wd", "epochs", "client_num", "round"], dtype=str))
# # for row in all_runs:
# #     print(row)
# #     np.savetxt(a_file, row)
# with open('all_runs_config.txt', 'w') as f:
#     for item in all_runs:
#         f.write("%s\n" % list(item))
range_boundary = int(sys.argv[1])
file1 = open('all_runs_config.txt', 'r')
Lines = file1.readlines()
steps = 25
gpu = sys.argv[2]
count = 0
# Strips the newline character
for line in Lines:
    if(range_boundary - steps < count < range_boundary):
        # configs = np.fromiter(line, dtype=np.float)
        configs = ast.literal_eval(line)
        if(configs[0] == 0):
            configs[0] = 'iid'
        else:
            configs[0] = 'non-iid'
        print(configs)
        run_name = 'FedAvg-{}-clients-{}-epochs-{}'.format(configs[0], configs[6], configs[5])
        os.system('python Fed-Horizontal-avg.py --partition_method {} --partition_alpha {} --batch_size {:.0f} --lr {} --wd {} --epochs {:.0f} --client_num_in_total {:.0f} --comm_round {:.0f} --gpu {} --run_name {}'.format(*list(configs), gpu, run_name))
    count +=1
    
    # os.system('python Fed-Horizontal-avg.py --partition_method non-iid --comm_round 50 --gpu 0')
print(' DONE ')