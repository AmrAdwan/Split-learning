#!/bin/bash

GROUP=$1

# for i in {0..100}
# do
#     let "config_id = (20 * $GROUP) + $i"
#     echo " This is the i: $i run for group: $GROUP ==> run: $config_id"
#     cd $HOME/fed-benchmarks && python ./Fed-Horizontal-avg.py --run_name "ALICE_RUN_03_CONFIG_$config_id" --config_id $config_id --data_dir /home/s2588862/data/cifar10
# done



# Split learning
for i in {0..30}
do
    let "config_id = (30 * $GROUP) + $i"
    echo " This is the i: $i run for group: $GROUP ==> run: $config_id"
    cd $HOME/fed-benchmarks/split-learning && python ./Split-NN-Benchmark.py --config_id $config_id
    # cd $HOME/jupyter/openml-fed/Experiments/split-learning && python ./Split-NN-Benchmark.py --config_id $config_id
    
done



# Split learning Controlled
# for i in {0..25}
# do
#     let "config_id = (25 * $GROUP) + $i"
#     echo " This is the i: $i run for group: $GROUP ==> run: $config_id"
#     # cd $HOME/fed-benchmarks/split-learning && python ./Split-NN-Benchmark.py --dataset_index_id $GROUP --config_id $config_id
#     cd $HOME/fed-benchmarks/split-learning && python ./Controlled-Split-NN-Benchmark.py  --config_id $config_id
    
# done


# Seed variant Split learning
# for i in {0..500}
# do
#     let "config_id = (500 * $GROUP) + $i"
#     echo " This is the i: $i run for group: $GROUP ==> run: $config_id"
#     cd $HOME/fed-benchmarks/split-learning && python ./Split-NN-Benchmark.py --dataset_index_id $GROUP --config_id $config_id
#     # cd $HOME/fed-benchmarks/split-learning && python ./Seed-Variant-Split-NN-Benchmark.py  --config_id $config_id
    
# done