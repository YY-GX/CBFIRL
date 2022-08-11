#!/bin/sh

#thearray=(0.01 0.05 0.1 0.5 1 5 10)
thearray=(0.5 1 5 10)

cd .. && pwd &&
for i in "${thearray[@]}"; # weight
do
    echo $i
    python scripts/cbf/train_safe_100_per.py --num_agents 9 --save_path scripts/cbf/models/model_0 --goal_reaching_weight $i
done &&
for i in $(seq 1 10); # iteration
do
    echo $i
done