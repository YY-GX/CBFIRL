#!/bin/sh
thearray=(1e-4 1e-1 10)
cd .. && pwd &&
for i in "${thearray[@]}"; # weight
do
    echo $i
    python scripts/cbf/train_safe_flexible_per.py --goal_reaching_weight $i --num_agents 17 --log_path 'data/obs16/03_08_2022_22_59_25'
done