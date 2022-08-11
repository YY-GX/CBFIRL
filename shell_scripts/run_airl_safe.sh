#!/bin/sh
thearray=(250 500 1000 2000 5000)
cd .. && pwd &&
for i in "${thearray[@]}"; # weight
do
    echo $i
    python scripts/airl_safe.py --fusion_num $i
done