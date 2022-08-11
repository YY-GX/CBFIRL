#!/bin/sh
EPOCH_NUM=50

for i in {1..5} # weight
do
  python scripts/train_airl.py --epoch_num $EPOCH_NUM --share_pth 'data/combine_1_2/v2/share' --airl_pth 'data/combine_1_2/v2/airl'
  echo '-----------------------------------------------------------------------------------------------------------------'
  python scripts/cbf/train_cbf.py --num_agents 17 --share_path 'data/combine_1_2/v2/share' --cbf_path 'data/combine_1_2/v2/cbf' --num $i
  echo '-----------------------------------------------------------------------------------------------------------------'
  python scripts/train_airl.py --epoch_num $EPOCH_NUM --share_pth 'data/combine_1_2/v2/share' --airl_pth 'data/combine_1_2/v2/airl'
  echo '-----------------------------------------------------------------------------------------------------------------'
done