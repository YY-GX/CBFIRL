#!/bin/sh
EPOCH_NUM=50

FOLDER="data/combine_1_2/100_v2/"

for i in {1..5} # weight
do
  python scripts/train_airl.py --epoch_num $EPOCH_NUM --share_pth "${FOLDER}share" --airl_pth "${FOLDER}airl"
  echo "-----------------------------------------------------------------------------------------------------------------"
  python scripts/cbf/train_cbf.py --num_agents 17 --share_path "${FOLDER}share" --cbf_path "${FOLDER}cbf" --num $i --is_hundred True --training_epoch 2000
  echo "-----------------------------------------------------------------------------------------------------------------"
  python scripts/train_airl.py --epoch_num $EPOCH_NUM --share_pth "${FOLDER}share" --airl_pth "${FOLDER}airl"
  echo "-----------------------------------------------------------------------------------------------------------------"
done

python scripts/airl_safe_test.py --policy_path "${FOLDER}share"