#!/bin/sh
EPOCH_NUM=500

FOLDER="data/cbf_test/no_goal_v2/"

#python scripts/train_airl.py --epoch_num $EPOCH_NUM --share_pth "${FOLDER}share" --airl_pth "${FOLDER}airl"
echo "-----------------------------------------------------------------------------------------------------------------"
python scripts/cbf/train_cbf.py --num_agents 17 --share_path "${FOLDER}share" --cbf_path "${FOLDER}cbf" --training_epoch 20000 --goal_reaching_weight 0
echo "-----------------------------------------------------------------------------------------------------------------"
python scripts/airl_safe_test.py --policy_path "${FOLDER}share"