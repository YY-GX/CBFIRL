#!/bin/sh
EPOCH_NUM=20

FOLDER="data/combine_1_2/debugged/no_goal_v2/"

python scripts/train_airl.py --epoch_num 300 --share_pth "${FOLDER}share" --airl_pth "${FOLDER}airl"

for i in {1..5} # weight
do
#  python scripts/train_airl.py --epoch_num $EPOCH_NUM --share_pth "${FOLDER}share" --airl_pth "${FOLDER}airl"
#  echo "-----------------------------------------------------------------------------------------------------------------"
  python scripts/cbf/train_cbf.py --num_agents 17 --share_path "${FOLDER}share" --cbf_path "${FOLDER}cbf" --num $i --is_hundred 0 --goal_reaching_weight 0
  echo "-----------------------------------------------------------------------------------------------------------------"
#  python scripts/train_airl.py --epoch_num 50 --share_pth "${FOLDER}share"
  python scripts/train_airl.py --epoch_num $EPOCH_NUM --share_pth "${FOLDER}share" --airl_pth "${FOLDER}airl"
  echo "-----------------------------------------------------------------------------------------------------------------"
done

python scripts/airl_safe_test.py --policy_path "${FOLDER}share"