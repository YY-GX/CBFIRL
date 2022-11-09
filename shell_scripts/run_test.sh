#!/bin/sh

i_array=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
i_array=(1 2 3 4 5 6 7 8 9 10)
#i_array=(11 12 13 14 15 16 17 18 19 20)


# 8obs
FOLDER="data/8obs/cbf_airl_2_"
FOLDER="data/8obs/just_airl_"

cd .. && pwd &&
for i in "${i_array[@]}"; # weight
do
    mainfolder="${FOLDER}$i/share"
#    s=$(($i-10))
    python scripts/airl_safe_test.py --policy_path "${mainfolder}" --seed $i --demo_path "src/demonstrations/8obs_acc_farther_target.pkl"
done


# 16obs

FOLDER="data/trpo_cbf/cbf_airl_based_on_airl_with_diff_seeds_"
FOLDER="data/trpo_cbf/cbf_airl_3_try1_"
#FOLDER="data/just_airl/airl_"

#cd .. && pwd &&
#for i in "${i_array[@]}"; # weight
#do
#    mainfolder="${FOLDER}$i/share"
#    python scripts/airl_safe_test.py --policy_path "${mainfolder}" --seed $i
#done
#
