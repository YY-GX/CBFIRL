#!/bin/sh


i=1
FOLDER="data/new_comb/"


w_cbf=(5e-7)


cd .. && pwd &&
for w in "${w_cbf[@]}";
  do
    mainfolder="${FOLDER}l$i"
    python scripts/comb_train_airl_batch.py \
     --cbf_weight $w \
     --epoch_num 101 \
     --cbf_pth "data/new_comb/baselines/cbf"\
     --restore_pth "data/new_comb/baselines/airl"\
     --is_restore True \
     --is_freeze_discriminator True \
     --log_pth "${mainfolder}/log" --share_pth "${mainfolder}/share" --airl_pth "${mainfolder}/airl"
    i=$(($i+1))
  done


#w_cbf=(1e-8 5e-8 5e-7 1e-6 5e-6 1e-5)  # j
w_cbf=(1e-7 2e-7 3e-7 4e-7)


cd .. && pwd &&
for w in "${w_cbf[@]}";
  do
    mainfolder="${FOLDER}k$i"
    python scripts/comb_train_airl_batch.py \
     --cbf_weight $w \
     --epoch_num 101 \
     --cbf_pth "data/new_comb/baselines/cbf"\
     --restore_pth "data/new_comb/baselines/airl"\
     --is_restore True \
     --is_freeze_discriminator False \
     --log_pth "${mainfolder}/log" --share_pth "${mainfolder}/share" --airl_pth "${mainfolder}/airl"
    i=$(($i+1))
  done




# # test
#i_array=(1 2)
#FOLDER="data/comb/16obs_airl_cbf_b"
#i=1
#
#cd .. && pwd &&
#for i in "${i_array[@]}"; # weight
#do
#    mainfolder="${FOLDER}$i/share"
#    python scripts/airl_safe_test.py --policy_path "${mainfolder}"
#done