#!/bin/sh


i=1
FOLDER="data/vel/"
lr_ls=(1e-6 5e-5 1e-4 5e-4 1e-3)
freq_ls=(1 5 10 20)
num_cbf=(64 128 256)


cd .. && pwd &&
for lr in "${lr_ls[@]}";
do
  for freq in "${freq_ls[@]}";
    do
      for num in "${num_cbf[@]}";
        do
          mainfolder="${FOLDER}v$i"
          python scripts/vel_train_airl.py \
           --lr_cbf $lr \
           --cbf_freq $freq \
           --num_states_for_training_each_iter $num \
           --start_cbf_epoch -1 \
           --epoch_num 201 \
           --cbf_pth "data/vel/baselines/cbf" --log_pth "${mainfolder}/log" --share_pth "${mainfolder}/share" --airl_pth "${mainfolder}/airl"
          i=$(($i+1))
        done
    done
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