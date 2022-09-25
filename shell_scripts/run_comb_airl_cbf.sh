#!/bin/sh


# finetune lr
#lr_array=(1e-6 1e-5 1e-4 2e-4 3e-4 5e-4 7e-4 9e-4 1e-3 2e-3 5e-3 1e-2)
# y3: 5e-5, y4: 1e-4
#lr_array=(5e-5 1e-4 1e-5 3e-5)
#lr_array=(1e-5 3e-5 5e-5 7e-5 1e-4 3e-4 5e-4)
#lr_array=(5e-5 5e-5)
#FOLDER="data/comb/16obs_airl_cbf_"
#i=1
#
#cd .. && pwd &&
#for lr in "${lr_array[@]}"; # weight
#do
#    echo $lr
#    mainfolder="${FOLDER}d$i"
#    python scripts/comb_train_airl.py --lr_cbf $lr --cbf_pth "data/comb/16obs_airl_cbf_debug/cbf" --log_pth "${mainfolder}/log" --share_pth "${mainfolder}/share" --airl_pth "${mainfolder}/airl"
#    i=$(($i+1))
#done
i=5
FOLDER="data/comb/16obs_airl_cbf_"
lr_ls=(5e-5 1e-4)
freq_ls=(3 5 10 20)
num_cbf=(64 128 256)

lr_ls=(5e-5)
freq_ls=(10)
num_cbf=(256)

cd .. && pwd &&
for lr in "${lr_ls[@]}";
do
  for freq in "${freq_ls[@]}";
    do
      for num in "${num_cbf[@]}";
        do
          mainfolder="${FOLDER}m$i"
          python scripts/comb_train_airl.py \
           --lr_cbf $lr \
           --cbf_freq $freq \
           --num_states_for_training_each_iter $num \
           --start_cbf_epoch -1 \
           --epoch_num 201 \
           --cbf_pth "data/comb/16obs_airl_cbf_debug/cbf" --log_pth "${mainfolder}/log" --share_pth "${mainfolder}/share" --airl_pth "${mainfolder}/airl"
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