#!/bin/bash




seeds=(1 2 3 4 5 6 7 8 9 10)

i=1
FOLDER="data/8obs/"

cd .. && pwd &&
for s in "${seeds[@]}";
  do
    mainfolder="${FOLDER}just_airl_$i"
    python scripts/train_airl.py \
     --seed $s \
     --epoch_num 100 \
     --share_pth "${mainfolder}/share" --airl_pth "${mainfolder}/airl" --demo_pth "src/demonstrations/8obs_acc_farther_target.pkl"
    i=$(($i+1))
  done
