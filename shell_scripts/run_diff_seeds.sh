#!/bin/sh

i=1
FOLDER="data/new_comb_new_demo/"

w_cbf=(1e-8)
entro_coeff=(0.001)
generator_iter=(10)
max_kl=(0.1)
#ent_methods=("regularized" "max")
seed_ls=(10 10)
ent_methods=("regularized")

cd .. && pwd &&
for w in "${w_cbf[@]}";
  do
    for entro in "${entro_coeff[@]}";
      do
        for num_gen in "${generator_iter[@]}";
          do
            for m_kl in "${max_kl[@]}";
              do
                for ent_m in "${ent_methods[@]}";
                  do
                    for seed in "${seed_ls[@]}";
                      do
#                        diffseed_airl_cbf_try2_
                        mainfolder="${FOLDER}test_repro_$i"
                        python scripts/comb_train_airl_batch.py \
                         --seed $seed \
                         --cbf_weight $w \
                         --epoch_num 10 \
                         --cbf_pth "data/new_comb_new_demo/cbf_posx_posy/cbf"\
                         --restore_pth "data/new_comb_new_demo/baseline/airl"\
                         --is_restore 1 \
                         --is_freeze_discriminator 1 \
                         --is_auto_tuning 1 \
                         --generator_train_itrs $num_gen \
                         --policy_ent_coeff $entro \
                         --max_kl_step $m_kl \
                         --ent_method $ent_m \
                         --is_use_two_step 1 \
                         --log_pth "${mainfolder}/log" --share_pth "${mainfolder}/share" --airl_pth "${mainfolder}/airl"

#                        mainfolder="${FOLDER}diffseed_just_airl_try2_$i"
#                        python scripts/comb_train_airl_batch.py \
#                         --seed $seed \
#                         --cbf_weight 0 \
#                         --epoch_num 20 \
#                         --cbf_pth "data/new_comb_new_demo/cbf_posx_posy/cbf"\
#                         --restore_pth "data/new_comb_new_demo/baseline/airl"\
#                         --is_restore 1 \
#                         --is_freeze_discriminator 0 \
#                         --is_auto_tuning 0 \
#                         --generator_train_itrs 10 \
#                         --policy_ent_coeff 0.0 \
#                         --max_kl_step 0.01 \
#                         --ent_method "no_entropy"\
#                         --is_use_two_step 1\
#                         --log_pth "${mainfolder}/log" --share_pth "${mainfolder}/share" --airl_pth "${mainfolder}/airl"

                        i=$(($i+1))
                      done
                  done
              done
          done
      done
  done
#
#i_array=(1 2 3 4 5 6 7 8 9 10)
##i_array=(1 2 3)
#FOLDER="data/new_comb_new_demo/diffseed_airl_cbf_try2_"
#i=1
#cd .. && pwd &&
#for i in "${i_array[@]}"; # weight
#do
#    mainfolder="${FOLDER}$i/share"
#    python scripts/airl_safe_test.py --policy_path "${mainfolder}" --seed $(($i-1))
#done
#
#i_array=(1 2 3 4 5 6 7 8 9 10)
##i_array=(1 2 3)
#FOLDER="data/new_comb_new_demo/diffseed_just_airl_try2_"
#i=1
#pwd &&
#for i in "${i_array[@]}"; # weight
#do
#    mainfolder="${FOLDER}$i/share"
#    python scripts/airl_safe_test.py --policy_path "${mainfolder}"  --seed $(($i-1))
#done