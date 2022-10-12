#!/bin/sh


i=1
FOLDER="data/new_comb_new_demo/"


w_cbf=(1e-8)
entro_coeff=(0.001 0.01 0.1 1 10)
generator_iter=(10)
max_kl=(0.01 0.1 1 10)
#ent_methods=("regularized" "max")
ent_methods=("max")

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
                    mainfolder="${FOLDER}TRPO-thresh2.0-more-ent-try-max-$i"
                    python scripts/comb_train_airl_batch.py \
                     --seed 10 \
                     --cbf_weight $w \
                     --epoch_num 201 \
                     --cbf_pth "data/new_comb_new_demo/cbf_posx_posy/cbf"\
                     --restore_pth "data/new_comb_new_demo/baseline/airl"\
                     --is_restore 1 \
                     --is_freeze_discriminator 1 \
                     --is_auto_tuning 1 \
                     --generator_train_itrs $num_gen \
                     --policy_ent_coeff $entro \
                     --max_kl_step $m_kl \
                     --ent_method $ent_m\
                     --is_use_two_step 1\
                     --log_pth "${mainfolder}/log" --share_pth "${mainfolder}/share" --airl_pth "${mainfolder}/airl"
                    i=$(($i+1))
                  done
              done
          done
      done
  done



#w_cbf=(0.003 1e-3 5e-4 1e-4 1e-5 1e-6)  # w files, first try auto tuning
#w_cbf=(7e-7 5e-7 3e-7 1e-7 5e-8 1e-8)  # ww files, i6-i11
#w_cbf=(1e-5 5e-6 1e-6 5e-7 1e-7)  # auto-m2
#
#w_cbf=(1e-5 5e-6 1e-6 5e-7 1e-7)
#
#
#cd .. && pwd &&
#for w in "${w_cbf[@]}";
#  do
#    mainfolder="${FOLDER}auto-unfreeze-$i"
#    python scripts/comb_train_airl_batch.py \
#     --cbf_weight $w \
#     --epoch_num 401 \
#     --cbf_pth "data/new_comb/baselines/cbf"\
#     --restore_pth "data/new_comb_new_demo/baseline/airl"\
#     --is_restore 1 \
#     --is_freeze_discriminator 0 \
#     --is_auto_tuning 1 \
#     --log_pth "${mainfolder}/log" --share_pth "${mainfolder}/share" --airl_pth "${mainfolder}/airl"
#    i=$(($i+1))
#  done



##w_cbf=(1e-8 5e-8 5e-7 1e-6 5e-6 1e-5)  # j
#w_cbf=(1e-7 2e-7 3e-7 4e-7)
#
#
#cd .. && pwd &&
#for w in "${w_cbf[@]}";
#  do
#    mainfolder="${FOLDER}k$i"
#    python scripts/comb_train_airl_batch.py \
#     --cbf_weight $w \
#     --epoch_num 101 \
#     --cbf_pth "data/new_comb/baselines/cbf"\
#     --restore_pth "data/new_comb/baselines/airl"\
#     --is_restore True \
#     --is_freeze_discriminator False \
#     --log_pth "${mainfolder}/log" --share_pth "${mainfolder}/share" --airl_pth "${mainfolder}/airl"
#    i=$(($i+1))
#  done




# # test
#i_array=(1 2 3 4 5)
#FOLDER="data/new_comb_new_demo/auto-"
#i=1
#
#cd .. && pwd &&
#for i in "${i_array[@]}"; # weight
#do
#    mainfolder="${FOLDER}$i/share"
#    python scripts/airl_safe_test.py --policy_path "${mainfolder}"
#done