#!/bin/bash


w_cbf=(1e-3 1e-4 1e-5 1e-6)
entro_coeff=(0.001 0.01)
generator_iter=(10)
max_kl=(0.01 0.1)
ent_methods=("regularized")

i=1
FOLDER="data/trpo_cbf/"

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
                    mainfolder="${FOLDER}tune_w_trpo_$i"
                    python scripts/train_trpo_cbf.py \
                     --seed 10 \
                     --cbf_weight $w \
                     --epoch_num 201 \
                     --cbf_pth "data/new_comb_new_demo/cbf_posx_posy/cbf"\
                     --restore_pth "data/trpo_cbf/pretrain_airl/airl"\
                     --is_restore 1 \
                     --is_freeze_discriminator 1 \
                     --is_auto_tuning 1 \
                     --generator_train_itrs $num_gen \
                     --policy_ent_coeff $entro \
                     --max_kl_step $m_kl \
                     --ent_method $ent_m \
                     --is_use_two_step 1 \
                     --log_pth "${mainfolder}/log" --share_pth "${mainfolder}/share" --airl_pth "${mainfolder}/airl"
                    i=$(($i+1))
                  done
              done
          done
      done
  done

