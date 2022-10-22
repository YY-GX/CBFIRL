#!/bin/sh



w_cbf=(10000 7500 5000 2500)
entro_coeff=(0.001 0.01)
generator_iter=(10)
max_kl=(0.01 0.1)
ent_methods=("regularized")
#
#w_cbf=(10000 100000 1000 100)
#entro_coeff=(0.001)
#generator_iter=(10)
#max_kl=(0.01)
#ent_methods=("regularized")

#w_cbf=(0)
#entro_coeff=(0 0.001 0.01)
#generator_iter=(10)
#max_kl=(0 0.001 0.01 0.1)
#ent_methods=("regularized")


w_cbf=(10000 10000)
entro_coeff=(0.01)
generator_iter=(10)
max_kl=(0.01)
ent_methods=("regularized")
seeds=(10)

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
                    mainfolder="${FOLDER}repro_$i"
                    python scripts/train_trpo_cbf.py \
                     --seed 10 \
                     --cbf_weight $w \
                     --epoch_num 10 \
                     --cbf_pth "data/trpo_cbf/pretrain_airl/cbf"\
                     --restore_pth "data/trpo_cbf/pretrain_airl/airl"\
                     --is_restore 1 \
                     --is_freeze_discriminator 1 \
                     --is_auto_tuning 1 \
                     --generator_train_itrs $num_gen \
                     --policy_ent_coeff $entro \
                     --max_kl_step $m_kl \
                     --ent_method $ent_m \
                     --is_use_two_step 0 \
                     --log_pth "${mainfolder}/log" --share_pth "${mainfolder}/share" --airl_pth "${mainfolder}/airl"
                    i=$(($i+1))
                  done
              done
          done
      done
  done

